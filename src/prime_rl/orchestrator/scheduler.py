from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import verifiers as vf
from verifiers.utils.logging_utils import print_time

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress
from prime_rl.orchestrator.concurrency import RolloutLimiter
from prime_rl.orchestrator.envs import EvalEnv, TrainEnvs
from prime_rl.orchestrator.vf_utils import get_seq_len
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool, client_identity
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RolloutGroup:
    """Pure state for a rollout group (one example x N rollouts)."""

    env_name: str
    example: dict
    requires_group_scoring: bool
    rollouts_per_example: int
    group_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    client: vf.ClientConfig | None = None
    off_policy_steps: int = 0
    completed: list[vf.RolloutOutput] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.completed) >= self.rollouts_per_example


@dataclass
class RolloutRequest:
    """Tracks a single rollout task — lives in queues, then in the inflight dict."""

    group_id: str
    cost: int  # limiter slots: 1 for individual, N for group scoring
    failure_count: int = 0


# ---------------------------------------------------------------------------
# Policy scheduler
# ---------------------------------------------------------------------------


class PolicyScheduler:
    """Watches for new policy checkpoints and applies weight updates.

    Polls the broadcast directory for new checkpoints. When found, updates
    inference server weights and drops stale rollouts via the train scheduler.
    """

    def __init__(
        self,
        train_scheduler: TrainScheduler,
        inference_pool: InferencePool,
        progress: Progress,
        output_dir: Path,
        max_async_level: int,
        lora_name: str | None = None,
    ):
        self.train_scheduler = train_scheduler
        self.inference_pool = inference_pool
        self.progress = progress
        self.output_dir = output_dir
        self.max_async_level = max_async_level
        self.lora_name = lora_name

        self.ckpt_step = 0
        self.update_weights_time = 0.0
        self.wait_for_ckpt_time = 0.0
        self.async_barrier_clear = asyncio.Event()
        self.async_barrier_clear.set()  # initially clear (no barrier)

    @property
    def async_level(self) -> int:
        return self.progress.step - self.ckpt_step

    async def wait_for_barrier(self) -> None:
        """Block until the async barrier is cleared."""
        await self.async_barrier_clear.wait()

    def _next_ckpt_step(self) -> int:
        """Compute the next checkpoint step to update to.

        Returns the maximum of the latest available checkpoint and the
        minimum required to stay within ``max_async_level``. When the
        orchestrator is far ahead, this may return a step whose checkpoint
        hasn't been written yet — the caller must wait for it.
        """
        latest = get_latest_ckpt_step(get_broadcast_dir(self.output_dir)) or 0
        min_required = max(self.progress.step - self.max_async_level, 0)
        return max(min_required, latest)

    async def start(self) -> None:
        """Background loop: poll for new checkpoints and apply weight updates.

        1. Compute next needed checkpoint (may not exist yet if orchestrator is ahead)
        2. If at async barrier — pause scheduling, wait for checkpoint to be written
        3. Update weights, drop stale groups, resume scheduling
        """
        while True:
            next_step = self._next_ckpt_step()
            if next_step <= self.ckpt_step:
                await asyncio.sleep(0.1)
                continue

            # When the next step is dictated by the async level (not just a new
            # checkpoint appearing), the checkpoint may not exist yet. Pause
            # scheduling and block the orchestrator until it's ready.
            min_required = max(self.progress.step - self.max_async_level, 0)
            if next_step == min_required:
                self.async_barrier_clear.clear()
                self.train_scheduler.pause()
                get_logger().info(f"At async barrier: pausing train scheduling, waiting for checkpoint {next_step}")
                t0 = time.perf_counter()
                await wait_for_path(get_step_path(get_broadcast_dir(self.output_dir), next_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - t0
                get_logger().debug(f"Waited for checkpoint {next_step} in {print_time(self.wait_for_ckpt_time)}")

            get_logger().info(f"Updating weights to training checkpoint {next_step}")
            await self._update_weights(next_step)
            get_logger().debug(f"Weights updated in {print_time(self.update_weights_time)}")

            self.train_scheduler.on_weights_updated()
            self.train_scheduler.resume()
            self.async_barrier_clear.set()

    async def _update_weights(self, ckpt_step: int) -> None:
        """Update the weights to the next training checkpoint."""
        t0 = time.perf_counter()
        weights_path = get_step_path(get_broadcast_dir(self.output_dir), ckpt_step)
        await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=ckpt_step)
        self.update_weights_time = time.perf_counter() - t0

        self.ckpt_step = ckpt_step
        if self.lora_name is not None:
            self.train_scheduler.model_name = self.lora_name
            self.inference_pool.update_model_name(self.lora_name)

    def get_metrics(self) -> dict[str, float]:
        return {
            "time/update_weights": self.update_weights_time,
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "scheduler/async_level": self.async_level,
        }


# ---------------------------------------------------------------------------
# Train scheduler
# ---------------------------------------------------------------------------


class TrainScheduler:
    """Always-on scheduler with two background loops: scheduling and completion processing.

    Scheduling is gated on two independent events:
    - ``_batch_needs_rollouts``: set while the current batch needs more rollouts.
      Automatically cleared when a batch fills, re-set by ``next_batch()``.
    - ``_policy_gate``: set unless an external caller (``PolicyScheduler``
      or eval) explicitly pauses scheduling via ``pause()``/``resume()``.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        train_envs: TrainEnvs,
        inference_pool: InferencePool,
        buffer: Buffer,
        progress: Progress,
        rollout_limiter: RolloutLimiter,
        batch_size: int | None,
        token_batch_size: int | None,
        rollouts_per_example: int,
        max_off_policy_steps: int,
        max_retries: int,
        model_name: str,
        json_logging: bool = False,
    ):
        self.train_envs = train_envs
        self.inference_pool = inference_pool
        self.buffer = buffer
        self.progress = progress
        self.limiter = rollout_limiter
        self._uses_token_batching = token_batch_size is not None
        self._batch_target = token_batch_size or batch_size
        self.rollouts_per_example = rollouts_per_example
        self.max_off_policy_steps = max_off_policy_steps
        self.max_retries = max_retries
        self.model_name = model_name
        self.json_logging = json_logging

        # Group + request tracking
        self._groups: dict[str, RolloutGroup] = {}
        self._requests: dict[asyncio.Task, RolloutRequest] = {}
        self._retry_queue: deque[RolloutRequest] = deque()
        self._schedule_queue: deque[RolloutRequest] = deque()
        self._client_load: Counter[tuple[str, str | None]] = Counter()

        # Batch accumulation
        self._batch_rollouts: list[vf.RolloutOutput] = []
        self._batch_progress = 0
        self._batch_full = asyncio.Event()  # set when batch reaches target, waited on by next_batch()
        self._batch_needs_rollouts = asyncio.Event()  # inverse of _batch_full, waited on by scheduling loop
        self._batch_needs_rollouts.set()
        self._pbar: ProgressTracker | None = None

        # Scheduling gate: cleared by pause(), set by resume()
        self._policy_gate = asyncio.Event()
        self._policy_gate.set()
        self._scheduling_task: asyncio.Task | None = None
        self._completion_task: asyncio.Task | None = None

        # Metrics (reset per step)
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.errored_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.total_rollouts_by_env: dict[str, int] = defaultdict(int)
        self._batch_start_time = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_inflight_rollouts(self) -> int:
        return sum(r.cost for r in self._requests.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background loops. Call once at the beginning of training."""
        self._reset_batch()
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())
        self._completion_task = asyncio.create_task(self._completion_loop())

    def pause(self) -> None:
        """Pause scheduling (policy gate). In-flight rollouts continue to completion."""
        self._policy_gate.clear()

    def resume(self) -> None:
        """Resume scheduling (policy gate)."""
        self._policy_gate.set()

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollout requests and release their slots."""
        all_tasks = list(self._requests.keys())
        count = sum(r.cost for r in self._requests.values())
        await safe_cancel_all(all_tasks)
        self._requests.clear()
        self._groups.clear()
        self._retry_queue.clear()
        self._schedule_queue.clear()
        self._client_load.clear()
        self.cancelled_rollouts_count += count
        self.limiter.release(count)

    async def next_batch(self) -> tuple[list[vf.RolloutOutput], float]:
        """Block until the current batch is complete, then return (rollouts, generation_time).

        Automatically resets for the next batch so scheduling resumes.
        Callers never need to call ``resume()`` for batch lifecycle —
        only for external pauses (eval, async barrier).
        """
        await self._batch_full.wait()

        rollouts = list(self._batch_rollouts)
        generation_time = time.perf_counter() - self._batch_start_time
        if self._pbar:
            self._pbar.close()
            self._pbar = None
        self._reset_batch()
        return rollouts, generation_time

    def _reset_batch(self) -> None:
        self._batch_rollouts.clear()
        self._batch_progress = 0
        self._batch_full.clear()
        self._batch_needs_rollouts.set()
        self._batch_start_time = time.perf_counter()
        self._pbar = ProgressTracker(
            total=self._batch_target,
            desc="Generating rollouts (train)",
            json_logging=self.json_logging,
            step=self.progress.step,
        )

    async def stop(self) -> None:
        """Stop all background tasks and cancel in-flight rollouts."""
        for task in [self._scheduling_task, self._completion_task]:
            if task is not None:
                await safe_cancel(task)
        self._scheduling_task = None
        self._completion_task = None
        await self.cancel_inflight_rollouts()

    # ------------------------------------------------------------------
    # Scheduling loop
    # ------------------------------------------------------------------

    async def _scheduling_loop(self) -> None:
        """Continuously schedule rollouts. Blocks when batch is full or paused."""
        while True:
            await self._batch_needs_rollouts.wait()
            await self._policy_gate.wait()

            request = self._next_request()
            group = self._groups[request.group_id]

            if group.client is None:
                group.client = await self._select_least_loaded_client()

            await self.limiter.acquire(request.cost)
            env = self.train_envs.get(group.env_name)
            task = asyncio.create_task(
                env.run(client=group.client, example=group.example, model_name=self.model_name, n=cost)
            )
            self._requests[task] = request
            self._client_load[client_identity(group.client)] += request.cost

    def _next_request(self) -> RolloutRequest:
        """Return the next request to schedule: retries first, then queued, then new."""
        for queue in (self._retry_queue, self._schedule_queue):
            while queue:
                request = queue.popleft()
                if request.group_id in self._groups:
                    return request
        return self._create_new_request()

    def _create_new_request(self) -> RolloutRequest:
        """Sample an example, create a group, and enqueue its requests."""
        example = self.buffer.sample_examples(n=1)[0]
        env = self.train_envs.get(example["env_name"])
        group = RolloutGroup(
            env_name=example["env_name"],
            example=example,
            requires_group_scoring=env.requires_group_scoring,
            rollouts_per_example=self.rollouts_per_example,
        )
        self._groups[group.group_id] = group

        if group.requires_group_scoring:
            return RolloutRequest(group_id=group.group_id, cost=group.rollouts_per_example)

        for _ in range(self.rollouts_per_example - 1):
            self._schedule_queue.append(RolloutRequest(group_id=group.group_id, cost=1))
        return RolloutRequest(group_id=group.group_id, cost=1)

    def _fire(self, group: RolloutGroup, cost: int) -> asyncio.Task:
        """Create an asyncio task for a rollout request."""

    # ------------------------------------------------------------------
    # Completion loop
    # ------------------------------------------------------------------

    async def _completion_loop(self) -> None:
        """Await completed tasks, process results, release slots."""
        while True:
            if not self._requests:
                await asyncio.sleep(0.1)
                continue

            done, _ = await asyncio.wait(self._requests, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                request = self._requests.pop(task, None)
                if request is None:
                    continue  # already removed by _drop_group

                self.limiter.release(request.cost)

                group = self._groups.get(request.group_id)
                if group is None:
                    continue

                if group.client is not None:
                    self._client_load[client_identity(group.client)] -= request.cost

                try:
                    result = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    get_logger().warning(f"Rollout failed: {e}")
                    self._drop_group(group.group_id)
                    continue

                self._process_rollouts(group, request, result)

    def _process_rollouts(self, group: RolloutGroup, request: RolloutRequest, rollouts: list[vf.RolloutOutput]) -> None:
        """Validate rollouts, handle retries, and finalize complete groups."""
        valid, num_failed = [], 0
        for rollout in rollouts:
            if not rollout["trajectory"] or rollout["error"] is not None:
                num_failed += 1
            else:
                rollout["env_name"] = group.env_name
                valid.append(rollout)

        self._record_rollout_metrics(group, rollouts)

        if num_failed:
            request.failure_count += 1
            if request.failure_count >= self.max_retries:
                self._drop_group(group.group_id)
                return
            if group.requires_group_scoring:
                group.completed.clear()
            else:
                group.completed.extend(valid)
            self._retry_queue.append(request)
        else:
            group.completed.extend(valid)

        if group.is_complete:
            self._finalize_group(group)

    def _record_rollout_metrics(self, group: RolloutGroup, rollouts: list[vf.RolloutOutput]) -> None:
        """Log warnings and track metrics for completed rollouts."""
        self.total_rollouts_by_env[group.env_name] += len(rollouts)
        for rollout in rollouts:
            if not rollout["trajectory"]:
                self.empty_rollouts_by_env[group.env_name] += 1
                get_logger().warning(
                    f"Empty trajectory in group {group.group_id} ({group.env_name}), re-scheduling "
                    f"({len(group.completed)}/{self.rollouts_per_example} complete)"
                )
            elif rollout["error"] is not None:
                self.errored_rollouts_by_env[group.env_name] += 1
                get_logger().warning(
                    f"Rollout error in group {group.group_id} ({group.env_name}), re-scheduling "
                    f"({len(group.completed)}/{self.rollouts_per_example} complete): "
                    f"{rollout['error']['error_chain_repr']}"
                )

    def _finalize_group(self, group: RolloutGroup) -> None:
        """Group complete — submit to buffer and accumulate batch."""
        self._groups.pop(group.group_id, None)
        self.buffer.update(group.completed)
        accepted = self.buffer.sample_rollouts(n=self.rollouts_per_example)

        self._batch_rollouts.extend(accepted)
        increment = sum(get_seq_len(r) for r in accepted) if self._uses_token_batching else len(accepted)
        self._batch_progress += increment
        if self._pbar:
            self._pbar.update(increment)

        if self._batch_progress >= self._batch_target:
            self._batch_full.set()
            self._batch_needs_rollouts.clear()

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def _drop_group(self, group_id: str) -> int:
        """Drop a group: cancel inflight tasks and release slots. Queued items are skipped lazily."""
        group = self._groups.pop(group_id, None)

        tasks_to_cancel = []
        count = 0
        for task, request in list(self._requests.items()):
            if request.group_id == group_id:
                tasks_to_cancel.append(task)
                count += request.cost
                self._requests.pop(task)

        if count:
            if group is not None and group.client is not None:
                self._client_load[client_identity(group.client)] -= count
            self.limiter.release(count)
            self.cancelled_rollouts_count += count

        for task in tasks_to_cancel:
            task.cancel()

        return count

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight requests."""
        clients = self.inference_pool.train_clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.train_clients
        return min(clients, key=lambda c: self._client_load[client_identity(c)])

    def on_weights_updated(self) -> None:
        """Called after a weight update: increment off-policy steps and drop stale groups."""
        for group in self._groups.values():
            group.off_policy_steps += 1

        # drop groups that have exceeded max_off_policy_steps
        stale_group_ids = [gid for gid, g in self._groups.items() if g.off_policy_steps >= self.max_off_policy_steps]
        if not stale_group_ids:
            return

        removed = sum(self._drop_group(gid) for gid in stale_group_ids)
        if removed:
            get_logger().warning(
                f"Cancelled {removed} stale rollout requests. Consider increasing max_off_policy_steps to avoid this."
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, float]:
        levels = [g.off_policy_steps for g in self._groups.values()]
        total_rollouts = sum(self.total_rollouts_by_env.values())
        metrics = {
            "scheduler/inflight_rollouts": self.num_inflight_rollouts,
            "scheduler/limiter_used": self.limiter.concurrency.used,
            "scheduler/limiter_remaining": self.limiter.remaining,
            "scheduler/cancelled_rollouts": self.cancelled_rollouts_count,
            "empty_rollouts/all": sum(self.empty_rollouts_by_env.values()) / max(total_rollouts, 1),
            "errored_rollouts/all": sum(self.errored_rollouts_by_env.values()) / max(total_rollouts, 1),
            "off_policy_level/all/max": max(levels) if levels else 0,
            "off_policy_level/all/mean": sum(levels) / len(levels) if levels else 0,
        }
        for env_name in self.total_rollouts_by_env:
            env_total = max(self.total_rollouts_by_env[env_name], 1)
            metrics[f"empty_rollouts/{env_name}"] = self.empty_rollouts_by_env.get(env_name, 0) / env_total
            metrics[f"errored_rollouts/{env_name}"] = self.errored_rollouts_by_env.get(env_name, 0) / env_total
        by_env: dict[str, list[int]] = {}
        for group in self._groups.values():
            by_env.setdefault(group.env_name, []).append(group.off_policy_steps)
        for env_name, steps in by_env.items():
            metrics[f"off_policy_level/{env_name}/max"] = max(steps)
            metrics[f"off_policy_level/{env_name}/mean"] = sum(steps) / len(steps)
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env.clear()
        self.errored_rollouts_by_env.clear()
        self.total_rollouts_by_env.clear()

        metrics.update(self.inference_pool.get_metrics())
        return metrics


# ---------------------------------------------------------------------------
# Eval scheduler (unchanged)
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single eval environment."""

    env_name: str
    rollouts: list[vf.RolloutOutput]
    total_rollouts: int
    rollouts_per_example: int
    eval_time: float


class EvalScheduler:
    """Schedules eval rollouts with shared concurrency and rate limiting."""

    def __init__(
        self,
        rollout_limiter: RolloutLimiter,
        inference_pool: InferencePool,
    ):
        self.logger = get_logger()
        self.limiter = rollout_limiter
        self.inference_pool = inference_pool

    async def evaluate_envs(
        self,
        eval_envs: list[EvalEnv],
        model_name: str,
    ) -> AsyncIterator[EvalResult]:
        """Run evals for multiple envs, yielding results as each env completes."""
        tasks = {asyncio.create_task(self.evaluate_env(env, model_name)) for env in eval_envs}
        try:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    yield task.result()
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def evaluate_env(self, eval_env: EvalEnv, model_name: str) -> EvalResult:
        """Run all rollouts for one eval env with concurrency control."""
        num_examples = len(eval_env.examples)
        rollouts_per_example = eval_env.config.rollouts_per_example
        total_rollouts = num_examples * rollouts_per_example
        cost = rollouts_per_example if eval_env.requires_group_scoring else 1

        self.logger.info(f"Evaluating {eval_env.name} ({num_examples=}, {rollouts_per_example=})")
        pbar = ProgressTracker(total=total_rollouts, desc=f"Evaluating {eval_env.name}")
        eval_start = time.perf_counter()

        async def run_one(example: dict, n: int) -> list[vf.RolloutOutput] | None:
            await self.limiter.acquire(cost, priority=True)
            try:
                client = await self.inference_pool.get_eval_client()
                outputs = await eval_env.run(client=client, example=example, model_name=model_name, n=n)
                pbar.update(n)
                return outputs
            except Exception as e:
                self.logger.warning(f"Rollout failed: {e}")
                pbar.update(n)
                return None
            finally:
                self.limiter.release(cost)

        if eval_env.requires_group_scoring:
            coros = [run_one(example, rollouts_per_example) for example in eval_env.examples]
        else:
            coros = [run_one(example, 1) for example in eval_env.examples for _ in range(rollouts_per_example)]

        try:
            results = await asyncio.gather(*coros)
        finally:
            pbar.close()

        rollouts = [o for group in results if group is not None for o in group]
        eval_time = time.perf_counter() - eval_start

        return EvalResult(
            env_name=eval_env.name,
            rollouts=rollouts,
            total_rollouts=total_rollouts,
            rollouts_per_example=rollouts_per_example,
            eval_time=eval_time,
        )
