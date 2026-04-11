from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from collections import Counter, defaultdict
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
from prime_rl.utils.client import InferencePool
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
class InflightRequest:
    """Base class for an in-flight rollout API call."""

    task: asyncio.Task
    client: vf.ClientConfig
    ckpt_step: int = 0

    @property
    @abstractmethod
    def rollout_count(self) -> int: ...


@dataclass
class InflightRolloutRequest(InflightRequest):
    """A single run_rollout call producing 1 rollout."""

    @property
    def rollout_count(self) -> int:
        return 1


@dataclass
class InflightGroupRequest(InflightRequest):
    """A single run_group call producing N rollouts."""

    rollouts_per_example: int = 1

    @property
    def rollout_count(self) -> int:
        return self.rollouts_per_example


@dataclass
class InflightGroup:
    """Tracks one example being evaluated with N rollouts."""

    example: dict
    env_name: str
    # Reuse the same client for all rollouts in a group to maximize prefix cache hits
    client: vf.ClientConfig | None = None

    inflight_requests: dict[asyncio.Task, InflightRequest] = field(default_factory=dict)
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)
    rollouts_to_schedule: int = 0

    @property
    def total_inflight_rollouts(self) -> int:
        return sum(r.rollout_count for r in self.inflight_requests.values())


# ---------------------------------------------------------------------------
# Policy scheduler
# ---------------------------------------------------------------------------


class PolicyScheduler:
    """Watches for new policy checkpoints and applies weight updates.

    Runs as a background loop owned by TrainScheduler. On policy update:
    pauses scheduling, waits for checkpoint, updates weights, drops stale
    rollouts, then resumes scheduling.
    """

    def __init__(
        self,
        train_scheduler: TrainScheduler,
        inference_pool: InferencePool,
        progress: Progress,
        output_dir: Path,
        max_async_level: int,
        strict_async_level: bool,
        model_name: str,
        lora_name: str | None = None,
    ):
        self.train_scheduler = train_scheduler
        self.inference_pool = inference_pool
        self.progress = progress
        self.output_dir = output_dir
        self.max_async_level = max_async_level
        self.strict_async_level = strict_async_level
        self.model_name = model_name
        self.lora_name = lora_name

        self.ckpt_step = 0
        self.update_weights_time = 0.0
        self.wait_for_ckpt_time = 0.0

    async def maybe_update(self, step: int) -> None:
        """Check for and apply any pending policy updates."""

        def get_next_ckpt_step() -> int:
            latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.output_dir)) or 0
            async_away_ckpt_step = max(step - self.max_async_level, 0)
            if self.strict_async_level:
                return async_away_ckpt_step
            return max(async_away_ckpt_step, latest_ckpt_step)

        while True:
            next_ckpt_step = get_next_ckpt_step()
            if next_ckpt_step <= self.ckpt_step:
                return
            await self._apply_update(next_ckpt_step, step)

    @property
    def async_level(self) -> int:
        return self.progress.step - self.ckpt_step

    @property
    def at_async_barrier(self) -> bool:
        return self.progress.step - self.ckpt_step == self.max_async_level

    def get_metrics(self) -> dict[str, float]:
        return {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "scheduler/async_level": self.async_level,
        }

    async def run(self) -> None:
        """Background loop. Reads current step from progress.step."""
        while True:
            await self.maybe_update(self.progress.step)
            await asyncio.sleep(1)

    async def _apply_update(self, next_ckpt_step: int, step: int) -> None:
        # Wait for checkpoint if we're at the async barrier
        async_away_ckpt_step = max(step - self.max_async_level, 0)
        if next_ckpt_step == async_away_ckpt_step:
            get_logger().info(
                f"At async barrier: Pausing new rollout scheduling while waiting for checkpoint {next_ckpt_step}"
            )
            self.train_scheduler.pause()
            t0 = time.perf_counter()
            await wait_for_path(get_step_path(get_broadcast_dir(self.output_dir), next_ckpt_step) / "STABLE")
            self.wait_for_ckpt_time = time.perf_counter() - t0
            get_logger().info(
                f"Cleared async barrier: Checkpoint {next_ckpt_step} ready after {print_time(self.wait_for_ckpt_time)}"
            )

        # Update weights on inference servers
        get_logger().info(f"Updating weights to step {next_ckpt_step}")
        t0 = time.perf_counter()
        await self.inference_pool.update_weights(
            weight_dir=get_step_path(get_broadcast_dir(self.output_dir), next_ckpt_step),
            lora_name=self.lora_name,
            step=next_ckpt_step,
        )
        self.update_weights_time = time.perf_counter() - t0

        self.ckpt_step = next_ckpt_step
        if self.lora_name is not None:
            self.model_name = self.lora_name
            self.train_scheduler.model_name = self.lora_name
            self.inference_pool.update_model_name(self.model_name)

        # Resume scheduling and drop stale rollouts
        self.train_scheduler.resume()
        await self.train_scheduler.drop_stale_groups(next_ckpt_step)
        get_logger().debug(
            f"Updated weights to step {next_ckpt_step} in {print_time(self.update_weights_time)}. Resuming scheduling of training rollouts"
        )


# ---------------------------------------------------------------------------
# Train scheduler
# ---------------------------------------------------------------------------


class TrainScheduler:
    """Always-on scheduler with two background loops: scheduling and completion processing.

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
        model_name: str,
        json_logging: bool = False,
    ):
        self.logger = get_logger()
        self.limiter = rollout_limiter
        self.train_envs = train_envs
        self.inference_pool = inference_pool
        self.buffer = buffer
        self.progress = progress
        self._uses_token_batching = token_batch_size is not None
        self._batch_target = token_batch_size or batch_size
        self.rollouts_per_example = rollouts_per_example
        self.max_off_policy_steps = max_off_policy_steps
        self.model_name = model_name
        self.json_logging = json_logging

        # Group tracking
        self._next_group_id = 0
        self._groups: dict[int, InflightGroup] = {}
        self._task_to_group: dict[asyncio.Task, int] = {}

        # Batch accumulation
        self._batch_rollouts: list[vf.RolloutOutput] = []
        self._batch_progress = 0
        self._batch_ready = asyncio.Event()
        self._pbar: ProgressTracker | None = None

        # Scheduling control
        self._scheduling_enabled = asyncio.Event()
        self._scheduling_enabled.set()
        self._scheduling_task: asyncio.Task | None = None
        self._completion_task: asyncio.Task | None = None

        self.ckpt_step = 0

        # Metrics (reset per step)
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.errored_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.total_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.last_batch_generation_time = 0.0
        self._batch_start_time = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_inflight_rollouts(self) -> int:
        return sum(g.total_inflight_rollouts for g in self._groups.values())

    def _off_policy_levels(self) -> list[int]:
        return [self.ckpt_step - r.ckpt_step for g in self._groups.values() for r in g.inflight_requests.values()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background loops. Call once at the beginning of training."""
        self._reset_batch()
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())
        self._completion_task = asyncio.create_task(self._completion_loop())

    def pause(self) -> None:
        """Pause scheduling. In-flight rollouts continue to completion."""
        self._scheduling_enabled.clear()

    def resume(self) -> None:
        """Resume scheduling."""
        self._scheduling_enabled.set()

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollout requests and release their slots."""
        all_tasks = list(self._task_to_group.keys())
        count = sum(g.total_inflight_rollouts for g in self._groups.values())
        await safe_cancel_all(all_tasks)
        for group in self._groups.values():
            group.inflight_requests.clear()
        self._task_to_group.clear()
        self._groups.clear()
        self.cancelled_rollouts_count += count
        self.limiter.release(count)

    async def wait_for_batch(self) -> list[vf.RolloutOutput]:
        """Block until the current batch is complete, then return it."""
        await self._batch_ready.wait()

        batch = list(self._batch_rollouts)
        self.last_batch_generation_time = time.perf_counter() - self._batch_start_time
        if self._pbar:
            self._pbar.close()
            self._pbar = None
        self._reset_batch()
        return batch

    def _reset_batch(self) -> None:
        self._batch_rollouts.clear()
        self._batch_progress = 0
        self._batch_ready.clear()
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
        """Continuously schedule rollouts. Blocks on acquire when capacity is full."""
        while True:
            await self._scheduling_enabled.wait()

            # Try to schedule from existing groups first (retries/pending)
            scheduled = False
            for group in list(self._groups.values()):
                if group.rollouts_to_schedule > 0:
                    await self._schedule_from_group(group)
                    scheduled = True
                    break

            if not scheduled:
                group = self._create_group()
                await self._schedule_from_group(group)

    def _create_group(self) -> InflightGroup:
        """Sample an example from the buffer and create a new group."""
        example = self.buffer.sample_examples(n=1)[0]
        group_id = self._next_group_id
        self._next_group_id += 1
        group = InflightGroup(
            example=example,
            env_name=example["env_name"],
            rollouts_to_schedule=self.rollouts_per_example,
        )
        self._groups[group_id] = group
        return group

    async def _schedule_from_group(self, group: InflightGroup) -> None:
        """Schedule one request from a group. Blocks until a concurrency slot is available."""
        env = self.train_envs.get(group.env_name)

        if group.client is not None:
            client = group.client
        else:
            client = await self._select_least_loaded_client()
            group.client = client

        if env.requires_group_scoring:
            rollout_count = group.rollouts_to_schedule
            await self.limiter.acquire(rollout_count)
            group.rollouts_to_schedule = 0
            task = asyncio.create_task(
                env.run_group(
                    client=client,
                    example=group.example,
                    model_name=self.model_name,
                    rollouts_per_example=rollout_count,
                )
            )
            request = InflightGroupRequest(
                task=task, client=client, ckpt_step=self.ckpt_step, rollouts_per_example=rollout_count
            )
        else:
            await self.limiter.acquire(1)
            group.rollouts_to_schedule -= 1
            task = asyncio.create_task(
                env.run_rollout(
                    client=client,
                    example=group.example,
                    model_name=self.model_name,
                )
            )
            request = InflightRolloutRequest(task=task, client=client, ckpt_step=self.ckpt_step)

        group.inflight_requests[task] = request
        group_id = self._group_id_for(group)
        self._task_to_group[task] = group_id

    # ------------------------------------------------------------------
    # Completion loop
    # ------------------------------------------------------------------

    async def _completion_loop(self) -> None:
        """Continuously process completed rollout tasks."""
        while True:
            tasks = list(self._task_to_group.keys())
            if not tasks:
                await asyncio.sleep(0)
                continue

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            await self._scheduling_enabled.wait()

            for task in done:
                group_id = self._task_to_group.pop(task, None)
                if group_id is None:
                    continue

                group = self._groups.get(group_id)
                if group is None:
                    continue

                request = group.inflight_requests.pop(task, None)
                if request is None:
                    continue

                self.limiter.release(request.rollout_count)
                await self._process_completion(group_id, group, request)

    async def _process_completion(self, group_id: int, group: InflightGroup, request: InflightRequest) -> None:
        """Process a completed request: validate, reschedule failures, accumulate batch."""
        env = self.train_envs.get(group.env_name)

        try:
            result = request.task.result()
            rollouts: list[vf.RolloutOutput] = result if isinstance(result, list) else [result]
        except asyncio.CancelledError:
            await self._drop_group(group_id)
            return
        except Exception as e:
            self.logger.warning(f"Rollout failed: {e}")
            await self._drop_group(group_id)
            return

        self.total_rollouts_by_env[group.env_name] += len(rollouts)

        valid_rollouts = []
        has_failures = False
        for rollout in rollouts:
            if len(rollout["trajectory"]) == 0:
                self.empty_rollouts_by_env[group.env_name] += 1
                has_failures = True
                self.logger.warning(
                    f"Empty trajectory in group {group_id} ({group.env_name}), re-scheduling "
                    f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete)"
                )
            elif rollout["error"] is not None:
                self.errored_rollouts_by_env[group.env_name] += 1
                has_failures = True
                self.logger.warning(
                    f"Rollout error in group {group_id} ({group.env_name}), re-scheduling "
                    f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete): "
                    f"{rollout['error']['error_chain_repr']}"
                )
            else:
                rollout["env_name"] = group.env_name
                valid_rollouts.append(rollout)

        if has_failures and env.requires_group_scoring:
            group.completed_rollouts.clear()
            group.rollouts_to_schedule = self.rollouts_per_example
            return

        group.rollouts_to_schedule += len(rollouts) - len(valid_rollouts)
        group.completed_rollouts.extend(valid_rollouts)

        if len(group.completed_rollouts) < self.rollouts_per_example:
            return

        # Group complete — add to batch
        completed = self._groups.pop(group_id).completed_rollouts
        self.buffer.update(completed)
        accepted = self.buffer.sample_rollouts(n=self.rollouts_per_example)

        self._batch_rollouts.extend(accepted)
        if self._uses_token_batching:
            increment = sum(get_seq_len(r) for r in accepted)
        else:
            increment = len(accepted)
        self._batch_progress += increment
        if self._pbar:
            self._pbar.update(increment)

        if self._batch_progress >= self._batch_target:
            self._batch_ready.set()

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def _group_id_for(self, group: InflightGroup) -> int:
        """Find group_id for a group object."""
        for gid, g in self._groups.items():
            if g is group:
                return gid
        raise ValueError("Group not found")

    async def _drop_group(self, group_id: int) -> int:
        """Drop a group: cancel pending requests, release slots."""
        group = self._groups.pop(group_id, None)
        if group is None:
            return 0

        tasks_to_cancel = list(group.inflight_requests.keys())
        rollout_count = sum(r.rollout_count for r in group.inflight_requests.values())

        for task in tasks_to_cancel:
            self._task_to_group.pop(task, None)
        group.inflight_requests.clear()

        await safe_cancel_all(tasks_to_cancel)
        if rollout_count:
            self.limiter.release(rollout_count)
            self.cancelled_rollouts_count += rollout_count
        return rollout_count

    @staticmethod
    def _client_identity(c: vf.ClientConfig) -> tuple[str, str | None]:
        return (c.api_base_url, c.extra_headers.get("X-data-parallel-rank"))

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight tasks."""
        clients = self.inference_pool.train_clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.train_clients
        inflight = Counter(
            self._client_identity(r.client) for g in self._groups.values() for r in g.inflight_requests.values()
        )
        return min(clients, key=lambda c: inflight[self._client_identity(c)])

    async def drop_stale_groups(self, current_ckpt_step: int) -> None:
        """Drop groups with requests older than max_off_policy_steps."""
        stale_group_ids = {
            gid
            for gid, group in self._groups.items()
            if any(
                current_ckpt_step - r.ckpt_step >= self.max_off_policy_steps for r in group.inflight_requests.values()
            )
        }
        if not stale_group_ids:
            return

        counts = await asyncio.gather(*(self._drop_group(gid) for gid in stale_group_ids))
        removed = sum(counts)
        if removed:
            self.logger.warning(
                f"Cancelled {removed} stale rollout requests (ckpt_step={current_ckpt_step}). "
                f"Consider increasing max_off_policy_steps to avoid this."
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, float]:
        total_rollouts = sum(self.total_rollouts_by_env.values())
        metrics = {
            "scheduler/inflight_rollouts": self.num_inflight_rollouts,
            "scheduler/limiter_used": self.limiter.concurrency.used,
            "scheduler/limiter_remaining": self.limiter.remaining,
            "scheduler/cancelled_rollouts": self.cancelled_rollouts_count,
            "empty_rollouts/all": sum(self.empty_rollouts_by_env.values()) / max(total_rollouts, 1),
            "errored_rollouts/all": sum(self.errored_rollouts_by_env.values()) / max(total_rollouts, 1),
            "off_policy_level/all/max": max(levels) if (levels := self._off_policy_levels()) else 0,
            "off_policy_level/all/mean": sum(levels) / len(levels) if levels else 0,
        }
        for env_name in self.total_rollouts_by_env:
            env_total = max(self.total_rollouts_by_env[env_name], 1)
            metrics[f"empty_rollouts/{env_name}"] = self.empty_rollouts_by_env.get(env_name, 0) / env_total
            metrics[f"errored_rollouts/{env_name}"] = self.errored_rollouts_by_env.get(env_name, 0) / env_total
        by_env: dict[str, list[int]] = {}
        for group in self._groups.values():
            for request in group.inflight_requests.values():
                by_env.setdefault(group.env_name, []).append(self.ckpt_step - request.ckpt_step)
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
        cost_per_coro = rollouts_per_example if eval_env.requires_group_scoring else 1

        self.logger.info(f"Evaluating {eval_env.name} ({num_examples=}, {rollouts_per_example=})")
        pbar = ProgressTracker(total=total_rollouts, desc=f"Evaluating {eval_env.name}")
        eval_start = time.perf_counter()

        if eval_env.requires_group_scoring:

            async def run_one(example: dict) -> list[vf.RolloutOutput] | None:
                await self.limiter.acquire(cost_per_coro, priority=True)
                try:
                    client = await self.inference_pool.get_eval_client()
                    outputs = await eval_env.run_group(
                        client=client,
                        example=example,
                        model_name=model_name,
                        rollouts_per_example=rollouts_per_example,
                    )
                    pbar.update(rollouts_per_example)
                    return outputs
                except Exception as e:
                    self.logger.warning(f"Group failed: {e}")
                    pbar.update(rollouts_per_example)
                    return None
                finally:
                    self.limiter.release(cost_per_coro)

            coros = [run_one(example) for example in eval_env.examples]

        else:

            async def run_one(example: dict) -> list[vf.RolloutOutput] | None:
                await self.limiter.acquire(1, priority=True)
                try:
                    client = await self.inference_pool.get_eval_client()
                    output = await eval_env.run_rollout(client=client, example=example, model_name=model_name)
                    pbar.update(1)
                    return [output]
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")
                    pbar.update(1)
                    return None
                finally:
                    self.limiter.release(1)

            coros = [run_one(example) for example in eval_env.examples for _ in range(rollouts_per_example)]

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
