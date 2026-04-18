import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.scheduler import GroupState, InflightRolloutInfo, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_async_level = 1
    scheduler.strict_async_level = False
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(output_dir=Path("/tmp/prime-rl-test"))
    scheduler.logger = MagicMock()
    scheduler.checkpoint_ready = asyncio.Event()
    scheduler.checkpoint_ready.set()
    scheduler.lora_name = None
    scheduler.model_name = "test-model"
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.enable_policy_updates = True
    return scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRolloutInfo(off_policy_steps=1, client_config=client, task="test", group_id=1),
            survivor_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRolloutInfo(
                    off_policy_steps=0,
                    client_config=client,
                    task="test",
                    group_id=3,
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_maybe_update_policy_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(scheduler.maybe_update_policy())
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert scheduler.ckpt_step == 8

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler.update_policy_task = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())


def test_generate_batch_drops_timeout_error_and_continues():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.enable_policy_updates = False
        scheduler.batch_size = 1
        scheduler.token_batch_size = None
        scheduler.rollouts_per_example = 1
        scheduler.json_logging = True
        scheduler.empty_rollouts_by_task = defaultdict(int)
        scheduler.errored_rollouts_by_task = defaultdict(int)
        scheduler.total_rollouts_by_task = defaultdict(int)
        scheduler.deferred_group_scoring_tasks = set()
        scheduler._fill_inflight_requests = AsyncMock()
        scheduler.buffer = MagicMock()
        scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        success_rollout = {
            "task": "debate",
            "trajectory": [{"tokens": None}],
            "error": None,
        }

        async def raise_timeout():
            raise TimeoutError("Environment timeout for run_group request after 30s")

        async def return_rollout():
            return success_rollout

        timeout_task = asyncio.create_task(raise_timeout())
        success_task = asyncio.create_task(return_rollout())

        scheduler.groups = {
            1: GroupState(example={"task": "debate"}, rollouts_to_schedule=0),
            2: GroupState(example={"task": "debate"}, rollouts_to_schedule=0),
        }
        scheduler.inflight_requests = {
            timeout_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="debate", group_id=1),
            success_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="debate", group_id=2),
        }
        scheduler.buffer.sample_rollouts.return_value = [success_rollout]

        with patch("prime_rl.orchestrator.scheduler.ProgressTracker") as progress_tracker:
            progress_tracker.return_value = SimpleNamespace(update=MagicMock(), close=MagicMock())
            batch = await scheduler.generate_batch(step=3)

        assert batch == [success_rollout]
        assert 1 not in scheduler.groups
        assert 2 not in scheduler.groups
        scheduler.buffer.update.assert_called_once_with([success_rollout])
        assert any(
            "Retryable rollout error in group 1" in str(call)
            and "TimeoutError" in str(call)
            for call in scheduler.logger.warning.call_args_list
        )

    asyncio.run(run())
