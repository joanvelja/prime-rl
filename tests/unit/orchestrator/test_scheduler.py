import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.scheduler import InflightRolloutInfo, Scheduler
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


def test_score_group_if_deferred_uses_pinned_env_version():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config = SimpleNamespace(verification=SimpleNamespace(enabled=True))

        rubric_v1 = SimpleNamespace(score_group=AsyncMock())
        rubric_v2 = SimpleNamespace(score_group=AsyncMock())
        env_registry = SimpleNamespace(
            should_defer_group_scoring=lambda task, version: task == "env_a" and version == 1,
            get_env_for_task=lambda task, version=None: {
                ("env_a", 1): SimpleNamespace(rubric=rubric_v1),
                ("env_a", 2): SimpleNamespace(rubric=rubric_v2),
            }[(task, version)],
        )
        scheduler.env = env_registry

        rollouts = [{"task": "env_a", "reward": 1.0}]
        await scheduler._score_group_if_deferred(rollouts, env_version=1)

        rubric_v1.score_group.assert_awaited_once_with(rollouts)
        rubric_v2.score_group.assert_not_called()

    asyncio.run(run())


def test_generate_batch_releases_version_when_buffer_update_raises():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config = SimpleNamespace(
            verification=SimpleNamespace(enabled=False),
            batch_size=1,
            token_batch_size=None,
        )
        scheduler.batch_size = 1
        scheduler.token_batch_size = None
        scheduler.rollouts_per_example = 1
        scheduler.json_logging = False
        scheduler.enable_policy_updates = False
        scheduler._fill_inflight_requests = AsyncMock()

        env_registry = SimpleNamespace(
            release_version=MagicMock(),
            should_defer_group_scoring=lambda task, version: False,
        )
        scheduler.env = env_registry

        rollout = {"task": "env_a", "trajectory": [{"tokens": {}}], "error": None}
        finished_task = asyncio.get_running_loop().create_future()
        finished_task.set_result(rollout)
        scheduler.inflight_requests = {
            finished_task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test"),
                task="env_a",
                group_id=0,
            )
        }
        scheduler.groups = {
            0: SimpleNamespace(
                example={"task": "env_a"},
                env_version=3,
                rollouts_to_schedule=0,
                completed_rollouts=[],
            )
        }
        scheduler.buffer = SimpleNamespace(
            update=MagicMock(side_effect=RuntimeError("boom")),
            sample_rollouts=MagicMock(),
        )

        try:
            with patch(
                "prime_rl.orchestrator.scheduler.asyncio.wait", new=AsyncMock(return_value=({finished_task}, set()))
            ):
                await scheduler.generate_batch(step=0)
        except RuntimeError as e:
            assert str(e) == "boom"
        else:
            raise AssertionError("Expected generate_batch to propagate buffer failure")

        env_registry.release_version.assert_called_once_with("env_a", 3)

    asyncio.run(run())
