import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
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


def make_rollout(reward: float, completion_len: int) -> dict:
    return {
        "reward": reward,
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [0],
                    "completion_ids": [1] * completion_len,
                },
                "response": None,
            }
        ],
    }


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


def test_select_rollouts_ignores_filtered_rollouts():
    scheduler = make_scheduler()
    scheduler.rollouts_per_example = 3
    scheduler.batch_size = 2
    scheduler.token_batch_size = None
    scheduler.config = SimpleNamespace(
        output_dir=Path("/tmp/prime-rl-test"),
        advantage=DefaultAdvantageConfig(),
        buffer=SimpleNamespace(min_abs_adv=0.0),
    )

    rollouts = [
        make_rollout(reward=1.0, completion_len=4),
        make_rollout(reward=0.5, completion_len=4),
        make_rollout(reward=0.0, completion_len=4),
    ]

    selected_mask, batch_progress = scheduler.select_rollouts(rollouts, remaining_batch_target=1)

    assert selected_mask == [True, False, False]
    assert batch_progress == 1


def test_select_rollouts_allows_final_overshoot():
    scheduler = make_scheduler()
    scheduler.rollouts_per_example = 3
    scheduler.batch_size = None
    scheduler.token_batch_size = 6
    scheduler.config = SimpleNamespace(
        output_dir=Path("/tmp/prime-rl-test"),
        advantage=DefaultAdvantageConfig(),
        buffer=SimpleNamespace(min_abs_adv=0.2),
    )

    rollouts = [
        make_rollout(reward=0.5, completion_len=3),
        make_rollout(reward=1.0, completion_len=4),
        make_rollout(reward=0.0, completion_len=5),
    ]

    selected_mask, batch_progress = scheduler.select_rollouts(rollouts, remaining_batch_target=6)

    assert selected_mask == [False, True, True]
    assert batch_progress == 11
