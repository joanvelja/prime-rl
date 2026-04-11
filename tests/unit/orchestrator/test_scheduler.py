import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.concurrency import ConcurrencyLimiter, RolloutLimiter
from prime_rl.orchestrator.scheduler import InflightGroup, InflightRolloutRequest, PolicyScheduler, TrainScheduler
from prime_rl.utils.async_utils import safe_cancel


def _make_train_scheduler() -> TrainScheduler:
    """Create a minimal TrainScheduler for testing (no __init__)."""
    scheduler = TrainScheduler.__new__(TrainScheduler)
    scheduler.limiter = RolloutLimiter(128)
    scheduler.logger = MagicMock()
    scheduler._scheduling_enabled = asyncio.Event()
    scheduler._scheduling_enabled.set()
    scheduler._groups = {}
    scheduler._task_to_group = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler._scheduling_task = None
    scheduler._completion_task = None
    scheduler._policy_loop_task = None
    scheduler.policy = None
    scheduler.step = 0
    scheduler.model_name = "test-model"
    scheduler.enable_policy_updates = False
    return scheduler


def _make_policy_scheduler(train_scheduler: TrainScheduler) -> PolicyScheduler:
    """Create a minimal PolicyScheduler for testing."""
    ps = PolicyScheduler.__new__(PolicyScheduler)
    ps.logger = MagicMock()
    ps._train = train_scheduler
    ps.inference_pool = MagicMock()
    ps.config = SimpleNamespace(output_dir=Path("/tmp/prime-rl-test"))
    ps.max_async_level = 1
    ps.strict_async_level = False
    ps.model_name = "test-model"
    ps.lora_name = None
    ps.ckpt_step = 7
    ps.update_weights_time = 0
    ps.wait_for_ckpt_time = 0
    ps._inflight_task = None
    ps._lock = asyncio.Lock()
    train_scheduler.policy = ps
    train_scheduler.step = 9
    return ps


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = _make_train_scheduler()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))

        stale_request = InflightRolloutRequest(task=stale_task, client=client, off_policy_steps=1)
        survivor_request = InflightRolloutRequest(task=survivor_task, client=client, off_policy_steps=0)

        scheduler._groups = {
            1: InflightGroup(
                example={},
                env_name="test",
                inflight_requests={stale_task: stale_request},
            ),
            2: InflightGroup(
                example={},
                env_name="test",
                inflight_requests={survivor_task: survivor_request},
            ),
        }
        scheduler._task_to_group = {stale_task: 1, survivor_task: 2}
        scheduler.limiter.concurrency.try_acquire(2)

        await scheduler._update_off_policy()

        assert 1 not in scheduler._groups
        assert 2 in scheduler._groups
        assert survivor_request.off_policy_steps == 1
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task):
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_maybe_update_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = _make_train_scheduler()
        ps = _make_policy_scheduler(scheduler)
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        ps.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(ps.maybe_update(step=9))
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(ps.maybe_update(step=9))
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert ps.ckpt_step == 8

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = _make_train_scheduler()
        ps = _make_policy_scheduler(scheduler)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        ps.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler._policy_loop_task = asyncio.create_task(ps.maybe_update(step=9))
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler._policy_loop_task is None
        assert ps._inflight_task is None

    asyncio.run(run())


def test_concurrency_limiter_basic():
    limiter = ConcurrencyLimiter(10)
    assert limiter.remaining == 10
    assert limiter.used == 0

    assert limiter.try_acquire(3)
    assert limiter.remaining == 7
    assert limiter.used == 3

    assert not limiter.try_acquire(8)
    assert limiter.remaining == 7

    limiter.release(3)
    assert limiter.remaining == 10
    assert limiter.used == 0


def test_concurrency_limiter_acquire_blocks():
    async def run():
        limiter = ConcurrencyLimiter(2)
        limiter.try_acquire(2)
        assert limiter.remaining == 0

        acquired = False

        async def waiter():
            nonlocal acquired
            await limiter.acquire(1)
            acquired = True

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert not acquired

        limiter.release(1)
        await asyncio.sleep(0)
        assert acquired

        limiter.release(1)
        task.cancel()

    asyncio.run(run())
