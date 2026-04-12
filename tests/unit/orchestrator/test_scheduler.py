import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.ckpt import Progress
from prime_rl.orchestrator.concurrency import ConcurrencyLimiter, RolloutLimiter
from prime_rl.orchestrator.scheduler import (
    PolicyScheduler,
    RolloutDispatcher,
    RolloutGroup,
    RolloutRequest,
    TrainScheduler,
)


def _make_dispatcher() -> RolloutDispatcher:
    return RolloutDispatcher(limiter=RolloutLimiter(128), inference_pool=MagicMock())


def _make_train_scheduler() -> TrainScheduler:
    """Create a minimal TrainScheduler for testing (no __init__)."""
    scheduler = TrainScheduler.__new__(TrainScheduler)
    scheduler.dispatcher = _make_dispatcher()
    scheduler._policy_gate = asyncio.Event()
    scheduler._policy_gate.set()
    scheduler._groups = {}
    scheduler._requests = {}
    scheduler._retry_queue = __import__("collections").deque()
    scheduler._schedule_queue = __import__("collections").deque()
    scheduler.max_off_policy_steps = 2
    scheduler.max_retries = 3
    scheduler.cancelled_rollouts_count = 0
    scheduler._scheduling_task = None
    scheduler._completion_task = None
    scheduler.progress = Progress(step=0)
    scheduler.model_name = "test-model"
    return scheduler


def _make_policy_scheduler(train_scheduler: TrainScheduler) -> PolicyScheduler:
    """Create a minimal PolicyScheduler for testing."""
    ps = PolicyScheduler.__new__(PolicyScheduler)
    ps.train_scheduler = train_scheduler
    ps.inference_pool = MagicMock()
    ps.output_dir = Path("/tmp/prime-rl-test")
    ps.lora_name = None
    ps.ckpt_step = 7
    ps.update_weights_time = 0
    ps.wait_for_ckpt_time = 0
    ps.async_barrier_clear = asyncio.Event()
    ps.async_barrier_clear.set()
    ps.max_async_level = 1
    ps.strict_async_level = False
    ps.progress = Progress(step=9)
    train_scheduler.progress = Progress(step=9)
    return ps


def test_drop_stale_groups():
    async def run() -> None:
        scheduler = _make_train_scheduler()

        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))

        stale_group = RolloutGroup(
            env_name="test",
            example={},
            requires_group_scoring=False,
            rollouts_per_example=1,
        )
        stale_group.off_policy_steps = 1  # at max_off_policy_steps

        survivor_group = RolloutGroup(
            env_name="test",
            example={},
            requires_group_scoring=False,
            rollouts_per_example=1,
        )
        survivor_group.off_policy_steps = 0  # below threshold

        scheduler._groups = {
            stale_group.group_id: stale_group,
            survivor_group.group_id: survivor_group,
        }
        scheduler._requests = {
            stale_task: RolloutRequest(group_id=stale_group.group_id, cost=1),
            survivor_task: RolloutRequest(group_id=survivor_group.group_id, cost=1),
        }
        scheduler.dispatcher.limiter.concurrency.try_acquire(2)

        scheduler.on_weights_updated()

        assert stale_group.group_id not in scheduler._groups
        assert survivor_group.group_id in scheduler._groups
        assert stale_task not in scheduler._requests
        assert survivor_task in scheduler._requests
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task):
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_policy_scheduler_applies_latest_checkpoint():
    async def run_test() -> None:
        scheduler = _make_train_scheduler()
        ps = _make_policy_scheduler(scheduler)
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)

        ps.inference_pool = MagicMock(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.on_weights_updated = MagicMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(ps.start())
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert applied_steps == [8]
        assert ps.ckpt_step == 8

    asyncio.run(run_test())


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

        ps.inference_pool = MagicMock(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.on_weights_updated = MagicMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new_callable=AsyncMock),
        ):
            policy_task = asyncio.create_task(ps.start())
            await started.wait()
            policy_task.cancel()
            try:
                await asyncio.wait_for(policy_task, timeout=0.2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        assert cancelled.is_set()

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
