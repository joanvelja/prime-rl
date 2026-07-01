import asyncio
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.watcher import WeightWatcher


def test_orchestrator_raises_when_component_task_fails():
    async def run() -> None:
        async def fail() -> None:
            raise ValueError("watcher boom")

        task = asyncio.create_task(fail(), name="watcher")
        await asyncio.sleep(0)

        orchestrator = object.__new__(Orchestrator)
        orchestrator.component_tasks = [task]

        with pytest.raises(RuntimeError, match="Pipeline component 'watcher' died") as exc_info:
            orchestrator.check_pipeline_health()

        assert isinstance(exc_info.value.__cause__, ValueError)

    asyncio.run(run())


def test_orchestrator_raises_when_component_task_exits_unexpectedly():
    async def run() -> None:
        async def exit_cleanly() -> None:
            return None

        task = asyncio.create_task(exit_cleanly(), name="watcher")
        await asyncio.sleep(0)

        orchestrator = object.__new__(Orchestrator)
        orchestrator.component_tasks = [task]

        with pytest.raises(RuntimeError, match="Pipeline component 'watcher' exited unexpectedly"):
            orchestrator.check_pipeline_health()

    asyncio.run(run())


def test_watcher_gauges_expose_progress_and_health():
    watcher = object.__new__(WeightWatcher)
    watcher.policy = SimpleNamespace(version=3)
    watcher.update_count = 2
    watcher.last_update_weights_time = 1.5
    watcher.last_wait_for_ckpt_time = 0.25
    watcher.ckpt_step = 2
    watcher.last_seen_broadcast_step = 4
    watcher.next_ckpt_step = 4
    watcher.current_update_step = 4
    watcher.last_error = None

    assert watcher.gauges() == {
        "watcher/policy_version": 3.0,
        "watcher/update_count": 2.0,
        "watcher/last_update_weights_time": 1.5,
        "watcher/last_wait_for_ckpt_time": 0.25,
        "watcher/last_seen_broadcast_step": 4.0,
        "watcher/next_ckpt_step": 4.0,
        "watcher/is_updating": 1.0,
        "watcher/current_update_step": 4.0,
        "watcher/healthy": 1.0,
    }


def test_watcher_gauges_mark_unhealthy_after_error():
    watcher = object.__new__(WeightWatcher)
    watcher.policy = SimpleNamespace(version=3)
    watcher.update_count = 2
    watcher.last_update_weights_time = 1.5
    watcher.last_wait_for_ckpt_time = 0.25
    watcher.ckpt_step = 2
    watcher.last_seen_broadcast_step = 4
    watcher.next_ckpt_step = 4
    watcher.current_update_step = None
    watcher.last_error = "RuntimeError('boom')"

    gauges = watcher.gauges()

    assert gauges["watcher/is_updating"] == 0.0
    assert gauges["watcher/current_update_step"] == 0.0
    assert gauges["watcher/healthy"] == 0.0


def test_watcher_retires_only_inactive_nccl_lora_versions():
    class FakeInference:
        def __init__(self):
            self.removed = []
            self.unloaded = []

        async def remove_lora_adapter(self, lora_name: str, lora_int_id: int) -> None:
            self.removed.append((lora_name, lora_int_id))

        async def unload_lora_adapter(self, lora_name: str) -> None:
            self.unloaded.append(lora_name)

    class FakeObserver:
        def active_policy_versions(self) -> set[int]:
            return {2}

    watcher = object.__new__(WeightWatcher)
    watcher.lora_name = "r8-1e-4"
    watcher.config = SimpleNamespace(weight_broadcast=SimpleNamespace(type="nccl"))
    watcher.policy = SimpleNamespace(version=3)
    watcher.observers = [FakeObserver()]
    watcher.inference = FakeInference()
    watcher.live_lora_steps = {1, 2, 3}

    asyncio.run(watcher.retire_unused_lora_versions())

    assert watcher.inference.removed == [("r8-1e-4__v00000001", 2)]
    assert watcher.inference.unloaded == []
    assert watcher.live_lora_steps == {2, 3}


def test_watcher_retires_only_inactive_filesystem_lora_versions():
    class FakeInference:
        def __init__(self):
            self.removed = []
            self.unloaded = []

        async def remove_lora_adapter(self, lora_name: str, lora_int_id: int) -> None:
            self.removed.append((lora_name, lora_int_id))

        async def unload_lora_adapter(self, lora_name: str) -> None:
            self.unloaded.append(lora_name)

    class FakeObserver:
        def active_policy_versions(self) -> set[int]:
            return {2}

    watcher = object.__new__(WeightWatcher)
    watcher.lora_name = "r8-1e-4"
    watcher.config = SimpleNamespace(weight_broadcast=SimpleNamespace(type="filesystem"))
    watcher.policy = SimpleNamespace(version=3)
    watcher.observers = [FakeObserver()]
    watcher.inference = FakeInference()
    watcher.live_lora_steps = {1, 2, 3}

    asyncio.run(watcher.retire_unused_lora_versions())

    assert watcher.inference.removed == []
    assert watcher.inference.unloaded == ["r8-1e-4__v00000001"]
    assert watcher.live_lora_steps == {2, 3}
