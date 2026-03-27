import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


class _DummyAfterEvent:
    def __init__(self) -> None:
        self.wait_calls = 0

    def current_stream_wait(self) -> None:
        self.wait_calls += 1


class _DummyBuffer:
    def __init__(self) -> None:
        self.combine_calls: list[dict[str, object]] = []
        self.dispatch_calls: list[dict[str, object]] = []
        self.combine_events: list[_DummyAfterEvent] = []
        self.dispatch_events: list[_DummyAfterEvent] = []

    def combine(self, **kwargs):
        self.combine_calls.append(kwargs)
        event = _DummyAfterEvent()
        self.combine_events.append(event)
        return kwargs["x"].clone(), None, event

    def dispatch(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        event = _DummyAfterEvent()
        self.dispatch_events.append(event)
        return kwargs["x"].clone(), None, None, None, None, event


@pytest.fixture(scope="module")
def deepep_module():
    module_path = Path(__file__).resolve().parents[3] / "src/prime_rl/trainer/distributed/deepep.py"

    deep_ep = ModuleType("deep_ep")
    deep_ep_utils = ModuleType("deep_ep.utils")

    class _FakeBuffer:
        @staticmethod
        def set_num_sms(num_sms: int) -> None:
            return None

    class _FakeEventHandle:
        pass

    class _FakeEventOverlap:
        def __init__(self, handle=None) -> None:
            self.handle = handle

        def current_stream_wait(self) -> None:
            return None

    deep_ep.Buffer = _FakeBuffer
    deep_ep_utils.EventHandle = _FakeEventHandle
    deep_ep_utils.EventOverlap = _FakeEventOverlap

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", deep_ep_utils)

    class _FakeLibrary:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def define(self, *_args, **_kwargs) -> None:
            return None

        def impl(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(torch.library, "Library", _FakeLibrary)
    monkeypatch.setattr(torch.library, "impl", lambda *_args, **_kwargs: (lambda func: func))
    monkeypatch.setattr(torch.library, "register_autograd", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.compiler, "disable", lambda *_args, **_kwargs: (lambda func: func))

    spec = importlib.util.spec_from_file_location("test_prime_rl_trainer_distributed_deepep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    try:
        yield module
    finally:
        monkeypatch.undo()


@pytest.fixture
def deepep_runtime(deepep_module):
    buffer = _DummyBuffer()
    deepep_module._buffer = buffer
    deepep_module._handle_cache.clear()
    deepep_module._pending_dispatch_events.clear()
    deepep_module._pending_combine_event = None
    try:
        yield deepep_module, buffer
    finally:
        deepep_module._buffer = None
        deepep_module._handle_cache.clear()
        deepep_module._pending_dispatch_events.clear()
        deepep_module._pending_combine_event = None


def test_private_sync_dispatch_waits_and_clears_pending_event(deepep_runtime) -> None:
    module, _buffer = deepep_runtime
    event = _DummyAfterEvent()
    handle_id = torch.tensor([1], dtype=torch.int64)
    module._pending_dispatch_events[handle_id.item()] = event

    module._sync_dispatch(handle_id)

    assert event.wait_calls == 1
    assert handle_id.item() not in module._pending_dispatch_events


def test_sync_combine_waits_and_clears_pending_event(deepep_runtime) -> None:
    module, _buffer = deepep_runtime
    event = _DummyAfterEvent()
    module._pending_combine_event = event

    module.sync_combine()

    assert event.wait_calls == 1
    assert module._pending_combine_event is None


def test_deep_ep_combine_autograd_function_uses_saved_handle_and_pending_event(deepep_runtime) -> None:
    module, buffer = deepep_runtime
    handle = object()
    handle_id = torch.tensor([2], dtype=torch.int64)
    module._handle_cache[handle_id.item()] = handle

    x = torch.randn(2, 3, requires_grad=True)
    combined = module._DeepEPCombine.apply(x, handle_id)

    assert module._pending_combine_event is buffer.combine_events[0]

    combined.sum().backward()

    assert buffer.combine_calls[0]["handle"] is handle
    assert buffer.dispatch_calls[0]["handle"] is handle
    assert buffer.dispatch_events[0].wait_calls == 1
    assert torch.equal(x.grad, torch.ones_like(x))
    assert handle_id.item() not in module._handle_cache
