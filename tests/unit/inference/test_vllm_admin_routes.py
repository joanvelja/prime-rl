import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest, UnloadLoRAAdapterRequest
from vllm.lora.request import LoRARequest


def _request(query_params: dict[str, str] | None = None):
    engine_client = SimpleNamespace(pause_generation=AsyncMock())
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine_client)),
        query_params=query_params or {},
    )
    return request, engine_client


def test_pause_route_defaults_to_wait_and_clear_cache():
    from prime_rl.inference.vllm.server import pause

    request, engine_client = _request()

    result = asyncio.run(pause(request))

    assert result == {"status": "paused", "mode": "wait", "clear_cache": True}
    engine_client.pause_generation.assert_awaited_once_with(mode="wait", clear_cache=True)


def test_pause_route_honors_explicit_pause_mode():
    from prime_rl.inference.vllm.server import pause

    request, engine_client = _request({"mode": "abort", "clear_cache": "false"})

    result = asyncio.run(pause(request))

    assert result == {"status": "paused", "mode": "abort", "clear_cache": False}
    engine_client.pause_generation.assert_awaited_once_with(mode="abort", clear_cache=False)


def test_load_lora_adapter_route_stages_prime_broadcast_path(monkeypatch, tmp_path):
    from prime_rl.inference.vllm import server

    shared = tmp_path / "shared" / "step_1"
    staged = tmp_path / "shm" / "adapter"
    handler = SimpleNamespace(load_lora_adapter=AsyncMock(return_value="ok"))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    monkeypatch.setattr(server, "models", lambda raw_request: handler)
    monkeypatch.setattr(server, "maybe_stage_lora_adapter", lambda source, name: staged)
    monkeypatch.setattr(server, "_log_lora_diagnostics", AsyncMock())

    result = asyncio.run(
        server.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name="debate-r64__v00000001", lora_path=shared.as_posix()),
            request,
        )
    )

    assert result == {"status": "ok"}
    sent_request = handler.load_lora_adapter.await_args.args[0]
    assert sent_request.lora_path == staged.as_posix()
    assert request.app.state.prime_rl_staged_lora_paths == {"debate-r64__v00000001": staged.as_posix()}


def test_unload_lora_adapter_route_cleans_staged_path(monkeypatch, tmp_path):
    from prime_rl.inference.vllm import server

    staged = tmp_path / "shm" / "adapter"
    staged.mkdir(parents=True)
    handler = SimpleNamespace(
        lora_requests={
            "debate-r64__v00000001": LoRARequest(
                lora_name="debate-r64__v00000001",
                lora_int_id=17,
                lora_path="/staged/adapter",
            )
        },
        unload_lora_adapter=AsyncMock(return_value="ok"),
    )
    engine_client = SimpleNamespace(remove_lora=AsyncMock(return_value=True))
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_client=engine_client,
                prime_rl_staged_lora_paths={"debate-r64__v00000001": staged.as_posix()},
            )
        )
    )
    cleaned: list[Path] = []

    monkeypatch.setattr(server, "models", lambda raw_request: handler)
    monkeypatch.setattr(server, "cleanup_staged_lora_adapter", lambda path: cleaned.append(path))
    monkeypatch.setattr(server, "_log_lora_diagnostics", AsyncMock())

    result = asyncio.run(
        server.unload_lora_adapter(UnloadLoRAAdapterRequest(lora_name="debate-r64__v00000001"), request)
    )

    assert result == {"status": "ok"}
    engine_client.remove_lora.assert_awaited_once_with(17)
    handler.unload_lora_adapter.assert_awaited_once()
    assert cleaned == [staged]
    assert request.app.state.prime_rl_staged_lora_paths == {}
