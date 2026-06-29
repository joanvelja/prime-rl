import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock


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
