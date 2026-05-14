import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import verifiers as vf

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils import client as client_module
from prime_rl.utils.client import (
    _is_retryable_admin_error,
    _is_retryable_lora_error,
    _post_admin_control,
    check_health,
    load_lora_adapter,
    setup_clients,
    update_weights,
)


class StubResponse:
    def raise_for_status(self):
        pass


def test_setup_clients_defaults_to_vllm_permissive_profile():
    clients = setup_clients(ClientConfig(base_url=["http://localhost:8000/v1"]))

    assert [client.profile for client in clients] == ["vllm_permissive"]


def test_setup_clients_allows_openai_strict_profile():
    clients = setup_clients(ClientConfig(base_url=["https://api.openai.com/v1"], api_profile="openai_strict"))

    assert [client.profile for client in clients] == ["openai_strict"]


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_is_retryable_admin_error_retries_transport_errors():
    request = httpx.Request("POST", "http://test/pause")
    assert _is_retryable_admin_error(httpx.RemoteProtocolError("disconnect", request=request)) is True


def test_is_retryable_admin_error_rejects_bad_request():
    response = httpx.Response(400, request=httpx.Request("POST", "http://test/pause"))
    error = httpx.HTTPStatusError("Bad request", request=response.request, response=response)
    assert _is_retryable_admin_error(error) is False


def test_post_admin_control_retries_transport_error(monkeypatch):
    monkeypatch.setattr(client_module, "ADMIN_CONTROL_INITIAL_BACKOFF_S", 0)

    mock_client = AsyncMock()
    mock_client.base_url = "http://test"
    request = httpx.Request("POST", "http://test/pause")
    mock_client.post.side_effect = [httpx.RemoteProtocolError("disconnect", request=request), StubResponse()]

    asyncio.run(_post_admin_control(mock_client, "/pause", operation="pause inference engine"))

    assert mock_client.post.call_count == 2


def test_check_health_requires_2xx_status(monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", no_sleep)
    request = httpx.Request("GET", "http://test/health")
    mock_client = AsyncMock()
    mock_client.base_url = "http://test"
    mock_client.get.return_value = httpx.Response(500, request=request)

    with pytest.raises(TimeoutError):
        asyncio.run(check_health([mock_client], interval=1, timeout=1))


def test_check_health_skips_missing_route():
    request = httpx.Request("GET", "http://test/health")
    mock_client = AsyncMock()
    mock_client.base_url = "http://test"
    mock_client.get.return_value = httpx.Response(404, request=request)

    asyncio.run(check_health([mock_client], interval=1, timeout=1))


def test_update_weights_resumes_after_pause_failure(monkeypatch):
    monkeypatch.setattr(client_module, "ADMIN_CONTROL_MAX_ATTEMPTS", 1)

    failing_client = AsyncMock()
    failing_client.base_url = "http://failing"
    healthy_client = AsyncMock()
    healthy_client.base_url = "http://healthy"
    resumed: list[str] = []

    request = httpx.Request("POST", "http://failing/pause")

    async def failing_post(path, **kwargs):
        if path == "/pause":
            raise httpx.RemoteProtocolError("disconnect", request=request)
        if path == "/resume":
            resumed.append("failing")
        return StubResponse()

    async def healthy_post(path, **kwargs):
        if path == "/resume":
            resumed.append("healthy")
        return StubResponse()

    failing_client.post = failing_post
    healthy_client.post = healthy_post

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(update_weights([failing_client, healthy_client], Path("/tmp/weights")))

    assert sorted(resumed) == ["failing", "healthy"]


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_name="qwen3_vl",
    )

    assert [client.client_type for client in clients] == ["renderer", "renderer"]
    assert [client.renderer for client in clients] == ["qwen3_vl", "qwen3_vl"]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.api_base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].extra_headers["X-Test"] == "test"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}


def test_setup_clients_assigns_renderer_model_name():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_name="qwen3_vl",
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert clients[0].renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_clients_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(client_config)

    assert clients == [
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            profile="vllm_permissive",
            renderer="auto",
            renderer_model_name=None,
            api_key_var="PRIME_API_KEY",
            api_base_url="http://worker-a:8000/v1",
            timeout=client_config.timeout,
            connect_timeout=client_config.connect_timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers={},
            extra_headers_from_state={},
        )
    ]
