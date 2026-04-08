from types import SimpleNamespace

import verifiers as vf

from prime_rl.utils.client import (
    RENDERER_UPSTREAM_BASE_URL_HEADER,
    ProxiedInferencePool,
)


class _FakeInferencePool:
    def __init__(self, clients: list[vf.ClientConfig]):
        self._clients = clients
        self.admin_clients = [SimpleNamespace(base_url="http://admin")]
        self.wait_for_ready_calls: list[tuple[str, int]] = []
        self.update_weights_calls: list[tuple[object, str | None, int]] = []
        self.stop_calls = 0
        self.updated_model_name = None

    @property
    def clients(self) -> list[vf.ClientConfig]:
        return self._clients

    def update_model_name(self, model_name: str) -> None:
        self.updated_model_name = model_name

    async def get_next_client(self) -> vf.ClientConfig:
        return self._clients[0]

    async def wait_for_ready(self, model_name: str, timeout: int = 1800) -> None:
        self.wait_for_ready_calls.append((model_name, timeout))

    async def update_weights(self, weight_dir, lora_name: str | None = None, step: int = 0) -> None:
        self.update_weights_calls.append((weight_dir, lora_name, step))

    def get_metrics(self) -> dict[str, float]:
        return {"ready_servers": 2.0}

    async def stop(self) -> None:
        self.stop_calls += 1


def _make_client(base_url: str, *, client_idx: int = 0, dp_rank: str | None = None) -> vf.ClientConfig:
    headers = {}
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = dp_rank
    return vf.ClientConfig(
        client_idx=client_idx,
        client_type="openai_chat_completions",
        api_key_var="PRIME_API_KEY",
        api_base_url=base_url,
        extra_headers=headers,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )


def test_proxied_inference_pool_wraps_clients_with_upstream_header():
    upstream_pool = _FakeInferencePool(
        [
            _make_client("http://worker-a:8000/v1", client_idx=0, dp_rank="0"),
            _make_client("http://worker-b:8000/v1", client_idx=1, dp_rank="0"),
        ]
    )

    proxy_pool = ProxiedInferencePool(upstream_pool, proxy_base_url="http://127.0.0.1:18100/v1")
    clients = proxy_pool.clients

    assert [client.api_base_url for client in clients] == ["http://127.0.0.1:18100/v1"] * 2
    assert [client.extra_headers[RENDERER_UPSTREAM_BASE_URL_HEADER] for client in clients] == [
        "http://worker-a:8000/v1",
        "http://worker-b:8000/v1",
    ]
    assert clients[0].extra_headers["X-data-parallel-rank"] == "0"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}
    assert upstream_pool.clients[0].api_base_url == "http://worker-a:8000/v1"
    assert RENDERER_UPSTREAM_BASE_URL_HEADER not in upstream_pool.clients[0].extra_headers


def test_proxied_inference_pool_delegates_admin_and_metrics():
    upstream_pool = _FakeInferencePool([_make_client("http://worker-a:8000/v1", client_idx=0)])
    proxy_pool = ProxiedInferencePool(upstream_pool, proxy_base_url="http://127.0.0.1:18100/v1")

    assert proxy_pool.admin_clients == upstream_pool.admin_clients
    assert proxy_pool.get_metrics() == {"ready_servers": 2.0}
