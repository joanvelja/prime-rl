from __future__ import annotations

import json

import httpx
import pytest
import verifiers as vf

from prime_rl.configs.orchestrator import (
    MultiAgentConfig,
    MultiAgentMemberBindingConfig,
    MultiAgentTargetConfig,
)
from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.actor_proxy import MultiAgentActorProxy
from prime_rl.orchestrator.member_bindings import (
    DEFAULT_TARGET_HEADER,
    LINEAGE_HEADER,
    ROLLOUT_ID_HEADER,
)


class _Logger:
    def info(self, _message: str) -> None:
        pass


class _FakeUpstreamClient:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
        kwargs.pop("timeout", None)
        return httpx.Request(method, url, **kwargs)

    async def send(self, request: httpx.Request, *, stream: bool) -> httpx.Response:
        assert stream
        self.requests.append(request)
        return httpx.Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
            request=request,
        )

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_actor_proxy_routes_by_member_and_rewrites_model_and_auth(monkeypatch):
    monkeypatch.setenv("LEARNER_KEY", "learner-secret")
    monkeypatch.setenv("OPPONENT_KEY", "opponent-secret")

    proxy = MultiAgentActorProxy(
        multi_agent=MultiAgentConfig(
            targets={
                "opponent": MultiAgentTargetConfig(
                    model="opponent-model",
                    client=ClientConfig(
                        base_url=["https://opponent.example/v1"],
                        api_key_var="OPPONENT_KEY",
                        headers={"X-Opponent": "opponent"},
                    ),
                )
            },
            member_bindings={
                "debater_b": MultiAgentMemberBindingConfig(
                    target="opponent",
                    trainable=False,
                )
            },
        ),
        default_model="learner-model",
        logger=_Logger(),
    )
    proxy._port = 32123
    fake_upstream = _FakeUpstreamClient()
    proxy._client = fake_upstream

    learner_config = vf.ClientConfig(
        api_base_url="http://learner.example/v1",
        api_key_var="LEARNER_KEY",
        extra_headers={"X-data-parallel-rank": "2"},
    )
    routed_config = proxy.client_config_for(learner_config, "learner-model")

    assert routed_config.api_base_url == "http://127.0.0.1:32123/v1"
    assert routed_config.extra_headers[DEFAULT_TARGET_HEADER]
    assert "X-data-parallel-rank" not in routed_config.extra_headers
    assert routed_config.extra_headers_from_state[ROLLOUT_ID_HEADER] == "trajectory_id"

    transport = httpx.ASGITransport(app=proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url=proxy.base_url) as client:
        await client.post(
            "/v1/chat/completions",
            headers={
                **routed_config.extra_headers,
                LINEAGE_HEADER: "debater_b",
                ROLLOUT_ID_HEADER: "episode-1",
                "Authorization": "Bearer learner-secret",
            },
            json={"model": "learner-model", "messages": []},
        )
        await client.post(
            "/v1/chat/completions",
            headers={
                **routed_config.extra_headers,
                LINEAGE_HEADER: "debater_a",
                ROLLOUT_ID_HEADER: "episode-1",
                "Authorization": "Bearer learner-secret",
            },
            json={"model": "learner-model", "messages": []},
        )

    opponent_request, learner_request = fake_upstream.requests
    assert str(opponent_request.url) == "https://opponent.example/v1/chat/completions"
    assert json.loads(opponent_request.content)["model"] == "opponent-model"
    assert opponent_request.headers["authorization"] == "Bearer opponent-secret"
    assert opponent_request.headers["x-opponent"] == "opponent"
    assert LINEAGE_HEADER not in opponent_request.headers
    assert DEFAULT_TARGET_HEADER not in opponent_request.headers
    assert ROLLOUT_ID_HEADER not in opponent_request.headers
    assert "x-data-parallel-rank" not in opponent_request.headers

    assert str(learner_request.url) == "http://learner.example/v1/chat/completions"
    assert json.loads(learner_request.content)["model"] == "learner-model"
    assert learner_request.headers["authorization"] == "Bearer learner-secret"
    assert learner_request.headers["x-data-parallel-rank"] == "2"
