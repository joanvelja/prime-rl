from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
from dataclasses import dataclass, field
from typing import Any

import httpx
import verifiers as vf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from prime_rl.configs.orchestrator import MultiAgentConfig, MultiAgentTargetConfig
from prime_rl.orchestrator.member_bindings import (
    DEFAULT_TARGET_HEADER,
    LINEAGE_HEADER,
    ROLLOUT_ID_HEADER,
    resolve_member_binding,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _ActorTarget:
    base_urls: tuple[str, ...]
    api_key_var: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 3600.0
    connect_timeout: float = 5.0
    model: str | None = None
    _next_url_idx: int = 0

    @classmethod
    def from_vf(cls, config: vf.ClientConfig, *, model: str) -> "_ActorTarget":
        return cls(
            base_urls=(config.api_base_url,),
            api_key_var=config.api_key_var,
            headers=dict(config.extra_headers),
            timeout=config.timeout,
            connect_timeout=config.connect_timeout,
            model=model,
        )

    @classmethod
    def from_config(cls, config: MultiAgentTargetConfig) -> "_ActorTarget":
        client = config.client
        return cls(
            base_urls=tuple(client.base_url),
            api_key_var=client.api_key_var,
            headers=dict(client.headers),
            timeout=float(client.timeout),
            connect_timeout=float(client.connect_timeout),
            model=config.model,
        )

    def next_base_url(self) -> str:
        if not self.base_urls:
            raise ValueError("Actor target has no base_url entries")
        url = self.base_urls[self._next_url_idx % len(self.base_urls)]
        self._next_url_idx += 1
        return url


class MultiAgentActorProxy:
    """OpenAI-compatible runtime router keyed by Verifiers member identity."""

    def __init__(
        self,
        *,
        multi_agent: MultiAgentConfig,
        default_model: str,
        logger,
    ) -> None:
        self.multi_agent = multi_agent
        self.default_model = default_model
        self.logger = logger
        self.app = FastAPI()
        self._targets: dict[str, _ActorTarget] = {
            name: _ActorTarget.from_config(target) for name, target in multi_agent.targets.items()
        }
        self._default_targets: dict[str, _ActorTarget] = {}
        self._port: int | None = None
        self._server: Any | None = None
        self._server_task: asyncio.Task | None = None
        self._client = httpx.AsyncClient(timeout=None)

        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/{path:path}", self._handle, methods=["GET", "POST"])

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("MultiAgentActorProxy has not been started")
        return f"http://127.0.0.1:{self._port}"

    async def _health(self) -> dict[str, str]:
        return {"status": "ok"}

    def _default_target_id(self, client_config: vf.ClientConfig, model: str) -> str:
        payload = {
            "api_base_url": client_config.api_base_url,
            "api_key_var": client_config.api_key_var,
            "extra_headers": client_config.extra_headers,
            "model": model,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def client_config_for(self, client_config: vf.ClientConfig, model: str) -> vf.ClientConfig:
        target_id = self._default_target_id(client_config, model)
        self._default_targets.setdefault(target_id, _ActorTarget.from_vf(client_config, model=model))

        extra_headers = {}
        extra_headers[DEFAULT_TARGET_HEADER] = target_id
        extra_headers_from_state = dict(client_config.extra_headers_from_state)
        extra_headers_from_state.setdefault(ROLLOUT_ID_HEADER, "trajectory_id")

        return client_config.model_copy(
            update={
                "api_base_url": f"{self.base_url}/v1",
                "endpoint_configs": [],
                "extra_headers": extra_headers,
                "extra_headers_from_state": extra_headers_from_state,
            }
        )

    async def start(self) -> None:
        if self._server_task is not None:
            return
        import uvicorn

        self._port = _free_port()
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if getattr(self._server, "started", False):
                self.logger.info(f"Started multi-agent actor proxy at {self.base_url}")
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("Timed out starting multi-agent actor proxy")

    async def stop(self) -> None:
        await self._client.aclose()
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await self._server_task
            self._server_task = None

    def _select_target(self, request: Request, body: dict[str, Any] | None) -> tuple[_ActorTarget, str]:
        member_id = request.headers.get(LINEAGE_HEADER)
        rollout_id = request.headers.get(ROLLOUT_ID_HEADER) or ""
        binding = resolve_member_binding(self.multi_agent, member_id=member_id, rollout_id=rollout_id)

        if binding.target is None:
            target_id = request.headers.get(DEFAULT_TARGET_HEADER)
            target = self._default_targets.get(target_id or "")
            if target is None:
                raise ValueError("Missing or unknown default actor target")
        else:
            target = self._targets.get(binding.target)
            if target is None:
                raise ValueError(f"Unknown multi-agent actor target: {binding.target!r}")

        model = binding.model or target.model or (body or {}).get("model") or self.default_model
        return target, str(model)

    @staticmethod
    def _forward_url(base_url: str, path: str, query: bytes) -> str:
        base = base_url.rstrip("/")
        if path.startswith("/chat/completions/tokens") and base.endswith("/v1"):
            base = base[:-3]
            url = f"{base}{path}"
        elif base.endswith("/v1") and path.startswith("/v1/"):
            url = f"{base}{path[3:]}"
        elif not base.endswith("/v1") and not path.startswith("/v1"):
            url = f"{base}/v1{path}"
        else:
            url = f"{base}{path}"
        if query:
            url = f"{url}?{query.decode()}"
        return url

    @staticmethod
    def _forward_headers(request: Request, target: _ActorTarget) -> dict[str, str]:
        blocked = {
            "host",
            "content-length",
            "connection",
            "authorization",
            DEFAULT_TARGET_HEADER.lower(),
            LINEAGE_HEADER.lower(),
            ROLLOUT_ID_HEADER.lower(),
            *{name.lower() for name in target.headers},
        }
        headers = {k: v for k, v in request.headers.items() if k.lower() not in blocked}
        headers.update(target.headers)
        if "authorization" not in {k.lower() for k in headers}:
            api_key = os.getenv(target.api_key_var) or "EMPTY"
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _handle(self, request: Request) -> Response:
        body_bytes = await request.body()
        body: dict[str, Any] | None = None
        if body_bytes:
            try:
                parsed = json.loads(body_bytes)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                body = parsed

        try:
            target, model = self._select_target(request, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if body is not None:
            body = {**body, "model": model}
            content = json.dumps(body).encode()
        else:
            content = body_bytes

        url = self._forward_url(target.next_base_url(), request.url.path, request.url.query)
        headers = self._forward_headers(request, target)
        timeout = httpx.Timeout(target.timeout, connect=target.connect_timeout)
        upstream = self._client.build_request(
            request.method,
            url,
            content=content,
            headers=headers,
            timeout=timeout,
        )
        response = await self._client.send(upstream, stream=True)

        response_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
        }
        if "text/event-stream" in response.headers.get("content-type", ""):
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=response_headers,
                background=BackgroundTask(response.aclose),
            )

        content_bytes = await response.aread()
        await response.aclose()
        return Response(content=content_bytes, status_code=response.status_code, headers=response_headers)
