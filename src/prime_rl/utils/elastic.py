"""
Elastic inference pool helpers: DNS-based service discovery and data classes.

The pool class itself lives in prime_rl.utils.client.InferencePool.
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from prime_rl.utils.logger import get_logger

# --- Discovery functions ---


def discover_server_ips(hostname: str) -> list[str]:
    """Discover server IPs via DNS lookup."""
    try:
        _, _, ips = socket.gethostbyname_ex(hostname)
        return sorted(ips)
    except socket.gaierror:
        return []


async def check_server_model(url: str, model_name: str, timeout: float = 5.0) -> tuple[bool, bool]:
    """Check if a server has a specific model loaded."""
    logger = get_logger()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            return model_name in models, len(models) > 0
    except Exception as e:
        logger.debug(f"Failed to check server {url}: {e}")
        return False, False


async def discover_ready_servers(hostname: str, port: int, model_name: str) -> list[str]:
    """Discover servers via DNS that have the requested model loaded."""
    loop = asyncio.get_event_loop()
    ips = await loop.run_in_executor(None, discover_server_ips, hostname)
    if not ips:
        return []

    checks = [check_server_model(f"http://{ip}:{port}", model_name) for ip in ips]
    results = await asyncio.gather(*checks, return_exceptions=True)

    with_model = set()
    for ip, result in zip(ips, results):
        if isinstance(result, BaseException):
            continue
        has_model, _ = result
        if has_model:
            with_model.add(f"http://{ip}:{port}/v1")

    return sorted(with_model)


# --- Data classes ---


@dataclass
class AdapterState:
    """State of a LoRA adapter (loaded or desired)."""

    name: str | None = None
    path: Path | None = None
    step: int = 0


ServerStatus = Literal["discovering", "syncing", "ready", "unhealthy"]


@dataclass
class ServerState:
    """State of an individual inference server."""

    ip: str
    url: str
    status: ServerStatus = "discovering"
    loaded_adapter: AdapterState | None = None
    sync_failures: int = 0
