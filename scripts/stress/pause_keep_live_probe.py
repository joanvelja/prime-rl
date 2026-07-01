"""Live vLLM pause/resume probe for NCCL-LoRA's keep-pause contract.

Boots a small local inference server, floods it with long generations, then
repeatedly pauses with mode=keep and resumes. This does not test the cross-node
NCCL collective; it tests the specific premise that keep-pause is fast and
preserves in-flight requests under live decode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def write_config(path: Path, *, model_name: str, port: int) -> None:
    path.write_text(
        f"""
enable_lora = true
max_loras = 4
max_cpu_loras = 8
max_lora_rank = 64
enable_prefix_caching = true
gpu_memory_utilization = 0.35
api_server_count = 1
data_parallel_rpc_port = {port + 1000}
seed = 0
enable_expert_parallel = false
enable_dbo = false
use_deep_gemm = false
enable_return_routed_experts = false
enable_fp32_lm_head = false

[server]
host = "0.0.0.0"
port = {port}
liveness_timeout_seconds = 30.0

[model]
name = "{model_name}"
dtype = "bfloat16"
max_model_len = 4096
enforce_eager = true
trust_remote_code = false

[parallel]
tp = 1
dp = 1

[log]
level = "info"
vf_level = "info"
json_logging = false
log_data = false
interval = 2.0

[weight_broadcast]
type = "nccl"

[vllm_extra]
max_num_seqs = 64
max_num_batched_tokens = 32768
generation_config = "vllm"
stream_interval = 1
enable_chunked_prefill = true
""".strip()
        + "\n"
    )


async def wait_healthy(base_url: str, proc: subprocess.Popen, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"inference server exited early with code {proc.returncode}")
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"inference server did not become healthy within {timeout_s:.0f}s")


async def get_model_id(base_url: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{base_url}/v1/models")
        response.raise_for_status()
        data = response.json()["data"]
        if not data:
            raise RuntimeError("/v1/models returned no models")
        return str(data[0]["id"])


async def scrape_metrics(base_url: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{base_url}/metrics")
        response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith("#"):
            continue
        if "num_requests" not in line and "gpu_cache_usage" not in line:
            continue
        match = re.match(r"(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\\{[^}]*\\})? (?P<value>[-+0-9.eE]+)$", line)
        if match:
            metrics[match.group("name")] = float(match.group("value"))
    return metrics


async def post_pause(base_url: str, *, mode: str, clear_cache: bool, timeout_s: float) -> float:
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=timeout_s, write=5.0, pool=2.0)) as client:
        response = await client.post(
            f"{base_url}/pause",
            params={"mode": mode, "clear_cache": str(clear_cache).lower()},
        )
        response.raise_for_status()
    return time.perf_counter() - start


async def post_resume(base_url: str) -> float:
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{base_url}/resume")
        response.raise_for_status()
    return time.perf_counter() - start


async def generate_once(
    base_url: str,
    *,
    model: str,
    token_ids: list[int],
    max_tokens: int,
    request_idx: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    body = {
        "model": model,
        "token_ids": token_ids,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.95,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)) as client:
            response = await client.post(f"{base_url}/inference/v1/generate", json=body)
            response.raise_for_status()
            payload = response.json()
        output_ids = payload.get("output_token_ids") or payload.get("token_ids") or []
        return {
            "request_idx": request_idx,
            "ok": True,
            "seconds": time.perf_counter() - start,
            "output_tokens": len(output_ids) if isinstance(output_ids, list) else None,
        }
    except BaseException as exc:
        return {
            "request_idx": request_idx,
            "ok": False,
            "seconds": time.perf_counter() - start,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


async def run_cycle(
    base_url: str,
    *,
    model: str,
    token_ids: list[int],
    cycle: int,
    concurrency: int,
    max_tokens: int,
    settle_s: float,
) -> dict[str, Any]:
    tasks = [
        asyncio.create_task(
            generate_once(base_url, model=model, token_ids=token_ids, max_tokens=max_tokens, request_idx=i)
        )
        for i in range(concurrency)
    ]
    await asyncio.sleep(settle_s)
    before = await scrape_metrics(base_url)
    pause_s = await post_pause(base_url, mode="keep", clear_cache=False, timeout_s=20.0)
    paused = await scrape_metrics(base_url)
    await asyncio.sleep(2.0)
    resume_s = await post_resume(base_url)
    results = await asyncio.gather(*tasks)
    after = await scrape_metrics(base_url)
    errors = [r for r in results if not r["ok"]]
    ok_latencies = [float(r["seconds"]) for r in results if r["ok"]]
    summary = {
        "cycle": cycle,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "pause_s": pause_s,
        "resume_s": resume_s,
        "ok": len(results) - len(errors),
        "errors": len(errors),
        "latency_p50_s": percentile(ok_latencies, 50),
        "latency_max_s": max(ok_latencies) if ok_latencies else None,
        "metrics_before": before,
        "metrics_paused": paused,
        "metrics_after": after,
        "first_error": errors[0] if errors else None,
    }
    emit("cycle", **summary)
    return summary


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


async def main_async(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "server.log"
    config_path = output_dir / "inference.toml"

    snapshot = snapshot_download(args.model, local_files_only=True)
    emit("model_cached", model=args.model, snapshot=snapshot)
    write_config(config_path, model_name=args.model, port=args.port)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "TRITON_CACHE_DIR": str(output_dir / "triton-cache"),
        }
    )
    command = ["uv", "run", "--no-sync", "inference", "@", str(config_path)]
    emit("server_start", command=command, log=str(log_path))
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            start_new_session=True,
        )

    base_url = f"http://127.0.0.1:{args.port}"
    try:
        await wait_healthy(base_url, proc, timeout_s=args.startup_timeout_s)
        model_id = await get_model_id(base_url)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        emit("server_healthy", base_url=base_url, model_id=model_id, prompt_tokens=len(token_ids))

        summaries = []
        for cycle in range(args.cycles):
            summaries.append(
                await run_cycle(
                    base_url,
                    model=model_id,
                    token_ids=token_ids,
                    cycle=cycle,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    settle_s=args.settle_s,
                )
            )
        failures = [s for s in summaries if s["errors"] or s["pause_s"] > args.max_pause_s]
        emit("verdict", passed=not failures, failures=failures)
        return 0 if not failures else 2
    finally:
        emit("server_stop")
        terminate_process_group(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--max-pause-s", type=float, default=5.0)
    parser.add_argument(
        "--prompt",
        default=(
            "Continue the following numbered list with short random-looking phrases. "
            "Do not summarize and do not stop early. 1."
        ),
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
