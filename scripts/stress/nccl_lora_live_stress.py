"""Single-node live stress for NCCL LoRA updates under active generation.

This boots a local vLLM server with the NCCL worker extension, initializes a
NCCL group (trainer rank + one or more inference GPUs), bootstraps a versioned
LoRA adapter, then repeatedly:

1. starts long generations against the previous adapter version,
2. runs the real client-side NCCL LoRA update path (pause keep -> NCCL_READY
   -> /update_lora -> resume),
3. broadcasts the next adapter tensors from the trainer rank over NCCL,
4. verifies the old-version generations complete successfully.

It is still one-node, so it cannot prove the 12-node production topology. With
``--num-servers 3`` it does exercise multi-peer NCCL fanout, per-peer admin
gathering, adapter registration, and routing while every peer is serving.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcastSender
from prime_rl.utils.client import init_nccl_broadcast, update_lora_adapter
from prime_rl.utils.lora import versioned_lora_adapter, versioned_lora_name


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def write_config(path: Path, *, model_name: str, port: int, rank: int, max_loras: int) -> None:
    path.write_text(
        f"""
enable_lora = true
max_loras = {max_loras}
max_cpu_loras = 16
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


async def wait_model_available(base_url: str, model_name: str, *, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            response = await client.get(f"{base_url}/v1/models")
            response.raise_for_status()
            models = [entry["id"] for entry in response.json()["data"]]
            if model_name in models:
                return
            await asyncio.sleep(0.5)
    raise TimeoutError(f"model {model_name!r} did not appear in /v1/models")


def make_sender(host: str, port: int, *, device_index: int, world_size: int, timeout: int) -> NCCLWeightBroadcastSender:
    torch.cuda.set_device(device_index)
    return NCCLWeightBroadcastSender(
        host=host,
        port=port,
        rank=0,
        world_size=world_size,
        device=torch.device(f"cuda:{device_index}"),
        timeout=timeout,
        dtype=torch.bfloat16,
    )


async def initialize_nccl(base_urls: list[str], host: str, port: int, *, sender_device_index: int, timeout: int):
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=30.0, pool=5.0)) as client:
        clients = [httpx.AsyncClient(base_url=base_url, timeout=client.timeout) for base_url in base_urls]
        init_task = asyncio.create_task(
            init_nccl_broadcast(clients, host=host, port=port, timeout=timeout, inference_world_size=len(clients))
        )
        try:
            sender = await asyncio.to_thread(
                make_sender,
                host,
                port,
                device_index=sender_device_index,
                world_size=len(clients) + 1,
                timeout=timeout,
            )
            await init_task
            return sender
        finally:
            for admin_client in clients:
                await admin_client.aclose()


def resolve_target_modules(value: str) -> list[str]:
    if value == "all-linear":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [module.strip() for module in value.split(",") if module.strip()]


def module_shape(config: Any, module: str) -> tuple[str, int, int]:
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // int(config.num_attention_heads)))
    kv_size = int(config.num_key_value_heads) * head_dim
    if module in {"q_proj", "o_proj"}:
        return "self_attn", hidden_size, hidden_size
    if module in {"k_proj", "v_proj"}:
        return "self_attn", hidden_size, kv_size
    if module in {"gate_proj", "up_proj"}:
        return "mlp", hidden_size, intermediate_size
    if module == "down_proj":
        return "mlp", intermediate_size, hidden_size
    raise ValueError(f"unsupported target module {module!r}")


def lora_state_dict(config: Any, *, rank: int, step: int, target_modules: list[str]) -> dict[str, torch.Tensor]:
    hidden_size = int(config.hidden_size)
    num_layers = int(config.num_hidden_layers)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(10_000 + step)
    state_dict = {}
    for layer_idx in range(num_layers):
        for module in target_modules:
            block, in_features, out_features = module_shape(config, module)
            prefix = f"base_model.model.model.layers.{layer_idx}.{block}.{module}"
            a = torch.randn((rank, in_features), generator=gen, dtype=torch.float32) * 0.01
            b = torch.randn((out_features, rank), generator=gen, dtype=torch.float32) * 0.01
            state_dict[f"{prefix}.lora_A.weight"] = a.to(device="cuda", dtype=torch.bfloat16)
            state_dict[f"{prefix}.lora_B.weight"] = b.to(device="cuda", dtype=torch.bfloat16)
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    return state_dict


def lora_header(base_name: str, *, step: int, rank: int, alpha: int, target_modules: list[str]) -> dict[str, Any]:
    return {
        **versioned_lora_adapter(base_name, step),
        "rank": rank,
        "alpha": alpha,
        "peft_config": {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": target_modules,
            "bias": "none",
        },
    }


async def send_lora_update(
    sender: NCCLWeightBroadcastSender,
    *,
    base_name: str,
    step: int,
    rank: int,
    alpha: int,
    config: Any,
    target_modules: list[str],
) -> float:
    torch.cuda.set_device(sender.communicator.device)
    state_dict = lora_state_dict(config, rank=rank, step=step, target_modules=target_modules)
    header = lora_header(base_name, step=step, rank=rank, alpha=alpha, target_modules=target_modules)
    start = time.perf_counter()
    await asyncio.to_thread(sender.broadcast_lora_update, step, header, state_dict)
    return time.perf_counter() - start


async def update_lora_via_real_client(
    base_urls: list[str],
    *,
    base_name: str,
    weight_dir: Path,
    step: int,
) -> float:
    clients = [httpx.AsyncClient(base_url=base_url, timeout=None) for base_url in base_urls]
    try:
        start = time.perf_counter()
        await update_lora_adapter(clients, base_name, weight_dir, step)
        return time.perf_counter() - start
    finally:
        for client in clients:
            await client.aclose()


async def nccl_update(
    sender: NCCLWeightBroadcastSender,
    base_urls: list[str],
    *,
    base_name: str,
    weight_dir: Path,
    step: int,
    rank: int,
    alpha: int,
    config: Any,
    target_modules: list[str],
) -> dict[str, float]:
    client_task = asyncio.create_task(
        update_lora_via_real_client(base_urls, base_name=base_name, weight_dir=weight_dir, step=step)
    )
    marker = weight_dir / "NCCL_READY"
    deadline = time.monotonic() + 30
    while not marker.exists():
        if client_task.done():
            await client_task
        if time.monotonic() > deadline:
            raise TimeoutError(f"{marker} was not created within 30s")
        await asyncio.sleep(0.05)
    send_s = await send_lora_update(
        sender,
        base_name=base_name,
        step=step,
        rank=rank,
        alpha=alpha,
        config=config,
        target_modules=target_modules,
    )
    client_s = await client_task
    return {"send_s": send_s, "client_s": client_s}


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
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=900.0, write=30.0, pool=5.0)) as client:
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


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def run_cycle(
    sender: NCCLWeightBroadcastSender,
    base_urls: list[str],
    *,
    base_name: str,
    output_dir: Path,
    config: Any,
    prev_step: int,
    next_step: int,
    rank: int,
    alpha: int,
    token_ids: list[int],
    concurrency: int,
    max_tokens: int,
    settle_s: float,
    target_modules: list[str],
) -> dict[str, Any]:
    old_model = versioned_lora_name(base_name, prev_step)
    tasks = [
        asyncio.create_task(
            generate_once(base_url, model=old_model, token_ids=token_ids, max_tokens=max_tokens, request_idx=i)
        )
        for server_idx, base_url in enumerate(base_urls)
        for i in range(concurrency)
    ]
    await asyncio.sleep(settle_s)
    update_stats = await nccl_update(
        sender,
        base_urls,
        base_name=base_name,
        weight_dir=output_dir / "broadcasts" / f"step_{next_step}",
        step=next_step,
        rank=rank,
        alpha=alpha,
        config=config,
        target_modules=target_modules,
    )
    await asyncio.gather(
        *(wait_model_available(base_url, versioned_lora_name(base_name, next_step)) for base_url in base_urls)
    )
    results = await asyncio.gather(*tasks)
    errors = [r for r in results if not r["ok"]]
    ok_latencies = [float(r["seconds"]) for r in results if r["ok"]]
    summary = {
        "prev_step": prev_step,
        "next_step": next_step,
        "old_model": old_model,
        "new_model": versioned_lora_name(base_name, next_step),
        "servers": len(base_urls),
        "concurrency_per_server": concurrency,
        "total_concurrency": len(base_urls) * concurrency,
        "max_tokens": max_tokens,
        **update_stats,
        "ok": len(results) - len(errors),
        "errors": len(errors),
        "latency_p50_s": percentile(ok_latencies, 50),
        "latency_max_s": max(ok_latencies) if ok_latencies else None,
        "first_error": errors[0] if errors else None,
    }
    emit("cycle", **summary)
    return summary


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
    server_cuda_indices = parse_cuda_indices(args)
    if args.sender_cuda_index in server_cuda_indices:
        raise ValueError(
            f"sender CUDA index {args.sender_cuda_index} overlaps server CUDA indices {server_cuda_indices}"
        )

    snapshot = snapshot_download(args.model, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    emit("model_cached", model=args.model, snapshot=snapshot, hidden_size=getattr(model_config, "hidden_size", None))
    target_modules = resolve_target_modules(args.target_modules)
    emit("lora_shape", rank=args.rank, target_modules=target_modules)
    procs: list[subprocess.Popen] = []
    base_urls = []
    try:
        for server_idx, cuda_index in enumerate(server_cuda_indices):
            port = args.port + server_idx
            log_path = output_dir / f"server_{server_idx}.log"
            config_path = output_dir / f"inference_{server_idx}.toml"
            write_config(config_path, model_name=args.model, port=port, rank=args.rank, max_loras=args.max_loras)
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(cuda_index),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "OMP_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                    "TRITON_CACHE_DIR": str(output_dir / f"triton-cache-{server_idx}"),
                }
            )
            command = ["uv", "run", "--no-sync", "inference", "@", str(config_path)]
            emit(
                "server_start", command=command, log=str(log_path), server_idx=server_idx, server_cuda_index=cuda_index
            )
            log_file = log_path.open("w")
            proc = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=Path(__file__).resolve().parents[2],
                start_new_session=True,
            )
            log_file.close()
            procs.append(proc)
            base_urls.append(f"http://127.0.0.1:{port}")

        await asyncio.gather(
            *(
                wait_healthy(base_url, proc, timeout_s=args.startup_timeout_s)
                for base_url, proc in zip(base_urls, procs)
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        emit("servers_healthy", base_urls=base_urls, prompt_tokens=len(token_ids))

        sender = await initialize_nccl(
            base_urls,
            host=args.nccl_host or socket.gethostname(),
            port=args.nccl_port,
            sender_device_index=args.sender_cuda_index,
            timeout=args.nccl_timeout,
        )
        emit("nccl_initialized", host=args.nccl_host or socket.gethostname(), port=args.nccl_port)

        await nccl_update(
            sender,
            base_urls,
            base_name=args.lora_name,
            weight_dir=output_dir / "broadcasts" / "step_0",
            step=0,
            rank=args.rank,
            alpha=args.alpha,
            config=model_config,
            target_modules=target_modules,
        )
        await asyncio.gather(
            *(wait_model_available(base_url, versioned_lora_name(args.lora_name, 0)) for base_url in base_urls)
        )
        emit("bootstrap_adapter_loaded", model=versioned_lora_name(args.lora_name, 0))

        summaries = []
        for step in range(1, args.cycles + 1):
            summaries.append(
                await run_cycle(
                    sender,
                    base_urls,
                    base_name=args.lora_name,
                    output_dir=output_dir,
                    config=model_config,
                    prev_step=step - 1,
                    next_step=step,
                    rank=args.rank,
                    alpha=args.alpha,
                    token_ids=token_ids,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    settle_s=args.settle_s,
                    target_modules=target_modules,
                )
            )
        failures = [s for s in summaries if s["errors"] or s["client_s"] > args.max_update_s]
        emit("verdict", passed=not failures, failures=failures)
        return 0 if not failures else 2
    finally:
        emit("server_stop")
        for proc in procs:
            terminate_process_group(proc)


def parse_cuda_indices(args: argparse.Namespace) -> list[int]:
    if args.server_cuda_indices:
        indices = [int(value.strip()) for value in args.server_cuda_indices.split(",") if value.strip()]
    elif args.num_servers == 1:
        indices = [args.server_cuda_index]
    else:
        indices = list(range(args.num_servers))
    if len(indices) != args.num_servers:
        raise ValueError(f"--num-servers={args.num_servers} but got CUDA indices {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"server CUDA indices must be unique, got {indices}")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=18280)
    parser.add_argument("--nccl-host", default="")
    parser.add_argument("--nccl-port", type=int, default=29280)
    parser.add_argument("--nccl-timeout", type=int, default=300)
    parser.add_argument("--server-cuda-index", type=int, default=0)
    parser.add_argument("--server-cuda-indices", default="")
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--sender-cuda-index", type=int, default=1)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--max-update-s", type=float, default=30.0)
    parser.add_argument("--lora-name", default="stress-lora")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--target-modules", default="q_proj")
    parser.add_argument("--max-loras", type=int, default=4)
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
