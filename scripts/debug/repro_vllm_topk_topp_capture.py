#!/usr/bin/env python3
"""Compare vLLM Triton top-k/top-p against the PyTorch reference on a capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "")

import torch
from vllm.v1.sample.ops import topk_topp_triton as triton_mod
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


def finite_counts(x: torch.Tensor) -> list[int]:
    return torch.isfinite(x).sum(dim=-1).detach().cpu().tolist()


def summarize(label: str, x: torch.Tensor) -> dict:
    return {
        "label": label,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite_counts": finite_counts(x),
        "all_neginf_rows": torch.all(x == float("-inf"), dim=-1).detach().cpu().tolist(),
        "any_nan_rows": torch.any(torch.isnan(x), dim=-1).detach().cpu().tolist(),
        "any_posinf_rows": torch.any(x == float("inf"), dim=-1).detach().cpu().tolist(),
    }


def summarize_bad_rows(label: str, x: torch.Tensor, bad_idx: list[int]) -> dict:
    if x.shape[0] == len(bad_idx):
        rows = x
    else:
        rows = x[bad_idx]
    return summarize(label, rows)


def select_rows(payload: dict, row_only: bool) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[int]]:
    pre = payload["pre_logits_batch"]
    bad_idx = payload["bad_idx"].to(torch.long).tolist()
    k = payload.get("k")
    p = payload.get("p")

    if row_only:
        rows = bad_idx[:1]
        pre = pre[rows]
        k = None if k is None else k[rows]
        p = None if p is None else p[rows]
        return pre, k, p, rows

    return pre, k, p, bad_idx


def run_once(pre: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    triton_logits = pre.clone()
    pytorch_logits = pre.clone()
    apply_top_k_top_p_triton(triton_logits, k, p)
    apply_top_k_top_p_pytorch(pytorch_logits, k, p)
    return triton_logits, pytorch_logits


def apply_top_k_top_p_triton_uncached(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert logits.ndim == 2
    assert logits.dtype == torch.float32

    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None
    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits, None

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits

    num_sm = triton_mod.num_compute_units(logits.device.index)
    num_programs = min(num_sm, batch_size)
    buffer_rows = min(triton_mod.next_power_of_2(num_programs), num_sm)
    buffer = logits.new_empty((buffer_rows, vocab_size))

    tables = triton_mod._TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        normal_cdf_to_sigma_table = logits.new_tensor(triton_mod._NORMAL_CDF_TO_SIGMA_TABLE)
        percentile_to_std_table = logits.new_tensor(triton_mod._PERCENTILE_TO_STD_TABLE)
        triton_mod._TRITON_TABLE_CACHE[logits.device] = (
            normal_cdf_to_sigma_table,
            percentile_to_std_table,
        )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    triton_mod._topk_topp_kernel[(num_programs,)](
        logits,
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
    )
    return logits, buffer


def stress_triton(
    pre: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    bad_idx: list[int],
    *,
    streams: int,
    iters: int,
    diverse_inputs: bool,
    uncached: bool,
) -> dict:
    cuda_streams = [torch.cuda.Stream() for _ in range(streams)]
    failures = []
    for base in range(0, iters, streams):
        outs: list[torch.Tensor] = []
        live_buffers: list[torch.Tensor] = []
        for stream_idx, stream in enumerate(cuda_streams):
            if diverse_inputs:
                logits = torch.roll(pre, shifts=(base + stream_idx) % pre.shape[0], dims=0).clone()
            else:
                logits = pre.clone()
            outs.append(logits)
        torch.cuda.synchronize()

        for logits, stream in zip(outs, cuda_streams, strict=True):
            with torch.cuda.stream(stream):
                if uncached:
                    _, buffer = apply_top_k_top_p_triton_uncached(logits, k, p)
                    if buffer is not None:
                        live_buffers.append(buffer)
                else:
                    apply_top_k_top_p_triton(logits, k, p)
        for stream in cuda_streams:
            stream.synchronize()
        live_buffers.clear()

        for offset, logits in enumerate(outs):
            iteration = base + offset
            if iteration >= iters:
                break
            empty = torch.all(logits == float("-inf"), dim=-1)
            if torch.any(empty):
                empty_idx = torch.nonzero(empty, as_tuple=False).flatten()
                failures.append(
                    {
                        "iteration": iteration,
                        "empty_rows": empty_idx.detach().cpu().tolist(),
                        "empty_row_count": int(empty.sum().item()),
                        "bad_row_finite_counts": finite_counts(logits[bad_idx]),
                    }
                )
                if len(failures) >= 8:
                    break
        if len(failures) >= 8:
            break

    return {
        "streams": streams,
        "iters_requested": iters,
        "diverse_inputs": diverse_inputs,
        "uncached_buffer": uncached,
        "failures": failures,
        "num_failures_recorded": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--row-only", action="store_true")
    parser.add_argument("--stress-streams", type=int, default=0)
    parser.add_argument("--stress-iters", type=int, default=200)
    parser.add_argument("--stress-identical-inputs", action="store_true")
    parser.add_argument("--stress-uncached-buffer", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    pre_cpu, k_cpu, p_cpu, bad_idx = select_rows(payload, args.row_only)

    device = torch.device("cuda")
    pre = pre_cpu.to(device=device, dtype=torch.float32)
    k = None if k_cpu is None else k_cpu.to(device=device)
    p = None if p_cpu is None else p_cpu.to(device=device, dtype=torch.float32)

    triton_logits, pytorch_logits = run_once(pre, k, p)

    post_bad = payload.get("post_logits_bad_rows")

    report = {
        "capture": str(args.capture),
        "row_only": args.row_only,
        "bad_idx_original": bad_idx,
        "metadata": payload.get("metadata", {}),
        "p": None if p_cpu is None else p_cpu.detach().cpu().tolist(),
        "k": None if k_cpu is None else k_cpu.detach().cpu().tolist(),
        "pre": summarize("pre", pre),
        "triton": summarize("triton", triton_logits),
        "pytorch": summarize("pytorch", pytorch_logits),
        "live_post_bad_rows": None
        if post_bad is None
        else summarize("live_post_bad_rows", post_bad.to(device=device, dtype=torch.float32)),
        "triton_bad_rows": summarize_bad_rows("triton_bad_rows", triton_logits, bad_idx),
        "pytorch_bad_rows": summarize_bad_rows("pytorch_bad_rows", pytorch_logits, bad_idx),
    }

    if args.stress_streams:
        report["stress"] = stress_triton(
            pre,
            k,
            p,
            bad_idx,
            streams=args.stress_streams,
            iters=args.stress_iters,
            diverse_inputs=not args.stress_identical_inputs,
            uncached=args.stress_uncached_buffer,
        )

    print(json.dumps(report, indent=2))

    triton_empty = torch.all(triton_logits == float("-inf"), dim=-1)
    pytorch_empty = torch.all(pytorch_logits == float("-inf"), dim=-1)
    if torch.any(triton_empty & ~pytorch_empty):
        raise SystemExit(42)


if __name__ == "__main__":
    main()
