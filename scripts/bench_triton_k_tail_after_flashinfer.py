#!/usr/bin/env python3
"""Benchmark a fused Triton K-space tail after FlashInfer top-k.

FlashInfer already gives exact top-K values and token ids; this script checks
whether one Triton kernel can replace the Python-launched K-space
softmax/top-p/sample/logprob tail.

The Triton kernel takes sorted top-K values and ids plus one uniform random
number per row, then returns the sampled token id and sampled-token logprob.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import torch
import triton
import triton.language as tl
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from scripts.bench_vllm_sampled_logprob_fastpath import (
    make_logits,
    parse_int_list,
    select_device,
)


@triton.jit
def _k_tail_uniform_kernel(
    vals,
    ids,
    uniforms,
    out_sampled_ids,
    out_sampled_logprobs,
    out_kept,
    K: tl.constexpr,
    K_BLOCK: tl.constexpr,
    TOP_P: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, K_BLOCK)
    mask = offsets < K

    row_vals = tl.load(vals + row * K + offsets, mask=mask, other=-float("inf"))
    row_ids = tl.load(ids + row * K + offsets, mask=mask, other=0)

    max_val = tl.max(row_vals, axis=0)
    weights = tl.exp(row_vals - max_val)
    weights = tl.where(mask, weights, 0.0)
    prefix = tl.cumsum(weights, 0)
    total_weight = tl.sum(weights, axis=0)
    keep = mask & ((prefix - weights) < TOP_P * total_weight)
    kept_weights = tl.where(keep, weights, 0.0)
    kept_prefix = tl.cumsum(kept_weights, 0)
    support_sum = tl.sum(kept_weights, axis=0)

    threshold = tl.load(uniforms + row) * support_sum
    sample_rank = tl.min(
        tl.where((kept_prefix >= threshold) & keep, offsets, K_BLOCK),
        axis=0,
    )
    sampled_id = tl.max(tl.where(offsets == sample_rank, row_ids, 0), axis=0)
    sampled_val = tl.max(
        tl.where(offsets == sample_rank, row_vals, -float("inf")),
        axis=0,
    )
    sampled_logprob = sampled_val - max_val - tl.log(support_sum)
    kept_count = tl.sum(tl.where(keep, 1, 0), axis=0)

    tl.store(out_sampled_ids + row, sampled_id)
    tl.store(out_sampled_logprobs + row, sampled_logprob)
    tl.store(out_kept + row, kept_count)


@dataclass(frozen=True)
class BenchResult:
    batch_size: int
    vocab_size: int
    top_k: int
    top_p: float
    device: str
    current_random_total_ms: float
    torch_uniform_total_ms: float
    triton_uniform_total_ms: float
    triton_uniform_total_with_rand_ms: float
    current_random_tail_ms: float
    torch_uniform_tail_ms: float
    triton_uniform_tail_ms: float
    torch_rand_ms: float
    topk_sort_ms: float
    output_pack_ms: float
    triton_vs_torch_tail_speedup: float
    triton_total_vs_current: float
    triton_total_with_rand_vs_current: float
    max_sampled_logprob_diff: float
    sampled_ids_match: bool
    kept_min: int
    kept_max: int


def load_flashinfer() -> Any:
    module = importlib.import_module("flashinfer")
    if not hasattr(module, "top_k"):
        raise RuntimeError("Installed FlashInfer does not expose flashinfer.top_k")
    return module


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_fn(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def make_uniforms(batch_size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(batch_size, generator=generator, dtype=torch.float32).to(device)


def flashinfer_topk_sorted(
    flashinfer: Any,
    logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vals, ids = flashinfer.top_k(logits.contiguous(), top_k, sorted=False)
    vals, order = vals.sort(dim=-1, descending=True)
    ids = ids.gather(-1, order)
    return vals, ids


def output_pack(sampled: torch.Tensor, sampled_logprobs: torch.Tensor) -> LogprobsTensors:
    sampled_i32 = sampled.to(torch.int32)
    return LogprobsTensors(
        logprob_token_ids=sampled_i32.unsqueeze(1),
        logprobs=sampled_logprobs.to(torch.float32).unsqueeze(1),
        selected_token_ranks=torch.ones_like(sampled_i32, dtype=torch.int32),
    )


def current_random_tail(
    vals: torch.Tensor,
    ids: torch.Tensor,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.softmax(vals, dim=-1, dtype=torch.float32)
    prefix = weights.cumsum(dim=-1)
    keep = (prefix - weights) < top_p
    support_vals = vals.masked_fill(~keep, -float("inf"))
    support_logprobs = support_vals - torch.logsumexp(
        support_vals,
        dim=-1,
        keepdim=True,
    )
    probs = torch.softmax(support_vals, dim=-1, dtype=torch.float32)
    sampled_in_topk = random_sample(probs, {})
    sampled = ids.gather(1, sampled_in_topk.unsqueeze(1)).squeeze(1).long()
    sampled_logprobs = support_logprobs.gather(
        1,
        sampled_in_topk.unsqueeze(1),
    ).squeeze(1)
    return sampled, sampled_logprobs, keep


def torch_uniform_tail(
    vals: torch.Tensor,
    ids: torch.Tensor,
    top_p: float,
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.softmax(vals, dim=-1, dtype=torch.float32)
    prefix = weights.cumsum(dim=-1)
    keep = (prefix - weights) < top_p
    kept_weights = weights.masked_fill(~keep, 0.0)
    kept_prefix = kept_weights.cumsum(dim=-1)
    support_sum = kept_weights.sum(dim=-1, keepdim=True)
    threshold = uniforms.unsqueeze(1) * support_sum
    sampled_rank = (kept_prefix >= threshold).to(torch.int64).argmax(dim=-1)
    sampled = ids.gather(1, sampled_rank.unsqueeze(1)).squeeze(1).long()

    support_vals = vals.masked_fill(~keep, -float("inf"))
    support_logprobs = support_vals - torch.logsumexp(
        support_vals,
        dim=-1,
        keepdim=True,
    )
    sampled_logprobs = support_logprobs.gather(
        1,
        sampled_rank.unsqueeze(1),
    ).squeeze(1)
    return sampled, sampled_logprobs, keep


def triton_uniform_tail(
    vals: torch.Tensor,
    ids: torch.Tensor,
    top_k: int,
    top_p: float,
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = vals.shape[0]
    sampled = torch.empty(batch_size, device=vals.device, dtype=torch.int64)
    sampled_logprobs = torch.empty(batch_size, device=vals.device, dtype=torch.float32)
    kept = torch.empty(batch_size, device=vals.device, dtype=torch.int32)
    _k_tail_uniform_kernel[(batch_size,)](
        vals,
        ids,
        uniforms,
        sampled,
        sampled_logprobs,
        kept,
        K=top_k,
        K_BLOCK=next_power_of_two(top_k),
        TOP_P=top_p,
        num_warps=1,
    )
    return sampled, sampled_logprobs, kept


def current_random_total(
    flashinfer: Any,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> LogprobsTensors:
    vals, ids = flashinfer_topk_sorted(flashinfer, logits, top_k)
    sampled, sampled_logprobs, _ = current_random_tail(vals, ids, top_p)
    return output_pack(sampled, sampled_logprobs)


def torch_uniform_total(
    flashinfer: Any,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    uniforms: torch.Tensor,
) -> LogprobsTensors:
    vals, ids = flashinfer_topk_sorted(flashinfer, logits, top_k)
    sampled, sampled_logprobs, _ = torch_uniform_tail(vals, ids, top_p, uniforms)
    return output_pack(sampled, sampled_logprobs)


def triton_uniform_total(
    flashinfer: Any,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    uniforms: torch.Tensor,
) -> LogprobsTensors:
    vals, ids = flashinfer_topk_sorted(flashinfer, logits, top_k)
    sampled, sampled_logprobs, _ = triton_uniform_tail(vals, ids, top_k, top_p, uniforms)
    return output_pack(sampled, sampled_logprobs)


def triton_uniform_total_with_rand(
    flashinfer: Any,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> LogprobsTensors:
    vals, ids = flashinfer_topk_sorted(flashinfer, logits, top_k)
    uniforms = torch.rand(vals.shape[0], device=vals.device, dtype=torch.float32)
    sampled, sampled_logprobs, _ = triton_uniform_tail(vals, ids, top_k, top_p, uniforms)
    return output_pack(sampled, sampled_logprobs)


def bench_case(
    batch_size: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> BenchResult:
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")

    flashinfer = load_flashinfer()
    logits = make_logits(batch_size, vocab_size, device, seed)
    uniforms = make_uniforms(batch_size, device, seed + 1_000_000)
    vals, ids = flashinfer_topk_sorted(flashinfer, logits, top_k)

    torch_sampled, torch_logprobs, torch_keep = torch_uniform_tail(
        vals,
        ids,
        top_p,
        uniforms,
    )
    triton_sampled, triton_logprobs, triton_kept = triton_uniform_tail(
        vals,
        ids,
        top_k,
        top_p,
        uniforms,
    )
    kept_counts = torch_keep.sum(dim=-1)
    if not torch.equal(kept_counts.to(torch.int32), triton_kept):
        raise RuntimeError(f"kept mismatch: torch={kept_counts.tolist()} triton={triton_kept.tolist()}")

    sampled_ids_match = bool(torch.equal(torch_sampled, triton_sampled))
    max_sampled_logprob_diff = (torch_logprobs - triton_logprobs).abs().max().item()
    sampled_for_pack, sampled_lp_for_pack, _ = triton_uniform_tail(
        vals,
        ids,
        top_k,
        top_p,
        uniforms,
    )

    current_random_total_ms = time_fn(
        lambda: current_random_total(flashinfer, logits, top_k, top_p),
        device,
        warmup,
        iters,
    )
    torch_uniform_total_ms = time_fn(
        lambda: torch_uniform_total(flashinfer, logits, top_k, top_p, uniforms),
        device,
        warmup,
        iters,
    )
    triton_uniform_total_ms = time_fn(
        lambda: triton_uniform_total(flashinfer, logits, top_k, top_p, uniforms),
        device,
        warmup,
        iters,
    )
    triton_uniform_total_with_rand_ms = time_fn(
        lambda: triton_uniform_total_with_rand(flashinfer, logits, top_k, top_p),
        device,
        warmup,
        iters,
    )
    current_random_tail_ms = time_fn(
        lambda: current_random_tail(vals, ids, top_p),
        device,
        warmup,
        iters,
    )
    torch_uniform_tail_ms = time_fn(
        lambda: torch_uniform_tail(vals, ids, top_p, uniforms),
        device,
        warmup,
        iters,
    )
    triton_uniform_tail_ms = time_fn(
        lambda: triton_uniform_tail(vals, ids, top_k, top_p, uniforms),
        device,
        warmup,
        iters,
    )
    torch_rand_ms = time_fn(
        lambda: torch.rand(batch_size, device=device, dtype=torch.float32),
        device,
        warmup,
        iters,
    )
    topk_sort_ms = time_fn(
        lambda: flashinfer_topk_sorted(flashinfer, logits, top_k),
        device,
        warmup,
        iters,
    )
    output_pack_ms = time_fn(
        lambda: output_pack(sampled_for_pack, sampled_lp_for_pack),
        device,
        warmup,
        iters,
    )

    return BenchResult(
        batch_size=batch_size,
        vocab_size=vocab_size,
        top_k=top_k,
        top_p=top_p,
        device=str(device),
        current_random_total_ms=current_random_total_ms,
        torch_uniform_total_ms=torch_uniform_total_ms,
        triton_uniform_total_ms=triton_uniform_total_ms,
        triton_uniform_total_with_rand_ms=triton_uniform_total_with_rand_ms,
        current_random_tail_ms=current_random_tail_ms,
        torch_uniform_tail_ms=torch_uniform_tail_ms,
        triton_uniform_tail_ms=triton_uniform_tail_ms,
        torch_rand_ms=torch_rand_ms,
        topk_sort_ms=topk_sort_ms,
        output_pack_ms=output_pack_ms,
        triton_vs_torch_tail_speedup=torch_uniform_tail_ms / triton_uniform_tail_ms
        if triton_uniform_tail_ms > 0
        else float("inf"),
        triton_total_vs_current=triton_uniform_total_ms / current_random_total_ms,
        triton_total_with_rand_vs_current=triton_uniform_total_with_rand_ms / current_random_total_ms,
        max_sampled_logprob_diff=max_sampled_logprob_diff,
        sampled_ids_match=sampled_ids_match,
        kept_min=int(kept_counts.min().item()),
        kept_max=int(kept_counts.max().item()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--vocab-size", type=int, default=248_320)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    results = [
        bench_case(
            batch_size=batch_size,
            vocab_size=args.vocab_size,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed + batch_size,
        )
        for batch_size in parse_int_list(args.batch_sizes)
    ]

    rows = [asdict(result) for result in results]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    print(
        "B\tV\tK\tp\tcurrent_total\ttriton_total\ttriton+rand\t"
        "current_tail\ttorch_tail\ttriton_tail\trand\ttopk_sort\t"
        "triton_tail_speedup\tsampled_match\tmax_lp_diff"
    )
    for result in results:
        print(
            f"{result.batch_size}\t{result.vocab_size}\t{result.top_k}\t"
            f"{result.top_p}\t{result.current_random_total_ms:.4f}\t"
            f"{result.triton_uniform_total_ms:.4f}\t"
            f"{result.triton_uniform_total_with_rand_ms:.4f}\t"
            f"{result.current_random_tail_ms:.4f}\t"
            f"{result.torch_uniform_tail_ms:.4f}\t"
            f"{result.triton_uniform_tail_ms:.4f}\t{result.torch_rand_ms:.4f}\t"
            f"{result.topk_sort_ms:.4f}\t{result.triton_vs_torch_tail_speedup:.2f}\t"
            f"{result.sampled_ids_match}\t{result.max_sampled_logprob_diff:.3g}"
        )


if __name__ == "__main__":
    main()
