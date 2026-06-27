#!/usr/bin/env python3
"""Prototype the vLLM sampled-logprob-only fast path.

It compares:

1. vLLM's current internal processed-logprob path:
   TopKTopPSampler(logprobs_mode="processed_logprobs") plus
   Sampler.gather_logprobs(..., num_logprobs=0) by default.
2. A Prime-shaped fast path that samples inside the top-K/top-P support and
   returns a LogprobsTensors-compatible width-1 sampled-token logprob.

It uses synthetic already-processed logits. It is not a replacement for a live
rollout profile, but it measures the integration seam that the earlier R3/R4
benchmark intentionally skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.v1.sample.sampler import Sampler


@dataclass(frozen=True)
class BenchResult:
    batch_size: int
    vocab_size: int
    top_k: int
    top_p: float
    num_logprobs: int
    device: str
    native_ms: float
    native_sampler_ms: float
    native_gather_ms: float
    fast_ms: float
    speedup: float
    gather_fraction: float
    max_forced_logprob_diff: float
    native_cols: int
    fast_cols: int
    native_kept_min: int
    native_kept_max: int
    fast_kept_min: int
    fast_kept_max: int


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def make_logits(batch_size: int, vocab_size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, generator=generator, dtype=torch.float32)
    return logits.to(device)


def make_topk_topp_tensors(
    batch_size: int,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((batch_size,), top_k, device=device, dtype=torch.int64),
        torch.full((batch_size,), top_p, device=device, dtype=torch.float32),
    )


def native_vllm_processed_logprobs(
    sampler: TopKTopPSampler,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    num_logprobs: int,
) -> tuple[torch.Tensor, LogprobsTensors, torch.Tensor]:
    batch_size = logits.shape[0]
    k, p = make_topk_topp_tensors(batch_size, top_k, top_p, logits.device)
    sampled, processed_logprobs = sampler(logits.clone(), {}, k, p)
    assert processed_logprobs is not None
    sampled = sampled.long()
    logprobs_tensors = Sampler.gather_logprobs(
        processed_logprobs,
        num_logprobs,
        token_ids=sampled,
    )
    return sampled, logprobs_tensors, processed_logprobs


def native_vllm_sampler_only(
    sampler: TopKTopPSampler,
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = logits.shape[0]
    k, p = make_topk_topp_tensors(batch_size, top_k, top_p, logits.device)
    sampled, processed_logprobs = sampler(logits.clone(), {}, k, p)
    assert processed_logprobs is not None
    return sampled.long(), processed_logprobs


def native_vllm_gather_only(
    processed_logprobs: torch.Tensor,
    sampled: torch.Tensor,
    num_logprobs: int,
) -> LogprobsTensors:
    return Sampler.gather_logprobs(processed_logprobs, num_logprobs, token_ids=sampled)


def topk_support(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vals, ids = torch.topk(logits, k=top_k, dim=-1)
    vals, order = vals.sort(dim=-1, descending=True)
    ids = ids.gather(-1, order)

    weights = torch.softmax(vals, dim=-1, dtype=torch.float32)
    prefix = weights.cumsum(dim=-1)
    keep = (prefix - weights) < top_p
    support_vals = vals.masked_fill(~keep, -float("inf"))
    support_logprobs = support_vals - torch.logsumexp(support_vals, dim=-1, keepdim=True)
    return ids, support_vals, support_logprobs, keep


def fast_width1_sampled_logprobs(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, LogprobsTensors, torch.Tensor]:
    ids, support_vals, support_logprobs, keep = topk_support(logits, top_k, top_p)
    probs = torch.softmax(support_vals, dim=-1, dtype=torch.float32)
    sampled_in_topk = random_sample(probs, {})
    sampled = ids.gather(1, sampled_in_topk.unsqueeze(1)).squeeze(1).long()
    sampled_logprobs = support_logprobs.gather(1, sampled_in_topk.unsqueeze(1)).squeeze(1)

    logprobs_tensors = LogprobsTensors(
        logprob_token_ids=sampled.to(torch.int32).unsqueeze(1),
        logprobs=sampled_logprobs.to(torch.float32).unsqueeze(1),
        selected_token_ranks=torch.ones_like(sampled, dtype=torch.int32),
    )
    return sampled, logprobs_tensors, keep


def forced_fast_logprobs(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids, _, support_logprobs, keep = topk_support(logits, top_k, top_p)
    match = ids == token_ids.unsqueeze(1)
    forced = support_logprobs.masked_fill(~match, -float("inf")).max(dim=1).values
    return forced, keep


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


def bench_case(
    batch_size: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
    num_logprobs: int,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> BenchResult:
    logits = make_logits(batch_size, vocab_size, device, seed)
    sampler = TopKTopPSampler("processed_logprobs")

    native_sampled, native_tensors, processed_logprobs = native_vllm_processed_logprobs(
        sampler,
        logits,
        top_k,
        top_p,
        num_logprobs,
    )
    forced_fast, fast_keep = forced_fast_logprobs(logits, top_k, top_p, native_sampled)
    native_sampled_logprobs = native_tensors.logprobs[:, 0]
    max_diff = (native_sampled_logprobs - forced_fast).abs().max().item()

    _, fast_tensors, _ = fast_width1_sampled_logprobs(logits, top_k, top_p)

    native_ms = time_fn(
        lambda: native_vllm_processed_logprobs(
            sampler,
            logits,
            top_k,
            top_p,
            num_logprobs,
        ),
        device,
        warmup,
        iters,
    )
    native_sampler_ms = time_fn(
        lambda: native_vllm_sampler_only(sampler, logits, top_k, top_p),
        device,
        warmup,
        iters,
    )
    native_gather_ms = time_fn(
        lambda: native_vllm_gather_only(processed_logprobs, native_sampled, num_logprobs),
        device,
        warmup,
        iters,
    )
    fast_ms = time_fn(
        lambda: fast_width1_sampled_logprobs(logits, top_k, top_p),
        device,
        warmup,
        iters,
    )

    native_kept = torch.isfinite(processed_logprobs).sum(dim=-1)
    fast_kept = fast_keep.sum(dim=-1)
    if not torch.equal(native_kept, fast_kept):
        raise RuntimeError(f"Support-size mismatch: native={native_kept.tolist()} fast={fast_kept.tolist()}")

    return BenchResult(
        batch_size=batch_size,
        vocab_size=vocab_size,
        top_k=top_k,
        top_p=top_p,
        num_logprobs=num_logprobs,
        device=str(device),
        native_ms=native_ms,
        native_sampler_ms=native_sampler_ms,
        native_gather_ms=native_gather_ms,
        fast_ms=fast_ms,
        speedup=native_ms / fast_ms if fast_ms > 0 else float("inf"),
        gather_fraction=native_gather_ms / native_ms if native_ms > 0 else float("nan"),
        max_forced_logprob_diff=max_diff,
        native_cols=native_tensors.logprobs.shape[1],
        fast_cols=fast_tensors.logprobs.shape[1],
        native_kept_min=int(native_kept.min().item()),
        native_kept_max=int(native_kept.max().item()),
        fast_kept_min=int(fast_kept.min().item()),
        fast_kept_max=int(fast_kept.max().item()),
    )


def select_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--vocab-size", type=int, default=248_320)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--num-logprobs",
        type=int,
        default=0,
        help="Internal vLLM max_num_logprobs. Prime chat logprobs=True maps to 0.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
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
            num_logprobs=args.num_logprobs,
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
        "B\tV\tK\tp\tnum_logprobs\tdevice\tnative_ms\tnative_sampler_ms\t"
        "native_gather_ms\tgather_fraction\tfast_ms\tspeedup\t"
        "max_forced_logprob_diff\tnative_cols\tfast_cols\tnative_kept\tfast_kept"
    )
    for result in results:
        print(
            f"{result.batch_size}\t{result.vocab_size}\t{result.top_k}\t"
            f"{result.top_p}\t{result.num_logprobs}\t{result.device}\t"
            f"{result.native_ms:.3f}\t"
            f"{result.native_sampler_ms:.3f}\t{result.native_gather_ms:.3f}\t"
            f"{result.gather_fraction:.2f}\t{result.fast_ms:.3f}\t{result.speedup:.2f}\t"
            f"{result.max_forced_logprob_diff:.3g}\t{result.native_cols}\t"
            f"{result.fast_cols}\t{result.native_kept_min}-{result.native_kept_max}\t"
            f"{result.fast_kept_min}-{result.fast_kept_max}"
        )


if __name__ == "__main__":
    main()
