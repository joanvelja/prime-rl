#!/usr/bin/env python3
"""Probe a vLLM Sampler.forward fast path backed by FlashInfer topK.

This is the integration version of ``bench_flashinfer_topk_sampled_logprob.py``:
it uses real ``SamplingMetadata``, ``SamplerOutput``, and ``LogprobsTensors`` at
the same seam as vLLM's sampler, but does not patch the installed vLLM package.
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

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import torch
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.v1.sample.sampler import Sampler

from scripts.bench_triton_k_tail_after_flashinfer import triton_uniform_tail


@dataclass(frozen=True)
class BenchResult:
    batch_size: int
    vocab_size: int
    top_k: int
    top_p: float
    device: str
    fastpath_allowed: bool
    native_ms: float
    flashinfer_fast_ms: float
    flashinfer_triton_tail_ms: float
    speedup: float
    triton_tail_speedup: float
    max_forced_logprob_diff: float
    native_cols: int
    fast_cols: int
    triton_tail_cols: int
    sampled_ids_shape: list[int]
    triton_tail_sampled_ids_shape: list[int]
    native_kept_min: int
    native_kept_max: int
    fast_kept_min: int
    fast_kept_max: int


def load_flashinfer():
    module = importlib.import_module("flashinfer")
    if not hasattr(module, "top_k"):
        raise RuntimeError("Installed FlashInfer does not expose flashinfer.top_k")
    return module


class FlashInferTopKSampledLogprobSampler(Sampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flashinfer = load_flashinfer()

    def can_use_flashinfer_sampled_logprob_fast_path(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode: str,
    ) -> bool:
        if logits.device.type != "cuda":
            return False
        if logprobs_mode != "processed_logprobs":
            return False
        if sampling_metadata.max_num_logprobs not in {0, 1}:
            return False
        if sampling_metadata.logprob_token_ids:
            return False
        if not sampling_metadata.all_random or sampling_metadata.all_greedy:
            return False
        if sampling_metadata.temperature is None:
            return False
        if torch.any(sampling_metadata.temperature < 1e-5):
            return False
        if sampling_metadata.top_k is None or sampling_metadata.top_p is None:
            return False

        top_k = sampling_metadata.top_k
        vocab_size = logits.shape[-1]
        if torch.any(top_k <= 0) or torch.any(top_k >= vocab_size):
            return False
        # FlashInfer top_k takes one scalar k. Prime's current debate config is
        # uniform; mixed top_k rows can fallback to native.
        if not torch.all(top_k == top_k[0]):
            return False
        return True

    def flashinfer_support(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert sampling_metadata.top_k is not None
        assert sampling_metadata.top_p is not None
        top_k = int(sampling_metadata.top_k[0].item())
        vals, ids = self._flashinfer.top_k(logits, top_k, sorted=False)
        vals, order = vals.sort(dim=-1, descending=True)
        ids = ids.gather(-1, order)

        weights = torch.softmax(vals, dim=-1, dtype=torch.float32)
        prefix = weights.cumsum(dim=-1)
        keep = (prefix - weights) < sampling_metadata.top_p.unsqueeze(1)
        support_vals = vals.masked_fill(~keep, -float("inf"))
        support_logprobs = support_vals - torch.logsumexp(
            support_vals,
            dim=-1,
            keepdim=True,
        )
        return ids, support_vals, support_logprobs, keep

    def sample_with_logprob(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, support_vals, support_logprobs, keep = self.flashinfer_support(
            logits,
            sampling_metadata,
        )
        probs = torch.softmax(support_vals, dim=-1, dtype=torch.float32)
        sampled_in_topk = random_sample(probs, sampling_metadata.generators)
        sampled = ids.gather(1, sampled_in_topk.unsqueeze(1)).squeeze(1).long()
        sampled_logprobs = support_logprobs.gather(
            1,
            sampled_in_topk.unsqueeze(1),
        ).squeeze(1)
        return sampled, sampled_logprobs, keep

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: str | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
        if self.can_use_flashinfer_sampled_logprob_fast_path(
            logits,
            sampling_metadata,
            logprobs_mode,
        ):
            sampled, sampled_logprobs, _ = self.sample_with_logprob(logits, sampling_metadata)
            sampled_i32 = sampled.to(torch.int32)
            logprobs_tensors = LogprobsTensors(
                logprob_token_ids=sampled_i32.unsqueeze(-1),
                logprobs=sampled_logprobs.to(torch.float32).unsqueeze(-1),
                selected_token_ranks=torch.ones_like(sampled_i32, dtype=torch.int32),
            )
            return SamplerOutput(
                sampled_token_ids=sampled_i32.unsqueeze(-1),
                logprobs_tensors=logprobs_tensors,
            )
        return super().forward(
            logits,
            sampling_metadata,
            predict_bonus_token=predict_bonus_token,
            logprobs_mode_override=logprobs_mode_override,
        )


class FlashInferTritonTailSampledLogprobSampler(FlashInferTopKSampledLogprobSampler):
    def sample_with_logprob(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert sampling_metadata.top_k is not None
        assert sampling_metadata.top_p is not None
        top_k = int(sampling_metadata.top_k[0].item())
        top_p = float(sampling_metadata.top_p[0].item())
        vals, ids = self._flashinfer.top_k(logits.contiguous(), top_k, sorted=False)
        vals, order = vals.sort(dim=-1, descending=True)
        ids = ids.gather(-1, order)
        uniforms = torch.rand(vals.shape[0], device=vals.device, dtype=torch.float32)
        sampled, sampled_logprobs, kept = triton_uniform_tail(
            vals,
            ids,
            top_k,
            top_p,
            uniforms,
        )
        return sampled, sampled_logprobs, kept


def make_metadata(
    batch_size: int,
    top_k: int,
    top_p: float,
    device: torch.device,
    max_num_logprobs: int = 0,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size, device=device, dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((batch_size,), top_p, device=device, dtype=torch.float32),
        top_k=torch.full((batch_size,), top_k, device=device, dtype=torch.int64),
        generators={},
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size, device=device, dtype=torch.float32),
        presence_penalties=torch.zeros(batch_size, device=device, dtype=torch.float32),
        repetition_penalties=torch.ones(batch_size, device=device, dtype=torch.float32),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
    )


def make_logits(batch_size: int, vocab_size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, generator=generator, dtype=torch.float32)
    return logits.to(device)


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


def forced_flashinfer_logprobs(
    sampler: FlashInferTopKSampledLogprobSampler,
    logits: torch.Tensor,
    metadata: SamplingMetadata,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids, _, support_logprobs, keep = sampler.flashinfer_support(logits, metadata)
    match = ids == token_ids.unsqueeze(1)
    forced = support_logprobs.masked_fill(~match, -float("inf")).max(dim=1).values
    return forced, keep


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
    logits = make_logits(batch_size, vocab_size, device, seed)
    metadata = make_metadata(batch_size, top_k, top_p, device)
    native = Sampler("processed_logprobs")
    fast = FlashInferTopKSampledLogprobSampler("processed_logprobs")
    triton_fast = FlashInferTritonTailSampledLogprobSampler("processed_logprobs")

    native_output = native(logits.clone(), metadata)
    fast_output = fast(logits.clone(), metadata)
    triton_fast_output = triton_fast(logits.clone(), metadata)
    assert native_output.logprobs_tensors is not None
    assert fast_output.logprobs_tensors is not None
    assert triton_fast_output.logprobs_tensors is not None
    _, native_processed_logprobs = TopKTopPSampler("processed_logprobs")(
        logits.clone(),
        metadata.generators,
        metadata.top_k,
        metadata.top_p,
    )
    assert native_processed_logprobs is not None

    native_sampled = native_output.sampled_token_ids.squeeze(1).long()
    forced_fast, keep = forced_flashinfer_logprobs(fast, logits, metadata, native_sampled)
    native_logprobs = native_output.logprobs_tensors.logprobs[:, 0]
    max_diff = (native_logprobs - forced_fast).abs().max().item()
    native_kept = torch.isfinite(native_processed_logprobs).sum(dim=-1)
    fast_kept = keep.sum(dim=-1)
    if not torch.equal(native_kept, fast_kept):
        raise RuntimeError(f"Support-size mismatch: native={native_kept.tolist()} fast={fast_kept.tolist()}")

    native_ms = time_fn(lambda: native(logits.clone(), metadata), device, warmup, iters)
    fast_ms = time_fn(lambda: fast(logits.clone(), metadata), device, warmup, iters)
    triton_fast_ms = time_fn(
        lambda: triton_fast(logits.clone(), metadata),
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
        fastpath_allowed=fast.can_use_flashinfer_sampled_logprob_fast_path(
            logits,
            metadata,
            "processed_logprobs",
        ),
        native_ms=native_ms,
        flashinfer_fast_ms=fast_ms,
        flashinfer_triton_tail_ms=triton_fast_ms,
        speedup=native_ms / fast_ms if fast_ms > 0 else float("inf"),
        triton_tail_speedup=native_ms / triton_fast_ms if triton_fast_ms > 0 else float("inf"),
        max_forced_logprob_diff=max_diff,
        native_cols=native_output.logprobs_tensors.logprobs.shape[1],
        fast_cols=fast_output.logprobs_tensors.logprobs.shape[1],
        triton_tail_cols=triton_fast_output.logprobs_tensors.logprobs.shape[1],
        sampled_ids_shape=list(fast_output.sampled_token_ids.shape),
        triton_tail_sampled_ids_shape=list(triton_fast_output.sampled_token_ids.shape),
        native_kept_min=int(native_kept.min().item()),
        native_kept_max=int(native_kept.max().item()),
        fast_kept_min=int(fast_kept.min().item()),
        fast_kept_max=int(fast_kept.max().item()),
    )


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def select_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type != "cuda":
        raise RuntimeError("This FlashInfer integration benchmark requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="134,256")
    parser.add_argument("--vocab-size", type=int, default=248_320)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
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
        "B\tV\tK\tp\tdevice\tfast_allowed\tnative_ms\tflashinfer_fast_ms\t"
        "flashinfer_triton_tail_ms\tspeedup\ttriton_tail_speedup\t"
        "max_forced_logprob_diff\tnative_cols\tfast_cols\ttriton_tail_cols\t"
        "sampled_shape\ttriton_tail_shape\tnative_support\tfast_support"
    )
    for result in results:
        print(
            f"{result.batch_size}\t{result.vocab_size}\t{result.top_k}\t"
            f"{result.top_p}\t{result.device}\t{result.fastpath_allowed}\t"
            f"{result.native_ms:.3f}\t{result.flashinfer_fast_ms:.3f}\t"
            f"{result.flashinfer_triton_tail_ms:.3f}\t{result.speedup:.2f}\t"
            f"{result.triton_tail_speedup:.2f}\t"
            f"{result.max_forced_logprob_diff:.3g}\t"
            f"{result.native_cols}\t{result.fast_cols}\t{result.triton_tail_cols}\t"
            f"{result.sampled_ids_shape}\t{result.triton_tail_sampled_ids_shape}\t"
            f"{result.native_kept_min}-{result.native_kept_max}\t"
            f"{result.fast_kept_min}-{result.fast_kept_max}"
        )


if __name__ == "__main__":
    main()
