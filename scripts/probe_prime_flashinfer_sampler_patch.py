#!/usr/bin/env python3
"""Probe the PrimeRL FlashInfer sampled-logprob monkey patch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from prime_rl.inference.vllm.flashinfer_sampler import (
    _DENSE_PRESENCE_ENV,
    _ENABLE_ENV,
    _PATCH_MARKER,
    _metadata_can_use_fast_path,
    apply_flashinfer_sampled_logprob_patch,
)
from scripts.bench_vllm_flashinfer_sampler_integration import (
    FlashInferTopKSampledLogprobSampler,
    forced_flashinfer_logprobs,
    make_logits,
    make_metadata,
)


@dataclass(frozen=True)
class OutputComparison:
    native_cols: int
    patched_cols: int
    sampled_token_mismatches: int
    logprob_token_id_mismatches: int
    selected_rank_mismatches: int
    max_logprob_diff: float | None


@dataclass(frozen=True)
class ProbeResult:
    batch_size: int
    vocab_size: int
    top_k: int
    top_p: float
    max_num_logprobs: int
    prompt_len: int
    unique_output_tokens: int
    presence_penalty: float
    device: str
    env_enabled: bool
    dense_presence_enabled: bool
    patch_marker: bool
    fastpath_allowed: bool
    max_forced_logprob_diff: float | None
    max_patched_logprob_diff: float | None
    max_expected_patched_logprob_diff: float | None
    expected_patched_sampled_token_mismatches: int
    expected_patched_logprob_token_id_mismatches: int
    expected_patched_selected_rank_mismatches: int
    native_cols: int
    patched_cols: int
    patched_sampled_ids_shape: list[int]


def select_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def make_unique_token_lists(
    batch_size: int,
    count: int,
    vocab_size: int,
    row_stride: int = 1009,
    token_stride: int = 7919,
) -> list[list[int]]:
    return [
        [(row * row_stride + index * token_stride) % vocab_size for index in range(count)] for row in range(batch_size)
    ]


def make_unique_token_tensor(
    batch_size: int,
    count: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        make_unique_token_lists(batch_size, count, vocab_size),
        device=device,
        dtype=torch.int64,
    )


def add_presence_policy(
    metadata,
    batch_size: int,
    vocab_size: int,
    prompt_len: int,
    unique_output_tokens: int,
    presence_penalty: float,
    device: torch.device,
) -> None:
    if prompt_len == 0 and unique_output_tokens == 0 and presence_penalty == 0.0:
        return
    metadata.no_penalties = presence_penalty == 0.0
    metadata.prompt_token_ids = make_unique_token_tensor(
        batch_size,
        prompt_len,
        vocab_size,
        device,
    )
    metadata.output_token_ids = make_unique_token_lists(
        batch_size,
        unique_output_tokens,
        vocab_size,
        row_stride=1543,
        token_stride=7919,
    )
    metadata.frequency_penalties = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.float32,
    )
    metadata.presence_penalties = torch.full(
        (batch_size,),
        presence_penalty,
        device=device,
        dtype=torch.float32,
    )
    metadata.repetition_penalties = torch.ones(
        batch_size,
        device=device,
        dtype=torch.float32,
    )


def processed_logits_for_fast_support(
    sampler: Sampler,
    logits: torch.Tensor,
    metadata,
) -> torch.Tensor:
    processed = logits.clone().to(torch.float32)
    processed = sampler.apply_logits_processors(
        processed,
        metadata,
        predict_bonus_token=False,
    )
    assert metadata.temperature is not None
    processed = sampler.apply_temperature(processed, metadata.temperature, metadata.all_random)
    for processor in metadata.logitsprocs.argmax_invariant:
        processed = processor.apply(processed)
    return processed


def compare_sampler_outputs(
    native: SamplerOutput,
    patched: SamplerOutput,
) -> OutputComparison:
    assert native.logprobs_tensors is not None
    assert patched.logprobs_tensors is not None
    native_logprobs = native.logprobs_tensors
    patched_logprobs = patched.logprobs_tensors

    if native_logprobs.logprobs.shape == patched_logprobs.logprobs.shape:
        max_logprob_diff = (native_logprobs.logprobs - patched_logprobs.logprobs).abs().max().item()
    else:
        max_logprob_diff = None

    return OutputComparison(
        native_cols=native_logprobs.logprobs.shape[1],
        patched_cols=patched_logprobs.logprobs.shape[1],
        sampled_token_mismatches=int((native.sampled_token_ids != patched.sampled_token_ids).sum().item()),
        logprob_token_id_mismatches=int(
            (native_logprobs.logprob_token_ids != patched_logprobs.logprob_token_ids).sum().item()
        )
        if native_logprobs.logprob_token_ids.shape == patched_logprobs.logprob_token_ids.shape
        else -1,
        selected_rank_mismatches=int(
            (native_logprobs.selected_token_ranks != patched_logprobs.selected_token_ranks).sum().item()
        )
        if native_logprobs.selected_token_ranks.shape == patched_logprobs.selected_token_ranks.shape
        else -1,
        max_logprob_diff=max_logprob_diff,
    )


def expected_output_for_sampled_ids(
    sampler: Sampler,
    logits: torch.Tensor,
    metadata,
    sampled_token_ids: torch.Tensor,
) -> SamplerOutput:
    processed_logits = processed_logits_for_fast_support(sampler, logits, metadata)
    _, processed_logprobs = sampler.topk_topp_sampler(
        processed_logits,
        metadata.generators,
        metadata.top_k,
        metadata.top_p,
    )
    assert processed_logprobs is not None
    sampled = sampled_token_ids.squeeze(1).long()
    max_num_logprobs = metadata.max_num_logprobs or 0
    logprobs_tensors = sampler.gather_logprobs(
        processed_logprobs,
        max_num_logprobs,
        token_ids=sampled,
    )
    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=logprobs_tensors,
    )


def run_probe(
    batch_size: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
    max_num_logprobs: int,
    prompt_len: int,
    unique_output_tokens: int,
    presence_penalty: float,
    dense_presence: bool,
    device: torch.device,
    seed: int,
) -> ProbeResult:
    logits = make_logits(batch_size, vocab_size, device, seed)
    metadata = make_metadata(batch_size, top_k, top_p, device, max_num_logprobs)
    add_presence_policy(
        metadata,
        batch_size,
        vocab_size,
        prompt_len,
        unique_output_tokens,
        presence_penalty,
        device,
    )

    native = Sampler("processed_logprobs")
    native_output = native(logits.clone(), metadata)
    assert native_output.logprobs_tensors is not None

    os.environ[_ENABLE_ENV] = "1"
    if dense_presence:
        os.environ[_DENSE_PRESENCE_ENV] = "1"
    else:
        os.environ.pop(_DENSE_PRESENCE_ENV, None)
    apply_flashinfer_sampled_logprob_patch()
    patched_marker = bool(getattr(Sampler.forward, _PATCH_MARKER, False))

    patched = Sampler("processed_logprobs")
    fastpath_allowed = _metadata_can_use_fast_path(
        logits,
        metadata,
        "processed_logprobs",
        predict_bonus_token=False,
    )
    patched_output = patched(logits.clone(), metadata)
    assert patched_output.logprobs_tensors is not None

    expected_patched_output = expected_output_for_sampled_ids(
        native,
        logits.clone(),
        metadata,
        patched_output.sampled_token_ids,
    )
    expected_patched_comparison = compare_sampler_outputs(
        expected_patched_output,
        patched_output,
    )

    max_diff = None
    max_patched_diff = None
    if device.type == "cuda":
        helper = FlashInferTopKSampledLogprobSampler("processed_logprobs")
        processed_logits = processed_logits_for_fast_support(native, logits, metadata)
        native_sampled = native_output.sampled_token_ids.squeeze(1).long()
        forced_fast, _ = forced_flashinfer_logprobs(
            helper,
            processed_logits,
            metadata,
            native_sampled,
        )
        native_logprobs = native_output.logprobs_tensors.logprobs[:, 0]
        max_diff = (native_logprobs - forced_fast).abs().max().item()
        patched_sampled = patched_output.sampled_token_ids.squeeze(1).long()
        forced_patched, _ = forced_flashinfer_logprobs(
            helper,
            processed_logits,
            metadata,
            patched_sampled,
        )
        patched_logprobs = patched_output.logprobs_tensors.logprobs[:, 0]
        max_patched_diff = (patched_logprobs - forced_patched).abs().max().item()

    return ProbeResult(
        batch_size=batch_size,
        vocab_size=vocab_size,
        top_k=top_k,
        top_p=top_p,
        max_num_logprobs=max_num_logprobs,
        prompt_len=prompt_len,
        unique_output_tokens=unique_output_tokens,
        presence_penalty=presence_penalty,
        device=str(device),
        env_enabled=True,
        dense_presence_enabled=dense_presence,
        patch_marker=patched_marker,
        fastpath_allowed=fastpath_allowed,
        max_forced_logprob_diff=max_diff,
        max_patched_logprob_diff=max_patched_diff,
        max_expected_patched_logprob_diff=expected_patched_comparison.max_logprob_diff,
        expected_patched_sampled_token_mismatches=expected_patched_comparison.sampled_token_mismatches,
        expected_patched_logprob_token_id_mismatches=expected_patched_comparison.logprob_token_id_mismatches,
        expected_patched_selected_rank_mismatches=expected_patched_comparison.selected_rank_mismatches,
        native_cols=native_output.logprobs_tensors.logprobs.shape[1],
        patched_cols=patched_output.logprobs_tensors.logprobs.shape[1],
        patched_sampled_ids_shape=list(patched_output.sampled_token_ids.shape),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-num-logprobs", type=int, default=0)
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument("--unique-output-tokens", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--dense-presence",
        action="store_true",
        help="Enable the production dense presence-penalty specialization.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        top_p=args.top_p,
        max_num_logprobs=args.max_num_logprobs,
        prompt_len=args.prompt_len,
        unique_output_tokens=args.unique_output_tokens,
        presence_penalty=args.presence_penalty,
        dense_presence=args.dense_presence,
        device=select_device(args.device),
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    print(f"B={result.batch_size} V={result.vocab_size} K={result.top_k} p={result.top_p}")
    print(f"max_num_logprobs={result.max_num_logprobs}")
    print(
        "presence_policy="
        f"prompt_len={result.prompt_len} "
        f"unique_output_tokens={result.unique_output_tokens} "
        f"presence_penalty={result.presence_penalty}"
    )
    print(f"device={result.device}")
    print(f"dense_presence_enabled={result.dense_presence_enabled}")
    print(f"patch_marker={result.patch_marker}")
    print(f"fastpath_allowed={result.fastpath_allowed}")
    if result.max_forced_logprob_diff is None:
        print("max_forced_logprob_diff=None")
    else:
        print(f"max_forced_logprob_diff={result.max_forced_logprob_diff:.3g}")
    if result.max_patched_logprob_diff is None:
        print("max_patched_logprob_diff=None")
    else:
        print(f"max_patched_logprob_diff={result.max_patched_logprob_diff:.3g}")
    if result.max_expected_patched_logprob_diff is None:
        print("max_expected_patched_logprob_diff=None")
    else:
        print(f"max_expected_patched_logprob_diff={result.max_expected_patched_logprob_diff:.3g}")
    print(
        "expected_patched_mismatches="
        f"sampled={result.expected_patched_sampled_token_mismatches} "
        f"logprob_ids={result.expected_patched_logprob_token_id_mismatches} "
        f"ranks={result.expected_patched_selected_rank_mismatches}"
    )
    print(f"native_cols={result.native_cols} patched_cols={result.patched_cols}")
    print(f"patched_sampled_ids_shape={result.patched_sampled_ids_shape}")
    print("status=PASS prime_flashinfer_sampler_patch")


if __name__ == "__main__":
    main()
