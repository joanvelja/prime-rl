#!/usr/bin/env python3
"""Run the PrimeRL FlashInfer sampled-logprob CUDA contract sweep."""

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

from scripts.probe_prime_flashinfer_sampler_patch import ProbeResult, run_probe, select_device

_BOUNDARY_TIE_GUARD_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_BOUNDARY_TIE_GUARD"
_TAIL_ENV = "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL"


@dataclass(frozen=True)
class SweepSpec:
    batch_size: int
    vocab_size: int
    top_k: int
    top_p: float
    max_num_logprobs: int
    dense_presence: bool
    seed: int

    @property
    def prompt_len(self) -> int:
        return 16 if self.dense_presence else 0

    @property
    def unique_output_tokens(self) -> int:
        return 8 if self.dense_presence else 0

    @property
    def presence_penalty(self) -> float:
        return 1.5 if self.dense_presence else 0.0


@dataclass(frozen=True)
class SweepSummary:
    rows: int
    failures: int
    atol: float
    max_expected_patched_logprob_diff: float | None
    max_forced_logprob_diff: float | None
    native_cols: list[int]
    patched_cols: list[int]


def production_vocab_specs(*, include_width1: bool) -> list[SweepSpec]:
    specs = [
        SweepSpec(16, 248320, 20, 0.95, 0, False, 0),
        SweepSpec(16, 248320, 20, 0.95, 0, True, 0),
        SweepSpec(64, 248320, 20, 0.95, 0, False, 20260624),
        SweepSpec(64, 248320, 20, 0.95, 0, True, 20260624),
        SweepSpec(64, 248320, 64, 0.95, 0, False, 424242),
        SweepSpec(64, 248320, 64, 0.95, 0, True, 424242),
        SweepSpec(128, 248320, 20, 0.95, 0, True, 12345),
        SweepSpec(128, 248320, 64, 0.95, 0, True, 99999),
    ]
    if include_width1:
        specs.extend(
            [
                SweepSpec(16, 248320, 20, 0.95, 1, False, 20260624),
                SweepSpec(16, 248320, 20, 0.95, 1, True, 20260624),
                SweepSpec(64, 248320, 64, 0.95, 1, False, 424242),
                SweepSpec(64, 248320, 64, 0.95, 1, True, 424242),
            ]
        )
    return specs


def probe_failed(result: ProbeResult, *, atol: float) -> bool:
    expected_cols = result.max_num_logprobs + 1
    return (
        not result.patch_marker
        or not result.fastpath_allowed
        or result.native_cols != expected_cols
        or result.patched_cols != expected_cols
        or result.max_expected_patched_logprob_diff is None
        or result.max_expected_patched_logprob_diff > atol
        or result.expected_patched_sampled_token_mismatches != 0
        or result.expected_patched_logprob_token_id_mismatches != 0
        or result.expected_patched_selected_rank_mismatches != 0
    )


def run_sweep(
    specs: list[SweepSpec],
    *,
    device_raw: str,
    atol: float,
    tail: str,
) -> tuple[list[ProbeResult], list[ProbeResult], SweepSummary]:
    os.environ[_BOUNDARY_TIE_GUARD_ENV] = "1"
    os.environ[_TAIL_ENV] = tail
    device = select_device(device_raw)

    results = [
        run_probe(
            batch_size=spec.batch_size,
            vocab_size=spec.vocab_size,
            top_k=spec.top_k,
            top_p=spec.top_p,
            max_num_logprobs=spec.max_num_logprobs,
            prompt_len=spec.prompt_len,
            unique_output_tokens=spec.unique_output_tokens,
            presence_penalty=spec.presence_penalty,
            dense_presence=spec.dense_presence,
            device=device,
            seed=spec.seed,
        )
        for spec in specs
    ]
    failures = [result for result in results if probe_failed(result, atol=atol)]
    expected_diffs = [
        result.max_expected_patched_logprob_diff
        for result in results
        if result.max_expected_patched_logprob_diff is not None
    ]
    forced_diffs = [result.max_forced_logprob_diff for result in results if result.max_forced_logprob_diff is not None]
    summary = SweepSummary(
        rows=len(results),
        failures=len(failures),
        atol=atol,
        max_expected_patched_logprob_diff=max(expected_diffs) if expected_diffs else None,
        max_forced_logprob_diff=max(forced_diffs) if forced_diffs else None,
        native_cols=sorted({result.native_cols for result in results}),
        patched_cols=sorted({result.patched_cols for result in results}),
    )
    return results, failures, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--tail", choices=["triton", "torch"], default="triton")
    parser.add_argument(
        "--include-width1",
        action="store_true",
        help="Also check max_num_logprobs=1 second-column construction.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object containing all rows and the summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON result artifact to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, failures, summary = run_sweep(
        production_vocab_specs(include_width1=args.include_width1),
        device_raw=args.device,
        atol=args.atol,
        tail=args.tail,
    )
    payload = {
        "results": [asdict(result) for result in results],
        "summary": asdict(summary),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            print(json.dumps(asdict(result), sort_keys=True))
        print("SUMMARY", json.dumps(asdict(summary), sort_keys=True))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
