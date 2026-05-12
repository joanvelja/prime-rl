"""Build an Omni-MATH-2 sensitivity subset from baseline rollouts.

Upstream Hendrycks sanity uses problems that the base model solves sometimes
but not always. This script applies the same idea to Omni-MATH-2: compute each
problem's empirical solve rate from a baseline rollout file, then keep source
dataset rows whose solve rate falls inside a target band.

Usage:
    uv run --no-sync python -m scripts.evals.make_omni_math2_perfectible_subset \
        --baseline-rollouts outputs/.../eval_rollouts.jsonl \
        --dataset benchmarks/datasets/omni_math2/omni_math2_stratified_600_seed42.jsonl \
        --output benchmarks/datasets/omni_math2/omni_math2_perfectible_seed42.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(
    *,
    baseline_rollouts: Path,
    dataset: Path,
    output: Path,
    low: float,
    high: float,
    min_rollouts: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _read_jsonl(baseline_rollouts):
        grouped[str(row["example_id"])].append(float(row.get("reward") or 0.0))

    solve_rates = {
        example_id: sum(rewards) / len(rewards)
        for example_id, rewards in grouped.items()
        if len(rewards) >= min_rollouts
    }
    selected_ids = {
        example_id
        for example_id, solve_rate in solve_rates.items()
        if low <= solve_rate <= high
    }

    source_rows = _read_jsonl(dataset)
    selected_rows = [row for row in source_rows if str(row.get("id")) in selected_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "baseline_rollouts": str(baseline_rollouts),
        "dataset": str(dataset),
        "output": str(output),
        "low": low,
        "high": high,
        "min_rollouts": min_rollouts,
        "baseline_examples": len(grouped),
        "eligible_examples": len(solve_rates),
        "selected_examples": len(selected_rows),
        "missing_selected_ids": sorted(selected_ids - {str(row.get("id")) for row in selected_rows}),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rollouts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=0.8)
    parser.add_argument("--min-rollouts", type=int, default=8)
    args = parser.parse_args()
    summary = run(
        baseline_rollouts=args.baseline_rollouts,
        dataset=args.dataset,
        output=args.output,
        low=args.low,
        high=args.high,
        min_rollouts=args.min_rollouts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
