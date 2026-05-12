"""Build a perfectible-subset dataset from baseline rollouts.

Implements the Hendrycks Sanity pattern (Defeating the Training-Inference
Mismatch, arxiv 2510.26788) for arbitrary (model, dataset) pairs: a problem
is "perfectible" if the base model solves it sometimes-but-not-always
(default band: 20-80% of rollouts). A working RL algorithm should be able
to push training accuracy on this subset above 95%, so it makes a good
hillclimb sanity check that rules out recipe-level bugs.

The workflow has two stages -- this script handles only the filter:

1. Generate baseline rollouts for your (model, dataset) pair via the
   `baselines` entrypoint, e.g.:
       uv run baselines @ configs/baselines/<your_config>.toml
   Output: `eval_rollouts.jsonl` with one row per (example, rollout).

2. Filter the source dataset by solve rate:
       uv run --no-sync python -m scripts.evals.make_perfectible_subset \\
           --baseline-rollouts outputs/.../eval_rollouts.jsonl \\
           --dataset PATH/TO/source.jsonl \\
           --output PATH/TO/perfectible.jsonl

Defaults match the omni-math2 baseline conventions (rollout id field
`example_id`, reward field `reward`, dataset id field `id`); override
--rollout-id-key / --rollout-reward-key / --dataset-id-key for other
schemas.

Solve rate is computed as the mean of the rollout rewards. For binary
0/1 rewards this is the fraction correct; for continuous rewards in
[0, 1] it is the mean reward. The 20-80% band is the original
Hendrycks Sanity choice.
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
    rollout_id_key: str,
    rollout_reward_key: str,
    dataset_id_key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _read_jsonl(baseline_rollouts):
        grouped[str(row[rollout_id_key])].append(float(row.get(rollout_reward_key) or 0.0))

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
    selected_rows = [row for row in source_rows if str(row.get(dataset_id_key)) in selected_ids]
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
        "rollout_id_key": rollout_id_key,
        "rollout_reward_key": rollout_reward_key,
        "dataset_id_key": dataset_id_key,
        "baseline_examples": len(grouped),
        "eligible_examples": len(solve_rates),
        "selected_examples": len(selected_rows),
        "missing_selected_ids": sorted(selected_ids - {str(row.get(dataset_id_key)) for row in selected_rows}),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-rollouts", type=Path, required=True, help="jsonl from `uv run baselines @ ...`")
    parser.add_argument("--dataset", type=Path, required=True, help="source jsonl to filter")
    parser.add_argument("--output", type=Path, required=True, help="filtered jsonl output path")
    parser.add_argument("--low", type=float, default=0.2, help="lower solve-rate bound (default: 0.2)")
    parser.add_argument("--high", type=float, default=0.8, help="upper solve-rate bound (default: 0.8)")
    parser.add_argument(
        "--min-rollouts",
        type=int,
        default=8,
        help="discard examples with fewer than this many rollouts (default: 8)",
    )
    parser.add_argument(
        "--rollout-id-key",
        default="example_id",
        help="rollout row key that joins to --dataset-id-key (default: example_id)",
    )
    parser.add_argument(
        "--rollout-reward-key",
        default="reward",
        help="rollout row key holding the per-rollout scalar reward (default: reward)",
    )
    parser.add_argument(
        "--dataset-id-key",
        default="id",
        help="source-dataset row key that joins to --rollout-id-key (default: id)",
    )
    args = parser.parse_args()
    summary = run(
        baseline_rollouts=args.baseline_rollouts,
        dataset=args.dataset,
        output=args.output,
        low=args.low,
        high=args.high,
        min_rollouts=args.min_rollouts,
        rollout_id_key=args.rollout_id_key,
        rollout_reward_key=args.rollout_reward_key,
        dataset_id_key=args.dataset_id_key,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
