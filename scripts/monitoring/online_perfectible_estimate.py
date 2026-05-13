#!/usr/bin/env python3
"""Online (mid-run) estimate of perfectible-subset size from a judge sqlite cache.

While a baselines run is in flight, records.jsonl is only flushed at the end.
But the judge persistent cache (judge_cells + judge_samples) is updated as
rollouts complete. This script parses that cache and reports per-problem
judge-side solve rates plus an extrapolated perfectible count.

Usage:
    uv run python scripts/monitoring/online_perfectible_estimate.py \\
        --judge-cache outputs/baselines/.../judge-cache/...sqlite3 \\
        [--rubric-family omni_math2_hybrid_math_v1] \\
        [--low 0.2 --high 0.8] \\
        [--min-judge-calls 4] \\
        [--total-problems 1000]

Caveats (be honest about what this measures):

- The judge is only invoked when math_verify is ambiguous (~50-60% of
  rollouts in typical omni-math2 settings). Problems where math_verify
  cleanly resolves every rollout do NOT appear in this cache. Solve-rate
  estimates here are biased toward the math-ambiguous half of the dataset.

- The runner processes problems roughly in source order, so early in a run
  only the first N/1000 problems are represented. The "% perfectible"
  number stabilizes only once most problems have been touched.

- "Perfectible" here = judge-side correct rate in [low, high]. The post-run
  scripts/evals/make_perfectible_subset.py operates on the FINAL reward
  field (math_verify + judge combined) and is the authoritative filter.
  This script is a directional progress indicator, not a final answer.

- Empirical undercount observed on omni-math2 / OLMo3-7B-Instruct-DPO:
  judge-side extrapolation predicted ~217-226 perfectible problems, the
  full-reward filter returned 340 (∼1.5× more). Bias is in the expected
  direction (judge only sees the math-ambiguous tail) and stabilizes
  within ~20 min of run start, but magnitude can be substantial. Treat
  the extrapolated count as a lower bound, not a point estimate.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def query_per_problem(judge_cache: Path, rubric_family: str) -> dict[str, dict[str, int]]:
    conn = sqlite3.connect(f"file:{judge_cache}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT c.question_id, s.verdict
            FROM judge_cells c JOIN judge_samples s ON c.cell_key = s.cell_key
            WHERE c.rubric_family = ?
            """,
            (rubric_family,),
        ).fetchall()
    finally:
        conn.close()

    per_q: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for qid, verdict in rows:
        per_q[qid]["total"] += 1
        if (verdict or "").strip().upper() == "CORRECT":
            per_q[qid]["correct"] += 1
    return per_q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-cache", type=Path, required=True, help="sqlite3 file (judge_persistent_cache_path)")
    parser.add_argument(
        "--rubric-family",
        default="omni_math2_hybrid_math_v1",
        help="filter judge_cells.rubric_family (default: omni_math2_hybrid_math_v1)",
    )
    parser.add_argument("--low", type=float, default=0.2, help="perfectible band lower bound (default: 0.2)")
    parser.add_argument("--high", type=float, default=0.8, help="perfectible band upper bound (default: 0.8)")
    parser.add_argument(
        "--min-judge-calls",
        type=int,
        default=4,
        help="discard problems with fewer than this many judge verdicts (default: 4)",
    )
    parser.add_argument(
        "--total-problems",
        type=int,
        default=None,
        help="total problems in the run (for extrapolation; default: skip)",
    )
    args = parser.parse_args()

    per_q = query_per_problem(args.judge_cache, args.rubric_family)
    total_verdicts = sum(d["total"] for d in per_q.values())
    print(f"Total judge verdicts in cache: {total_verdicts}")
    print(f"Distinct problems with any judge call: {len(per_q)}")

    filtered = {q: d for q, d in per_q.items() if d["total"] >= args.min_judge_calls}
    print(f"Problems with >={args.min_judge_calls} judge calls: {len(filtered)}")
    if not filtered:
        return 0

    buckets: Counter[float] = Counter()
    perfectible = 0
    for d in filtered.values():
        rate = d["correct"] / d["total"]
        buckets[min(int(rate * 10), 9) / 10] += 1
        if args.low <= rate <= args.high:
            perfectible += 1

    print(f"\nJudge-side solve-rate histogram (perfectible band [{args.low}, {args.high}] marked with *):")
    for b in sorted(buckets):
        in_band = args.low <= b < args.high or (b == 0.9 and args.high >= 1.0)
        marker = "*" if in_band else " "
        bar = "#" * buckets[b]
        print(f"  {marker} [{b:.1f}-{b + 0.1:.1f}) : {buckets[b]:4d} {bar[:60]}")

    print(
        f"\nPerfectible (in [{args.low}, {args.high}] judge-side): {perfectible} / {len(filtered)} "
        f"({perfectible * 100 / len(filtered):.1f}%)"
    )

    if args.total_problems is not None:
        extrap = int(perfectible * args.total_problems / len(filtered))
        print(f"Extrapolation to {args.total_problems} problems (assumes this fraction holds): ~{extrap} perfectible")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
