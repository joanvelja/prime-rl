#!/usr/bin/env python3
"""Walk per-step offline-eval records and print qualitative trajectory summaries.

Groups rollouts by example id and shows how each example's outputs evolved
across checkpoints: correctness, truncation, token counts, domain breakdowns,
parser/judge counters, and transcript pointers.

Usage:
    uv run scripts/evals/analyze_qual_trajectories.py \\
        --rollouts-root outputs/.../offline_eval_600x8/default \\
        --steps 300 350 400 450 \\
        [--limit 12] [--truncate-len 180] [--transcript-samples 4] \\
        [--out-dir outputs/.../qual_trajectories]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_step(root: Path, step: int) -> list[dict]:
    path = root / f"step_{step:06d}" / "records.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def short(s: str | None, n: int) -> str:
    if not s:
        return ""
    return " ".join(s.split())[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollouts-root", type=Path, required=True, help="root dir containing step_*/records.jsonl")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="checkpoint steps to walk")
    parser.add_argument("--limit", type=int, default=12, help="max examples per trajectory bucket (default: 12)")
    parser.add_argument("--truncate-len", type=int, default=180, help="max chars for response snippets (default: 180)")
    parser.add_argument("--transcript-samples", type=int, default=4, help="examples per bucket in transcript pointers (default: 4)")
    parser.add_argument("--out-dir", type=Path, default=None, help="write output to this directory instead of stdout")
    args = parser.parse_args()

    steps = args.steps
    first_step = steps[0]
    last_step = steps[-1]
    step_label = "/".join(str(s) for s in steps)

    by_example: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    by_step: dict[int, list[dict]] = {}
    for step in steps:
        rows = load_step(args.rollouts_root, step)
        by_step[step] = rows
        for row in rows:
            by_example[str(row["example_id"])][step].append(row)

    out = sys.stdout
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = (args.out_dir / "qual_trajectories.md").open("w")

    def pr(*a: object, **kw: object) -> None:
        print(*a, **kw, file=out)

    pr("# Offline Qualitative Trajectory Selector")
    pr()

    pr("## Step-Level Summary")
    pr("| step | rows | p@8 | mean correct/sample | trunc | mean output tokens | parsed null |")
    pr("|---:|---:|---:|---:|---:|---:|---:|")
    for step, rows in by_step.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row["example_id"])].append(row)
        p8 = mean(any(r.get("correct") for r in group) for group in grouped.values())
        correct = mean(bool(r.get("correct")) for r in rows)
        trunc = mean(bool(r.get("is_truncated")) for r in rows)
        out_tok = mean(float(r.get("output_tokens") or 0) for r in rows)
        parsed_null = mean(r.get("parsed_answer") is None for r in rows)
        pr(f"| {step} | {len(rows)} | {p8:.4f} | {correct:.4f} | {trunc:.4f} | {out_tok:.1f} | {parsed_null:.4f} |")
    pr()

    trajectories = []
    for example_id, step_map in by_example.items():
        if any(step not in step_map for step in steps):
            continue
        counts = {step: sum(bool(r.get("correct")) for r in step_map[step]) for step in steps}
        truncs = {step: sum(bool(r.get("is_truncated")) for r in step_map[step]) for step in steps}
        tokens = {step: mean(float(r.get("output_tokens") or 0) for r in step_map[step]) for step in steps}
        domain = step_map[first_step][0].get("info", {}).get("primary_domain")
        difficulty = step_map[first_step][0].get("info", {}).get("difficulty")
        source = step_map[first_step][0].get("info", {}).get("source_bucket")
        trajectories.append(
            {
                "example_id": example_id,
                "counts": counts,
                "truncs": truncs,
                "tokens": tokens,
                "domain": domain,
                "difficulty": difficulty,
                "source": source,
            }
        )

    def print_examples(title: str, items: list[dict]) -> None:
        pr(f"## {title}")
        pr(f"| example_id | domain | diff | source | correct {step_label} | trunc {step_label} | mean toks {step_label} |")
        pr("|---:|---|---:|---|---|---|---|")
        for item in items[: args.limit]:
            c = "/".join(str(item["counts"][s]) for s in steps)
            t = "/".join(str(item["truncs"][s]) for s in steps)
            tok = "/".join(f"{item['tokens'][s]:.0f}" for s in steps)
            pr(f"| {item['example_id']} | {item['domain']} | {item['difficulty']} | {item['source']} | {c} | {t} | {tok} |")
        pr()

    improving = sorted(
        [x for x in trajectories if x["counts"][last_step] > x["counts"][first_step]],
        key=lambda x: (x["counts"][last_step] - x["counts"][first_step], x["counts"][last_step]),
        reverse=True,
    )
    regressing = sorted(
        [x for x in trajectories if x["counts"][last_step] < x["counts"][first_step]],
        key=lambda x: (x["counts"][first_step] - x["counts"][last_step], x["counts"][first_step]),
        reverse=True,
    )
    peaked_mid = sorted(
        [x for x in trajectories if len(steps) > 2 and x["counts"][steps[1]] > max(x["counts"][s] for s in steps if s != steps[1])],
        key=lambda x: x["counts"][steps[1]] - max(x["counts"][s] for s in steps if s != steps[1]),
        reverse=True,
    )
    trunc_worse = sorted(
        [x for x in trajectories if x["truncs"][last_step] > x["truncs"][first_step]],
        key=lambda x: x["truncs"][last_step] - x["truncs"][first_step],
        reverse=True,
    )
    trunc_better = sorted(
        [x for x in trajectories if x["truncs"][last_step] < x["truncs"][first_step]],
        key=lambda x: x["truncs"][first_step] - x["truncs"][last_step],
        reverse=True,
    )

    print_examples(f"Largest Correctness Improvements {first_step}→{last_step}", improving)
    print_examples(f"Largest Correctness Regressions {first_step}→{last_step}", regressing)
    print_examples(f"Step-{steps[1]}-Only Peaks", peaked_mid)
    print_examples(f"Truncation Worsened {first_step}→{last_step}", trunc_worse)
    print_examples(f"Truncation Improved {first_step}→{last_step}", trunc_better)

    pr("## Domain Counts")
    for step in steps:
        grouped = defaultdict(list)
        for row in by_step[step]:
            grouped[row.get("info", {}).get("primary_domain")].append(row)
        pr(f"### step {step}")
        for domain, rows in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
            correct = mean(bool(r.get("correct")) for r in rows)
            trunc = mean(bool(r.get("is_truncated")) for r in rows)
            pr(f"- {domain}: n={len(rows)} correct={correct:.4f} trunc={trunc:.4f}")
        pr()

    pr("## Parser/Judge Counters")
    for step, rows in by_step.items():
        metrics = Counter()
        for row in rows:
            if row.get("parsed_answer") is None:
                metrics["parsed_null"] += 1
            if row.get("judge_response") is not None:
                metrics["judge_response"] += 1
            if row.get("metrics", {}).get("math_verify_score") == 1.0:
                metrics["math_verify_correct"] += 1
            if row.get("metrics", {}).get("judge_score") == 1.0:
                metrics["judge_correct"] += 1
        pr(f"- step {step}: {dict(metrics)}")

    pr()
    pr("## Quick Transcript Pointers")
    k = args.transcript_samples
    interesting = []
    for bucket in (improving[:k], regressing[:k], peaked_mid[:k], trunc_worse[:k], trunc_better[:k]):
        interesting.extend(bucket)
    seen = set()
    for item in interesting:
        if item["example_id"] in seen:
            continue
        seen.add(item["example_id"])
        pr(f"### example {item['example_id']}")
        for step in steps:
            rows = sorted(by_example[item["example_id"]][step], key=lambda r: r["sample_idx"])
            first_correct = next((r for r in rows if r.get("correct")), None)
            first_trunc = next((r for r in rows if r.get("is_truncated")), None)
            row = first_correct or first_trunc or rows[0]
            pr(
                f"- step {step} sample {row['sample_idx']} correct={row.get('correct')} "
                f"trunc={row.get('is_truncated')} parsed={row.get('parsed_answer')!r} "
                f"tokens={row.get('output_tokens')} path={args.rollouts_root / f'step_{step:06d}' / 'records.jsonl'}"
            )
            pr(f"  response: {short(row.get('response'), args.truncate_len)}")
        pr()

    if out is not sys.stdout:
        out.close()
        print(args.out_dir / "qual_trajectories.md")


if __name__ == "__main__":
    main()
