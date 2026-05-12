#!/usr/bin/env python3
"""Qualitative time-axis audit across train, online-eval, and offline-eval rollouts.

Combines train rollouts, online eval rollouts, and offline checkpoint eval
rollouts into one time-axis qualitative dump: tokens, truncation rates, parsed
answers, correctness, and transcript motifs. Output is a markdown report.

Usage:
    uv run scripts/evals/analyze_qual_time_axis.py \\
        --root outputs/omni_math2_rlvr_canary/20260509_1825 \\
        --train-rollouts-dir dpo_default_14i2t_bs256_eval50_1000step/run_default/rollouts \\
        --offline-rollouts-dir offline_eval_600x8/default \\
        --out tmp/qual_time_axis.md \\
        --train-steps 0 1 10 50 100 150 200 250 300 350 400 450 500 \\
        --online-eval-steps 51 101 151 201 251 301 351 401 \\
        --offline-steps 300 350 400 450

Paths for --train-rollouts-dir and --offline-rollouts-dir can be absolute or
relative to --root.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def completion_text(row: dict[str, Any]) -> str:
    value = row.get("completion")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if content:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    response = row.get("response")
    if isinstance(response, str):
        return response
    return ""


def token_usage(row: dict[str, Any], key: str) -> float:
    if key in row and row[key] is not None:
        return float(row[key])
    usage = row.get("token_usage")
    if isinstance(usage, dict) and usage.get(key) is not None:
        return float(usage[key])
    return 0.0


def is_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    if row.get("reward") is not None:
        return float(row["reward"]) > 0.0
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metrics.get("judge_score") is not None:
        return float(metrics["judge_score"]) > 0.0
    return False


def is_truncated(row: dict[str, Any]) -> bool:
    return bool(row.get("is_truncated") or row.get("truncated"))


def answer_value(row: dict[str, Any]) -> str | None:
    for key in ("parsed_answer", "answer"):
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    return None


def p_quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[idx]


def unbiased_pass_at_k(correct: int, total: int, k: int) -> float:
    if total <= 0:
        return 0.0
    if correct <= 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def tail_flags(text: str, tail_chars: int) -> dict[str, bool]:
    lower = text.lower()
    tail = text[-tail_chars:]
    words = re.findall(r"[A-Za-z0-9_\\{}^+\-=/]+", tail)
    grams = Counter(tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4)))
    repeated_5gram = bool(grams and grams.most_common(1)[0][1] >= 12)
    repeated_symbol = bool(re.search(r"([-_=*]{8,})(?:\s*\1){2,}", tail))
    self_correction = sum(lower.count(x) for x in ("wait", "but", "however", "actually")) >= 8
    enumeration = len(re.findall(r"\bcase\b|\bsubcase\b|^\s*[-*]\s+", lower, re.M)) >= 18
    final_answer = bool(re.search(r"final answer|\\boxed\s*{", lower))
    theorem_soup = len(
        re.findall(
            r"cauchy|jensen|am-gm|pigeonhole|inclusion-exclusion|vieta|fermat|euler|"
            r"legendre|burnside|generating function|induction|contradiction",
            lower,
        )
    )
    return {
        "repeated_tail": repeated_5gram or repeated_symbol,
        "self_correction": self_correction,
        "enumeration": enumeration,
        "final_answer_marker": final_answer,
        "theorem_mentions_3p": theorem_soup >= 3,
    }


def summarize_rows(rows: list[dict[str, Any]], tail_chars: int) -> dict[str, Any]:
    outs = [token_usage(r, "output_tokens") for r in rows]
    flags = [tail_flags(completion_text(r), tail_chars) for r in rows]
    answers = [answer_value(r) for r in rows]
    return {
        "n": len(rows),
        "correct": sum(is_correct(r) for r in rows),
        "trunc": sum(is_truncated(r) for r in rows),
        "answer_null": sum(a is None for a in answers),
        "unique_answers": len({a for a in answers if a is not None}),
        "mean_out": mean(outs) if outs else 0.0,
        "median_out": median(outs) if outs else 0.0,
        "p95_out": p_quantile(outs, 0.95),
        "max_out": max(outs) if outs else 0.0,
        "final_answer_marker": sum(f["final_answer_marker"] for f in flags),
        "repeated_tail": sum(f["repeated_tail"] for f in flags),
        "self_correction": sum(f["self_correction"] for f in flags),
        "enumeration": sum(f["enumeration"] for f in flags),
        "theorem_mentions_3p": sum(f["theorem_mentions_3p"] for f in flags),
    }


def format_summary_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| split | step | n | correct/sample | p@8 | trunc | answer_null | unique_answers | mean_out_tok | p95_out_tok | final_marker | repeat_tail | self_correct | enum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            "| {split} | {step} | {n} | {correct:.4f} | {p8:.4f} | {trunc:.4f} | "
            "{answer_null:.4f} | {unique_answers} | {mean_out:.1f} | {p95_out:.0f} | "
            "{final_marker:.3f} | {repeat_tail:.3f} | {self_correct:.3f} | {enum:.3f} |".format(
                split=r["split"],
                step=r["step"],
                n=r["n"],
                correct=r["correct_rate"],
                p8=r.get("pass8", 0.0),
                trunc=r["trunc_rate"],
                answer_null=r["answer_null_rate"],
                unique_answers=r["unique_answers"],
                mean_out=r["mean_out"],
                p95_out=r["p95_out"],
                final_marker=r["final_answer_marker_rate"],
                repeat_tail=r["repeated_tail_rate"],
                self_correct=r["self_correction_rate"],
                enum=r["enumeration_rate"],
            )
        )
    return out


def add_rates(summary: dict[str, Any], split: str, step: int, pass8: float = 0.0) -> dict[str, Any]:
    n = summary["n"] or 1
    return {
        **summary,
        "split": split,
        "step": step,
        "correct_rate": summary["correct"] / n,
        "trunc_rate": summary["trunc"] / n,
        "answer_null_rate": summary["answer_null"] / n,
        "final_answer_marker_rate": summary["final_answer_marker"] / n,
        "repeated_tail_rate": summary["repeated_tail"] / n,
        "self_correction_rate": summary["self_correction"] / n,
        "enumeration_rate": summary["enumeration"] / n,
        "pass8": pass8,
    }


def passk_from_rows(rows: list[dict[str, Any]]) -> dict[int, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("example_id"))].append(r)
    result = {}
    for k in (1, 2, 3, 4, 5, 6, 8):
        vals = []
        for sample_rows in grouped.values():
            c = sum(is_correct(x) for x in sample_rows)
            n = len(sample_rows)
            vals.append(unbiased_pass_at_k(c, n, min(k, n)))
        result[k] = mean(vals) if vals else 0.0
    return result


def best_examples_by_delta(
    offline_dir: Path,
    offline_steps: list[int],
    step_a: int,
    step_b: int,
    limit: int = 8,
) -> tuple[list[tuple[int, str, list[int]]], list[tuple[int, str, list[int]]]]:
    counts: dict[str, list[int]] = defaultdict(list)
    for step in offline_steps:
        rows = list(iter_jsonl(offline_dir / f"step_{step:06d}/records.jsonl"))
        by_ex: dict[str, int] = defaultdict(int)
        for r in rows:
            by_ex[str(r["example_id"])] += int(is_correct(r))
        for ex, c in by_ex.items():
            counts[ex].append(c)
    idx_a = offline_steps.index(step_a)
    idx_b = offline_steps.index(step_b)
    scored = [(vals[idx_b] - vals[idx_a], ex, vals) for ex, vals in counts.items() if len(vals) == len(offline_steps)]
    return sorted(scored, reverse=True)[:limit], sorted(scored)[:limit]


def representative(rows: list[dict[str, Any]], prefer_trunc: bool = False) -> dict[str, Any] | None:
    if not rows:
        return None
    if prefer_trunc:
        rows = sorted(rows, key=lambda r: (not is_truncated(r), -token_usage(r, "output_tokens")))
    else:
        rows = sorted(rows, key=lambda r: (-token_usage(r, "output_tokens"), not is_truncated(r)))
    return rows[0]


def compact_snip(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def resolve_dir(path: Path, root: Path) -> Path:
    """Resolve path: use directly if absolute, otherwise resolve against root."""
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="run output root directory")
    parser.add_argument("--train-rollouts-dir", type=Path, required=True, help="train rollouts dir (absolute or relative to --root)")
    parser.add_argument("--offline-rollouts-dir", type=Path, required=True, help="offline eval rollouts dir (absolute or relative to --root)")
    parser.add_argument("--out", type=Path, required=True, help="markdown output path")
    parser.add_argument("--train-steps", type=int, nargs="+", required=True, help="train step numbers to include")
    parser.add_argument("--online-eval-steps", type=int, nargs="+", required=True, help="online eval step numbers to include")
    parser.add_argument("--offline-steps", type=int, nargs="+", required=True, help="offline eval step numbers to include")
    parser.add_argument("--interesting-examples", type=str, nargs="*", default=[], help="example IDs to show detailed offline trajectories for")
    parser.add_argument("--snip-limit", type=int, default=420, help="char limit for compact_snip (default: 420)")
    parser.add_argument("--anchor-head-chars", type=int, default=1200, help="chars to show from head of transcript anchors (default: 1200)")
    parser.add_argument("--anchor-tail-chars", type=int, default=1200, help="chars to show from tail of transcript anchors (default: 1200)")
    parser.add_argument("--trajectory-tail-chars", type=int, default=900, help="chars to show from tail of offline trajectories (default: 900)")
    parser.add_argument("--tail-flag-chars", type=int, default=2000, help="chars to scan for tail-flag motifs (default: 2000)")
    parser.add_argument("--delta-steps", type=int, nargs=2, default=None, metavar=("FROM", "TO"), help="offline step pair for biggest-flips comparison (default: first and last --offline-steps)")
    parser.add_argument("--flip-limit", type=int, default=8, help="number of biggest-delta examples to show (default: 8)")
    args = parser.parse_args()

    train_dir = resolve_dir(args.train_rollouts_dir, args.root)
    offline_dir = resolve_dir(args.offline_rollouts_dir, args.root)
    delta_from, delta_to = args.delta_steps if args.delta_steps else (args.offline_steps[0], args.offline_steps[-1])

    lines: list[str] = [
        "# Qualitative Time-Axis Audit",
        "",
        "Scope: current default run, init/train rollouts, online eval rollouts, and offline checkpoint evals. Length is measured in generated tokens, not characters.",
        "",
    ]

    table_rows = []
    for step in args.train_steps:
        path = train_dir / f"step_{step}/train_rollouts.jsonl"
        if not path.exists():
            continue
        rows = list(iter_jsonl(path))
        table_rows.append(add_rates(summarize_rows(rows, args.tail_flag_chars), "train", step))

    for step in args.online_eval_steps:
        path = train_dir / f"step_{step}/eval_rollouts.jsonl"
        if not path.exists():
            continue
        rows = list(iter_jsonl(path))
        pk = passk_from_rows(rows)
        table_rows.append(add_rates(summarize_rows(rows, args.tail_flag_chars), "online_eval", step, pk.get(8, 0.0)))

    for step in args.offline_steps:
        path = offline_dir / f"step_{step:06d}/records.jsonl"
        rows = list(iter_jsonl(path))
        pk = passk_from_rows(rows)
        table_rows.append(add_rates(summarize_rows(rows, args.tail_flag_chars), "offline_600x8", step, pk.get(8, 0.0)))

    lines.extend(format_summary_table(table_rows))
    lines.append("")

    lines.append("## Pass@k, Offline 600x8")
    lines.append("")
    lines.append("| step | p@1 | p@2 | p@3 | p@4 | p@5 | p@6 | p@8 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for step in args.offline_steps:
        rows = list(iter_jsonl(offline_dir / f"step_{step:06d}/records.jsonl"))
        pk = passk_from_rows(rows)
        lines.append(
            f"| {step} | "
            + " | ".join(f"{pk[k]:.4f}" for k in (1, 2, 3, 4, 5, 6, 8))
            + " |"
        )
    lines.append("")

    lines.append("## Biggest Offline Flips")
    lines.append("")
    improves, regresses = best_examples_by_delta(offline_dir, args.offline_steps, delta_from, delta_to, args.flip_limit)
    lines.append(f"Improved {delta_from}->{delta_to}:")
    for delta, ex, vals in improves:
        lines.append(f"- example {ex}: counts {vals}, delta {delta:+d}")
    lines.append("")
    lines.append(f"Regressed {delta_from}->{delta_to}:")
    for delta, ex, vals in regresses:
        lines.append(f"- example {ex}: counts {vals}, delta {delta:+d}")
    lines.append("")

    lines.append("## Transcript Anchors")
    lines.append("")
    lines.append("These are deliberately short anchors for manual reading, selected by token budget/truncation or known correctness flips.")
    lines.append("")

    anchor_specs = [
        ("train", args.train_steps[0], train_dir / f"step_{args.train_steps[0]}/train_rollouts.jsonl"),
        ("train", args.train_steps[-1], train_dir / f"step_{args.train_steps[-1]}/train_rollouts.jsonl"),
        ("online_eval", args.online_eval_steps[0], train_dir / f"step_{args.online_eval_steps[0]}/eval_rollouts.jsonl"),
        ("online_eval", args.online_eval_steps[-1], train_dir / f"step_{args.online_eval_steps[-1]}/eval_rollouts.jsonl"),
    ]
    for split, step, path in anchor_specs:
        if not path.exists():
            continue
        rows = list(iter_jsonl(path))
        row = representative(rows, prefer_trunc=True)
        if row is None:
            continue
        text = completion_text(row)
        flags = tail_flags(text, args.tail_flag_chars)
        lines.append(
            f"- {split} step {step}, example {row.get('example_id')}, "
            f"reward={row.get('reward')}, truncated={is_truncated(row)}, "
            f"out_tok={token_usage(row, 'output_tokens'):.0f}, answer={answer_value(row)!r}, "
            f"flags={','.join(k for k, v in flags.items() if v)}"
        )
        lines.append(f"  - head: {compact_snip(text[:args.anchor_head_chars], args.snip_limit)}")
        lines.append(f"  - tail: {compact_snip(text[-args.anchor_tail_chars:], args.snip_limit)}")
    lines.append("")

    lines.append("## Offline Example Trajectories")
    lines.append("")
    for ex in args.interesting_examples:
        lines.append(f"### example {ex}")
        for step in args.offline_steps:
            rows = [r for r in iter_jsonl(offline_dir / f"step_{step:06d}/records.jsonl") if str(r.get("example_id")) == ex]
            if not rows:
                continue
            c = sum(is_correct(r) for r in rows)
            tr = sum(is_truncated(r) for r in rows)
            ans = Counter(answer_value(r) for r in rows)
            chosen = representative(rows, prefer_trunc=bool(tr))
            assert chosen is not None
            flags = tail_flags(completion_text(chosen), args.tail_flag_chars)
            lines.append(
                f"- step {step}: correct={c}/8 trunc={tr}/8 mean_out={mean([token_usage(r, 'output_tokens') for r in rows]):.0f} "
                f"top_answers={ans.most_common(3)} chosen_sample={chosen.get('sample_idx')} "
                f"flags={','.join(k for k, v in flags.items() if v)}"
            )
            lines.append(f"  - tail: {compact_snip(completion_text(chosen)[-args.trajectory_tail_chars:], args.snip_limit)}")
        lines.append("")

    args.out.write_text("\n".join(lines) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
