"""Summarize PRIME-RL online eval_rollouts.jsonl artifacts.

The online W&B rows are useful, but they do not include uncertainty bands or
enough scorer diagnostics for checkpoint decisions. This script reads the
persisted rollout JSONL files, aggregates pass@k by example id, and optionally
bootstraps paired deltas against a reference checkpoint.

Usage:
    uv run --no-sync python -m scripts.evals.analyze_online_eval_rollouts \
        --rollouts-root outputs/.../run_default/rollouts \
        --baseline-step 50 \
        --output-json tmp/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from prime_rl.orchestrator.eval_utils import compute_pass_at_k

_STEP_RE = re.compile(r"step_(\d+)$")


def _step_from_path(path: Path) -> int:
    match = _STEP_RE.search(path.parent.name)
    if match is None:
        raise ValueError(f"Could not parse step from {path}")
    return int(match.group(1))


def _ckpt_step(row: dict[str, Any], fallback_step: int) -> int:
    cache_salt = ((row.get("sampling_args") or {}).get("extra_body") or {}).get("cache_salt")
    if isinstance(cache_salt, str) and cache_salt.isdigit():
        return int(cache_salt)
    if isinstance(cache_salt, int):
        return cache_salt
    return fallback_step


def _metric(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        value = (row.get("metrics") or {}).get(name)
    return float(value) if value is not None else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def _load_eval_file(path: Path) -> tuple[int, list[dict[str, Any]]]:
    fallback_step = _step_from_path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return fallback_step, []
    ckpt_step = _ckpt_step(rows[0], fallback_step)
    return ckpt_step, rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_example: dict[str, list[float]] = defaultdict(list)
    scoring_ms: list[float] = []
    generation_ms: list[float] = []
    completion_lens: list[float] = []
    stop_counts: dict[str, int] = defaultdict(int)
    metric_values: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        by_example[str(row["example_id"])].append(float(row.get("reward") or 0.0))
        timing = row.get("timing") or {}
        if timing.get("scoring_ms") is not None:
            scoring_ms.append(float(timing["scoring_ms"]))
        if timing.get("generation_ms") is not None:
            generation_ms.append(float(timing["generation_ms"]))
        token_usage = row.get("token_usage") or {}
        if token_usage.get("output_tokens") is not None:
            completion_lens.append(float(token_usage["output_tokens"]))
        stop_condition = row.get("stop_condition")
        if stop_condition:
            stop_counts[str(stop_condition)] += 1
        for metric_name in ("math_verify_score", "choice_alias_score", "text_alias_score", "judge_score"):
            value = _metric(row, metric_name)
            if value is not None:
                metric_values[metric_name].append(value)

    pass_by_example = [compute_pass_at_k(rewards) for rewards in by_example.values()]
    pass_at_k = {
        key: statistics.fmean(example_pass[key] for example_pass in pass_by_example if key in example_pass)
        for key in sorted({key for example_pass in pass_by_example for key in example_pass})
    }
    total = len(rows)
    return {
        "rollouts": total,
        "examples": len(by_example),
        "avg_reward": statistics.fmean(float(row.get("reward") or 0.0) for row in rows),
        "pass_at_k": pass_at_k,
        "truncated_rate": statistics.fmean(float(bool(row.get("is_truncated"))) for row in rows),
        "no_response_rate": statistics.fmean(float(not row.get("completion")) for row in rows),
        "stop_condition_rates": {key: value / total for key, value in sorted(stop_counts.items())},
        "scoring_ms": {
            "mean": statistics.fmean(scoring_ms) if scoring_ms else None,
            "p95": _quantile(scoring_ms, 0.95),
            "max": max(scoring_ms) if scoring_ms else None,
        },
        "generation_ms": {
            "mean": statistics.fmean(generation_ms) if generation_ms else None,
            "p95": _quantile(generation_ms, 0.95),
            "max": max(generation_ms) if generation_ms else None,
        },
        "output_tokens": {
            "mean": statistics.fmean(completion_lens) if completion_lens else None,
            "p95": _quantile(completion_lens, 0.95),
            "max": max(completion_lens) if completion_lens else None,
        },
        "metric_means": {
            key: statistics.fmean(values)
            for key, values in sorted(metric_values.items())
            if values
        },
    }


def _passes_by_example(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(float(row.get("reward") or 0.0))
    return {
        example_id: compute_pass_at_k(rewards)[key]
        for example_id, rewards in grouped.items()
        if key in compute_pass_at_k(rewards)
    }


def _bootstrap_delta(
    baseline_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    pass_key: str,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    baseline = _passes_by_example(baseline_rows, pass_key)
    target = _passes_by_example(target_rows, pass_key)
    ids = sorted(set(baseline) & set(target))
    if not ids:
        raise ValueError(f"No overlapping examples for {pass_key}")
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        drawn = [rng.choice(ids) for _ in ids]
        deltas.append(statistics.fmean(target[i] - baseline[i] for i in drawn))
    deltas.sort()
    return {
        "mean": statistics.fmean(deltas),
        "ci95": [deltas[int(0.025 * samples)], deltas[int(0.975 * samples)]],
        "overlap_examples": len(ids),
    }


def run(
    rollouts_root: Path,
    *,
    baseline_step: int | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    files = sorted(rollouts_root.glob("step_*/eval_rollouts.jsonl"), key=_step_from_path)
    if not files:
        raise ValueError(f"No eval_rollouts.jsonl files under {rollouts_root}")

    loaded: dict[tuple[str, int], list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for path in files:
        ckpt_step, rows = _load_eval_file(path)
        if not rows:
            continue
        rows_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_env[row.get("env_name") or "unknown"].append(row)
        for env_name, env_rows in rows_by_env.items():
            loaded[(env_name, ckpt_step)] = env_rows
            summaries.setdefault(env_name, {})[str(ckpt_step)] = _summarize_rows(env_rows)

    if baseline_step is not None:
        for env_name in list(summaries):
            baseline_rows = loaded.get((env_name, baseline_step))
            if baseline_rows is None:
                continue
            for ckpt_step in summaries[env_name]:
                step = int(ckpt_step)
                if step == baseline_step:
                    continue
                target_rows = loaded[(env_name, step)]
                deltas = {}
                for pass_key in ("pass@1", "pass@2", "pass@4", "pass@8"):
                    if pass_key in summaries[env_name][ckpt_step]["pass_at_k"]:
                        deltas[pass_key] = _bootstrap_delta(
                            baseline_rows,
                            target_rows,
                            pass_key=pass_key,
                            samples=bootstrap_samples,
                            seed=seed + step,
                        )
                summaries[env_name][ckpt_step]["delta_vs_baseline"] = deltas

    return {"rollouts_root": str(rollouts_root), "baseline_step": baseline_step, "envs": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts-root", type=Path, required=True)
    parser.add_argument("--baseline-step", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summary = run(
        args.rollouts_root,
        baseline_step=args.baseline_step,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
