#!/usr/bin/env python3
"""Watch local W&B eval tables and summarize held-out eval strata.

Polls on-disk eval tables from a training run, parses them, computes
per-stratum pass@k summaries against a stratified eval dataset, and emits
markdown/CSV/JSON summaries.

Usage:
    uv run scripts/monitoring/watch_eval_strata.py \\
        --run-dir outputs/.../run_default \\
        --eval-dataset benchmarks/datasets/omni_math2/omni_math2_stratified_600_seed42.jsonl \\
        --out-dir outputs/.../eval_strata_watch \\
        [--k-values 1 2 4 8] [--easy-threshold 0.875] [--hard-threshold 0.0625] \\
        [--interval 120] [--once]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in _read_jsonl(path)}


def _pass_at_k(correct: int, total: int, k: int) -> float:
    if total <= 0:
        return math.nan
    if k == 1:
        return correct / total
    if total < k:
        return correct / total
    incorrect = total - correct
    if incorrect < k:
        return 1.0
    return 1.0 - math.comb(incorrect, k) / math.comb(total, k)


def _pool_from_rate(rate: float, easy_threshold: float, hard_threshold: float) -> str:
    if rate >= easy_threshold:
        return "early_easy"
    if rate <= hard_threshold:
        return "early_hard"
    return "early_normal"


def _load_eval_table(path: Path) -> dict[str, Any]:
    table = json.loads(path.read_text())
    columns = table["columns"]
    idx = {name: columns.index(name) for name in columns}
    data = table["data"]
    if not data:
        raise ValueError(f"empty eval table: {path}")

    envs = sorted({row[idx["env"]] for row in data})
    steps = sorted({int(row[idx["step"]]) for row in data})
    if len(envs) != 1:
        raise ValueError(f"expected one env in {path}, got {envs}")
    if len(steps) != 1:
        raise ValueError(f"expected one step in {path}, got {steps}")

    by_example: dict[str, list[float]] = defaultdict(list)
    for row in data:
        by_example[str(row[idx["example_id"]])].append(float(row[idx["reward"]]))

    raw_step = steps[0]
    ckpt_step = raw_step - 1 if raw_step > 0 else raw_step
    return {
        "path": str(path),
        "suite": envs[0],
        "raw_step": raw_step,
        "step": ckpt_step,
        "examples": by_example,
        "rows": len(data),
    }


def _find_tables(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("wandb/run-*/files/media/table/eval/*.table.json"))


def _baseline_pools(
    tables: list[dict[str, Any]], easy_threshold: float, hard_threshold: float
) -> dict[str, str]:
    baseline_tables = [t for t in tables if t["suite"] == "omni-math2-baseline100"]
    if not baseline_tables:
        return {}
    first = min(baseline_tables, key=lambda t: t["step"])
    pools = {}
    for example_id, rewards in first["examples"].items():
        pools[example_id] = _pool_from_rate(mean(rewards), easy_threshold, hard_threshold)
    return pools


def _group_key(
    stratum_type: str,
    example_id: str,
    metadata: dict[str, dict[str, Any]],
    early_pools: dict[str, str],
) -> str | None:
    if stratum_type == "overall":
        return "overall"
    if stratum_type == "early_pool":
        return early_pools.get(example_id, "early_unlabeled")
    meta = metadata.get(example_id, {})
    if stratum_type == "difficulty":
        return str(meta.get("difficulty_label", meta.get("difficulty", "unknown")))
    if stratum_type == "primary_domain":
        return str(meta.get("primary_domain", "unknown"))
    if stratum_type == "source_bucket":
        return str(meta.get("source_bucket", meta.get("source", "unknown")))
    return None


def _summarize_group(
    rewards_by_example: dict[str, list[float]], k_values: tuple[int, ...],
) -> dict[str, Any]:
    per_k = {k: [] for k in k_values}
    total_samples = 0
    total_correct = 0
    for rewards in rewards_by_example.values():
        total = len(rewards)
        correct = int(sum(1 for reward in rewards if reward > 0.0))
        total_samples += total
        total_correct += correct
        for k in k_values:
            per_k[k].append(_pass_at_k(correct, total, k))
    row: dict[str, Any] = {
        "examples": len(rewards_by_example),
        "samples": total_samples,
        "correct_samples": total_correct,
    }
    for k in k_values:
        vals = [v for v in per_k[k] if math.isfinite(v)]
        row[f"pass@{k}"] = mean(vals) if vals else math.nan
    return row


def _summaries(
    tables: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    early_pools: dict[str, str],
    k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    stratum_types = ("overall", "early_pool", "difficulty", "primary_domain", "source_bucket")
    for table in tables:
        for stratum_type in stratum_types:
            grouped: dict[str, dict[str, list[float]]] = defaultdict(dict)
            for example_id, rewards in table["examples"].items():
                key = _group_key(stratum_type, example_id, metadata, early_pools)
                if key is None:
                    continue
                grouped[key][example_id] = rewards
            for stratum, examples in grouped.items():
                stats = _summarize_group(examples, k_values)
                rows.append(
                    {
                        "suite": table["suite"],
                        "step": table["step"],
                        "raw_step": table["raw_step"],
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "table_path": table["path"],
                        **stats,
                    }
                )
    return rows


def _attach_deltas(rows: list[dict[str, Any]], k_values: tuple[int, ...]) -> None:
    first_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r["step"]):
        key = (row["suite"], row["stratum_type"], row["stratum"])
        first_by_key.setdefault(key, row)
    for row in rows:
        base = first_by_key[(row["suite"], row["stratum_type"], row["stratum"])]
        for k in k_values:
            row[f"delta_pass@{k}_vs_first"] = row[f"pass@{k}"] - base[f"pass@{k}"]


def _write_csv(rows: list[dict[str, Any]], path: Path, k_values: tuple[int, ...]) -> None:
    fields = [
        "suite",
        "step",
        "raw_step",
        "stratum_type",
        "stratum",
        "examples",
        "samples",
        "correct_samples",
        *[f"pass@{k}" for k in k_values],
        *[f"delta_pass@{k}_vs_first" for k in k_values],
        "table_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, precision: int = 4) -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{precision}f}"


def _latest_suite_rows(rows: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    suite_rows = [row for row in rows if row["suite"] == suite]
    if not suite_rows:
        return []
    step = max(row["step"] for row in suite_rows)
    return [row for row in suite_rows if row["step"] == step]


def _rows_of_type(rows: list[dict[str, Any]], stratum_type: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["stratum_type"] == stratum_type]


def _sort_difficulty(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _alerts(rows: list[dict[str, Any]]) -> list[str]:
    alerts = []
    latest_baseline = _latest_suite_rows(rows, "omni-math2-baseline100")
    for row in latest_baseline:
        if row["stratum_type"] != "early_pool":
            continue
        if row["stratum"] == "early_easy" and row["delta_pass@8_vs_first"] < -0.05:
            alerts.append(
                f"early_easy held-out pass@8 dropped by {_fmt(row['delta_pass@8_vs_first'])} at step {row['step']}."
            )
        if row["stratum"] == "early_easy" and row["delta_pass@1_vs_first"] < -0.05:
            alerts.append(
                f"early_easy held-out pass@1 dropped by {_fmt(row['delta_pass@1_vs_first'])} at step {row['step']}."
            )
        if row["stratum"] == "early_normal" and row["delta_pass@8_vs_first"] < -0.10:
            alerts.append(
                f"early_normal held-out pass@8 dropped by {_fmt(row['delta_pass@8_vs_first'])} at step {row['step']}."
            )
        if row["stratum"] == "early_hard" and row["delta_pass@8_vs_first"] > 0.05:
            alerts.append(
                f"early_hard held-out pass@8 improved by {_fmt(row['delta_pass@8_vs_first'])} at step {row['step']}."
            )
    return alerts


def _append_table(lines: list[str], rows: list[dict[str, Any]], title: str, max_rows: int | None = None) -> None:
    lines.extend(["", f"## {title}", "", "| stratum | examples | p@1 | p@2 | p@4 | p@8 | d p@1 | d p@8 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows[:max_rows]:
        lines.append(
            "| `{stratum}` | {examples} | {p1} | {p2} | {p4} | {p8} | {dp1} | {dp8} |".format(
                stratum=row["stratum"],
                examples=row["examples"],
                p1=_fmt(row["pass@1"]),
                p2=_fmt(row["pass@2"]),
                p4=_fmt(row["pass@4"]),
                p8=_fmt(row["pass@8"]),
                dp1=_fmt(row["delta_pass@1_vs_first"]),
                dp8=_fmt(row["delta_pass@8_vs_first"]),
            )
        )


def _write_markdown(
    rows: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    early_pool_counts: dict[str, int],
    alerts: list[str],
    path: Path,
) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()

    lines = [
        "# Eval Strata Watch",
        "",
        f"Updated UTC: `{updated_at}`",
        f"Eval tables found: `{len(tables)}`",
        "",
        "## Scope",
        "",
        "- Online eval is held out from train: `omni_math2_train_excluding_baseline600_seed42.jsonl` excludes `omni_math2_stratified_600_seed42.jsonl`.",
        "- Therefore this report does not join eval rows to train easy/normal/hard pools.",
        "- `early_pool` is the held-out analogue, assigned from the first baseline100 eval using the same reward thresholds.",
        "",
        "## Alerts",
        "",
    ]
    if alerts:
        lines.extend(f"- {alert}" for alert in alerts)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Held-Out Early Pools",
            "",
            "| pool | examples |",
            "|---|---:|",
        ]
    )
    for pool in ("early_easy", "early_normal", "early_hard"):
        lines.append(f"| `{pool}` | {early_pool_counts.get(pool, 0)} |")

    baseline_rows = _latest_suite_rows(rows, "omni-math2-baseline100")
    full_rows = _latest_suite_rows(rows, "omni-math2-full600-p8")

    if baseline_rows:
        step = baseline_rows[0]["step"]
        lines.extend(["", f"# Latest Baseline100 Step {step}"])
        _append_table(lines, _rows_of_type(baseline_rows, "overall"), "Overall")
        _append_table(lines, sorted(_rows_of_type(baseline_rows, "early_pool"), key=lambda r: r["stratum"]), "Held-Out Early Pool")
        _append_table(
            lines,
            sorted(_rows_of_type(baseline_rows, "difficulty"), key=lambda r: _sort_difficulty(r["stratum"])),
            "Difficulty",
        )
        _append_table(
            lines,
            sorted(_rows_of_type(baseline_rows, "primary_domain"), key=lambda r: r["stratum"]),
            "Primary Domain",
        )

    if full_rows:
        step = full_rows[0]["step"]
        lines.extend(["", f"# Latest Full600-p8 Step {step}"])
        _append_table(lines, _rows_of_type(full_rows, "overall"), "Overall")
        _append_table(
            lines,
            sorted(_rows_of_type(full_rows, "difficulty"), key=lambda r: _sort_difficulty(r["stratum"])),
            "Difficulty",
        )
        _append_table(
            lines,
            sorted(_rows_of_type(full_rows, "primary_domain"), key=lambda r: r["stratum"]),
            "Primary Domain",
        )
        early = [r for r in _rows_of_type(full_rows, "early_pool") if r["stratum"] != "early_unlabeled"]
        if early:
            _append_table(lines, sorted(early, key=lambda r: r["stratum"]), "Full600 Subset With Early-Pool Labels")

    path.write_text("\n".join(lines) + "\n")


def _snapshot(args: argparse.Namespace) -> dict[str, Any]:
    k_values = args.k_values
    metadata = _load_metadata(args.eval_dataset)
    tables = [_load_eval_table(path) for path in _find_tables(args.run_dir)]
    tables = sorted(tables, key=lambda t: (t["suite"], t["step"], t["path"]))
    early_pools = _baseline_pools(tables, args.easy_threshold, args.hard_threshold)
    rows = _summaries(tables, metadata, early_pools, k_values)
    _attach_deltas(rows, k_values)
    alerts = _alerts(rows)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tables": [
            {k: v for k, v in table.items() if k != "examples"}
            | {"examples": len(table["examples"])}
            for table in tables
        ],
        "early_pool_counts": dict(sorted((pool, list(early_pools.values()).count(pool)) for pool in set(early_pools.values()))),
        "alerts": alerts,
        "rows": rows,
    }


def _write_snapshot(snapshot: dict[str, Any], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n")
    _write_csv(snapshot["rows"], args.out_dir / "by_step_strata.csv", args.k_values)
    _write_markdown(
        snapshot["rows"],
        snapshot["tables"],
        snapshot["early_pool_counts"],
        snapshot["alerts"],
        args.out_dir / "latest.md",
    )
    with (args.out_dir / "snapshots.jsonl").open("a") as f:
        f.write(json.dumps({k: v for k, v in snapshot.items() if k != "rows"}, sort_keys=True, default=str) + "\n")
    latest_baseline = _latest_suite_rows(snapshot["rows"], "omni-math2-baseline100")
    overall = next((row for row in latest_baseline if row["stratum_type"] == "overall"), None)
    with (args.out_dir / "watch.log").open("a") as f:
        f.write(
            "{time} tables={tables} baseline_step={step} p8={p8} alerts={alerts}\n".format(
                time=snapshot["updated_at"],
                tables=len(snapshot["tables"]),
                step=overall["step"] if overall else "",
                p8=_fmt(overall["pass@8"]) if overall else "",
                alerts="; ".join(snapshot["alerts"]) if snapshot["alerts"] else "none",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="training run output directory")
    parser.add_argument("--eval-dataset", type=Path, required=True, help="stratified eval dataset (.jsonl)")
    parser.add_argument("--out-dir", type=Path, required=True, help="snapshot output directory")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8], help="k values for pass@k (default: 1 2 4 8)")
    parser.add_argument("--easy-threshold", type=float, default=0.875, help="reward rate threshold for early_easy pool (default: 0.875)")
    parser.add_argument("--hard-threshold", type=float, default=0.0625, help="reward rate threshold for early_hard pool (default: 0.0625)")
    parser.add_argument("--interval", type=int, default=120, help="poll interval in seconds (default: 120)")
    parser.add_argument("--once", action="store_true", help="snapshot once and exit")
    args = parser.parse_args()
    args.k_values = tuple(args.k_values)

    while True:
        started = time.monotonic()
        try:
            snapshot = _snapshot(args)
            _write_snapshot(snapshot, args)
            latest_baseline = _latest_suite_rows(snapshot["rows"], "omni-math2-baseline100")
            overall = next((row for row in latest_baseline if row["stratum_type"] == "overall"), None)
            print(
                "{time} tables={tables} baseline_step={step} p8={p8} alerts={alerts}".format(
                    time=snapshot["updated_at"],
                    tables=len(snapshot["tables"]),
                    step=overall["step"] if overall else "",
                    p8=_fmt(overall["pass@8"]) if overall else "",
                    alerts=len(snapshot["alerts"]),
                ),
                flush=True,
            )
        except Exception as exc:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            message = f"{datetime.now(timezone.utc).isoformat()} ERROR {type(exc).__name__}: {exc}"
            with (args.out_dir / "watch.log").open("a") as f:
                f.write(message + "\n")
            print(message, flush=True)
        if args.once:
            return
        elapsed = time.monotonic() - started
        time.sleep(max(1.0, args.interval - elapsed))


if __name__ == "__main__":
    main()
