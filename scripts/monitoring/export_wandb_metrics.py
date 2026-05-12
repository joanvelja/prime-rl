#!/usr/bin/env python3
"""Export and summarize W&B history for an RLVR run.

Pulls run metadata, the summary dict, and the full history via scan_history,
then writes:

- `history.jsonl`           - raw rows from scan_history
- `summary.json`            - run.summary dict
- `run_metadata.json`       - entity/project/id/name/state/url
- `metric_inventory.csv`    - per-key counts, first/last/min/max/mean
- `metric_inventory.json`   - same inventory as JSON
- `METRICS_INVENTORY.md`    - markdown family overview + interesting-keys table

Usage:
    uv run scripts/monitoring/export_wandb_metrics.py \\
        --run-path entity/project/run_id \\
        --out-dir outputs/.../wandb_metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb


class Stat:
    def __init__(self) -> None:
        self.count = 0
        self.numeric_count = 0
        self.non_numeric_count = 0
        self.first_step: int | None = None
        self.last_step: int | None = None
        self.first_value: Any = None
        self.last_value: Any = None
        self.min_value = math.inf
        self.max_value = -math.inf
        self.sum_value = 0.0
        self.samples: list[str] = []

    def add(self, step: int | None, value: Any) -> None:
        if value is None:
            return
        self.count += 1
        if self.first_step is None:
            self.first_step = step
            self.first_value = value
        self.last_step = step
        self.last_value = value
        if isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, int | float) and math.isfinite(float(value)):
            numeric = float(value)
        else:
            numeric = None
        if numeric is None:
            self.non_numeric_count += 1
            if len(self.samples) < 3:
                self.samples.append(_short(value))
            return
        self.numeric_count += 1
        self.min_value = min(self.min_value, numeric)
        self.max_value = max(self.max_value, numeric)
        self.sum_value += numeric

    def row(self, key: str) -> dict[str, Any]:
        mean_value = self.sum_value / self.numeric_count if self.numeric_count else ""
        return {
            "key": key,
            "count": self.count,
            "numeric_count": self.numeric_count,
            "non_numeric_count": self.non_numeric_count,
            "first_step": "" if self.first_step is None else self.first_step,
            "last_step": "" if self.last_step is None else self.last_step,
            "first_value": _short(self.first_value),
            "last_value": _short(self.last_value),
            "min": "" if not self.numeric_count else self.min_value,
            "max": "" if not self.numeric_count else self.max_value,
            "mean": mean_value,
            "samples": " | ".join(self.samples),
            "family": key.split("/", 1)[0] if "/" in key else "",
        }


def _short(value: Any, limit: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, dict | list) else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _interesting(key: str) -> bool:
    needles = [
        "filter",
        "filtered",
        "reward",
        "advantage",
        "eval",
        "pass",
        "correct",
        "score",
        "tokens",
        "token",
        "length",
        "trunc",
        "inference",
        "rollout",
        "buffer",
        "off_policy",
        "trainer",
        "loss",
        "grad",
        "mfu",
        "throughput",
        "time",
        "duration",
        "step",
        "kl",
        "entropy",
        "lr",
    ]
    lower = key.lower()
    return any(n in lower for n in needles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-path", required=True, help="wandb run path: entity/project/run_id")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--api-timeout", type=int, default=90, help="wandb api timeout in seconds")
    parser.add_argument("--page-size", type=int, default=5000, help="scan_history page size")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=args.api_timeout)
    run = api.run(args.run_path)

    metadata = {
        "path": args.run_path,
        "entity": run.entity,
        "project": run.project,
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(getattr(run, "created_at", "")),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(dict(run.summary), indent=2, sort_keys=True, default=str) + "\n")

    stats: dict[str, Stat] = defaultdict(Stat)
    rows = 0
    history_path = args.out_dir / "history.jsonl"
    with history_path.open("w") as f:
        for row in run.scan_history(page_size=args.page_size):
            rows += 1
            step_value = row.get("_step", row.get("step"))
            try:
                step = int(step_value) if step_value is not None else None
            except (TypeError, ValueError):
                step = None
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            for key, value in row.items():
                stats[key].add(step, value)

    inventory_rows = [stat.row(key) for key, stat in sorted(stats.items())]
    fields = [
        "key",
        "family",
        "count",
        "numeric_count",
        "non_numeric_count",
        "first_step",
        "last_step",
        "first_value",
        "last_value",
        "min",
        "max",
        "mean",
        "samples",
    ]
    with (args.out_dir / "metric_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory_rows)
    (args.out_dir / "metric_inventory.json").write_text(json.dumps(inventory_rows, indent=2, sort_keys=True) + "\n")

    family_counts = defaultdict(lambda: {"keys": 0, "points": 0})
    for row in inventory_rows:
        family = row["family"] or "(root)"
        family_counts[family]["keys"] += 1
        family_counts[family]["points"] += int(row["count"])

    lines = [
        "# W&B Metrics Inventory",
        "",
        f"Run: `{args.run_path}`",
        f"Name: `{run.name}`",
        f"State: `{run.state}`",
        f"History rows exported: `{rows}`",
        f"Metric keys observed: `{len(inventory_rows)}`",
        "",
        "Files:",
        "",
        "- `history.jsonl`: raw W&B history rows from `scan_history`.",
        "- `metric_inventory.csv`: per-key counts, first/last/min/max/mean.",
        "- `metric_inventory.json`: same inventory as JSON.",
        "- `summary.json`: W&B summary dict.",
        "",
        "## Families",
        "",
        "| family | keys | points |",
        "|---|---:|---:|",
    ]
    for family, vals in sorted(family_counts.items(), key=lambda kv: (-kv[1]["points"], kv[0])):
        lines.append(f"| `{family}` | {vals['keys']} | {vals['points']} |")

    lines.extend(
        [
            "",
            "## Potentially Useful Keys",
            "",
            "| key | count | first_step | last_step | first | last | min | max | mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inventory_rows:
        if not _interesting(row["key"]):
            continue
        lines.append(
            "| `{key}` | {count} | {first_step} | {last_step} | {first_value} | {last_value} | {min} | {max} | {mean} |".format(
                **row
            )
        )
    (args.out_dir / "METRICS_INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(args.out_dir)
    print(f"rows={rows} keys={len(inventory_rows)}")


if __name__ == "__main__":
    main()
