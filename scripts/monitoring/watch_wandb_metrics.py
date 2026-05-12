#!/usr/bin/env python3
"""Live W&B watcher for an OmniMath2 RLVR run.

Polls a wandb run on an interval, dumps a snapshot of the latest train/eval
metrics and a markdown summary to OUT_DIR/latest.{json,md}, appends each tick
to OUT_DIR/snapshots.jsonl, and prints a one-line status to stdout.

Usage:
    uv run scripts/monitoring/watch_wandb_metrics.py \\
        --run-path entity/project/run_id \\
        --out-dir outputs/.../wandb_metrics/live \\
        [--interval 120] [--once]

The train/eval key maps assume the omni-math2 reward/eval naming. Swap them
for another workload by editing TRAIN_KEYS / EVAL_PREFIXES.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import wandb

TRAIN_KEYS = {
    "reward": "reward/omni-math2-train/mean",
    "reward_max": "reward/omni-math2-train/max",
    "reward_min": "reward/omni-math2-train/min",
    "math_verify": "metrics/omni-math2-train/math_verify_score",
    "judge": "metrics/omni-math2-train/judge_score",
    "trunc": "is_truncated/omni-math2-train/mean",
    "evicted_easy": "evicted_examples/omni-math2-train/easy",
    "evicted_hard": "evicted_examples/omni-math2-train/hard",
    "filtered_easy": "filtered_rollouts/omni-math2-train/easy",
    "filtered_hard": "filtered_rollouts/omni-math2-train/hard",
    "pool_easy": "pool/omni-math2-train/easy",
    "pool_hard": "pool/omni-math2-train/hard",
    "pool_normal": "pool/omni-math2-train/normal",
    "off_policy_mean": "off_policy_level/omni-math2-train/mean",
    "off_policy_max": "off_policy_level/omni-math2-train/max",
    "importance_mean": "importance_ratio/mean",
    "importance_raw_mean": "importance_ratio_raw/mean",
    "importance_clip_mean": "importance_ratio_clipped/mean",
    "masked_mean": "is_masked/mean",
    "masked_high_mean": "is_masked_high/mean",
    "masked_low_mean": "is_masked_low/mean",
    "mismatch_kl": "mismatch_kl/mean",
    "entropy": "entropy/mean",
    "mfu": "perf/mfu",
    "throughput": "perf/throughput",
    "wait_ckpt": "time/wait_for_ckpt",
    "wait_batch": "time/wait_for_batch",
    "forward_backward": "time/forward_backward",
    "inference_running": "inference/num_requests_running",
    "inference_waiting": "inference/num_requests_waiting",
    "decode_tps": "inference/decode_throughput_tps",
}

ONLINE_EVAL_PREFIX = "eval/omni-math2-baseline100"
FULL_EVAL_PREFIX = "eval/omni-math2-full600-p8"
EVAL_KEYS = ("pass@1", "pass@2", "pass@4", "pass@8", "is_truncated/mean", "completion_len/mean", "time")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _step(row: dict[str, Any]) -> int | None:
    for key in ("progress/ckpt_step", "step", "_step"):
        value = _as_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def _fmt(value: Any, precision: int = 4) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{precision}f}"


def _latest_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in rows:
        value = _as_float(row.get(key))
        if value is None:
            continue
        latest = {
            "key": key,
            "step": _step(row),
            "wandb_step": row.get("_step"),
            "value": value,
        }
    return latest


def _current_ckpt(rows: list[dict[str, Any]]) -> int | None:
    value = _latest_metric(rows, "progress/ckpt_step")
    if value is not None:
        return int(value["value"])
    last_step = None
    for row in rows:
        step = _step(row)
        if step is not None:
            last_step = step
    return last_step


def _bin_means(rows: list[dict[str, Any]], current_step: int | None) -> tuple[str, dict[str, float]]:
    if current_step is None:
        return "unknown", {}
    bin_start = (current_step // 50) * 50
    if current_step >= bin_start and current_step % 50 == 0 and current_step != 0:
        bin_start = current_step - 49
    bin_end = current_step
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        step = _step(row)
        if step is None or step < bin_start or step > bin_end:
            continue
        for name, key in TRAIN_KEYS.items():
            value = _as_float(row.get(key))
            if value is not None:
                values[name].append(value)
    means = {name: mean(vals) for name, vals in values.items() if vals}
    return f"{bin_start}-{bin_end}", means


def _latest_eval(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in rows:
        if f"{prefix}/pass@8" not in row:
            continue
        latest = {"step": _step(row), "wandb_step": row.get("_step")}
        for key in EVAL_KEYS:
            value = _as_float(row.get(f"{prefix}/{key}"))
            if value is not None:
                latest[key] = value
    return latest


def _eval_history(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    events = []
    for row in rows:
        if f"{prefix}/pass@8" not in row:
            continue
        event = {"step": _step(row), "wandb_step": row.get("_step")}
        for key in EVAL_KEYS:
            value = _as_float(row.get(f"{prefix}/{key}"))
            if value is not None:
                event[key] = value
        events.append(event)
    return events


def _alerts(latest: dict[str, dict[str, Any]], bin_means: dict[str, float], online_eval: dict[str, Any] | None) -> list[str]:
    alerts = []
    pool_normal = latest.get("pool_normal", {}).get("value")
    if pool_normal is not None and pool_normal < 0.05:
        alerts.append("normal pool is below 5%; train reward is now heavily selection-conditioned.")

    off_policy_mean = latest.get("off_policy_mean", {}).get("value")
    off_policy_max = latest.get("off_policy_max", {}).get("value")
    if off_policy_mean is not None and off_policy_mean > 0.25:
        alerts.append(f"off-policy mean is elevated at {_fmt(off_policy_mean)}.")
    if off_policy_max is not None and off_policy_max > 4:
        alerts.append(f"off-policy max is high at {_fmt(off_policy_max)}.")

    masked = latest.get("masked_mean", {}).get("value")
    if masked is not None and masked > 0.25:
        alerts.append(f"IS mask rate is high at {_fmt(masked)}.")

    trunc = latest.get("trunc", {}).get("value")
    if trunc is not None and trunc > 0.12:
        alerts.append(f"train truncation is elevated at {_fmt(trunc)}.")

    if online_eval is not None and online_eval.get("pass@8", 1.0) < 0.63:
        alerts.append(f"latest online pass@8 is weak at {_fmt(online_eval.get('pass@8'))}.")

    if bin_means.get("mfu", 100.0) < 25.0 and bin_means.get("wait_ckpt", 0.0) > 120.0:
        alerts.append("MFU is depressed while wait_for_ckpt is high; likely pipeline imbalance or eval/checkpoint stall.")

    return alerts


def _snapshot(run_path: str, api_timeout: int) -> dict[str, Any]:
    api = wandb.Api(timeout=api_timeout)
    run = api.run(run_path)
    rows = list(run.scan_history(page_size=5000))
    current_step = _current_ckpt(rows)
    bin_label, bin_means = _bin_means(rows, current_step)
    latest = {name: metric for name, key in TRAIN_KEYS.items() if (metric := _latest_metric(rows, key)) is not None}
    online_history = _eval_history(rows, ONLINE_EVAL_PREFIX)
    full_history = _eval_history(rows, FULL_EVAL_PREFIX)
    online_eval = online_history[-1] if online_history else None
    full_eval = full_history[-1] if full_history else None
    now = datetime.now(timezone.utc).isoformat()
    return {
        "updated_at": now,
        "run_path": run_path,
        "run_name": run.name,
        "run_state": run.state,
        "run_url": run.url,
        "history_rows": len(rows),
        "current_step": current_step,
        "current_bin": bin_label,
        "latest": latest,
        "bin_means": bin_means,
        "online_eval": online_eval,
        "online_eval_history": online_history,
        "full_eval": full_eval,
        "full_eval_history": full_history,
        "alerts": _alerts(latest, bin_means, online_eval),
    }


def _write_markdown(snapshot: dict[str, Any], path: Path) -> None:
    latest = snapshot["latest"]
    bin_means = snapshot["bin_means"]
    lines = [
        "# Live W&B Watch",
        "",
        f"Updated UTC: `{snapshot['updated_at']}`",
        f"Run: `{snapshot['run_name']}`",
        f"State: `{snapshot['run_state']}`",
        f"URL: {snapshot['run_url']}",
        f"History rows scanned: `{snapshot['history_rows']}`",
        f"Current ckpt step: `{snapshot['current_step']}`",
        f"Current 50-step window: `{snapshot['current_bin']}`",
        "",
        "## Alerts",
        "",
    ]
    if snapshot["alerts"]:
        lines.extend(f"- {alert}" for alert in snapshot["alerts"])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Training Signals",
            "",
            "| metric | latest | latest step | window mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in TRAIN_KEYS:
        point = latest.get(name, {})
        lines.append(
            f"| `{name}` | {_fmt(point.get('value'))} | {point.get('step', '')} | {_fmt(bin_means.get(name))} |"
        )

    lines.extend(
        [
            "",
            "## Eval Signals",
            "",
            "| suite | step | p@1 | p@2 | p@4 | p@8 | trunc | len | time s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for suite, event in (("baseline100", snapshot["online_eval"]), ("full600-p8", snapshot["full_eval"])):
        if event is None:
            lines.append(f"| `{suite}` |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| `{suite}` | {step} | {p1} | {p2} | {p4} | {p8} | {trunc} | {length} | {seconds} |".format(
                suite=suite,
                step=event.get("step", ""),
                p1=_fmt(event.get("pass@1")),
                p2=_fmt(event.get("pass@2")),
                p4=_fmt(event.get("pass@4")),
                p8=_fmt(event.get("pass@8")),
                trunc=_fmt(event.get("is_truncated/mean")),
                length=_fmt(event.get("completion_len/mean"), precision=1),
                seconds=_fmt(event.get("time"), precision=1),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `filtered_rollouts/easy|hard` here means difficulty-filtered, not literal gibberish/repetition filtering.",
            "- Train reward and train verifier scores are conditioned on the normal pool; compare them against held-out eval before making learning claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_snapshot(snapshot: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n")
    _write_markdown(snapshot, out_dir / "latest.md")
    with (out_dir / "snapshots.jsonl").open("a") as f:
        f.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
    with (out_dir / "watch.log").open("a") as f:
        f.write(
            "{time} step={step} state={state} p8={p8} reward={reward} alerts={alerts}\n".format(
                time=snapshot["updated_at"],
                step=snapshot["current_step"],
                state=snapshot["run_state"],
                p8=_fmt((snapshot["online_eval"] or {}).get("pass@8")),
                reward=_fmt(snapshot["latest"].get("reward", {}).get("value")),
                alerts="; ".join(snapshot["alerts"]) if snapshot["alerts"] else "none",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-path", required=True, help="wandb run path: entity/project/run_id")
    parser.add_argument("--out-dir", type=Path, required=True, help="snapshot output directory")
    parser.add_argument("--interval", type=int, default=120, help="poll interval in seconds (default: 120)")
    parser.add_argument("--once", action="store_true", help="snapshot once and exit")
    parser.add_argument("--api-timeout", type=int, default=120, help="wandb api timeout in seconds")
    args = parser.parse_args()

    while True:
        started = time.monotonic()
        try:
            snapshot = _snapshot(run_path=args.run_path, api_timeout=args.api_timeout)
            _write_snapshot(snapshot, args.out_dir)
            print(
                "{time} step={step} state={state} p8={p8} reward={reward} alerts={alerts}".format(
                    time=snapshot["updated_at"],
                    step=snapshot["current_step"],
                    state=snapshot["run_state"],
                    p8=_fmt((snapshot["online_eval"] or {}).get("pass@8")),
                    reward=_fmt(snapshot["latest"].get("reward", {}).get("value")),
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
