from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

TRAINER_STEP_RE = re.compile(
    r"Step (?P<step>\d+) \| Time: (?P<time>[0-9.]+)s .*? \| Throughput: (?P<throughput>[0-9.]+) tokens/s"
)
ORCHESTRATOR_STEP_RE = re.compile(
    r"Step (?P<step>\d+) \| Time: (?P<time>[0-9.]+)s .*? Seq\. Length: (?P<seq_len>[0-9.]+) tokens/sample"
)


@dataclass
class StepMetric:
    step: int
    time_seconds: float
    throughput: float


@dataclass
class RunSummary:
    label: str
    trainer_steps: list[StepMetric]
    inference_steps: list[StepMetric]
    trainer_mean_tokens_per_second: float
    inference_mean_tokens_per_second: float


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_trainer_steps(log_path: Path, skip_steps: int) -> list[StepMetric]:
    metrics: list[StepMetric] = []
    with log_path.open() as f:
        for raw_line in f:
            line = strip_ansi(raw_line)
            match = TRAINER_STEP_RE.search(line)
            if match is None:
                continue
            step = int(match.group("step"))
            if step < skip_steps:
                continue
            metrics.append(
                StepMetric(
                    step=step,
                    time_seconds=float(match.group("time")),
                    throughput=float(match.group("throughput")),
                )
            )
    return metrics


def parse_orchestrator_steps(log_path: Path, batch_size: int, skip_steps: int) -> list[StepMetric]:
    metrics: list[StepMetric] = []
    with log_path.open() as f:
        for raw_line in f:
            line = strip_ansi(raw_line)
            match = ORCHESTRATOR_STEP_RE.search(line)
            if match is None:
                continue
            step = int(match.group("step"))
            if step < skip_steps:
                continue
            time_seconds = float(match.group("time"))
            seq_len = float(match.group("seq_len"))
            throughput = batch_size * seq_len / time_seconds
            metrics.append(
                StepMetric(
                    step=step,
                    time_seconds=time_seconds,
                    throughput=throughput,
                )
            )
    return metrics


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_run(label: str, run_dir: Path, batch_size: int, skip_trainer_steps: int, skip_inference_steps: int) -> RunSummary:
    trainer_log = run_dir / "slurm" / "latest_train_node_rank_0.log"
    orchestrator_log = run_dir / "slurm" / "latest_orchestrator.log"

    trainer_steps = parse_trainer_steps(trainer_log, skip_trainer_steps)
    inference_steps = parse_orchestrator_steps(orchestrator_log, batch_size, skip_inference_steps)

    return RunSummary(
        label=label,
        trainer_steps=trainer_steps,
        inference_steps=inference_steps,
        trainer_mean_tokens_per_second=mean([metric.throughput for metric in trainer_steps]),
        inference_mean_tokens_per_second=mean([metric.throughput for metric in inference_steps]),
    )


def write_csv(summaries: list[RunSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "trainer_mean_tokens_per_second",
                "inference_mean_tokens_per_second",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "label": summary.label,
                    "trainer_mean_tokens_per_second": f"{summary.trainer_mean_tokens_per_second:.2f}",
                    "inference_mean_tokens_per_second": f"{summary.inference_mean_tokens_per_second:.2f}",
                }
            )


def write_json(summaries: list[RunSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for summary in summaries:
        payload.append(
            {
                "label": summary.label,
                "trainer_steps": [asdict(step) for step in summary.trainer_steps],
                "inference_steps": [asdict(step) for step in summary.inference_steps],
                "trainer_mean_tokens_per_second": summary.trainer_mean_tokens_per_second,
                "inference_mean_tokens_per_second": summary.inference_mean_tokens_per_second,
            }
        )
    output_path.write_text(json.dumps(payload, indent=2))


def _bar_svg(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="6" fill="{fill}" />'


def _text_svg(x: float, y: float, text: str, size: int = 14, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#1f2937">{text}</text>'
    )


def write_svg(summaries: list[RunSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 920
    height = 520
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    categories = [
        ("Trainer", "trainer_mean_tokens_per_second"),
        ("Inference", "inference_mean_tokens_per_second"),
    ]
    max_value = max(
        max(getattr(summary, metric_name) for summary in summaries)
        for _, metric_name in categories
    )
    max_value = max(max_value, 1.0)
    y_max = math.ceil(max_value / 1000) * 1000 if max_value > 1000 else math.ceil(max_value / 100) * 100
    y_max = max(y_max, max_value)

    colors = ["#2563eb", "#dc2626", "#059669", "#d97706"]
    group_count = len(categories)
    run_count = len(summaries)
    group_width = plot_width / group_count
    bar_slot = min(60.0, group_width / max(run_count + 1, 2))
    bar_width = bar_slot * 0.8

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        _text_svg(margin_left, 36, "GLM5 Index Cache Throughput", size=24, weight=700),
        _text_svg(
            margin_left,
            58,
            "Steady-state averages from trainer step logs and orchestrator rollout step logs",
            size=13,
            weight=400,
        ),
    ]

    for tick_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tick_value = y_max * tick_ratio
        y = margin_top + plot_height * (1 - tick_ratio)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#cbd5e1" stroke-width="1" />'
        )
        svg_parts.append(_text_svg(margin_left - 12, y + 5, f"{tick_value:.0f}", size=12, anchor="end"))

    for group_idx, (group_label, metric_name) in enumerate(categories):
        group_x = margin_left + group_idx * group_width
        center_x = group_x + group_width / 2
        svg_parts.append(_text_svg(center_x, height - 30, group_label, size=14, anchor="middle", weight=600))

        start_x = center_x - (run_count * bar_slot) / 2
        for run_idx, summary in enumerate(summaries):
            value = getattr(summary, metric_name)
            bar_height = 0 if y_max == 0 else plot_height * (value / y_max)
            x = start_x + run_idx * bar_slot
            y = margin_top + plot_height - bar_height
            color = colors[run_idx % len(colors)]
            svg_parts.append(_bar_svg(x, y, bar_width, bar_height, color))
            svg_parts.append(_text_svg(x + bar_width / 2, y - 8, f"{value:.0f}", size=12, anchor="middle", weight=600))

    legend_x = width - margin_right - 240
    legend_y = 36
    for idx, summary in enumerate(summaries):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 24
        svg_parts.append(_bar_svg(legend_x, y - 12, 16, 16, color))
        svg_parts.append(_text_svg(legend_x + 24, y + 1, summary.label, size=13))

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Prime RL baseline/index-cache logs into a throughput report.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--cached-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="No Index Cache")
    parser.add_argument("--cached-label", default="Index Cache")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-trainer-steps", type=int, default=1)
    parser.add_argument("--skip-inference-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        summarize_run(
            label=args.baseline_label,
            run_dir=args.baseline_dir,
            batch_size=args.batch_size,
            skip_trainer_steps=args.skip_trainer_steps,
            skip_inference_steps=args.skip_inference_steps,
        ),
        summarize_run(
            label=args.cached_label,
            run_dir=args.cached_dir,
            batch_size=args.batch_size,
            skip_trainer_steps=args.skip_trainer_steps,
            skip_inference_steps=args.skip_inference_steps,
        ),
    ]

    write_csv(summaries, args.output_dir / "index_cache_throughput.csv")
    write_json(summaries, args.output_dir / "index_cache_throughput.json")
    write_svg(summaries, args.output_dir / "index_cache_throughput.svg")


if __name__ == "__main__":
    main()
