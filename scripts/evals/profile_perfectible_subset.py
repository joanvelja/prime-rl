"""Profile a perfectible-subset selection: stats + SVG plots + HF dataset card.

Reads the baseline rollouts + the selected subset + the source dataset, computes
per-problem and per-domain statistics, renders inline SVG charts, and writes a
README.md suitable for direct upload as the HuggingFace dataset card.

Usage:
    uv run python scripts/evals/profile_perfectible_subset.py \\
        --records-jsonl outputs/baselines/.../records.jsonl \\
        --selected-jsonl benchmarks/datasets/.../perfectible.jsonl \\
        --source-jsonl benchmarks/datasets/.../full_train.jsonl \\
        --out-dir benchmarks/datasets/.../perfectible_card \\
        --model-name allenai/Olmo-3-7B-Instruct-DPO \\
        --hf-repo joanvelja/omni-math2-olmo3-perfectible-seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evals.analyze_baseline_matrix import compact_float, esc, scale, svg_doc


def primary_domain(domain_field: Any) -> str:
    if not domain_field:
        return "unknown"
    items = domain_field if isinstance(domain_field, list) else [domain_field]
    if not items:
        return "unknown"
    first = str(items[0])
    parts = [p.strip() for p in first.split("->")]
    if len(parts) >= 2 and parts[0].lower() == "mathematics":
        return parts[1]
    return first.split(" ")[0][:30]


def domain_chain(domain_field: Any, depth: int = 2) -> str:
    if not domain_field:
        return "unknown"
    items = domain_field if isinstance(domain_field, list) else [domain_field]
    first = str(items[0]) if items else "unknown"
    parts = [p.strip() for p in first.split("->")]
    take = parts[: depth + 1] if parts and parts[0].lower() == "mathematics" else parts[:depth]
    return " > ".join(take[1 : depth + 1] if parts and parts[0].lower() == "mathematics" else take)


def difficulty_bucket(d: Any) -> str:
    try:
        v = float(d)
    except (TypeError, ValueError):
        return "unknown"
    if v < 3:
        return "easy (<3)"
    if v < 5:
        return "medium (3-5)"
    if v < 7:
        return "hard (5-7)"
    if v < 9:
        return "very hard (7-9)"
    return "extreme (>=9)"


def bar_chart_vertical(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    width: int = 640,
    height: int = 320,
    color: str = "#1f77b4",
) -> str:
    """Vertical bar chart with categorical x-axis."""
    pad_l, pad_r, pad_t, pad_b = 70, 24, 36, 80
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = max(values) * 1.1 if values else 1.0
    n = len(values)
    bar_w = plot_w / max(n, 1) * 0.78
    gap = plot_w / max(n, 1) * 0.22

    body = [
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{esc(title)}</text>',
    ]
    body.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#333"/>')
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = y_max * tick
        ypx = pad_t + plot_h - (yv / y_max) * plot_h if y_max else pad_t + plot_h
        body.append(
            f'<line x1="{pad_l - 4}" y1="{ypx}" x2="{pad_l}" y2="{ypx}" stroke="#333"/>'
            f'<text x="{pad_l - 8}" y="{ypx + 4}" text-anchor="end" font-size="10">{compact_float(yv)}</text>'
        )
    body.append(
        f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" font-size="11" transform="rotate(-90 14 {pad_t + plot_h / 2})">{esc(y_label)}</text>'
    )

    for i, (lbl, v) in enumerate(zip(labels, values)):
        x = pad_l + i * (bar_w + gap) + gap / 2
        h = (v / y_max) * plot_h if y_max else 0
        y = pad_t + plot_h - h
        body.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}"/>')
        body.append(
            f'<text x="{x + bar_w / 2}" y="{y - 4}" text-anchor="middle" font-size="10">{compact_float(v)}</text>'
        )
        body.append(
            f'<text x="{x + bar_w / 2}" y="{pad_t + plot_h + 14}" text-anchor="middle" font-size="10" transform="rotate(35 {x + bar_w / 2} {pad_t + plot_h + 14})">{esc(lbl)}</text>'
        )
    return svg_doc(width, height, "\n".join(body))


def bar_chart_horizontal(
    *,
    labels: list[str],
    values: list[int | float],
    title: str,
    x_label: str,
    width: int = 720,
    row_h: int = 22,
    color: str = "#1f77b4",
) -> str:
    """Horizontal bar chart sorted descending."""
    pad_l, pad_r, pad_t, pad_b = 220, 24, 36, 32
    plot_w = width - pad_l - pad_r
    plot_h = max(row_h * len(labels), row_h)
    height = pad_t + plot_h + pad_b
    x_max = max(values) * 1.05 if values else 1.0

    body = [
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{esc(title)}</text>',
    ]
    body.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#333"/>')
    for i, (lbl, v) in enumerate(zip(labels, values)):
        y = pad_t + i * row_h + 2
        w = (v / x_max) * plot_w if x_max else 0
        body.append(f'<rect x="{pad_l}" y="{y}" width="{w}" height="{row_h - 6}" fill="{color}"/>')
        body.append(
            f'<text x="{pad_l - 6}" y="{y + (row_h - 4) / 2 + 4}" text-anchor="end" font-size="11">{esc(lbl)}</text>'
        )
        body.append(
            f'<text x="{pad_l + w + 4}" y="{y + (row_h - 4) / 2 + 4}" font-size="10">{compact_float(v)}</text>'
        )
    body.append(
        f'<text x="{pad_l + plot_w / 2}" y="{pad_t + plot_h + 22}" text-anchor="middle" font-size="11">{esc(x_label)}</text>'
    )
    return svg_doc(width, height, "\n".join(body))


def line_chart(
    *,
    xs: list[float],
    ys: list[float],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 640,
    height: int = 320,
    color: str = "#1f77b4",
    annotate: bool = True,
) -> str:
    """Single line plot with markers."""
    pad_l, pad_r, pad_t, pad_b = 64, 24, 36, 56
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = 0.0, max(ys) * 1.08 if ys else 1.0

    body = [
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{esc(title)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#333"/>',
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = y_hi * t
        ypx = pad_t + plot_h - (yv / y_hi) * plot_h if y_hi else pad_t + plot_h
        body.append(
            f'<line x1="{pad_l - 4}" y1="{ypx}" x2="{pad_l}" y2="{ypx}" stroke="#333"/>'
            f'<text x="{pad_l - 8}" y="{ypx + 4}" text-anchor="end" font-size="10">{compact_float(yv)}</text>'
        )
    for xv in xs:
        xpx = scale(xv, (x_lo, x_hi), (pad_l, pad_l + plot_w))
        body.append(
            f'<line x1="{xpx}" y1="{pad_t + plot_h}" x2="{xpx}" y2="{pad_t + plot_h + 4}" stroke="#333"/>'
            f'<text x="{xpx}" y="{pad_t + plot_h + 16}" text-anchor="middle" font-size="10">{compact_float(xv)}</text>'
        )
    pts = " ".join(
        f"{scale(x, (x_lo, x_hi), (pad_l, pad_l + plot_w))},{pad_t + plot_h - (y / y_hi) * plot_h if y_hi else pad_t + plot_h}"
        for x, y in zip(xs, ys)
    )
    body.append(f'<polyline points="{pts}" stroke="{color}" stroke-width="2" fill="none"/>')
    for x, y in zip(xs, ys):
        cx = scale(x, (x_lo, x_hi), (pad_l, pad_l + plot_w))
        cy = pad_t + plot_h - (y / y_hi) * plot_h if y_hi else pad_t + plot_h
        body.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{color}"/>')
        if annotate:
            body.append(f'<text x="{cx}" y="{cy - 8}" text-anchor="middle" font-size="9">{compact_float(y)}</text>')
    body.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-size="11">{esc(x_label)}</text>'
    )
    body.append(
        f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" font-size="11" transform="rotate(-90 14 {pad_t + plot_h / 2})">{esc(y_label)}</text>'
    )
    return svg_doc(width, height, "\n".join(body))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hf-repo", default=None, help="for the readme header (e.g. user/dataset)")
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=0.8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    selected = [json.loads(line) for line in args.selected_jsonl.read_text().splitlines() if line.strip()]
    selected_ids = {str(r["id"]) for r in selected}
    source = [json.loads(line) for line in args.source_jsonl.read_text().splitlines() if line.strip()]

    # per-problem rollout aggregates from records.jsonl
    per_q_reward: dict[str, list[float]] = defaultdict(list)
    per_q_trunc: dict[str, list[bool]] = defaultdict(list)
    per_q_tokens: dict[str, list[float]] = defaultdict(list)
    per_q_judge_called: dict[str, int] = defaultdict(int)
    with args.records_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            qid = str(r.get("example_id"))
            per_q_reward[qid].append(float(r.get("reward") or 0.0))
            per_q_trunc[qid].append(bool(r.get("is_truncated")))
            per_q_tokens[qid].append(float(r.get("output_tokens") or 0.0))
            if r.get("judge_decision") is not None:
                per_q_judge_called[qid] += 1

    sampled_ids = set(per_q_reward.keys())
    sampled_rows = [r for r in source if str(r.get("id")) in sampled_ids]

    sel_rates = [mean(per_q_reward[qid]) for qid in selected_ids if qid in per_q_reward]
    sel_correct_counts = [sum(1 for r in per_q_reward[qid] if r >= 0.5) for qid in selected_ids if qid in per_q_reward]
    sel_tokens_mean = [mean(per_q_tokens[qid]) for qid in selected_ids if qid in per_q_tokens]
    sel_trunc_rate = [sum(per_q_trunc[qid]) / len(per_q_trunc[qid]) for qid in selected_ids if qid in per_q_trunc]
    sel_judge_rate = [per_q_judge_called.get(qid, 0) / max(len(per_q_reward[qid]), 1) for qid in selected_ids]

    sampled_rates = [mean(per_q_reward[qid]) for qid in sampled_ids if qid in per_q_reward]

    # Histogram of solve rates
    def hist(rates: list[float], buckets: list[tuple[float, float]]) -> list[int]:
        out = [0] * len(buckets)
        for r in rates:
            for i, (lo, hi) in enumerate(buckets):
                if lo <= r < hi or (hi == 1.0 and r == 1.0):
                    out[i] += 1
                    break
        return out

    decile_buckets = [(i / 10, (i + 1) / 10) for i in range(10)]
    sel_hist = hist(sel_rates, decile_buckets)
    sampled_hist = hist(sampled_rates, decile_buckets)

    # Domain mix
    sel_domains = Counter(primary_domain(r.get("domain")) for r in selected)
    sampled_domains = Counter(primary_domain(r.get("domain")) for r in sampled_rows)
    source_domains = Counter(primary_domain(r.get("domain")) for r in source)

    # Source mix
    sel_sources = Counter((r.get("source") or "unknown") for r in selected)
    sampled_sources = Counter((r.get("source") or "unknown") for r in sampled_rows)

    # Difficulty mix
    sel_difficulty = Counter(difficulty_bucket(r.get("difficulty")) for r in selected)
    sampled_difficulty = Counter(difficulty_bucket(r.get("difficulty")) for r in sampled_rows)

    # pass@k for selected (unbiased estimator)
    def pass_at_k(n: int, c: int, k: int) -> float:
        if k > n or c <= 0:
            return 0.0 if c <= 0 else 1.0
        if n - c < k:
            return 1.0
        total = 1.0
        for i in range(n - c + 1, n + 1):
            total *= 1.0 - k / i
        return 1.0 - total

    selected_per_problem_correct = [(40, c) for c in sel_correct_counts]
    pass_ks = [1, 2, 4, 8, 16, 32, 40]
    pass_at_k_values = [mean(pass_at_k(n, c, k) for n, c in selected_per_problem_correct) for k in pass_ks]

    # === Plots ===
    plots = args.out_dir / "plots"

    # 1) Solve-rate histogram for selected
    (plots / "solve_rate_selected.svg").write_text(
        bar_chart_vertical(
            labels=[f"[{lo:.1f},{hi:.1f})" for lo, hi in decile_buckets],
            values=[float(c) for c in sel_hist],
            title=f"Solve-rate distribution (selected {len(selected)} problems)",
            y_label="# problems",
            color="#2ca02c",
        )
    )

    # 2) Solve-rate histogram for full sampled-1000 (band shaded by color)
    sampled_colors = ["#d62728" if not (args.low <= b[0] and b[1] <= args.high) else "#2ca02c" for b in decile_buckets]
    (plots / "solve_rate_sampled.svg").write_text(
        bar_chart_vertical(
            labels=[f"[{lo:.1f},{hi:.1f})" for lo, hi in decile_buckets],
            values=[float(c) for c in sampled_hist],
            title=f"Solve-rate over all {len(sampled_rows)} sampled problems (green = perfectible)",
            y_label="# problems",
            color="#888",
        )
    )

    # 3) Domain mix (top 12 by selected count)
    top_doms = [d for d, _ in sel_domains.most_common(12)]
    (plots / "domain_mix.svg").write_text(
        bar_chart_horizontal(
            labels=top_doms,
            values=[sel_domains[d] for d in top_doms],
            title=f"Top math domains in selected ({len(selected)} problems)",
            x_label="# problems",
        )
    )

    # 4) Source mix (top 15)
    top_srcs = [s for s, _ in sel_sources.most_common(15)]
    (plots / "source_mix.svg").write_text(
        bar_chart_horizontal(
            labels=top_srcs,
            values=[sel_sources[s] for s in top_srcs],
            title=f"Top competition sources in selected ({len(selected)} problems)",
            x_label="# problems",
            color="#9467bd",
        )
    )

    # 5) Difficulty mix
    diff_order = ["easy (<3)", "medium (3-5)", "hard (5-7)", "very hard (7-9)", "extreme (>=9)", "unknown"]
    (plots / "difficulty_mix.svg").write_text(
        bar_chart_vertical(
            labels=diff_order,
            values=[float(sel_difficulty.get(d, 0)) for d in diff_order],
            title=f"Difficulty distribution in selected ({len(selected)} problems)",
            y_label="# problems",
            color="#ff7f0e",
        )
    )

    # 6) Output token distribution (binned)
    bins = list(range(0, 16001, 1500))
    token_hist = [0] * (len(bins) - 1)
    for t in sel_tokens_mean:
        for i in range(len(bins) - 1):
            if bins[i] <= t < bins[i + 1]:
                token_hist[i] += 1
                break
    (plots / "output_tokens.svg").write_text(
        bar_chart_vertical(
            labels=[f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)],
            values=[float(c) for c in token_hist],
            title="Mean output tokens per problem (selected)",
            y_label="# problems",
            color="#17becf",
        )
    )

    # 7) Truncation rate distribution
    trunc_bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    trunc_hist = [0] * (len(trunc_bins) - 1)
    for r in sel_trunc_rate:
        for i in range(len(trunc_bins) - 1):
            if trunc_bins[i] <= r < trunc_bins[i + 1]:
                trunc_hist[i] += 1
                break
    (plots / "truncation_rate.svg").write_text(
        bar_chart_vertical(
            labels=[f"[{trunc_bins[i]:.0%},{trunc_bins[i + 1]:.0%})" for i in range(len(trunc_bins) - 1)],
            values=[float(c) for c in trunc_hist],
            title="Per-problem length-truncation rate (selected)",
            y_label="# problems",
            color="#e377c2",
        )
    )

    # 8) Pass@k for selected subset
    (plots / "pass_at_k.svg").write_text(
        line_chart(
            xs=[float(k) for k in pass_ks],
            ys=pass_at_k_values,
            title=f"pass@k for selected {len(selected)}-problem subset",
            x_label="k",
            y_label="pass@k",
        )
    )

    # === README ===
    readme = []
    repo_header = f"# {args.hf_repo}\n\n" if args.hf_repo else "# Perfectible subset\n\n"
    readme.append(repo_header)
    readme.append(
        "Hendrycks-Sanity perfectible-subset of Omni-MATH-2: problems where the base model "
        f"solves between {args.low:.0%} and {args.high:.0%} of attempts. Useful for "
        "recipe-correctness diagnostics on RL algorithms (a working algorithm should push "
        "training accuracy on this subset above 95%).\n\n"
    )
    readme.append("## Provenance\n\n")
    readme.append("| | |\n|---|---|\n")
    readme.append(f"| Base model | `{args.model_name}` |\n")
    readme.append(f"| Source dataset | `omni_math2_train_excluding_baseline600_seed42.jsonl` ({len(source)} problems) |\n")
    readme.append(f"| Sampled this run (seed=42) | **{len(sampled_rows)}** problems × 40 rollouts each |\n")
    readme.append(f"| Selected band | `[{args.low}, {args.high}]` mean reward |\n")
    readme.append(f"| Min rollouts threshold | 8 |\n")
    readme.append(f"| Scoring | math_verify (SymPy) + gpt-5.4-mini judge fallback (`omni_math2_hybrid_math_v1` rubric) |\n")
    readme.append(f"| Sampling | t=1.0, top_p=0.95, max_completion_tokens=15360 |\n")
    readme.append(f"| **Selected** | **{len(selected)}** problems ({len(selected) * 100 / len(sampled_rows):.1f}% of sampled) |\n\n")

    readme.append("## Solve-rate distribution\n\n")
    readme.append(
        f"Per-problem mean reward across 40 rollouts. By construction every selected problem "
        f"falls in [{args.low}, {args.high}].\n\n"
    )
    readme.append(
        f"- range: [{min(sel_rates):.3f}, {max(sel_rates):.3f}]\n"
        f"- mean: {mean(sel_rates):.3f} | median: {median(sel_rates):.3f} | stdev: {stdev(sel_rates):.3f}\n\n"
    )
    readme.append("![solve_rate_selected](plots/solve_rate_selected.svg)\n\n")
    readme.append(
        "For context, the full 1000 sampled problems show the heavy left tail — most problems are unsolvable "
        f"for {args.model_name.split('/')[-1]} (~50% in [0.0, 0.1)). The 'perfectible' band is the central region.\n\n"
    )
    readme.append("![solve_rate_sampled](plots/solve_rate_sampled.svg)\n\n")

    readme.append("## pass@k on selected\n\n")
    readme.append("Unbiased Chen et al. (2021) estimator over 40 rollouts.\n\n")
    readme.append("| k | pass@k |\n|---:|---:|\n")
    for k, v in zip(pass_ks, pass_at_k_values):
        readme.append(f"| {k} | {v:.4f} |\n")
    readme.append("\n![pass_at_k](plots/pass_at_k.svg)\n\n")

    readme.append("## Domain mix (top 12)\n\n")
    readme.append("Primary domain extracted from the 'Mathematics -> X -> ...' chain in the original tags.\n\n")
    readme.append("![domain_mix](plots/domain_mix.svg)\n\n")

    readme.append("## Competition source mix (top 15)\n\n")
    readme.append("![source_mix](plots/source_mix.svg)\n\n")

    readme.append("## Difficulty mix\n\n")
    readme.append("Original omni-math difficulty score (1-10).\n\n")
    readme.append("![difficulty_mix](plots/difficulty_mix.svg)\n\n")

    readme.append("## Output token distribution\n\n")
    readme.append(
        f"Mean output tokens per problem (averaged over 40 rollouts). Note the 15360-token cap — "
        f"{sum(1 for r in sel_trunc_rate if r > 0):d} of {len(sel_trunc_rate)} problems had at least one truncated rollout.\n\n"
    )
    readme.append("![output_tokens](plots/output_tokens.svg)\n\n")

    readme.append("## Truncation rate per problem\n\n")
    readme.append(
        "Fraction of the 40 rollouts that hit `max_completion_tokens=15360` before emitting EOS. "
        "High truncation suggests the problem is at the edge of the model's reasoning budget.\n\n"
    )
    readme.append(
        f"- mean: {mean(sel_trunc_rate):.3f} | median: {median(sel_trunc_rate):.3f} | max: {max(sel_trunc_rate):.3f}\n\n"
    )
    readme.append("![truncation_rate](plots/truncation_rate.svg)\n\n")

    readme.append("## Judge-fallback rate\n\n")
    readme.append(
        "Fraction of rollouts where `math_verify` was ambiguous and the LLM judge (`gpt-5.4-mini`) "
        "had to be invoked. High judge rate → answers in symbolic forms that don't reduce cleanly.\n\n"
    )
    readme.append(
        f"- mean: {mean(sel_judge_rate):.3f} | median: {median(sel_judge_rate):.3f}\n\n"
    )

    readme.append("## Selected problems × source dataset coverage\n\n")
    readme.append("| | Selected | Sampled (1k) | Source (full) |\n|---|---:|---:|---:|\n")
    readme.append(f"| Total | {len(selected)} | {len(sampled_rows)} | {len(source)} |\n")
    for d in sorted(set(sel_domains) | set(sampled_domains) | set(source_domains), key=lambda x: -sel_domains.get(x, 0))[:10]:
        sel_pct = sel_domains.get(d, 0) * 100 / len(selected) if selected else 0
        samp_pct = sampled_domains.get(d, 0) * 100 / len(sampled_rows) if sampled_rows else 0
        src_pct = source_domains.get(d, 0) * 100 / len(source) if source else 0
        readme.append(
            f"| {d} | {sel_domains.get(d, 0)} ({sel_pct:.1f}%) | "
            f"{sampled_domains.get(d, 0)} ({samp_pct:.1f}%) | "
            f"{source_domains.get(d, 0)} ({src_pct:.1f}%) |\n"
        )
    readme.append("\n")

    readme.append("## Usage\n\n")
    readme.append("```python\n")
    readme.append("from datasets import load_dataset\n")
    if args.hf_repo:
        readme.append(f"ds = load_dataset(\"{args.hf_repo}\", split=\"train\")\n")
    readme.append("# Each row: id, problem, answer, solution, domain, source, difficulty, tags\n")
    readme.append("```\n\n")

    readme.append("## Methodology\n\n")
    readme.append(
        "1. Sample 1000 problems from the source dataset deterministically (seed=42).\n"
        f"2. Generate 40 rollouts per problem with `{args.model_name}` at t=1.0, top_p=0.95, "
        "max_completion_tokens=15360 via the prime-rl baselines harness.\n"
        f"3. Score each rollout with `math_verify` (SymPy symbolic comparison) and, when "
        "math_verify is ambiguous, fall back to a `gpt-5.4-mini` judge using the "
        "`omni_math2_hybrid_math_v1` grader rubric (the same rubric the prime-rl RLVR "
        "training loop uses).\n"
        f"4. Keep problems with mean reward in [{args.low}, {args.high}] and ≥8 rollouts.\n\n"
        "This follows the Hendrycks Sanity Check pattern from Yao et al. 2025 "
        "(*Defeating the Training-Inference Mismatch via FP16*, arxiv 2510.26788, §4).\n"
    )

    (args.out_dir / "README.md").write_text("".join(readme))
    print(f"wrote {args.out_dir / 'README.md'}")
    print(f"wrote {len(list(plots.glob('*.svg')))} plots in {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
