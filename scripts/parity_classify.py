#!/usr/bin/env python3
"""parity_classify.py — empirical train<->inference parity diagnostic for prime-rl.

prime-rl async RL trains a model whose train-forward logprobs must match the
vLLM inference-forward logprobs of the SAME sampled tokens. The per-token gap is
`mismatch_kl` (k3 KL). This script classifies, from multi-step `token_export`
JSONLs, whether a model is RL-clean or has a parity problem.

Hard-won lesson baked into the logic: judge MULTI-STEP, not step 0. A port can be
clean at step 0 then blow up later via "catastrophe tokens" (trainer assigns
~e^-33 to tokens vLLM sampled at p~=0.98 -> argmax flips). The median can stay
low while the tail explodes. A distinct failure is uniform elevation (median
itself high) which is a measurement / capture ARTIFACT, not a real parity gap.

Verdicts: CLEAN | DISCRETE-BUG-suspect | FORWARD-DIVERGENCE | ARTIFACT.

A FORWARD-DIVERGENCE verdict means the train<->inference forward genuinely diverges
(clean-then-tail-growth-catastrophe). It does NOT mean "irreducible". This is exactly
the signature gemma-4-26B showed, and that turned out to be a fixable discrete vLLM bug
(router root_size clobbered to 1.0 by warm layerwise reload), not a bf16 kernel gap.
Always LOCALIZE before concluding inherent: cold-load-vs-warm-reload A/B, in-memory
restore of the suspect tensor, or per-layer hidden-state parity on one catastrophe seq.

Stdlib only (json, math, statistics, glob, argparse, os, sys). No numpy/pandas.

Usage:
    python parity_classify.py --exports <dir> [--reference <dir>] [--out report.md]

<dir> is either a run root (containing token_exports/) or a token_exports/ dir
itself; both are accepted.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — tunable thresholds. Every verdict cites which of these drove it.
# Calibrated against three labeled prime-rl runs (see module docstring / report).
# ---------------------------------------------------------------------------
CONFIG = {
    # --- step-0 faithfulness gate (is the static port already unfaithful?) ---
    # If step-0 mean mismatch_kl exceeds this, the trainer<->vLLM forward pass
    # disagree before any RL drift -> suspect a discrete implementation bug.
    "step0_mean_unfaithful": 0.1,

    # --- ARTIFACT gate: uniform elevation, not tail-shaped ---
    # If the *median* (p50) mismatch_kl over the run is this high, the bulk of
    # tokens are mismatched, not just a tail. That is a broken capture / replay
    # measurement artifact, not a real parity gap. (replay-on: p50 ~ 12.)
    "artifact_median": 1.0,

    # --- catastrophe-token detection ---
    # A token is a "catastrophe" if its mismatch_kl exceeds this...
    "catastrophe_mismatch_kl": 5.0,
    # ...OR it is an argmax-flip proxy: trainer collapsed the token while
    # inference was near-certain (or vice versa).
    "argmax_flip_trainer_lp": -10.0,   # trainer_logprobs < this  -> trainer-collapsed
    "argmax_flip_inference_lp": -0.5,  # inference_logprobs > this -> inference-confident
    # Per-step catastrophe fraction above this counts the step as "has catastrophe".
    "catastrophe_frac_flag": 1e-4,
    # Aggregate catastrophe fraction (late steps) above this => non-negligible.
    "catastrophe_frac_significant": 1e-3,

    # --- growth gate: late-vs-early mean ratio ---
    # Define early = steps <= growth_early_max, late = steps >= growth_late_min.
    "growth_early_max": 1,
    "growth_late_min": 3,
    # Late/early ratio above this == "grew" (drift, not flat).
    "growth_ratio_flag": 3.0,
    # Absolute late-mean above this == "elevated late" (real magnitude, not noise).
    "late_mean_elevated": 0.05,

    # --- importance_ratio centeredness sanity (mean should be ~1) ---
    "ir_center_tol": 0.25,  # |mean(IR) - 1| > tol -> flag (informational)

    # --- tail concentration ---
    "tail_top_frac": 0.01,  # top 1% of tokens by mismatch_kl
}

NEAR_ZERO = 1e-12


# ---------------------------------------------------------------------------
# small stdlib stats helpers (mirror agg_is_family.py null-handling)
# ---------------------------------------------------------------------------
def _clean(xs):
    return [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]


def mean(xs):
    xs = _clean(xs)
    return sum(xs) / len(xs) if xs else None


def quantile(sorted_xs, q):
    """Linear-interpolated quantile on an already-sorted list. None if empty."""
    n = len(sorted_xs)
    if n == 0:
        return None
    if n == 1:
        return sorted_xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


# ---------------------------------------------------------------------------
# parsing — mirror agg_is_family.py: iterate loss_mask=True tokens per step
# ---------------------------------------------------------------------------
def find_token_exports(d):
    """Accept either a run root or a token_exports dir; return the exports dir."""
    cand = os.path.join(d, "token_exports")
    if os.path.isdir(cand):
        return cand
    if os.path.basename(os.path.normpath(d)) == "token_exports" and os.path.isdir(d):
        return d
    # maybe they pointed straight at a dir that *is* the exports (has step_*)
    if glob.glob(os.path.join(d, "step_*")):
        return d
    raise FileNotFoundError(f"No token_exports/ (or step_*) found under {d!r}")


def parse_run(d):
    """Return per-step stats dict. Each step: list of per-token mismatch_kl and
    catastrophe/argmax counts over loss_mask=True tokens."""
    exports = find_token_exports(d)
    files = sorted(glob.glob(os.path.join(exports, "step_*", "rank_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No step_*/rank_*.jsonl under {exports!r}")

    by_step_files = defaultdict(list)
    for f in files:
        step = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        by_step_files[step].append(f)

    cat_kl = CONFIG["catastrophe_mismatch_kl"]
    flip_tlp = CONFIG["argmax_flip_trainer_lp"]
    flip_ilp = CONFIG["argmax_flip_inference_lp"]

    steps = {}
    for step in sorted(by_step_files):
        mkl = []          # all loss_mask mismatch_kl (finite)
        n_tok = 0         # loss_mask tokens (incl. null mismatch_kl)
        n_null_mkl = 0    # loss_mask tokens with non-finite/None mismatch_kl
        n_seq = 0
        ir = []
        is_masked_true = 0
        n_cat = 0                # any catastrophe (kl OR flip)
        n_flip_trainer = 0       # trainer-collapsed flip
        n_flip_inference = 0     # inference-collapsed flip
        n_cat_klthresh = 0       # catastrophe via raw kl threshold
        envs = defaultdict(int)

        for f in by_step_files[step]:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    n_seq += 1
                    envs[rec.get("env_name")] += 1
                    lm = rec["loss_mask"]
                    mkl_arr = rec["mismatch_kl"]
                    tlp_arr = rec["trainer_logprobs"]
                    ilp_arr = rec["inference_logprobs"]
                    ir_arr = rec["importance_ratio"]
                    ism_arr = rec.get("is_masked")
                    for i, m in enumerate(lm):
                        if not m:
                            continue
                        n_tok += 1
                        v = mkl_arr[i]
                        if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
                            n_null_mkl += 1
                        else:
                            mkl.append(v)
                        vir = ir_arr[i]
                        if vir is not None and isinstance(vir, (int, float)) and math.isfinite(vir):
                            ir.append(vir)
                        if ism_arr is not None and ism_arr[i]:
                            is_masked_true += 1
                        # catastrophe / argmax-flip classification
                        tlp = tlp_arr[i]
                        ilp = ilp_arr[i]
                        is_cat = False
                        if v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v > cat_kl:
                            is_cat = True
                            n_cat_klthresh += 1
                        # trainer-collapsed: trainer tiny, inference confident
                        trainer_collapsed = (
                            tlp is not None and ilp is not None
                            and tlp < flip_tlp and ilp > flip_ilp
                        )
                        # inference-collapsed: inference tiny, trainer confident
                        inference_collapsed = (
                            tlp is not None and ilp is not None
                            and ilp < flip_tlp and tlp > flip_ilp
                        )
                        if trainer_collapsed:
                            n_flip_trainer += 1
                            is_cat = True
                        if inference_collapsed:
                            n_flip_inference += 1
                            is_cat = True
                        if is_cat:
                            n_cat += 1

        mkl_sorted = sorted(mkl)
        n_fin = len(mkl_sorted)
        total_mass = sum(mkl_sorted)
        # tail mass: top tail_top_frac of tokens
        k = max(1, int(round(CONFIG["tail_top_frac"] * n_fin))) if n_fin else 0
        tail_mass = sum(mkl_sorted[-k:]) if k else 0.0
        tail_frac = (tail_mass / total_mass) if total_mass > NEAR_ZERO else None

        steps[step] = {
            "n_seq": n_seq,
            "n_tok": n_tok,
            "n_null_mkl": n_null_mkl,
            "n_finite": n_fin,
            "mean": mean(mkl_sorted),
            "p50": quantile(mkl_sorted, 0.50),
            "p99": quantile(mkl_sorted, 0.99),
            "max": mkl_sorted[-1] if n_fin else None,
            "tail_mass_frac": tail_frac,
            "ir_mean": mean(ir),
            "is_masked_frac": (is_masked_true / n_tok) if n_tok else None,
            "cat_frac": (n_cat / n_tok) if n_tok else None,
            "cat_count": n_cat,
            "cat_kl_count": n_cat_klthresh,
            "flip_trainer": n_flip_trainer,
            "flip_inference": n_flip_inference,
            "envs": dict(envs),
            # keep the sorted finite list for aggregate stats
            "_mkl_sorted": mkl_sorted,
        }
    return steps


# ---------------------------------------------------------------------------
# aggregate features across steps
# ---------------------------------------------------------------------------
def aggregate_features(steps):
    sk = sorted(steps)
    all_mkl = sorted(v for s in sk for v in steps[s]["_mkl_sorted"])
    n = len(all_mkl)
    total_mass = sum(all_mkl)
    k = max(1, int(round(CONFIG["tail_top_frac"] * n))) if n else 0
    tail_frac = (sum(all_mkl[-k:]) / total_mass) if (k and total_mass > NEAR_ZERO) else None

    early = [s for s in sk if s <= CONFIG["growth_early_max"]]
    late = [s for s in sk if s >= CONFIG["growth_late_min"]]

    def step_weighted_mean(step_list):
        vals = [v for s in step_list for v in steps[s]["_mkl_sorted"]]
        return mean(vals)

    early_mean = step_weighted_mean(early) if early else None
    late_mean = step_weighted_mean(late) if late else None
    growth_ratio = None
    if early_mean is not None and late_mean is not None and early_mean > NEAR_ZERO:
        growth_ratio = late_mean / early_mean

    # aggregate catastrophe fraction over late steps (or all if no late)
    cat_steps = late if late else sk
    n_tok_cat = sum(steps[s]["n_tok"] for s in cat_steps)
    n_cat_cat = sum(steps[s]["cat_count"] for s in cat_steps)
    late_cat_frac = (n_cat_cat / n_tok_cat) if n_tok_cat else None

    step0 = steps[sk[0]] if sk else None

    return {
        "steps_sorted": sk,
        "step0_mean": step0["mean"] if step0 else None,
        "overall_mean": mean(all_mkl),
        "overall_p50": quantile(all_mkl, 0.50),
        "overall_p99": quantile(all_mkl, 0.99),
        "overall_max": all_mkl[-1] if n else None,
        "overall_tail_frac": tail_frac,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "growth_ratio": growth_ratio,
        "late_cat_frac": late_cat_frac,
        "total_flip_trainer": sum(steps[s]["flip_trainer"] for s in sk),
        "total_flip_inference": sum(steps[s]["flip_inference"] for s in sk),
        "ir_mean": mean([m for s in sk if (m := steps[s]["ir_mean"]) is not None]),
        "null_mkl_total": sum(steps[s]["n_null_mkl"] for s in sk),
    }


# ---------------------------------------------------------------------------
# verdict logic — ordered; first match wins; records the driving threshold
# ---------------------------------------------------------------------------
def classify(agg):
    c = CONFIG
    reasons = []

    s0 = agg["step0_mean"]
    p50 = agg["overall_p50"]
    growth = agg["growth_ratio"]
    late_mean = agg["late_mean"]
    late_cat = agg["late_cat_frac"]

    # 1) ARTIFACT: median itself high -> uniform elevation, not tail-shaped.
    #    Checked FIRST: a broken capture can also have high step-0 mean, but the
    #    distinguishing signature is the elevated MEDIAN (bulk mismatched).
    if p50 is not None and p50 > c["artifact_median"]:
        reasons.append(
            f"overall p50(median) mismatch_kl={fmt(p50)} > artifact_median={c['artifact_median']} "
            f"-> bulk of tokens mismatched (uniform elevation), not a tail"
        )
        return "ARTIFACT", reasons

    # 2) DISCRETE-BUG-suspect: step-0 already unfaithful (port broken statically).
    if s0 is not None and s0 > c["step0_mean_unfaithful"]:
        reasons.append(
            f"step-0 mean mismatch_kl={fmt(s0)} > step0_mean_unfaithful={c['step0_mean_unfaithful']} "
            f"-> port unfaithful before any RL drift (recommend static audit)"
        )
        return "DISCRETE-BUG-suspect", reasons

    # step-0 is clean from here. Decide CLEAN vs FORWARD-DIVERGENCE via growth +
    # catastrophe tail.
    grew = growth is not None and growth >= c["growth_ratio_flag"]
    elevated_late = late_mean is not None and late_mean > c["late_mean_elevated"]
    has_cat = late_cat is not None and late_cat > c["catastrophe_frac_significant"]

    # 3) FORWARD-DIVERGENCE: clean start, then growth and/or catastrophe tail while the
    #    median stays low (tail-shaped). A REAL train<->inference forward divergence --
    #    LOCALIZE it (cold-load A/B / suspect-tensor restore / per-layer parity). It is
    #    OFTEN a fixable discrete bug (gemma-4 root_size reload clobber had exactly this
    #    signature), NOT necessarily an irreducible bf16 kernel gap.
    if (grew or elevated_late) and (has_cat or elevated_late):
        drivers = []
        if grew:
            drivers.append(f"growth_ratio={fmt(growth,2)} >= growth_ratio_flag={c['growth_ratio_flag']}")
        if elevated_late:
            drivers.append(f"late_mean={fmt(late_mean)} > late_mean_elevated={c['late_mean_elevated']}")
        if has_cat:
            drivers.append(
                f"late catastrophe_frac={fmt(late_cat,6)} > catastrophe_frac_significant="
                f"{c['catastrophe_frac_significant']}"
            )
        reasons.append(
            "step-0 clean (mean=%s <= %s) but tail-shaped drift: %s; median stayed low "
            "(p50=%s) -> catastrophe-TAIL signature; LOCALIZE before concluding inherent "
            "(often a fixable discrete bug, cf. gemma-4 root_size reload clobber)"
            % (fmt(s0), c["step0_mean_unfaithful"], "; ".join(drivers), fmt(p50))
        )
        return "FORWARD-DIVERGENCE", reasons

    # 4) CLEAN: clean start, bounded growth, ~no catastrophe.
    reasons.append(
        "step-0 clean (mean=%s <= %s), bounded growth (ratio=%s < %s), "
        "late_mean=%s <= %s, ~no catastrophe (late_cat_frac=%s <= %s)"
        % (
            fmt(s0), c["step0_mean_unfaithful"],
            fmt(growth, 2) if growth is not None else "n/a", c["growth_ratio_flag"],
            fmt(late_mean), c["late_mean_elevated"],
            fmt(late_cat, 6) if late_cat is not None else "n/a", c["catastrophe_frac_significant"],
        )
    )
    return "CLEAN", reasons


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------
def render_step_table(steps):
    lines = []
    hdr = (
        "| step | seq | tok | null | mean | p50 | p99 | max | tail% | cat_frac | "
        "flipT | flipI | IR_mean | is_masked% |"
    )
    sep = "|" + "---|" * 12
    lines.append(hdr)
    lines.append(sep)
    for s in sorted(steps):
        d = steps[s]
        lines.append(
            "| {st} | {sq} | {tk} | {nl} | {mn} | {p50} | {p99} | {mx} | {tl} | {cf} | "
            "{ft} | {fi} | {ir} | {im} |".format(
                st=s, sq=d["n_seq"], tk=d["n_tok"], nl=d["n_null_mkl"],
                mn=fmt(d["mean"]), p50=fmt(d["p50"]), p99=fmt(d["p99"]), mx=fmt(d["max"]),
                tl=fmt(d["tail_mass_frac"], 3),
                cf=fmt(d["cat_frac"], 6),
                ft=d["flip_trainer"], fi=d["flip_inference"],
                ir=fmt(d["ir_mean"], 3),
                im=fmt(d["is_masked_frac"], 4),
            )
        )
    return "\n".join(lines)


def render_report(name, steps, agg, verdict, reasons, ref=None):
    L = []
    L.append(f"# Parity diagnostic — {name}")
    L.append("")
    L.append(f"**VERDICT: {verdict}**")
    L.append("")
    L.append("Driving threshold(s):")
    for r in reasons:
        L.append(f"- {r}")
    L.append("")
    L.append("## Per-step signature (loss_mask=True tokens)")
    L.append("")
    L.append(render_step_table(steps))
    L.append("")
    L.append("## Aggregate features")
    L.append("")
    L.append(f"- step-0 mean mismatch_kl: {fmt(agg['step0_mean'])}")
    L.append(f"- overall mean / p50 / p99 / max: "
             f"{fmt(agg['overall_mean'])} / {fmt(agg['overall_p50'])} / "
             f"{fmt(agg['overall_p99'])} / {fmt(agg['overall_max'])}")
    L.append(f"- early mean (steps<={CONFIG['growth_early_max']}): {fmt(agg['early_mean'])}")
    L.append(f"- late mean (steps>={CONFIG['growth_late_min']}): {fmt(agg['late_mean'])}")
    L.append(f"- growth ratio (late/early): "
             f"{fmt(agg['growth_ratio'],2) if agg['growth_ratio'] is not None else 'n/a'}")
    L.append(f"- late catastrophe fraction: "
             f"{fmt(agg['late_cat_frac'],6) if agg['late_cat_frac'] is not None else 'n/a'}")
    L.append(f"- catastrophe sign split (total): "
             f"trainer-collapsed={agg['total_flip_trainer']}, "
             f"inference-collapsed={agg['total_flip_inference']}")
    L.append(f"- overall top-1% tail mass fraction: "
             f"{fmt(agg['overall_tail_frac'],3) if agg['overall_tail_frac'] is not None else 'n/a'}")
    L.append(f"- importance_ratio mean (should be ~1): {fmt(agg['ir_mean'],4)}")
    ir = agg["ir_mean"]
    if ir is not None and abs(ir - 1.0) > CONFIG["ir_center_tol"]:
        L.append(f"  - WARNING: |IR_mean - 1| = {fmt(abs(ir-1.0),4)} > "
                 f"ir_center_tol={CONFIG['ir_center_tol']} (IR off-center)")
    L.append(f"- null/non-finite mismatch_kl tokens (divergence signal): {agg['null_mkl_total']}")
    L.append("")

    if ref is not None:
        ragg, rname, rverdict = ref["agg"], ref["name"], ref["verdict"]
        L.append("## Reference contrast")
        L.append("")
        L.append(f"Reference: **{rname}** (verdict: {rverdict})")
        L.append("")
        L.append("| feature | target | reference | delta (tgt-ref) |")
        L.append("|---|---|---|---|")

        def row(label, a, b):
            if a is None or b is None:
                d = "n/a"
            else:
                d = fmt(a - b)
            L.append(f"| {label} | {fmt(a)} | {fmt(b)} | {d} |")
        row("step-0 mean", agg["step0_mean"], ragg["step0_mean"])
        row("overall mean", agg["overall_mean"], ragg["overall_mean"])
        row("overall p50", agg["overall_p50"], ragg["overall_p50"])
        row("overall p99", agg["overall_p99"], ragg["overall_p99"])
        row("overall max", agg["overall_max"], ragg["overall_max"])
        row("late mean", agg["late_mean"], ragg["late_mean"])
        gr_t = agg["growth_ratio"]
        gr_r = ragg["growth_ratio"]
        L.append(f"| growth ratio | {fmt(gr_t,2) if gr_t is not None else 'n/a'} | "
                 f"{fmt(gr_r,2) if gr_r is not None else 'n/a'} | "
                 f"{fmt(gr_t-gr_r,2) if (gr_t is not None and gr_r is not None) else 'n/a'} |")
        lc_t = agg["late_cat_frac"]
        lc_r = ragg["late_cat_frac"]
        L.append(f"| late cat_frac | {fmt(lc_t,6) if lc_t is not None else 'n/a'} | "
                 f"{fmt(lc_r,6) if lc_r is not None else 'n/a'} | "
                 f"{fmt(lc_t-lc_r,6) if (lc_t is not None and lc_r is not None) else 'n/a'} |")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_one(d):
    steps = parse_run(d)
    agg = aggregate_features(steps)
    verdict, reasons = classify(agg)
    return steps, agg, verdict, reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", required=True,
                    help="run root or token_exports/ dir for the target model")
    ap.add_argument("--reference", default=None,
                    help="run root or token_exports/ dir for a reference model (contrast)")
    ap.add_argument("--out", default=None, help="write markdown report to this path")
    # expose every CONFIG knob as a CLI flag
    for key, val in CONFIG.items():
        ap.add_argument(f"--{key}", type=type(val) if not isinstance(val, bool) else float,
                        default=None, help=f"override CONFIG['{key}'] (default {val})")
    args = ap.parse_args(argv)

    # apply CLI overrides into CONFIG
    for key in CONFIG:
        v = getattr(args, key, None)
        if v is not None:
            CONFIG[key] = type(CONFIG[key])(v)

    name = os.path.basename(os.path.normpath(args.exports))
    steps, agg, verdict, reasons = run_one(args.exports)

    ref = None
    if args.reference:
        rname = os.path.basename(os.path.normpath(args.reference))
        rsteps, ragg, rverdict, rreasons = run_one(args.reference)
        ref = {"name": rname, "steps": rsteps, "agg": ragg,
               "verdict": rverdict, "reasons": rreasons}

    report = render_report(name, steps, agg, verdict, reasons, ref=ref)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[written: {args.out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
