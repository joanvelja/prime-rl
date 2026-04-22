"""Tier-2 debate metric aggregator.

Consumes rollout primitives emitted by verifiers' ``DebateRubric``
(``first_answer/{member}``, ``final_answer/{member}``, ``flipped/{member}``,
per-member rewards, MAR episode_categorical winner) and produces
per-training-step scalars for W&B + a sidecar JSONL.

Scope: what survives without judge logprobs. Calibration (Brier,
Reliability, Resolution) is tracked separately and requires a
logprob-capable judge backend (see joanvelja/verifiers#5).

Design: one pure function ``compute_step_metrics`` for aggregation,
plus ``write_step_metrics`` for the fire-and-forget write path. The
orchestrator calls ``write_step_metrics`` once per ``save_rollouts``
site. Debate-specific: skips rollouts where the env isn't a debate env
(no ``winner`` in ``mar_score.episode_categorical``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import verifiers as vf

from prime_rl.orchestrator.vf_utils import get_model_completion_len
from prime_rl.utils.monitor.base import Monitor


DebateRollout = dict[str, Any]  # a RolloutOutput that carries debate fields


def _mar_categorical(rollout: vf.RolloutOutput) -> dict[str, Any]:
    """Return mar_score.episode_categorical or {} if absent."""
    mar = rollout.get("mar_score")
    if mar is None:
        return {}
    if isinstance(mar, dict):
        return mar.get("episode_categorical") or {}
    # pydantic model path (local env.run_rollout; orchestrator serializes to dict)
    cats = getattr(mar, "episode_categorical", None)
    return cats or {}


def _is_debate_rollout(rollout: vf.RolloutOutput) -> bool:
    """A rollout is debate-shaped iff it carries a judge decision.

    We accept 'winner' in episode_categorical as the authoritative marker;
    absence = single-agent env = silently skip.
    """
    return "winner" in _mar_categorical(rollout)


def _truth_member(rollout: vf.RolloutOutput) -> str | None:
    """The member whose committed answer matches ground truth.

    ``info.learner_seat`` alone doesn't identify truth — truth_member is
    assigned at dataset-prep time by gpqa_debate and stored via
    ``truth_member``'s mapping to per-member ``final_correct``. We
    infer: the debater with ``final_correct=1`` is truth-side (when
    exactly one debater is correct — the resolvable case).
    """
    final_correct = {}
    for member in ("debater_a", "debater_b"):
        key = f"final_correct/{member}"
        if key in rollout:
            final_correct[member] = rollout[key]
    if len(final_correct) != 2:
        return None
    a, b = final_correct["debater_a"], final_correct["debater_b"]
    if a == b:
        return None  # both correct or both wrong — un-resolvable
    return "debater_a" if a > b else "debater_b"


def _winner(rollout: vf.RolloutOutput) -> str | None:
    """Judge's decision: 'debater_a' | 'debater_b' | 'tie' | None."""
    return _mar_categorical(rollout).get("winner")


def _seat_of_truth(rollout: vf.RolloutOutput) -> str | None:
    """'a' or 'b' — which seat the truth_member occupies. Same as
    truth_member's suffix; exposed separately for clarity at call sites."""
    tm = _truth_member(rollout)
    if tm is None:
        return None
    return tm.split("_")[1]  # "debater_a" -> "a"


def _completion_tokens_by_member(rollout: vf.RolloutOutput) -> dict[str, int]:
    """Sum of completion token lengths per member across the trajectory.

    Requires ``trajectory`` in the rollout. Returns {} when trajectory
    is stripped (dump_trajectory=False). Callers should guard on empty.
    """
    out: dict[str, int] = {}
    for step in rollout.get("trajectory") or []:
        extras = step.get("extras") or {}
        mid = extras.get("member_id")
        if not mid:
            continue
        tokens = step.get("tokens") or {}
        completion_ids = tokens.get("completion_ids") if tokens else None
        if completion_ids is not None:
            out[mid] = out.get(mid, 0) + len(completion_ids)
    return out


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0 when either list is
    constant-valued (undefined) or empty. Dependency-free rank with
    average-rank tie handling."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0

    def _ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx2 = sum((rx[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ry[i] - my) ** 2 for i in range(n))
    if dx2 == 0 or dy2 == 0:
        return 0.0
    return num / math.sqrt(dx2 * dy2)


def compute_step_metrics(rollouts: Iterable[vf.RolloutOutput]) -> dict[str, float]:
    """Aggregate tier-2 debate metrics over a batch of rollouts.

    Inputs: an iterable of RolloutOutputs. Non-debate rollouts (those
    missing ``mar_score.episode_categorical.winner``) are silently
    skipped — this lets mixed-env runs call us uniformly.

    Outputs: flat ``{metric_name: float}``, safe to pass to monitor.log().

    Metric families:
      - twc_3way, twc_2way_cond: 3-way and conditional-on-non-tie TWC
      - tie_rate: Pr[W=tie | C_a != C_b]
      - position_bias: |TWC(truth=a) - TWC(truth=b)|
      - mind_change_{good,bad}_rate: per-debater wrong→right and right→wrong
      - length_bias_corr: Spearman(length_delta, winner-is-seat-a)
      - parse_fail_rate: fraction of rollouts with errored mar_score
      - truncation_rate: fraction with is_truncated=True
      - avg_turns/{member}: mean turn count
      - completion_tokens_mean/{member}: mean model-generated token length
      - n_rollouts, n_resolvable: sample-size diagnostics
    """
    debate_rollouts = [r for r in rollouts if _is_debate_rollout(r)]
    n = len(debate_rollouts)
    if n == 0:
        return {}

    # Resolvable = exactly one debater correct (C_a != C_b). Ties + both-correct +
    # both-wrong go into separate diagnostic counts but out of TWC.
    resolvable = [r for r in debate_rollouts if _truth_member(r) is not None]
    n_resolvable = len(resolvable)

    metrics: dict[str, float] = {
        "n_rollouts": float(n),
        "n_resolvable": float(n_resolvable),
        "resolvable_rate": n_resolvable / n,
    }

    # ---- TWC family (only meaningful over resolvable) ------------------
    if n_resolvable > 0:
        # 3-way: Pr[W == truth] including ties-as-failure
        correct_any = [1.0 if _winner(r) == _truth_member(r) else 0.0 for r in resolvable]
        metrics["twc_3way"] = _safe_mean(correct_any)
        metrics["twc_3way_null"] = 1.0 / 3.0  # reference line for plots

        # 2-way conditional: drop tie decisions, null = 1/2
        non_tie = [r for r in resolvable if _winner(r) != "tie"]
        if non_tie:
            correct_2 = [1.0 if _winner(r) == _truth_member(r) else 0.0 for r in non_tie]
            metrics["twc_2way_cond"] = _safe_mean(correct_2)
            metrics["twc_2way_cond_null"] = 0.5
            metrics["n_non_tie"] = float(len(non_tie))

        # Tie rate
        tie_cnt = sum(1 for r in resolvable if _winner(r) == "tie")
        metrics["tie_rate"] = tie_cnt / n_resolvable

        # Position bias: difference in TWC by truth seat
        by_seat = {"a": [], "b": []}
        for r in resolvable:
            seat = _seat_of_truth(r)
            if seat in by_seat:
                by_seat[seat].append(1.0 if _winner(r) == _truth_member(r) else 0.0)
        if by_seat["a"] and by_seat["b"]:
            twc_a = _safe_mean(by_seat["a"])
            twc_b = _safe_mean(by_seat["b"])
            metrics["twc_by_seat_a"] = twc_a
            metrics["twc_by_seat_b"] = twc_b
            metrics["position_bias"] = abs(twc_a - twc_b)

    # ---- Mind-change: requires per-rollout first/final_correct --------
    # (Both are 0/1; mind-change-good = wrong→right; bad = right→wrong.)
    for member in ("debater_a", "debater_b"):
        good = 0
        bad = 0
        total = 0
        for r in debate_rollouts:
            ic = r.get(f"initial_correct/{member}")
            fc = r.get(f"final_correct/{member}")
            if ic is None or fc is None:
                continue
            total += 1
            if ic == 0.0 and fc == 1.0:
                good += 1
            elif ic == 1.0 and fc == 0.0:
                bad += 1
        if total:
            metrics[f"mind_change_good_rate/{member}"] = good / total
            metrics[f"mind_change_bad_rate/{member}"] = bad / total

    # ---- Parse-fail + truncation -------------------------------------
    errored = sum(1 for r in debate_rollouts if r.get("error") is not None)
    metrics["error_rate"] = errored / n
    truncated = sum(1 for r in debate_rollouts if r.get("is_truncated"))
    metrics["truncation_rate"] = truncated / n

    # ---- Per-member turns + completion tokens -----------------------
    for member in ("debater_a", "debater_b", "judge"):
        turns = [r[f"turns/{member}"] for r in debate_rollouts if f"turns/{member}" in r]
        if turns:
            metrics[f"avg_turns/{member}"] = _safe_mean(turns)

    # Completion tokens: only when trajectory is dumped. Silently skip if absent.
    token_sums: dict[str, list[int]] = {"debater_a": [], "debater_b": [], "judge": []}
    for r in debate_rollouts:
        per_member = _completion_tokens_by_member(r)
        for member, ids in per_member.items():
            if member in token_sums:
                token_sums[member].append(ids)
    for member, sums in token_sums.items():
        if sums:
            metrics[f"completion_tokens_mean/{member}"] = _safe_mean([float(x) for x in sums])

    # ---- Length bias (Spearman of length_delta vs winner_is_a) -------
    # Only meaningful when trajectory is dumped (completion token counts available)
    # and we have non-tie decisions to regress against.
    paired = []
    for r in debate_rollouts:
        if _winner(r) == "tie" or _winner(r) is None:
            continue
        tokens = _completion_tokens_by_member(r)
        la, lb = tokens.get("debater_a"), tokens.get("debater_b")
        if la is None or lb is None:
            continue
        paired.append((float(la - lb), 1.0 if _winner(r) == "debater_a" else 0.0))
    if len(paired) >= 2:
        metrics["length_bias_corr"] = _spearman(
            [p[0] for p in paired], [p[1] for p in paired]
        )

    # ---- Flip rates per debater (from rubric's flipped metric) -------
    for member in ("debater_a", "debater_b"):
        flips = [r[f"flipped/{member}"] for r in debate_rollouts if f"flipped/{member}" in r]
        if flips:
            metrics[f"flip_rate/{member}"] = _safe_mean(flips)

    return metrics


def write_step_metrics(
    rollouts: list[vf.RolloutOutput],
    path: Path,
    step: int,
    monitor: Monitor,
    prefix: str = "debate",
) -> None:
    """Compute + persist tier-2 step metrics.

    Writes one JSONL line to ``path`` (sidecar to ``*_rollouts.jsonl``)
    and logs the same scalars to the monitor under ``{prefix}/{key}``.
    No-op when no debate-shaped rollouts are present (mixed-env safety).
    """
    scalars = compute_step_metrics(rollouts)
    if not scalars:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"step": step, **scalars}
    with open(path, "w") as f:
        json.dump(row, f)
        f.write("\n")
    # Prefix for W&B namespacing so debate metrics don't collide with
    # per-member MA rubric fanout (which also uses reward/debater_a etc.).
    namespaced = {f"{prefix}/{k}": v for k, v in scalars.items()}
    monitor.log(namespaced, step)
