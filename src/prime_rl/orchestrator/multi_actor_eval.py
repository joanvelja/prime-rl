"""Two-grain eval metrics for multi-actor episodes.

Episode table: keyed by (base_example_id, episode_id)
Member table:  keyed by (base_example_id, episode_id, member_id, role_id)

pass@k is episode-level. In constant-sum self-play, flattening members
gives 50% by construction — meaningless. The correct grain: define a
binary episode success predicate, group episodes by base_example_id,
compute pass@k on that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from verifiers.types import EpisodeResult, MemberResult


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Identical to eval_utils._pass_at_k — duplicated here to avoid pulling in
    the heavy eval_utils import chain (configs, vf_utils, logger, monitor).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _compute_pass_at_k(rewards: list[float]) -> dict[str, float]:
    n = len(rewards)
    c = sum(r == 1.0 for r in rewards)
    ks = [2**i for i in range(n.bit_length())]
    return {f"pass@{k}": _pass_at_k(n, c, k) for k in ks}



def _member_completion_len(member: MemberResult) -> int:
    """Sum of completion token counts across trajectory steps."""
    total = 0
    for step in member.trajectory:
        tokens = step.get("tokens")
        if tokens is not None:
            total += len(tokens["completion_ids"])
        elif step.get("response") is not None:
            resp = step["response"]
            usage = resp.get("usage")
            if usage is not None:
                total += usage.get("completion_tokens", 0)
    return total


def _member_is_truncated(member: MemberResult) -> bool:
    """True if any trajectory step was truncated."""
    return any(step.get("is_truncated", False) for step in member.trajectory)


def evaluate_multi_actor_episodes(
    results: list[EpisodeResult],
    env_name: str,
    rollouts_per_example: int,
    episode_success_fn: Callable[[EpisodeResult], bool],
) -> dict[str, float]:
    """Build two-table eval metrics from multi-actor episode results.

    Returns dict of monitor-ready metrics keyed by logging namespace.
    """
    if not results:
        return {}

    # --- Episode table ---
    episode_rows: list[dict[str, Any]] = []
    for r in results:
        outcome = r.outcome or {}
        # Fallback sums across all members — each member's trajectory is
        # only the steps that member authored. ``members[0]`` alone
        # under-reports by ≈N on an N-actor schedule.
        total_turns = outcome.get(
            "total_turns",
            sum(len(m.trajectory) for m in r.members),
        )
        episode_rows.append({
            "base_example_id": r.base_example_id,
            "episode_id": r.episode_id,
            "success": float(episode_success_fn(r)),
            "total_turns": total_turns,
        })
    episode_df = pd.DataFrame(episode_rows)

    # --- Member table ---
    member_rows: list[dict[str, Any]] = []
    for r in results:
        for m in r.members:
            member_rows.append({
                "base_example_id": r.base_example_id,
                "episode_id": r.episode_id,
                "member_id": m.member_id,
                "role_id": m.role_id,
                "reward": m.reward,
                "completion_len": _member_completion_len(m),
                "is_truncated": _member_is_truncated(m),
            })
    member_df = pd.DataFrame(member_rows)

    # --- Metrics dict ---
    metrics: dict[str, float] = {}
    prefix = f"eval/{env_name}"

    # Episode-level aggregate metrics
    metrics[f"{prefix}/episode/avg@{rollouts_per_example}"] = float(episode_df["success"].mean())
    metrics[f"{prefix}/episode/total_turns/mean"] = float(episode_df["total_turns"].mean())

    # Episode-level pass@k: group by base_example_id, treat success as binary reward
    pass_at_k_by_example = (
        episode_df.groupby("base_example_id")
        .apply(lambda g: _compute_pass_at_k(g["success"].tolist()), include_groups=False)
        .apply(pd.Series)
    )
    for col in pass_at_k_by_example.columns:
        metrics[f"{prefix}/episode/{col}"] = float(pass_at_k_by_example[col].mean())

    # Per-role metrics
    for role_id, role_group in member_df.groupby("role_id"):
        role_prefix = f"{prefix}/role/{role_id}"
        reward_series = role_group["reward"].dropna()
        if not reward_series.empty:
            metrics[f"{role_prefix}/reward/mean"] = float(reward_series.mean())
        metrics[f"{role_prefix}/completion_len/mean"] = float(role_group["completion_len"].mean())
        metrics[f"{role_prefix}/is_truncated/mean"] = float(role_group["is_truncated"].mean())

    return metrics
