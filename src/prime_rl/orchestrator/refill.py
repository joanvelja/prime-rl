from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isclose
from typing import Any

import verifiers as vf


@dataclass
class TrainBatchRefillSelection:
    rollouts: list[vf.RolloutOutput]
    training_units: list[vf.RolloutOutput]
    rollout_to_unit_idxs: list[list[int]]
    dropped_groups: list[list[vf.RolloutOutput]]
    metrics: dict[str, float]


def _group_key(rollout: vf.RolloutOutput, rollout_idx: int, rollouts_per_example: int) -> Any:
    group_id = rollout.get("group_id")
    if group_id is not None:
        return ("group_id", group_id)
    return ("chunk", rollout_idx // rollouts_per_example)


def _all_close(values: list[float], target: float) -> bool:
    return bool(values) and all(isclose(value, target, abs_tol=1e-8) for value in values)


def _reward_stats(rollouts: list[vf.RolloutOutput]) -> tuple[float, float]:
    values = [float(rollout["reward"]) for rollout in rollouts if rollout.get("reward") is not None]
    return float(sum(values)), float(len(values))


def select_trainable_rollout_groups(
    *,
    train_rollouts: list[vf.RolloutOutput],
    training_units: list[vf.RolloutOutput],
    rollout_to_unit_idxs: list[list[int]],
    rollouts_per_example: int,
    target_rollouts: int,
) -> TrainBatchRefillSelection:
    """Select whole rollout groups that contain at least one trainable unit.

    This is the post-advantage/post-filter part of DAPO-style refill: groups
    whose rollouts all became filtered are dropped from the current batch, but
    nothing here mutates prompt pools or evicts examples from future sampling.
    """
    groups: OrderedDict[Any, list[int]] = OrderedDict()
    for rollout_idx, rollout in enumerate(train_rollouts):
        groups.setdefault(_group_key(rollout, rollout_idx, rollouts_per_example), []).append(rollout_idx)

    selected_rollout_idxs: list[int] = []
    selected_unit_idxs: list[int] = []
    dropped_groups: list[list[vf.RolloutOutput]] = []
    candidate_groups = len(groups)
    filtered_groups = 0
    filtered_easy_groups = 0
    filtered_hard_groups = 0
    filtered_zero_advantage_groups = 0
    overflow_groups = 0
    overflow_rollouts: list[vf.RolloutOutput] = []
    accepted_groups = 0
    filtered_rollouts: list[vf.RolloutOutput] = []
    selected_unit_idx_set: set[int] = set()

    for rollout_idxs in groups.values():
        unit_idxs: list[int] = []
        for rollout_idx in rollout_idxs:
            unit_idxs.extend(rollout_to_unit_idxs[rollout_idx])

        has_trainable_unit = any(not training_units[unit_idx].get("is_filtered", False) for unit_idx in unit_idxs)
        if not has_trainable_unit:
            filtered_groups += 1
            dropped_group = [train_rollouts[rollout_idx] for rollout_idx in rollout_idxs]
            dropped_groups.append(dropped_group)
            filtered_rollouts.extend(dropped_group)
            rewards = [float(train_rollouts[rollout_idx]["reward"]) for rollout_idx in rollout_idxs]
            advantages = [float(training_units[unit_idx].get("advantage", 0.0)) for unit_idx in unit_idxs]
            if _all_close(rewards, 1.0):
                filtered_easy_groups += 1
            elif _all_close(rewards, 0.0):
                filtered_hard_groups += 1
            if _all_close(advantages, 0.0):
                filtered_zero_advantage_groups += 1
            continue

        if len(selected_rollout_idxs) + len(rollout_idxs) > target_rollouts:
            overflow_groups += 1
            overflow_rollouts.extend(train_rollouts[rollout_idx] for rollout_idx in rollout_idxs)
            continue

        selected_rollout_idxs.extend(rollout_idxs)
        accepted_groups += 1
        for unit_idx in unit_idxs:
            if unit_idx not in selected_unit_idx_set:
                selected_unit_idx_set.add(unit_idx)
                selected_unit_idxs.append(unit_idx)

    unit_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_unit_idxs)}
    selected_rollout_idx_set = set(selected_rollout_idxs)
    selected_rollouts = [train_rollouts[idx] for idx in selected_rollout_idxs]
    selected_training_units = [training_units[idx] for idx in selected_unit_idxs]
    selected_rollout_to_unit_idxs = [
        [unit_idx_map[unit_idx] for unit_idx in rollout_to_unit_idxs[rollout_idx] if unit_idx in unit_idx_map]
        for rollout_idx in selected_rollout_idxs
        if rollout_idx in selected_rollout_idx_set
    ]

    prompts_consumed_per_accepted_group = (
        candidate_groups / accepted_groups if accepted_groups > 0 else float(candidate_groups)
    )
    candidate_reward_sum, candidate_reward_count = _reward_stats(train_rollouts)
    accepted_reward_sum, accepted_reward_count = _reward_stats(selected_rollouts)
    filtered_reward_sum, filtered_reward_count = _reward_stats(filtered_rollouts)
    overflow_reward_sum, overflow_reward_count = _reward_stats(overflow_rollouts)

    metrics = {
        "train_batch_refill/candidate_groups": float(candidate_groups),
        "train_batch_refill/accepted_groups": float(accepted_groups),
        "train_batch_refill/filtered_groups": float(filtered_groups),
        "train_batch_refill/filtered_easy_groups": float(filtered_easy_groups),
        "train_batch_refill/filtered_hard_groups": float(filtered_hard_groups),
        "train_batch_refill/filtered_zero_advantage_groups": float(filtered_zero_advantage_groups),
        "train_batch_refill/overflow_groups": float(overflow_groups),
        "train_batch_refill/candidate_rollouts": float(len(train_rollouts)),
        "train_batch_refill/accepted_rollouts": float(len(selected_rollouts)),
        "train_batch_refill/overflow_rollouts": float(len(overflow_rollouts)),
        "train_batch_refill/prompts_consumed_per_accepted_group": float(prompts_consumed_per_accepted_group),
        "train_batch_refill/reward_unconditioned_on_filtering/sum": candidate_reward_sum,
        "train_batch_refill/reward_unconditioned_on_filtering/count": candidate_reward_count,
        "train_batch_refill/reward_conditioned_on_filtering/sum": accepted_reward_sum,
        "train_batch_refill/reward_conditioned_on_filtering/count": accepted_reward_count,
        "train_batch_refill/reward_filtered_out/sum": filtered_reward_sum,
        "train_batch_refill/reward_filtered_out/count": filtered_reward_count,
        "train_batch_refill/reward_overflow/sum": overflow_reward_sum,
        "train_batch_refill/reward_overflow/count": overflow_reward_count,
    }
    return TrainBatchRefillSelection(
        rollouts=selected_rollouts,
        training_units=selected_training_units,
        rollout_to_unit_idxs=selected_rollout_to_unit_idxs,
        dropped_groups=dropped_groups,
        metrics=metrics,
    )
