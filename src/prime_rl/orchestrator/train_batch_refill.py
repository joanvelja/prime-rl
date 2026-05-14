from __future__ import annotations

from dataclasses import dataclass

import verifiers as vf

from prime_rl.configs.orchestrator import AdvantageConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.refill import select_trainable_rollout_groups


@dataclass
class RefilledTrainBatch:
    train_rollouts: list[vf.RolloutOutput]
    dropped_groups: list[list[vf.RolloutOutput]]
    metrics: dict[str, float]


def refill_train_batch_from_candidates(
    *,
    candidates: list[vf.RolloutOutput],
    target_num_rollouts: int,
    rollouts_per_example: int,
    advantage_config: AdvantageConfig | None,
    rollout_filters: list[RolloutFilter],
) -> RefilledTrainBatch:
    """Compute advantages/filters and keep only trainable candidate groups."""
    compute_advantages(candidates, rollouts_per_example, advantage_config)
    apply_filters(rollout_filters, candidates)
    selection = select_trainable_rollout_groups(
        train_rollouts=candidates,
        training_units=list(candidates),
        rollout_to_unit_idxs=[[i] for i in range(len(candidates))],
        rollouts_per_example=rollouts_per_example,
        target_rollouts=target_num_rollouts,
    )
    return RefilledTrainBatch(
        train_rollouts=selection.rollouts,
        dropped_groups=selection.dropped_groups,
        metrics=selection.metrics,
    )
