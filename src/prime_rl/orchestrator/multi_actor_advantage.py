"""Role-conditioned Advantage Estimation (RAE / SPIRAL).

Separate from the GRPO advantage path (advantage.py) — different computation
model (per-sample with metadata vs. [P,N] tensor), different state lifecycle.
The orchestrator multi-actor path calls compute_rae_advantages instead of
compute_advantages. Both coexist.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from verifiers.types import MemberRollout

RAEKey = tuple[str, int | str, str]
"""(task, example_id, role_id). ``task`` is the env name (MemberRollout['task']),
which partitions the baseline store across environments — otherwise two envs
with overlapping example_ids would contaminate each other's baselines.
``example_id`` is int|str to match MemberRollout (HuggingFace UID-style ids
like \"mmlu_0001\" propagate verbatim end-to-end)."""


@dataclass
class RAEState:
    """Persistent EMA baselines keyed by (task, example_id, role_id).

    Cold start: missing keys default to 0.0 baseline, so the first
    advantage for a new (task, example, role) tuple equals the raw reward.
    """

    baselines: dict[RAEKey, float] = field(default_factory=dict)
    momentum: float = 0.9

    def update(self, key: RAEKey, reward: float) -> None:
        prev = self.baselines.get(key, 0.0)
        self.baselines[key] = self.momentum * prev + (1 - self.momentum) * reward


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
) -> list[float]:
    """Compute per-member advantages and update EMA baselines.

    A_i = R_i - b[(task_i, example_id_i, role_id_i)]

    Baselines are read BEFORE the batch, then updated AFTER all advantages
    are computed. This prevents within-batch ordering effects.
    """
    advantages: list[float] = []
    updates: list[tuple[RAEKey, float]] = []

    for mr in member_rollouts:
        reward = mr["reward"]
        if reward is None:
            raise ValueError(
                f"MemberRollout has reward=None "
                f"(episode={mr['episode_id']}, member={mr['member_id']})"
            )
        key: RAEKey = (mr["task"], mr["example_id"], mr["role_id"])
        baseline = state.baselines.get(key, 0.0)
        advantages.append(reward - baseline)
        updates.append((key, reward))

    # Aggregate rewards per key so update order doesn't matter when
    # the same (task, example_id, role_id) appears multiple times in a batch.
    key_sums: dict[RAEKey, float] = defaultdict(float)
    key_counts: dict[RAEKey, int] = defaultdict(int)
    for key, reward in updates:
        key_sums[key] += reward
        key_counts[key] += 1

    for key, total in key_sums.items():
        state.update(key, total / key_counts[key])

    return advantages
