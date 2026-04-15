"""Role-conditioned Advantage Estimation (RAE / SPIRAL).

Separate from the GRPO advantage path (advantage.py) — different computation
model (per-sample with metadata vs. [P,N] tensor), different state lifecycle.
The orchestrator multi-actor path calls compute_rae_advantages instead of
compute_advantages. Both coexist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prime_rl.orchestrator.multi_actor_bridge import MemberRollout


@dataclass
class RAEState:
    """Persistent EMA baselines keyed by (example_id, role_id).

    Cold start: missing keys default to 0.0 baseline, so the first
    advantage for a new (example, role) pair equals the raw reward.
    """

    baselines: dict[tuple[int, str], float] = field(default_factory=dict)
    momentum: float = 0.9

    def update(self, key: tuple[int, str], reward: float) -> None:
        prev = self.baselines.get(key, 0.0)
        self.baselines[key] = self.momentum * prev + (1 - self.momentum) * reward


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
) -> list[float]:
    """Compute per-member advantages and update EMA baselines.

    A_i = R_i - b[(example_id_i, role_id_i)]

    Baselines are read BEFORE the batch, then updated AFTER all advantages
    are computed. This prevents within-batch ordering effects.
    """
    advantages: list[float] = []
    updates: list[tuple[tuple[int, str], float]] = []

    for mr in member_rollouts:
        reward = mr["reward"]
        if reward is None:
            raise ValueError(
                f"MemberRollout has reward=None "
                f"(episode={mr['episode_id']}, member={mr['member_id']})"
            )
        key = (mr["example_id"], mr["role_id"])
        baseline = state.baselines.get(key, 0.0)
        advantages.append(reward - baseline)
        updates.append((key, reward))

    # Aggregate rewards per key so update order doesn't matter when
    # the same (example_id, role_id) appears multiple times in a batch.
    key_sums: dict[tuple[int, str], float] = {}
    key_counts: dict[tuple[int, str], int] = {}
    for key, reward in updates:
        key_sums[key] = key_sums.get(key, 0.0) + reward
        key_counts[key] = key_counts.get(key, 0) + 1

    for key in key_sums:
        state.update(key, key_sums[key] / key_counts[key])

    return advantages
