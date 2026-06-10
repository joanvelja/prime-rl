"""Role-conditioned advantage estimation for multi-agent training.

Rank-7 RAE: shrunk leave-one-out + historical prior, with the antithetic
merge for zero-sum member pairs. For a finalized group of G episodes
sharing key ``(env_name, example_id)``, each episode contributing a
zero-sum pair with canonical reward ``r_i``::

    A_i = r_i - [lam * b + (1 - lam) * loo_mean_i]
    lam = n_eff / (n_eff + G - 1)        (lam = 1 when G = 1)

where ``b`` is the persistent per-key baseline and ``loo_mean_i`` is the
mean of the *other* episodes' canonical rewards. The non-canonical member
gets ``-A_i``. The baseline folds once per group close::

    b <- beta * b + (1 - beta) * group_mean

Order-invariant by construction: every episode subtracts the same
pre-group baseline, and the fold happens after all advantages are set.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from verifiers import rollout_to_member_rollouts
from verifiers.types import MemberRollout

from prime_rl.configs.multi_agent import MultiAgentConfig
from prime_rl.orchestrator.member_generation import is_trainable_member as is_bound_trainable_member

MemberTrainability = Callable[[Mapping, str], bool]

RAEKey = tuple[str, int | str]


@dataclass
class GroupFold:
    """Baseline fold record for one ``(env_name, example_id)`` group close."""

    key: RAEKey
    baseline_before: float
    baseline_after: float
    group_size: int
    group_mean: float
    cold: bool


@dataclass
class RAEState:
    """Persistent baselines keyed by (env_name, example_id).

    ``last_folds`` records the folds from the most recent
    ``compute_rae_advantages`` call (observability hook; not checkpointed).
    """

    baselines: dict[RAEKey, float] = field(default_factory=dict)
    beta: float = 0.9
    n_eff: float = 6.0
    last_folds: list[GroupFold] = field(default_factory=list)


def fan_out_for_multi_agent(
    rollouts: list[Mapping],
    *,
    is_trainable_member: MemberTrainability | None = None,
) -> tuple[list[MemberRollout], list[list[int]]]:
    """Project episode rollouts to per-member training units."""
    training_units: list[MemberRollout] = []
    rollout_to_unit_idxs: list[list[int]] = []
    for rollout in rollouts:
        members = rollout_to_member_rollouts(rollout)
        if is_trainable_member is not None:
            members = [member for member in members if is_trainable_member(rollout, member["member_id"])]
        env_name = rollout.get("env_name")
        if env_name is not None:
            if not isinstance(env_name, str):
                raise TypeError(f"rollout env_name must be a string for RAE identity, got {type(env_name).__name__}")
            for member in members:
                member["env_name"] = env_name
        rollout_to_unit_idxs.append(list(range(len(training_units), len(training_units) + len(members))))
        training_units.extend(members)
    return training_units, rollout_to_unit_idxs


def fan_out_trainable_for_multi_agent(
    rollouts: list[Mapping],
    config: MultiAgentConfig,
) -> tuple[list[MemberRollout], list[list[int]]]:
    """Project episode rollouts to the member rows that can enter training."""
    return fan_out_for_multi_agent(
        rollouts,
        is_trainable_member=lambda rollout, member_id: is_bound_trainable_member(config, rollout, member_id),
    )


def _rae_key(member_rollout: Mapping) -> RAEKey:
    env_name = member_rollout.get("env_name")
    if env_name is None:
        env_name = member_rollout["task"]
    if not isinstance(env_name, str):
        raise TypeError(
            "RAE baseline identity requires a string env_name; "
            f"got {type(env_name).__name__} from member rollout identity fields"
        )
    return (env_name, member_rollout["example_id"])


def _canonical_rewards(
    key: RAEKey,
    episode_rows: dict[str, list[int]],
    member_rollouts: list[MemberRollout],
) -> tuple[str, dict[str, float]]:
    """Validate the zero-sum pair structure and project each episode to its
    canonical (lexicographically smallest member id) reward."""
    member_ids = sorted({member_rollouts[i]["member_id"] for rows in episode_rows.values() for i in rows})
    if len(member_ids) > 2:
        raise ValueError(
            f"group {key} has member ids {member_ids}: "
            "rank-7 RAE requires zero-sum member pairs (at most 2 distinct member ids per group)"
        )
    canonical = member_ids[0]

    rewards: dict[str, float] = {}
    for episode_id, rows in episode_rows.items():
        by_member = {member_rollouts[i]["member_id"]: member_rollouts[i]["reward"] for i in rows}
        if len(by_member) != len(rows) or len(rows) > 2:
            raise ValueError(
                f"episode {episode_id} in group {key} has member rows "
                f"{[member_rollouts[i]['member_id'] for i in rows]}: "
                "rank-7 RAE requires zero-sum member pairs (one row per member, at most 2 per episode)"
            )
        if len(by_member) == 2:
            (id_a, r_a), (id_b, r_b) = sorted(by_member.items())
            if r_a != -r_b:
                raise ValueError(
                    f"episode {episode_id} in group {key} has rewards "
                    f"{id_a}={r_a}, {id_b}={r_b}: "
                    "rank-7 RAE requires zero-sum member pairs (r_a == -r_b)"
                )
            rewards[episode_id] = by_member[canonical]
        else:
            # One side quarantined: recover the canonical reward via sign
            ((present_id, present_reward),) = by_member.items()
            rewards[episode_id] = present_reward if present_id == canonical else -present_reward
    return canonical, rewards


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
) -> list[float]:
    """Compute rank-7 RAE advantages and fold each group's baseline once.

    See the module docstring for the estimator. Advantages are returned in
    input order; fold records for this call land on ``state.last_folds``.
    """
    episode_rows_by_key: dict[RAEKey, dict[str, list[int]]] = {}
    for idx, member_rollout in enumerate(member_rollouts):
        key = _rae_key(member_rollout)
        episode_rows_by_key.setdefault(key, {}).setdefault(member_rollout["episode_id"], []).append(idx)

    advantages: list[float] = [0.0] * len(member_rollouts)
    state.last_folds = []
    for key, episode_rows in episode_rows_by_key.items():
        canonical, rewards = _canonical_rewards(key, episode_rows, member_rollouts)
        group_size = len(rewards)
        baseline = state.baselines.get(key, 0.0)
        cold = key not in state.baselines
        lam = 1.0 if group_size == 1 else state.n_eff / (state.n_eff + group_size - 1)
        total = sum(rewards.values())
        for episode_id, rows in episode_rows.items():
            reward = rewards[episode_id]
            loo_mean = (total - reward) / (group_size - 1) if group_size > 1 else 0.0
            advantage = reward - (lam * baseline + (1.0 - lam) * loo_mean)
            for i in rows:
                advantages[i] = advantage if member_rollouts[i]["member_id"] == canonical else -advantage
        group_mean = total / group_size
        folded = state.beta * baseline + (1.0 - state.beta) * group_mean
        state.baselines[key] = folded
        state.last_folds.append(
            GroupFold(
                key=key,
                baseline_before=baseline,
                baseline_after=folded,
                group_size=group_size,
                group_mean=group_mean,
                cold=cold,
            )
        )
    return advantages
