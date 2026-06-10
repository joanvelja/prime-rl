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

The baseline frame is derived from the UNFILTERED zero-sum pair
(``extract_episode_pairs_for_multi_agent``, fed by each episode's
``mar_score``), never from the trainability-filtered member rows: under
``train_one`` a whole group can collapse to a single seat, and a
present-set-derived frame would fold the persistent baseline with a
flipped sign. The canonical member id is persisted per key in
``RAEState.canonical_members``; a group whose full pair would change the
frame fails loud.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from verifiers import rollout_to_member_rollouts
from verifiers.types import MARScore, MemberRollout

from prime_rl.configs.multi_agent import MultiAgentConfig
from prime_rl.orchestrator.member_generation import fixed_member_targets
from prime_rl.orchestrator.member_generation import is_trainable_member as is_bound_trainable_member

MemberTrainability = Callable[[Mapping, str], bool]

RAEKey = tuple[str, int | str]

# episode_id -> {member_id: reward} for the full zero-sum pair
EpisodePairs = Mapping[str, Mapping[str, float]]


@dataclass
class GroupFold:
    """Baseline fold record for one ``(env_name, example_id)`` group close."""

    key: RAEKey
    canonical: str
    baseline_before: float
    baseline_after: float
    group_size: int
    group_mean: float
    cold: bool


@dataclass
class RAEState:
    """Persistent baselines keyed by (env_name, example_id).

    ``canonical_members`` pins the baseline's sign frame per key; it is
    checkpointed alongside the baselines. ``last_folds`` records the folds
    from the most recent ``compute_rae_advantages`` call (observability
    hook; not checkpointed).
    """

    baselines: dict[RAEKey, float] = field(default_factory=dict)
    canonical_members: dict[RAEKey, str] = field(default_factory=dict)
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


def extract_episode_pairs_for_multi_agent(
    rollouts: list[Mapping],
    config: MultiAgentConfig,
) -> dict[str, dict[str, float]]:
    """Project each episode's full ``mar_score`` to its zero-sum pair
    rewards, keyed by episode id (= ``trajectory_id``).

    Runs BEFORE trainability filtering: the estimator needs the complete
    pair for frame derivation and antisymmetry validation even when
    ``train_one`` leaves a single seat per episode. Fixed members (e.g.
    the judge) are not part of the pair.
    """
    fixed_members = fixed_member_targets(config)
    pairs: dict[str, dict[str, float]] = {}
    for rollout in rollouts:
        episode_id = rollout["trajectory_id"]
        if episode_id in pairs:
            raise ValueError(f"duplicate trajectory_id {episode_id!r} — episode identity must be unique in a group")
        mar_raw = rollout["mar_score"]
        mar = mar_raw if isinstance(mar_raw, MARScore) else MARScore.model_validate(mar_raw)
        pairs[episode_id] = {m.member_id: m.reward for m in mar.members if m.member_id not in fixed_members}
    return pairs


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


def _validated_pair(key: RAEKey, episode_id: str, episode_pairs: EpisodePairs) -> dict[str, float]:
    if episode_id not in episode_pairs:
        raise ValueError(
            f"episode {episode_id} in group {key} has no entry in episode_pairs — "
            "rank-7 RAE requires the unfiltered zero-sum pair from the episode's mar_score"
        )
    pair = dict(episode_pairs[episode_id])
    if not pair or len(pair) > 2:
        raise ValueError(
            f"episode {episode_id} in group {key} has pair members {sorted(pair)}: "
            "rank-7 RAE requires zero-sum member pairs (2 members, or 1 when one side is quarantined)"
        )
    if len(pair) == 2:
        (id_a, r_a), (id_b, r_b) = sorted(pair.items())
        if r_a != -r_b:
            raise ValueError(
                f"episode {episode_id} in group {key} has rewards {id_a}={r_a}, {id_b}={r_b}: "
                "rank-7 RAE requires zero-sum member pairs (r_a == -r_b)"
            )
    return pair


def _resolve_canonical(key: RAEKey, pairs: dict[str, dict[str, float]], state: RAEState) -> str:
    """Resolve the baseline's sign frame for one group, stable across steps.

    Derived as the lexicographically smallest member id over the group's
    full pairs; validated against (and persisted to) the per-key frame in
    ``state.canonical_members``. A group of quarantined single-member
    episodes cannot establish a frame on its own — it inherits the
    persisted one (sign-derivation handles the non-canonical seat).
    """
    union_ids = sorted({member_id for pair in pairs.values() for member_id in pair})
    if len(union_ids) > 2:
        raise ValueError(
            f"group {key} has pair members {union_ids} across episodes: rank-7 RAE requires one zero-sum pair per key"
        )
    persisted = state.canonical_members.get(key)
    if persisted is None:
        return union_ids[0]
    if len(union_ids) == 2 and union_ids[0] != persisted:
        raise ValueError(
            f"group {key} has full pair {union_ids}, which would change the canonical member "
            f"from {persisted!r} to {union_ids[0]!r}; the persisted baseline frame cannot flip"
        )
    return persisted


def compute_rae_advantages(
    member_rollouts: list[MemberRollout],
    state: RAEState,
    *,
    episode_pairs: EpisodePairs,
) -> list[float]:
    """Compute rank-7 RAE advantages and fold each group's baseline once.

    ``member_rollouts`` are the (possibly trainability-filtered) rows that
    receive advantages; ``episode_pairs`` carries every episode's full
    zero-sum pair (see ``extract_episode_pairs_for_multi_agent``) so frame
    derivation and antisymmetry validation never depend on which seats
    survived filtering. Advantages are returned in input order; fold
    records for this call land on ``state.last_folds``.
    """
    episode_rows_by_key: dict[RAEKey, dict[str, list[int]]] = {}
    for idx, member_rollout in enumerate(member_rollouts):
        key = _rae_key(member_rollout)
        episode_rows_by_key.setdefault(key, {}).setdefault(member_rollout["episode_id"], []).append(idx)

    advantages: list[float] = [0.0] * len(member_rollouts)
    state.last_folds = []
    for key, episode_rows in episode_rows_by_key.items():
        pairs = {episode_id: _validated_pair(key, episode_id, episode_pairs) for episode_id in episode_rows}
        canonical = _resolve_canonical(key, pairs, state)
        rewards: dict[str, float] = {}
        for episode_id, pair in pairs.items():
            if canonical in pair:
                rewards[episode_id] = pair[canonical]
            else:
                # Quarantined single-member episode: recover via sign
                ((_, present_reward),) = pair.items()
                rewards[episode_id] = -present_reward

        group_size = len(rewards)
        baseline = state.baselines.get(key, 0.0)
        cold = key not in state.baselines
        lam = 1.0 if group_size == 1 else state.n_eff / (state.n_eff + group_size - 1)
        total = sum(rewards.values())
        for episode_id, rows in episode_rows.items():
            pair = pairs[episode_id]
            row_ids = [member_rollouts[i]["member_id"] for i in rows]
            if len(set(row_ids)) != len(row_ids):
                raise ValueError(f"episode {episode_id} in group {key} has duplicate member rows {sorted(row_ids)}")
            reward = rewards[episode_id]
            loo_mean = (total - reward) / (group_size - 1) if group_size > 1 else 0.0
            advantage = reward - (lam * baseline + (1.0 - lam) * loo_mean)
            for i in rows:
                member_id = member_rollouts[i]["member_id"]
                if member_id not in pair:
                    raise ValueError(
                        f"episode {episode_id} in group {key}: member row {member_id!r} "
                        f"is not in the episode's zero-sum pair {sorted(pair)}"
                    )
                advantages[i] = advantage if member_id == canonical else -advantage

        group_mean = total / group_size
        folded = state.beta * baseline + (1.0 - state.beta) * group_mean
        state.canonical_members[key] = canonical
        state.baselines[key] = folded
        state.last_folds.append(
            GroupFold(
                key=key,
                canonical=canonical,
                baseline_before=baseline,
                baseline_after=folded,
                group_size=group_size,
                group_mean=group_mean,
                cold=cold,
            )
        )
    return advantages
