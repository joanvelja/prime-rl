"""Rank-7 RAE acceptance tests: shrunk leave-one-out + historical prior,
order-invariant, antithetic merge for zero-sum member pairs."""

import random

import pytest
from verifiers.types import MemberRollout

from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    fan_out_for_multi_agent,
)

ENV_NAME = "debate_v1"
KEY = (ENV_NAME, 1)


def _member(
    *,
    example_id: int | str = 1,
    member_id: str = "debater_a",
    reward: float = 1.0,
    episode_id: str = "ep-0",
    task: object = ENV_NAME,
    env_name: str | None = None,
) -> MemberRollout:
    rollout = MemberRollout(
        example_id=example_id,
        task=task,
        trajectory=[],
        sampling_args={"temperature": 0.7},
        error=None,
        reward=reward,
        episode_id=episode_id,
        member_id=member_id,
    )
    if env_name is not None:
        rollout["env_name"] = env_name
    return rollout


def _pair(
    episode_id: str, reward: float, *, example_id: int | str = 1, env_name: str | None = None
) -> list[MemberRollout]:
    """One zero-sum episode: canonical member ``debater_a`` gets ``reward``."""
    return [
        _member(member_id="debater_a", reward=reward, episode_id=episode_id, example_id=example_id, env_name=env_name),
        _member(member_id="debater_b", reward=-reward, episode_id=episode_id, example_id=example_id, env_name=env_name),
    ]


def _group(canonical_rewards: list[float]) -> list[MemberRollout]:
    rows: list[MemberRollout] = []
    for i, reward in enumerate(canonical_rewards):
        rows.extend(_pair(f"ep-{i}", reward))
    return rows


def test_permutation_equivariance():
    """Advantages and post-call baselines are invariant under any reordering
    of the member rows — exact comparison, no fp tolerance."""
    rows = _group([1.0, -1.0, 0.0, 1.0])

    def run(ordered_rows: list[MemberRollout]) -> tuple[dict[tuple[str, str], float], dict, list]:
        state = RAEState(baselines={KEY: 0.25})
        advantages = compute_rae_advantages(ordered_rows, state)
        by_identity = {(r["episode_id"], r["member_id"]): a for r, a in zip(ordered_rows, advantages)}
        return by_identity, dict(state.baselines), state.last_folds

    reference, reference_baselines, _ = run(rows)
    for seed in range(5):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        result, baselines, folds = run(shuffled)
        assert result == reference
        assert baselines == reference_baselines
        assert [(f.baseline_before, f.baseline_after, f.group_size) for f in folds] == [
            (0.25, reference_baselines[KEY], 4)
        ]


def test_antithetic_invariant():
    rows = _group([1.0, -1.0, 0.0, 1.0])
    state = RAEState(baselines={KEY: 0.25})

    advantages = compute_rae_advantages(rows, state)

    # Per pair: A_b is the exact negation of A_a
    for i in range(0, len(rows), 2):
        assert advantages[i] + advantages[i + 1] == 0.0
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)


def test_g1_reduces_to_pure_prior():
    state = RAEState(baselines={KEY: 0.4})

    advantages = compute_rae_advantages(_pair("ep-0", 1.0), state)

    # G=1: lam pinned to 1, LOO term has zero weight. A = r - b = 1.0 - 0.4 = 0.6
    assert advantages == [pytest.approx(0.6), pytest.approx(-0.6)]


def test_g1_with_neff_zero_still_uses_prior():
    state = RAEState(baselines={KEY: 0.4}, n_eff=0.0)

    advantages = compute_rae_advantages(_pair("ep-0", 1.0), state)

    # lam = 1 at G=1 regardless of n_eff (0/0 would otherwise be NaN)
    assert advantages == [pytest.approx(0.6), pytest.approx(-0.6)]


def test_neff_zero_reduces_to_rloo():
    state = RAEState(baselines={KEY: 0.7}, n_eff=0.0)

    advantages = compute_rae_advantages(_group([1.0, -1.0, 1.0]), state)

    # lam = 0: A_i = r_i - mean(r_{-i}); the 0.7 prior carries zero weight.
    #   A_0 = 1 - (-1+1)/2 = 1.0;  A_1 = -1 - (1+1)/2 = -2.0;  A_2 = 1 - (1-1)/2 = 1.0
    assert advantages == [1.0, -1.0, -2.0, 2.0, 1.0, -1.0]


def test_fold_semantics_and_mid_group_isolation():
    state = RAEState(baselines={KEY: 0.5})

    advantages = compute_rae_advantages(_group([1.0, 1.0]), state)

    # Both episodes subtract the same pre-group b0 = 0.5 — identical advantages
    # prove the fold is not consulted mid-group. lam = 6/(6+1) = 6/7:
    #   A = 1 - (6/7 * 0.5 + 1/7 * 1.0) = 1 - 4/7 = 3/7 = 0.42857142857142855
    assert advantages[0] == advantages[2]
    assert advantages[0] == pytest.approx(0.42857142857142855)
    # Fold once at group close: b1 = beta*b0 + (1-beta)*mean = 0.9*0.5 + 0.1*1.0 = 0.55
    assert state.baselines[KEY] == pytest.approx(0.55)
    (fold,) = state.last_folds
    assert (fold.key, fold.group_size, fold.cold) == (KEY, 2, False)
    assert fold.baseline_before == 0.5
    assert fold.baseline_after == pytest.approx(0.55)
    assert fold.group_mean == pytest.approx(1.0)


def test_cold_key_baseline_is_zero():
    state = RAEState()

    advantages = compute_rae_advantages(_group([1.0, -1.0]), state)

    # Cold key: b0 = 0. lam = 6/7:
    #   A_0 = 1 - (0 + 1/7 * (-1)) = 8/7 = 1.1428571428571428
    #   A_1 = -1 - (0 + 1/7 * 1) = -8/7
    assert advantages[0] == pytest.approx(1.1428571428571428)
    assert advantages[2] == pytest.approx(-1.1428571428571428)
    # Fold: b1 = 0.9*0 + 0.1*mean([1,-1]) = 0.0 — but the key is now warm
    assert state.baselines[KEY] == pytest.approx(0.0)
    assert state.last_folds[0].cold is True


def test_zero_sum_violation_raises():
    state = RAEState()
    rows = [
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
        _member(member_id="debater_b", reward=1.0, episode_id="ep-0"),
    ]

    with pytest.raises(ValueError, match="zero-sum member pairs") as excinfo:
        compute_rae_advantages(rows, state)
    assert "ep-0" in str(excinfo.value)
    assert "debater_a=1.0" in str(excinfo.value)


def test_more_than_two_member_ids_raise():
    state = RAEState()
    rows = _pair("ep-0", 1.0) + [_member(member_id="judge", reward=0.0, episode_id="ep-0")]

    with pytest.raises(ValueError, match="zero-sum member pairs"):
        compute_rae_advantages(rows, state)


def test_duplicate_member_row_in_episode_raises():
    state = RAEState()
    rows = [
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
        _member(member_id="debater_a", reward=-1.0, episode_id="ep-0"),
    ]

    with pytest.raises(ValueError, match="one row per member"):
        compute_rae_advantages(rows, state)


def test_quarantined_single_member_episode_uses_sign():
    state = RAEState()
    # ep-1's debater_a row was quarantined; its canonical reward is -r_b = -1.0
    rows = _pair("ep-0", 1.0) + [_member(member_id="debater_b", reward=1.0, episode_id="ep-1")]

    advantages = compute_rae_advantages(rows, state)

    # Canonical rewards [1, -1], cold b0 = 0, lam = 6/7:
    #   A_canonical(ep-1) = -1 - 1/7 = -8/7; the present debater_b row gets +8/7
    assert advantages[2] == pytest.approx(1.1428571428571428)
    assert advantages[0] == pytest.approx(1.1428571428571428)


def test_single_seat_group_trains_on_canonical_rewards():
    """train_one mode: only one seat per episode survives fan-out."""
    state = RAEState(baselines={KEY: 0.5})
    rows = [
        _member(member_id="debater_a", reward=1.0, episode_id="ep-0"),
        _member(member_id="debater_a", reward=-1.0, episode_id="ep-1"),
    ]

    advantages = compute_rae_advantages(rows, state)

    # Single member id: it is the canonical member. lam = 6/7:
    #   A_0 = 1 - (6/7*0.5 + 1/7*(-1)) = 1 - 2/7 = 5/7 = 0.7142857142857143
    #   A_1 = -1 - (6/7*0.5 + 1/7*1) = -1 - 4/7 = -11/7 = -1.5714285714285714
    assert advantages[0] == pytest.approx(0.7142857142857143)
    assert advantages[1] == pytest.approx(-1.5714285714285714)
    # Fold: b1 = 0.9*0.5 + 0.1*0 = 0.45
    assert state.baselines[KEY] == pytest.approx(0.45)


def test_baselines_partition_by_env_and_example():
    state = RAEState()
    rows = (
        _pair("ep-0", 1.0, env_name="env_a", example_id=1)
        + _pair("ep-1", -1.0, env_name="env_a", example_id=2)
        + _pair("ep-2", 1.0, env_name="env_b", example_id=1)
    )

    compute_rae_advantages(rows, state)

    # Each (env_name, example_id) is its own G=1 group; fold = 0.1 * r
    assert state.baselines == {
        ("env_a", 1): pytest.approx(0.1),
        ("env_a", 2): pytest.approx(-0.1),
        ("env_b", 1): pytest.approx(0.1),
    }
    assert len(state.last_folds) == 3


def test_fan_out_carries_env_name_for_rae_identity():
    state = RAEState()
    episode = {
        "env_name": "gpqa_debate",
        "example_id": 1,
        "task": {"question": "q"},
        "trajectory": [{"extras": {"member_id": m}} for m in ("prover", "verifier")],
        "sampling_args": {"temperature": 0.7},
        "error": None,
        "trajectory_id": "ep-0",
        "mar_score": {
            "members": [
                {"member_id": "prover", "reward": 1.0},
                {"member_id": "verifier", "reward": -1.0},
            ],
            "episode_scalar": 1.0,
        },
    }

    units, mapping = fan_out_for_multi_agent([episode])
    compute_rae_advantages(units, state)

    assert mapping == [[0, 1]]
    assert set(state.baselines) == {("gpqa_debate", 1)}


def test_fan_out_respects_trainable_member_predicate():
    episode = {
        "env_name": ENV_NAME,
        "example_id": 1,
        "task": ENV_NAME,
        "trajectory": [{"extras": {"member_id": m}} for m in ("debater_a", "debater_b", "judge")],
        "sampling_args": {"temperature": 0.7},
        "error": None,
        "trajectory_id": "ep-0",
        "mar_score": {
            "members": [
                {"member_id": "debater_a", "reward": 1.0},
                {"member_id": "debater_b", "reward": -1.0},
                {"member_id": "judge", "reward": 0.0},
            ],
            "episode_scalar": 1.0,
        },
    }

    units, mapping = fan_out_for_multi_agent(
        [episode],
        is_trainable_member=lambda _rollout, member_id: member_id == "debater_a",
    )

    assert mapping == [[0]]
    assert [unit["member_id"] for unit in units] == ["debater_a"]


def test_rae_requires_string_identity_when_task_is_the_fallback():
    state = RAEState()
    rollout = _member(task={"question": "q"}, env_name=None)

    with pytest.raises(TypeError, match="string env_name"):
        compute_rae_advantages([rollout], state)
