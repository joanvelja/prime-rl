"""Tests for RAEState + compute_rae_advantages.

This is the per-(task, example_id, member_id) baseline path that the
multi-agent training loop will use as the bridge consumer. Algorithm:
A_i = R_i - b[(task_i, example_id_i, member_id_i)], EMA baseline update.
"""

import pytest
from verifiers.types import MemberRollout

from prime_rl.orchestrator.multi_agent_advantage import RAEState, compute_rae_advantages

ENV_NAME = "debate_v1"
TEMPERATURE = 0.7


def _make_rollout(
    *,
    example_id: int | str = 1,
    member_id: str = "prover",
    reward: float = 1.0,
    episode_id: str = "ep-0",
    task: str = ENV_NAME,
) -> MemberRollout:
    return MemberRollout(
        example_id=example_id,
        task=task,
        trajectory=[],
        sampling_args={"temperature": TEMPERATURE},
        error=None,
        reward=reward,
        episode_id=episode_id,
        member_id=member_id,
    )


# ---------------------------------------------------------------------------
# RAE invariants
# ---------------------------------------------------------------------------


def test_cold_start_advantage_equals_reward():
    """No baseline yet → baseline defaults to 0 → advantage = reward."""
    state = RAEState(baselines={})
    rollouts = [
        _make_rollout(reward=1.0, member_id="prover"),
        _make_rollout(reward=0.0, member_id="verifier"),
    ]
    advs = compute_rae_advantages(rollouts, state)
    assert advs == [1.0, 0.0]


def test_baselines_update_after_batch():
    """After one batch, baselines should reflect EMA update."""
    state = RAEState(baselines={}, momentum=0.9)
    rollouts = [_make_rollout(reward=1.0, member_id="prover")]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.1)


def test_second_batch_uses_updated_baseline():
    state = RAEState(baselines={}, momentum=0.9)
    advs1 = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    assert advs1 == [1.0]
    advs2 = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    assert advs2 == [pytest.approx(0.9)]
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.19)


def test_degenerate_group_always_positive():
    """Member always wins (reward=1.0). EMA baseline < 1.0 for finite t,
    so advantage stays positive — the agent keeps learning."""
    state = RAEState(baselines={}, momentum=0.9)
    for _ in range(20):
        advs = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    assert advs[0] > 0
    assert state.baselines[(ENV_NAME, 1, "prover")] < 1.0


def test_per_member_baselines_independent():
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [
        _make_rollout(example_id=1, member_id="prover", reward=1.0),
        _make_rollout(example_id=1, member_id="verifier", reward=0.0),
    ]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[(ENV_NAME, 1, "verifier")] == pytest.approx(0.0)


def test_per_example_baselines_independent():
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [
        _make_rollout(example_id=1, member_id="prover", reward=1.0),
        _make_rollout(example_id=2, member_id="prover", reward=0.0, episode_id="ep-1"),
    ]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[(ENV_NAME, 2, "prover")] == pytest.approx(0.0)


def test_per_task_baselines_independent():
    """Same (example_id, member_id) under different tasks must NOT share baselines."""
    state = RAEState(baselines={}, momentum=0.5)
    r_a = _make_rollout(example_id=1, member_id="prover", reward=1.0, task="env_a")
    r_b = _make_rollout(example_id=1, member_id="prover", reward=0.0, task="env_b")
    compute_rae_advantages([r_a, r_b], state)
    assert state.baselines[("env_a", 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[("env_b", 1, "prover")] == pytest.approx(0.0)


def test_within_batch_ordering_invariant():
    """All advantages in a batch use the SAME pre-batch baselines.
    Order within the batch must not matter."""
    state_fwd = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5}, momentum=0.9)
    state_rev = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5}, momentum=0.9)
    r1 = _make_rollout(reward=1.0, episode_id="ep-0")
    r2 = _make_rollout(reward=0.0, episode_id="ep-1")
    advs_fwd = compute_rae_advantages([r1, r2], state_fwd)
    advs_rev = compute_rae_advantages([r2, r1], state_rev)
    assert advs_fwd == [pytest.approx(0.5), pytest.approx(-0.5)]
    assert advs_rev == [pytest.approx(-0.5), pytest.approx(0.5)]
    assert state_fwd.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(
        state_rev.baselines[(ENV_NAME, 1, "prover")]
    )


def test_none_reward_raises():
    """Even though MemberRollout typed reward as float, the bridge boundary
    is dict-typed at runtime — defensive None-check stays."""
    state = RAEState(baselines={})
    rollouts = [_make_rollout(reward=None)]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reward=None"):
        compute_rae_advantages(rollouts, state)


def test_empty_batch_preserves_baselines():
    state = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5})
    advs = compute_rae_advantages([], state)
    assert advs == []
    assert state.baselines[(ENV_NAME, 1, "prover")] == 0.5


def test_str_example_id_keys_baseline_correctly():
    """HuggingFace UID-style example_ids (e.g. 'mmlu_0001') flow through
    the RAE key as-is; baseline lookup matches by tuple equality."""
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [_make_rollout(example_id="mmlu_0001", reward=1.0)]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, "mmlu_0001", "prover")] == pytest.approx(0.5)


def test_repeated_key_in_batch_uses_mean_for_baseline_update():
    """Two rollouts with the same (task, example_id, member_id): both get
    advantages computed from the SAME pre-batch baseline; baseline is
    then updated using the MEAN of the rewards (not last-write-wins)."""
    state = RAEState(baselines={}, momentum=0.5)
    rs = [
        _make_rollout(reward=1.0, episode_id="ep-0"),
        _make_rollout(reward=0.0, episode_id="ep-1"),
    ]
    advs = compute_rae_advantages(rs, state)
    # Both used pre-batch baseline = 0.0
    assert advs == [1.0, 0.0]
    # Baseline updated with mean(reward) = 0.5
    # new = 0.5 * 0 + 0.5 * 0.5 = 0.25
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.25)


def test_zero_reward_from_errored_rollout_keys_correctly():
    """Errored mar_score produces MemberRollouts with reward=0.0 — RAE
    should treat them like any other zero-reward sample (no special case)."""
    state = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.7}, momentum=0.5)
    advs = compute_rae_advantages([_make_rollout(reward=0.0)], state)
    assert advs == [pytest.approx(-0.7)]
