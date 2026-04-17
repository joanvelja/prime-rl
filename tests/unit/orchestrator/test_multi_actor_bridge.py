import pytest

from verifiers.types import EpisodeResult, MemberResult, TrajectoryStep
from prime_rl.orchestrator.multi_actor_bridge import (
    MemberRollout,
    episodes_to_member_rollouts,
    rollout_to_member_rollouts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENV_NAME = "debate_v1"
TEMPERATURE = 0.7


def _make_step(trajectory_id: str = "traj-0") -> TrajectoryStep:
    """Minimal valid TrajectoryStep for structural tests."""
    return TrajectoryStep(
        prompt=[],
        completion=[],
        response={
            "id": "r0",
            "created": 0,
            "model": "test",
            "message": {
                "role": "assistant",
                "content": "hi",
                "finish_reason": "stop",
                "is_truncated": False,
            },
        },
        tokens=None,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id=trajectory_id,
        extras={},
    )


def _make_episode(
    base_example_id: int = 1,
    episode_id: str = "ep-0",
    members: list[MemberResult] | None = None,
) -> EpisodeResult:
    if members is None:
        members = [
            MemberResult(
                member_id="alice",
                role_id="prover",
                seat_id="A",
                trajectory=[_make_step("alice-traj")],
                reward=1.0,
            ),
            MemberResult(
                member_id="bob",
                role_id="verifier",
                seat_id="B",
                trajectory=[_make_step("bob-traj")],
                reward=0.0,
            ),
        ]
    return EpisodeResult(
        base_example_id=base_example_id,
        episode_id=episode_id,
        members=members,
    )


# ---------------------------------------------------------------------------
# Tests — episodes_to_member_rollouts
# ---------------------------------------------------------------------------


def test_single_episode_produces_one_rollout_per_member():
    episode = _make_episode()
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    assert len(rollouts) == 2


def test_rollout_has_correct_training_fields():
    episode = _make_episode()
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    r = rollouts[0]

    assert r["example_id"] == 1
    assert r["task"] == ENV_NAME
    assert r["sampling_args"] == {"temperature": TEMPERATURE}
    assert r["error"] is None
    assert r["reward"] == 1.0
    assert len(r["trajectory"]) == 1


def test_rollout_has_correct_metadata():
    episode = _make_episode()
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)

    alice, bob = rollouts
    assert alice["episode_id"] == "ep-0"
    assert alice["member_id"] == "alice"
    assert alice["role_id"] == "prover"

    assert bob["member_id"] == "bob"
    assert bob["role_id"] == "verifier"


def test_task_is_env_name_not_role_id():
    """Critical invariant: buffer/scheduler assert task == env_name."""
    episode = _make_episode()
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    for r in rollouts:
        assert r["task"] == ENV_NAME
        assert r["task"] != r["role_id"]


def test_multiple_episodes_flatten_correctly():
    episodes = [
        _make_episode(base_example_id=1, episode_id="ep-0"),
        _make_episode(base_example_id=2, episode_id="ep-1"),
    ]
    rollouts = episodes_to_member_rollouts(episodes, ENV_NAME, TEMPERATURE)
    assert len(rollouts) == 4

    ids = [(r["example_id"], r["episode_id"], r["member_id"]) for r in rollouts]
    assert ids == [
        (1, "ep-0", "alice"),
        (1, "ep-0", "bob"),
        (2, "ep-1", "alice"),
        (2, "ep-1", "bob"),
    ]


def test_string_example_id_passes_through():
    """EpisodeResult.base_example_id is int | str — bridge must not coerce."""
    episode = _make_episode()
    episode.base_example_id = "string-id"
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    assert all(r["example_id"] == "string-id" for r in rollouts)


def test_none_example_id_raises():
    episode = _make_episode()
    episode.base_example_id = None
    with pytest.raises(TypeError, match="must not be None"):
        episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)


def test_empty_results_returns_empty():
    assert episodes_to_member_rollouts([], ENV_NAME, TEMPERATURE) == []


def test_member_with_none_reward():
    member = MemberResult(
        member_id="c",
        role_id="judge",
        seat_id="C",
        trajectory=[],
        reward=None,
    )
    episode = _make_episode(members=[member])
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    assert rollouts[0]["reward"] is None


def test_trajectory_passed_through_unchanged():
    """Trajectory should be the exact same list object, not a copy."""
    episode = _make_episode()
    original_traj = episode.members[0].trajectory
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    assert rollouts[0]["trajectory"] is original_traj


# ---------------------------------------------------------------------------
# Tests — RAE advantage (compute_rae_advantages + RAEState)
# ---------------------------------------------------------------------------

from prime_rl.orchestrator.multi_actor_advantage import RAEState, compute_rae_advantages


def _make_rollout(
    example_id: int = 1,
    role_id: str = "prover",
    reward: float = 1.0,
    episode_id: str = "ep-0",
    member_id: str = "alice",
) -> MemberRollout:
    return MemberRollout(
        example_id=example_id,
        task=ENV_NAME,
        trajectory=[],
        sampling_args={"temperature": TEMPERATURE},
        error=None,
        reward=reward,
        episode_id=episode_id,
        member_id=member_id,
        role_id=role_id,
    )


def test_rae_cold_start_advantage_equals_reward():
    """No baseline yet → baseline defaults to 0 → advantage = reward."""
    state = RAEState(baselines={})
    rollouts = [
        _make_rollout(reward=1.0, role_id="prover"),
        _make_rollout(reward=0.0, role_id="verifier", member_id="bob"),
    ]
    advs = compute_rae_advantages(rollouts, state)
    assert advs == [1.0, 0.0]


def test_rae_baselines_update_after_batch():
    """After one batch, baselines should reflect EMA update."""
    state = RAEState(baselines={}, momentum=0.9)
    rollouts = [_make_rollout(reward=1.0, role_id="prover")]
    compute_rae_advantages(rollouts, state)

    # EMA: 0.9 * 0.0 + 0.1 * 1.0 = 0.1
    assert state.baselines[(1, "prover")] == pytest.approx(0.1)


def test_rae_second_batch_uses_updated_baseline():
    state = RAEState(baselines={}, momentum=0.9)

    # Batch 1: cold start
    rollouts1 = [_make_rollout(reward=1.0)]
    advs1 = compute_rae_advantages(rollouts1, state)
    assert advs1 == [1.0]  # baseline was 0

    # Batch 2: baseline is 0.1
    rollouts2 = [_make_rollout(reward=1.0)]
    advs2 = compute_rae_advantages(rollouts2, state)
    assert advs2 == [pytest.approx(0.9)]  # 1.0 - 0.1

    # Baseline after two updates: 0.9 * 0.1 + 0.1 * 1.0 = 0.19
    assert state.baselines[(1, "prover")] == pytest.approx(0.19)


def test_rae_degenerate_group_always_positive():
    """Alice always wins (reward=1.0). EMA baseline < 1.0 for finite t,
    so advantage stays positive — the agent keeps learning."""
    state = RAEState(baselines={}, momentum=0.9)
    for _ in range(20):
        advs = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    # After 20 updates, baseline converges toward 1.0 but never reaches it
    assert advs[0] > 0
    assert state.baselines[(1, "prover")] < 1.0


def test_rae_per_role_baselines_are_independent():
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [
        _make_rollout(example_id=1, role_id="prover", reward=1.0, member_id="alice"),
        _make_rollout(example_id=1, role_id="verifier", reward=0.0, member_id="bob"),
    ]
    compute_rae_advantages(rollouts, state)

    assert state.baselines[(1, "prover")] == pytest.approx(0.5)
    assert state.baselines[(1, "verifier")] == pytest.approx(0.0)


def test_rae_per_example_baselines_are_independent():
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [
        _make_rollout(example_id=1, role_id="prover", reward=1.0),
        _make_rollout(example_id=2, role_id="prover", reward=0.0, episode_id="ep-1"),
    ]
    compute_rae_advantages(rollouts, state)

    assert state.baselines[(1, "prover")] == pytest.approx(0.5)
    assert state.baselines[(2, "prover")] == pytest.approx(0.0)


def test_rae_within_batch_ordering_invariant():
    """All advantages in a batch use the SAME pre-batch baselines.
    Order within the batch must not matter."""
    state_fwd = RAEState(baselines={(1, "prover"): 0.5}, momentum=0.9)
    state_rev = RAEState(baselines={(1, "prover"): 0.5}, momentum=0.9)

    r1 = _make_rollout(reward=1.0, episode_id="ep-0")
    r2 = _make_rollout(reward=0.0, episode_id="ep-1")

    advs_fwd = compute_rae_advantages([r1, r2], state_fwd)
    advs_rev = compute_rae_advantages([r2, r1], state_rev)

    assert advs_fwd == [pytest.approx(0.5), pytest.approx(-0.5)]
    assert advs_rev == [pytest.approx(-0.5), pytest.approx(0.5)]

    # Both states should have same final baseline
    assert state_fwd.baselines[(1, "prover")] == pytest.approx(
        state_rev.baselines[(1, "prover")]
    )


def test_rae_none_reward_raises():
    state = RAEState(baselines={})
    rollouts = [_make_rollout(reward=None)]
    with pytest.raises(ValueError, match="reward=None"):
        compute_rae_advantages(rollouts, state)


def test_rae_empty_batch():
    state = RAEState(baselines={(1, "prover"): 0.5})
    advs = compute_rae_advantages([], state)
    assert advs == []
    assert state.baselines[(1, "prover")] == 0.5


# ---------------------------------------------------------------------------
# Tests — rollout_to_member_rollouts (DebateEnv bridge)
# ---------------------------------------------------------------------------


def _make_tagged_step(
    member_id: str,
    role_id: str,
    phase: str = "opening",
    trajectory_id: str = "traj-0",
) -> TrajectoryStep:
    """TrajectoryStep with DebateEnv-style extras tags."""
    return TrajectoryStep(
        prompt=[],
        completion=[],
        response={
            "id": "r0",
            "created": 0,
            "model": "test",
            "message": {
                "role": "assistant",
                "content": f"{member_id} says hi",
                "finish_reason": "stop",
                "is_truncated": False,
            },
        },
        tokens=None,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id=trajectory_id,
        extras={"member_id": member_id, "role_id": role_id, "phase": phase},
    )


def _make_rollout_output(
    steps: list[TrajectoryStep] | None = None,
    example_id: int = 42,
    temperature: float = 0.7,
    trajectory_id: str = "debate-0",
    metrics: dict | None = None,
) -> dict:
    """Minimal RolloutOutput dict as DebateEnv would produce."""
    if steps is None:
        steps = [
            _make_tagged_step("A", "prover", "opening"),
            _make_tagged_step("B", "verifier", "opening"),
            _make_tagged_step("A", "prover", "rebuttal"),
            _make_tagged_step("B", "verifier", "rebuttal"),
        ]
    if metrics is None:
        metrics = {"reward/A": 1.0, "reward/B": 0.0}
    return {
        "trajectory": steps,
        "sampling_args": {"temperature": temperature},
        "example_id": example_id,
        "trajectory_id": trajectory_id,
        "metrics": metrics,
    }


def test_rollout_bridge_splits_by_member():
    output = _make_rollout_output()
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    assert len(rollouts) == 2
    member_ids = {r["member_id"] for r in rollouts}
    assert member_ids == {"A", "B"}


def test_rollout_bridge_correct_step_counts():
    output = _make_rollout_output()
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    by_member = {r["member_id"]: r for r in rollouts}
    assert len(by_member["A"]["trajectory"]) == 2
    assert len(by_member["B"]["trajectory"]) == 2


def test_rollout_bridge_training_fields():
    output = _make_rollout_output()
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    a = next(r for r in rollouts if r["member_id"] == "A")

    assert a["example_id"] == 42
    assert a["task"] == ENV_NAME
    assert a["sampling_args"] == {"temperature": 0.7}
    assert a["error"] is None
    assert a["reward"] == 1.0
    assert a["episode_id"] == "debate-0"
    assert a["role_id"] == "prover"


def test_rollout_bridge_per_member_rewards():
    output = _make_rollout_output()
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    by_member = {r["member_id"]: r for r in rollouts}
    assert by_member["A"]["reward"] == 1.0
    assert by_member["B"]["reward"] == 0.0


def test_rollout_bridge_missing_reward_is_none():
    output = _make_rollout_output(metrics={})
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    for r in rollouts:
        assert r["reward"] is None


def test_rollout_bridge_empty_trajectory():
    output = _make_rollout_output(steps=[])
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    assert rollouts == []


def test_rollout_bridge_missing_member_id_raises():
    bad_step = _make_tagged_step("A", "prover")
    bad_step["extras"] = {}  # no member_id
    output = _make_rollout_output(steps=[bad_step])
    with pytest.raises(ValueError, match="member_id"):
        rollout_to_member_rollouts(output, ENV_NAME)


def test_rollout_bridge_missing_sampling_args_raises():
    output = _make_rollout_output()
    del output["sampling_args"]
    with pytest.raises(ValueError, match="sampling_args"):
        rollout_to_member_rollouts(output, ENV_NAME)


def test_rollout_bridge_preserves_step_order():
    """Steps within a member must maintain temporal order."""
    s1 = _make_tagged_step("A", "prover", "opening")
    s2 = _make_tagged_step("B", "verifier", "opening")
    s3 = _make_tagged_step("A", "prover", "rebuttal")
    s4 = _make_tagged_step("B", "verifier", "rebuttal")
    output = _make_rollout_output(steps=[s1, s2, s3, s4])
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    a = next(r for r in rollouts if r["member_id"] == "A")
    assert a["trajectory"][0]["extras"]["phase"] == "opening"
    assert a["trajectory"][1]["extras"]["phase"] == "rebuttal"


def test_rollout_bridge_role_id_from_first_step():
    output = _make_rollout_output()
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    a = next(r for r in rollouts if r["member_id"] == "A")
    b = next(r for r in rollouts if r["member_id"] == "B")
    assert a["role_id"] == "prover"
    assert b["role_id"] == "verifier"


def test_rollout_bridge_default_episode_id():
    """Missing trajectory_id defaults to empty string."""
    output = _make_rollout_output()
    del output["trajectory_id"]
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    assert rollouts[0]["episode_id"] == ""


def test_rollout_bridge_task_is_env_name():
    """Same invariant as EpisodeResult bridge: task == env_name."""
    output = _make_rollout_output()
    rollouts = rollout_to_member_rollouts(output, ENV_NAME)
    for r in rollouts:
        assert r["task"] == ENV_NAME
