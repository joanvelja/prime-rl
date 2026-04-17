"""End-to-end validation: synthetic EpisodeResult through bridge → RAE → eval."""

import pytest

from verifiers.types import (
    EpisodeResult,
    MemberResult,
    TrajectoryStep,
    TrajectoryStepTokens,
)
from prime_rl.orchestrator.multi_actor_bridge import (
    MemberRollout,
    episodes_to_member_rollouts,
)
from prime_rl.orchestrator.multi_actor_advantage import RAEState, compute_rae_advantages
from prime_rl.orchestrator.multi_actor_eval import evaluate_multi_actor_episodes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENV_NAME = "debate_v1"
TEMPERATURE = 0.7


def _majority_positive(r: EpisodeResult) -> bool:
    """Majority of members have reward > 0."""
    rewards = [m.reward for m in r.members if m.reward is not None]
    if not rewards:
        return False
    return sum(rw > 0 for rw in rewards) >= len(rewards) / 2


def _make_step(trajectory_id: str, tokens: TrajectoryStepTokens | None = None) -> TrajectoryStep:
    """Minimal valid TrajectoryStep. Optionally pre-fill tokens."""
    return TrajectoryStep(
        prompt=[],
        completion=[],
        response={
            "id": "r0",
            "created": 0,
            "model": "test",
            "message": {
                "role": "assistant",
                "content": "hello",
                "finish_reason": "stop",
                "is_truncated": False,
            },
        },
        tokens=tokens,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id=trajectory_id,
        extras={},
    )


def _make_tokens(
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_mask: list[int] | None = None,
) -> TrajectoryStepTokens:
    """Build pre-filled tokens for gradient scoping tests."""
    if completion_mask is None:
        completion_mask = [1] * len(completion_ids)
    return TrajectoryStepTokens(
        prompt_ids=prompt_ids,
        prompt_mask=[0] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * len(completion_ids),
        overlong_prompt=False,
        is_truncated=False,
        routed_experts=None,
    )


def _make_two_actor_episode(
    base_example_id: int = 1,
    episode_id: str = "ep-0",
    alice_reward: float = 1.0,
    bob_reward: float = 0.0,
    alice_tokens: list[TrajectoryStepTokens | None] | None = None,
    bob_tokens: list[TrajectoryStepTokens | None] | None = None,
) -> EpisodeResult:
    """Deterministic 2-actor, 2-turn episode."""
    a_toks = alice_tokens or [None, None]
    b_toks = bob_tokens or [None, None]
    return EpisodeResult(
        base_example_id=base_example_id,
        episode_id=episode_id,
        members=[
            MemberResult(
                member_id="alice",
                role_id="prover",
                seat_id="A",
                trajectory=[
                    _make_step("alice-t0", a_toks[0]),
                    _make_step("alice-t1", a_toks[1]),
                ],
                reward=alice_reward,
            ),
            MemberResult(
                member_id="bob",
                role_id="verifier",
                seat_id="B",
                trajectory=[
                    _make_step("bob-t0", b_toks[0]),
                    _make_step("bob-t1", b_toks[1]),
                ],
                reward=bob_reward,
            ),
        ],
        outcome={"winner": "alice" if alice_reward > bob_reward else "bob", "total_turns": 2},
    )


# ---------------------------------------------------------------------------
# Tests — Bridge → RAE (training path)
# ---------------------------------------------------------------------------


def test_e2e_bridge_produces_correct_rollout_shape():
    """EpisodeResult → MemberRollout: correct count and fields."""
    episodes = [
        _make_two_actor_episode(base_example_id=1, episode_id="ep-0"),
        _make_two_actor_episode(base_example_id=2, episode_id="ep-1"),
    ]
    rollouts = episodes_to_member_rollouts(episodes, ENV_NAME, TEMPERATURE)
    assert len(rollouts) == 4

    for r in rollouts:
        assert r["task"] == ENV_NAME
        assert r["sampling_args"]["temperature"] == TEMPERATURE
        assert r["error"] is None
        assert len(r["trajectory"]) == 2


def test_e2e_bridge_then_rae_cold_start():
    """Full path: bridge → RAE. Cold start advantages == rewards."""
    episodes = [
        _make_two_actor_episode(base_example_id=1, episode_id="ep-0"),
        _make_two_actor_episode(base_example_id=2, episode_id="ep-1"),
    ]
    rollouts = episodes_to_member_rollouts(episodes, ENV_NAME, TEMPERATURE)
    state = RAEState(baselines={}, momentum=0.9)
    advs = compute_rae_advantages(rollouts, state)
    assert advs == [1.0, 0.0, 1.0, 0.0]


def test_e2e_bridge_then_rae_across_batches():
    """Two batches: second batch uses updated baselines."""
    state = RAEState(baselines={}, momentum=0.9)

    batch1 = episodes_to_member_rollouts(
        [_make_two_actor_episode(base_example_id=1, episode_id="ep-0")],
        ENV_NAME, TEMPERATURE,
    )
    compute_rae_advantages(batch1, state)

    batch2 = episodes_to_member_rollouts(
        [_make_two_actor_episode(base_example_id=1, episode_id="ep-1")],
        ENV_NAME, TEMPERATURE,
    )
    advs2 = compute_rae_advantages(batch2, state)
    # Alice: 1.0 - 0.1 = 0.9,  Bob: 0.0 - 0.0 = 0.0
    assert advs2[0] == pytest.approx(0.9)
    assert advs2[1] == pytest.approx(0.0)


def test_e2e_rae_on_known_baselines():
    """RAE with pre-set baselines produces exact expected values."""
    state = RAEState(
        baselines={(ENV_NAME, 1, "prover"): 0.5, (ENV_NAME, 1, "verifier"): 0.3},
        momentum=0.9,
    )
    episodes = [_make_two_actor_episode(base_example_id=1, episode_id="ep-0")]
    rollouts = episodes_to_member_rollouts(episodes, ENV_NAME, TEMPERATURE)
    advs = compute_rae_advantages(rollouts, state)

    assert advs[0] == pytest.approx(0.5)   # 1.0 - 0.5
    assert advs[1] == pytest.approx(-0.3)  # 0.0 - 0.3


# ---------------------------------------------------------------------------
# Tests — Eval bridge (two-table output)
# ---------------------------------------------------------------------------


def test_e2e_eval_metric_namespaces():
    """Eval bridge produces correctly namespaced metrics."""
    episodes = [
        _make_two_actor_episode(base_example_id=1, episode_id="ep-0"),
        _make_two_actor_episode(base_example_id=1, episode_id="ep-1"),
        _make_two_actor_episode(
            base_example_id=2, episode_id="ep-2",
            alice_reward=0.0, bob_reward=1.0,
        ),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, ENV_NAME, rollouts_per_example=2, episode_success_fn=_majority_positive,
    )

    # Episode-level keys
    assert f"eval/{ENV_NAME}/episode/avg@2" in metrics
    assert f"eval/{ENV_NAME}/episode/total_turns/mean" in metrics
    assert metrics[f"eval/{ENV_NAME}/episode/total_turns/mean"] == pytest.approx(2.0)

    # Per-role keys
    assert f"eval/{ENV_NAME}/role/prover/reward/mean" in metrics
    assert f"eval/{ENV_NAME}/role/verifier/reward/mean" in metrics
    assert f"eval/{ENV_NAME}/role/prover/completion_len/mean" in metrics
    assert f"eval/{ENV_NAME}/role/verifier/is_truncated/mean" in metrics


def test_e2e_eval_per_role_reward_aggregation():
    """Per-role mean rewards computed correctly across episodes."""
    episodes = [
        _make_two_actor_episode(base_example_id=1, episode_id="ep-0",
                                alice_reward=1.0, bob_reward=0.0),
        _make_two_actor_episode(base_example_id=1, episode_id="ep-1",
                                alice_reward=1.0, bob_reward=0.0),
        _make_two_actor_episode(base_example_id=2, episode_id="ep-2",
                                alice_reward=0.0, bob_reward=1.0),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, ENV_NAME, rollouts_per_example=2, episode_success_fn=_majority_positive,
    )

    # Prover: (1.0 + 1.0 + 0.0) / 3
    assert metrics[f"eval/{ENV_NAME}/role/prover/reward/mean"] == pytest.approx(2.0 / 3)
    # Verifier: (0.0 + 0.0 + 1.0) / 3
    assert metrics[f"eval/{ENV_NAME}/role/verifier/reward/mean"] == pytest.approx(1.0 / 3)


def test_e2e_eval_pass_at_k_episode_level():
    """pass@k is episode-level, NOT member-level (which gives 50% by construction)."""
    episodes = [
        _make_two_actor_episode(base_example_id=1, episode_id=f"ep-{i}")
        for i in range(4)
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, ENV_NAME, rollouts_per_example=4, episode_success_fn=_majority_positive,
    )
    assert metrics[f"eval/{ENV_NAME}/episode/pass@1"] == pytest.approx(1.0)


def test_e2e_eval_all_failures():
    """Both members reward=0 → episode fails under default predicate."""
    episodes = [
        _make_two_actor_episode(
            base_example_id=1, episode_id="ep-0",
            alice_reward=0.0, bob_reward=0.0,
        ),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, ENV_NAME, rollouts_per_example=1, episode_success_fn=_majority_positive,
    )
    assert metrics[f"eval/{ENV_NAME}/episode/avg@1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests — Gradient scoping (completion_mask)
# ---------------------------------------------------------------------------


def test_gradient_scoping_completion_mask_per_member():
    """Each member's trajectory carries only their own completion tokens.

    In multi-actor, alice's trajectory has alice's completions (mask=1)
    and opponent messages appear as prompt tokens (mask=0). The bridge
    passes trajectories through unchanged, so if the tokens are pre-filled
    correctly by the env, gradient scoping is preserved.
    """
    # Alice: turn 0 has 3 prompt tokens + 2 completion tokens
    #        turn 1 has 5 prompt tokens (original + bob's reply) + 3 completion tokens
    alice_t0 = _make_tokens(
        prompt_ids=[10, 11, 12],
        completion_ids=[100, 101],
        completion_mask=[1, 1],
    )
    alice_t1 = _make_tokens(
        prompt_ids=[10, 11, 12, 50, 51],  # includes bob's tokens as prompt
        completion_ids=[102, 103, 104],
        completion_mask=[1, 1, 1],
    )

    # Bob: turn 0 has 3 prompt tokens + 2 completion tokens
    #      turn 1 has 5 prompt tokens (original + alice's reply) + 2 completion tokens
    bob_t0 = _make_tokens(
        prompt_ids=[20, 21, 22],
        completion_ids=[200, 201],
        completion_mask=[1, 1],
    )
    bob_t1 = _make_tokens(
        prompt_ids=[20, 21, 22, 100, 101],  # includes alice's tokens as prompt
        completion_ids=[202, 203],
        completion_mask=[1, 1],
    )

    episode = _make_two_actor_episode(
        alice_tokens=[alice_t0, alice_t1],
        bob_tokens=[bob_t0, bob_t1],
    )

    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    alice_rollout, bob_rollout = rollouts

    # Alice's trajectory: completion tokens are her own
    for step in alice_rollout["trajectory"]:
        tokens = step["tokens"]
        assert tokens is not None
        # prompt_mask all 0 (not trained on)
        assert all(m == 0 for m in tokens["prompt_mask"])
        # completion_mask all 1 (trained on — alice's own tokens)
        assert all(m == 1 for m in tokens["completion_mask"])

    # Bob's trajectory: same invariant
    for step in bob_rollout["trajectory"]:
        tokens = step["tokens"]
        assert tokens is not None
        assert all(m == 0 for m in tokens["prompt_mask"])
        assert all(m == 1 for m in tokens["completion_mask"])

    # Cross-check: alice's completion tokens don't appear in alice's prompt
    # but DO appear in bob's prompt (turn 1)
    alice_completion_ids = set()
    for step in alice_rollout["trajectory"]:
        alice_completion_ids.update(step["tokens"]["completion_ids"])

    bob_t1_prompt = bob_rollout["trajectory"][1]["tokens"]["prompt_ids"]
    # Bob's turn 1 prompt includes alice's turn 0 completion tokens
    assert 100 in bob_t1_prompt
    assert 101 in bob_t1_prompt

    # But alice's own prompts never include her completion tokens
    for step in alice_rollout["trajectory"]:
        alice_prompt_set = set(step["tokens"]["prompt_ids"])
        assert alice_prompt_set.isdisjoint(alice_completion_ids)


def test_gradient_scoping_no_cross_contamination():
    """Completion IDs from one member must not appear as completions of the other."""
    alice_t0 = _make_tokens(prompt_ids=[1, 2], completion_ids=[100, 101])
    bob_t0 = _make_tokens(prompt_ids=[3, 4], completion_ids=[200, 201])

    episode = _make_two_actor_episode(
        alice_tokens=[alice_t0, None],
        bob_tokens=[bob_t0, None],
    )
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    alice_rollout, bob_rollout = rollouts

    alice_comp_ids = set()
    for step in alice_rollout["trajectory"]:
        if step["tokens"] is not None:
            alice_comp_ids.update(step["tokens"]["completion_ids"])

    bob_comp_ids = set()
    for step in bob_rollout["trajectory"]:
        if step["tokens"] is not None:
            bob_comp_ids.update(step["tokens"]["completion_ids"])

    assert alice_comp_ids.isdisjoint(bob_comp_ids)


# ---------------------------------------------------------------------------
# Tests — interleave_rollout contract (structural validation)
#
# interleave_rollout (trajectories.py) imports torch and can't run in the
# lightweight test runner. These tests validate that MemberRollout output
# satisfies every field interleave_rollout reads, with correct types and
# the extension property satisfied for multi-turn trajectories.
# ---------------------------------------------------------------------------


def _make_extending_tokens(turn: int) -> TrajectoryStepTokens:
    """Build tokens where turn N+1 extends turn N (prefix monotonicity).

    Turn 0: prompt=[1,2,3], completion=[10,11]
    Turn 1: prompt=[1,2,3,10,11,20,21], completion=[30,31,32]
        (turn 0 prompt + turn 0 completion + opponent reply + own completion)

    This mirrors real debate: each turn's prompt is the full history.
    """
    if turn == 0:
        return TrajectoryStepTokens(
            prompt_ids=[1, 2, 3],
            prompt_mask=[0, 0, 0],
            completion_ids=[10, 11],
            completion_mask=[1, 1],
            completion_logprobs=[-0.5, -0.3],
            overlong_prompt=False,
            is_truncated=False,
            routed_experts=None,
        )
    # Turn 1 extends turn 0: prompt starts with turn 0's prompt + completion
    return TrajectoryStepTokens(
        prompt_ids=[1, 2, 3, 10, 11, 20, 21],
        prompt_mask=[0, 0, 0, 0, 0, 0, 0],
        completion_ids=[30, 31, 32],
        completion_mask=[1, 1, 1],
        completion_logprobs=[-0.2, -0.4, -0.1],
        overlong_prompt=False,
        is_truncated=False,
        routed_experts=None,
    )


def test_interleave_contract_rollout_has_required_keys():
    """MemberRollout must provide all keys interleave_rollout reads."""
    episode = _make_two_actor_episode(
        alice_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
        bob_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
    )
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)

    for r in rollouts:
        # interleave_rollout reads these at the top level
        assert "trajectory" in r
        assert "error" in r
        assert "sampling_args" in r
        assert "temperature" in r["sampling_args"]
        assert "example_id" in r

        # interleave_rollout checks output["error"] is not None
        assert r["error"] is None

        # sampling_args.temperature must be a float
        assert isinstance(r["sampling_args"]["temperature"], float)


def test_interleave_contract_step_tokens_complete():
    """Each step's tokens dict must have all fields interleave_rollout reads."""
    episode = _make_two_actor_episode(
        alice_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
        bob_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
    )
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)

    required_token_keys = {
        "prompt_ids", "prompt_mask",
        "completion_ids", "completion_mask", "completion_logprobs",
        "routed_experts",
    }

    for r in rollouts:
        for step_idx, step in enumerate(r["trajectory"]):
            tokens = step["tokens"]
            assert tokens is not None, f"step {step_idx} has tokens=None"
            for key in required_token_keys:
                assert key in tokens, f"step {step_idx} missing tokens[{key!r}]"

            # Length consistency
            assert len(tokens["completion_ids"]) == len(tokens["completion_mask"])
            assert len(tokens["completion_ids"]) == len(tokens["completion_logprobs"])
            assert len(tokens["prompt_ids"]) == len(tokens["prompt_mask"])

            # Types: interleave_rollout calls list(), bool(), on these
            assert isinstance(tokens["prompt_ids"], list)
            assert isinstance(tokens["completion_ids"], list)
            assert all(isinstance(x, int) for x in tokens["prompt_ids"])
            assert all(isinstance(x, int) for x in tokens["completion_ids"])
            assert all(isinstance(x, (int, float)) for x in tokens["completion_logprobs"])


def test_interleave_contract_extension_property():
    """Turn N+1's prompt_ids must start with turn N's (prompt_ids + completion_ids).

    This is the extension property that interleave_rollout uses to merge
    multi-turn trajectories into a single TrainingSample.
    """
    episode = _make_two_actor_episode(
        alice_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
        bob_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
    )
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)

    for r in rollouts:
        traj = r["trajectory"]
        for i in range(len(traj) - 1):
            t_curr = traj[i]["tokens"]
            t_next = traj[i + 1]["tokens"]

            prefix = t_curr["prompt_ids"] + t_curr["completion_ids"]
            next_prompt = t_next["prompt_ids"]

            assert next_prompt[:len(prefix)] == prefix, (
                f"Extension property violated between steps {i} and {i+1}: "
                f"prefix={prefix}, next_prompt_start={next_prompt[:len(prefix)]}"
            )


def test_interleave_contract_completion_mask_gradient_scoping():
    """completion_mask=True only on the member's own generated tokens.

    interleave_rollout converts mask values via bool() — prompt tokens that
    appear between turns get mask=False. This test validates that the token
    data is structured so interleave_rollout would produce correct gradients.
    """
    alice_t0 = _make_extending_tokens(0)
    alice_t1 = _make_extending_tokens(1)

    episode = _make_two_actor_episode(
        alice_tokens=[alice_t0, alice_t1],
        bob_tokens=[_make_extending_tokens(0), _make_extending_tokens(1)],
    )
    rollouts = episodes_to_member_rollouts([episode], ENV_NAME, TEMPERATURE)
    alice_rollout = rollouts[0]

    # Simulate what interleave_rollout does: merge turns via extension property
    t0 = alice_rollout["trajectory"][0]["tokens"]
    t1 = alice_rollout["trajectory"][1]["tokens"]
    prefix_len = len(t0["prompt_ids"]) + len(t0["completion_ids"])

    # After merging, the combined completion would be:
    # [t0 completion (mask=True)] + [new prompt tokens from t1 (mask=False)] + [t1 completion (mask=True)]
    new_prompt_between = t1["prompt_ids"][prefix_len:]
    merged_mask = (
        [bool(m) for m in t0["completion_mask"]]         # alice's turn 0 tokens: trained
        + [False] * len(new_prompt_between)               # opponent reply: NOT trained
        + [bool(m) for m in t1["completion_mask"]]        # alice's turn 1 tokens: trained
    )

    # Verify: only completion tokens (positions from t0 and t1) are True
    n_trained = sum(merged_mask)
    n_untrained = len(merged_mask) - n_trained
    assert n_trained == len(t0["completion_ids"]) + len(t1["completion_ids"])
    assert n_untrained == len(new_prompt_between)
