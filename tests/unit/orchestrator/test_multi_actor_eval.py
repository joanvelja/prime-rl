from verifiers.types import EpisodeResult, MemberResult, TrajectoryStep

from prime_rl.orchestrator.multi_actor_eval import evaluate_multi_actor_episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(*, completion_ids: list[int] | None = None, is_truncated: bool = False) -> TrajectoryStep:
    """Build a minimal TrajectoryStep dict for testing."""
    tokens = None
    if completion_ids is not None:
        tokens = {
            "prompt_ids": [],
            "prompt_mask": [],
            "completion_ids": completion_ids,
            "completion_mask": [1] * len(completion_ids),
            "completion_logprobs": [0.0] * len(completion_ids),
            "overlong_prompt": False,
            "is_truncated": is_truncated,
            "routed_experts": None,
        }
    return {
        "prompt": [],
        "completion": [],
        "response": {"id": "r", "created": 0, "model": "m", "message": {"role": "assistant", "finish_reason": "stop", "is_truncated": is_truncated}},
        "tokens": tokens,
        "reward": None,
        "advantage": None,
        "is_truncated": is_truncated,
        "trajectory_id": "t",
        "extras": {},
    }


def _make_member(member_id: str, role_id: str, reward: float, steps: list[TrajectoryStep] | None = None) -> MemberResult:
    return MemberResult(
        member_id=member_id,
        role_id=role_id,
        seat_id=member_id,
        trajectory=steps or [_make_step(completion_ids=[1, 2, 3])],
        reward=reward,
    )


def _make_episode(base_id: int, episode_id: str, members: list[MemberResult], outcome: dict | None = None) -> EpisodeResult:
    return EpisodeResult(
        base_example_id=base_id,
        episode_id=episode_id,
        members=members,
        outcome=outcome,
    )


def _majority_positive(r: EpisodeResult) -> bool:
    """Majority of members have reward > 0."""
    rewards = [m.reward for m in r.members if m.reward is not None]
    if not rewards:
        return False
    return sum(rw > 0 for rw in rewards) >= len(rewards) / 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_two_table_structure():
    """Metrics dict contains episode-level and per-role keys."""
    episodes = [
        _make_episode(1, "ep-0", [
            _make_member("A", "prover", 1.0),
            _make_member("B", "verifier", 0.0),
        ]),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=1, episode_success_fn=_majority_positive,
    )

    # Episode-level keys
    assert "eval/debate/episode/avg@1" in metrics
    assert "eval/debate/episode/total_turns/mean" in metrics
    assert "eval/debate/episode/pass@1" in metrics

    # Per-role keys
    assert "eval/debate/role/prover/reward/mean" in metrics
    assert "eval/debate/role/prover/completion_len/mean" in metrics
    assert "eval/debate/role/prover/is_truncated/mean" in metrics
    assert "eval/debate/role/verifier/reward/mean" in metrics
    assert "eval/debate/role/verifier/completion_len/mean" in metrics
    assert "eval/debate/role/verifier/is_truncated/mean" in metrics


def test_pass_at_k_is_episode_level():
    """Construct 4 episodes where truth (prover) always wins.

    If pass@k were member-level, it would be 0.5 (half win, half lose).
    Episode-level pass@k should be 1.0 because every episode is a success.
    """
    episodes = [
        _make_episode(1, f"ep-{i}", [
            _make_member("A", "prover", 1.0),
            _make_member("B", "verifier", 0.0),
        ])
        for i in range(4)
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=4,
        episode_success_fn=_majority_positive,
    )

    # Episode success: majority of members have reward > 0 → prover wins → success=1.0
    assert metrics["eval/debate/episode/avg@4"] == 1.0

    # pass@k should all be 1.0 (every episode succeeds)
    pass_keys = [k for k in metrics if "pass@" in k]
    assert len(pass_keys) > 0
    for k in pass_keys:
        assert metrics[k] == 1.0, f"{k} = {metrics[k]}, expected 1.0"


def test_pass_at_k_mixed_outcomes():
    """2 examples, 2 episodes each. Example 1: both succeed. Example 2: one succeeds, one fails.

    Uses a custom success_fn (prover won) to make the distinction meaningful:
    the default predicate treats any member with reward > 0 as partial success,
    which in constant-sum games means both sides "succeed".
    """
    def prover_won(r: EpisodeResult) -> bool:
        return any(m.role_id == "prover" and (m.reward or 0) > 0 for m in r.members)

    episodes = [
        # Example 1: both episodes prover wins
        _make_episode(1, "ep-1a", [_make_member("A", "prover", 1.0), _make_member("B", "verifier", 0.0)]),
        _make_episode(1, "ep-1b", [_make_member("A", "prover", 1.0), _make_member("B", "verifier", 0.0)]),
        # Example 2: first prover wins, second verifier wins
        _make_episode(2, "ep-2a", [_make_member("A", "prover", 1.0), _make_member("B", "verifier", 0.0)]),
        _make_episode(2, "ep-2b", [_make_member("A", "prover", 0.0), _make_member("B", "verifier", 1.0)]),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=2, episode_success_fn=prover_won,
    )

    # pass@1: example 1 → c=2,n=2 → 1.0; example 2 → c=1,n=2 → 0.5
    # mean = (1.0 + 0.5) / 2 = 0.75
    assert abs(metrics["eval/debate/episode/pass@1"] - 0.75) < 1e-9

    # pass@2: example 1 → 1.0; example 2 → 1.0
    assert abs(metrics["eval/debate/episode/pass@2"] - 1.0) < 1e-9


def test_per_role_metrics():
    """Per-role reward/completion_len/is_truncated are computed correctly."""
    step_a = _make_step(completion_ids=[1, 2, 3, 4, 5])  # 5 tokens
    step_b = _make_step(completion_ids=[1, 2], is_truncated=True)  # 2 tokens, truncated

    episodes = [
        _make_episode(1, "ep-0", [
            _make_member("A", "prover", 1.0, steps=[step_a]),
            _make_member("B", "verifier", 0.0, steps=[step_b]),
        ]),
        _make_episode(1, "ep-1", [
            _make_member("A", "prover", 0.5, steps=[step_a]),
            _make_member("B", "verifier", 0.5, steps=[step_b]),
        ]),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=2, episode_success_fn=_majority_positive,
    )

    assert metrics["eval/debate/role/prover/reward/mean"] == 0.75  # (1.0 + 0.5) / 2
    assert metrics["eval/debate/role/verifier/reward/mean"] == 0.25  # (0.0 + 0.5) / 2

    assert metrics["eval/debate/role/prover/completion_len/mean"] == 5.0
    assert metrics["eval/debate/role/verifier/completion_len/mean"] == 2.0

    assert metrics["eval/debate/role/prover/is_truncated/mean"] == 0.0
    assert metrics["eval/debate/role/verifier/is_truncated/mean"] == 1.0


def test_monitor_logging_keys_format():
    """All returned keys match the expected eval/{env}/... namespace."""
    episodes = [
        _make_episode(1, "ep-0", [
            _make_member("A", "debater_a", 1.0),
            _make_member("B", "debater_b", 0.0),
        ]),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "my_env", rollouts_per_example=1, episode_success_fn=_majority_positive,
    )

    for key in metrics:
        assert key.startswith("eval/my_env/"), f"Key {key} does not start with eval/my_env/"

    # Verify specific namespace patterns
    episode_keys = [k for k in metrics if "/episode/" in k]
    role_keys = [k for k in metrics if "/role/" in k]
    assert len(episode_keys) >= 3  # avg@k, total_turns/mean, pass@k
    assert len(role_keys) >= 6  # 2 roles × 3 metrics each


def test_custom_episode_success_fn():
    """Custom success function overrides default majority-wins logic."""
    episodes = [
        _make_episode(1, "ep-0", [
            _make_member("A", "prover", 0.0),
            _make_member("B", "verifier", 1.0),
        ], outcome={"winner_role": "verifier"}),
    ]

    # Custom: success = prover won (which is False here)
    def prover_won(r: EpisodeResult) -> bool:
        return (r.outcome or {}).get("winner_role") == "prover"

    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=1, episode_success_fn=prover_won,
    )
    assert metrics["eval/debate/episode/avg@1"] == 0.0

    # With majority-positive: verifier has reward > 0 → success=True
    metrics_majority = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=1, episode_success_fn=_majority_positive,
    )
    assert metrics_majority["eval/debate/episode/avg@1"] == 1.0


def test_empty_results():
    """Empty input returns empty metrics."""
    assert evaluate_multi_actor_episodes(
        [], "debate", rollouts_per_example=1, episode_success_fn=_majority_positive,
    ) == {}


def test_outcome_total_turns_used():
    """total_turns is read from outcome dict when available."""
    episodes = [
        _make_episode(1, "ep-0", [
            _make_member("A", "prover", 1.0, steps=[_make_step(), _make_step()]),
        ], outcome={"total_turns": 7}),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=1, episode_success_fn=_majority_positive,
    )
    assert metrics["eval/debate/episode/total_turns/mean"] == 7.0


def test_completion_len_sums_across_steps():
    """completion_len sums token counts across multiple trajectory steps."""
    steps = [
        _make_step(completion_ids=[1, 2, 3]),  # 3 tokens
        _make_step(completion_ids=[4, 5]),      # 2 tokens
    ]
    episodes = [
        _make_episode(1, "ep-0", [
            _make_member("A", "prover", 1.0, steps=steps),
        ]),
    ]
    metrics = evaluate_multi_actor_episodes(
        episodes, "debate", rollouts_per_example=1, episode_success_fn=_majority_positive,
    )
    assert metrics["eval/debate/role/prover/completion_len/mean"] == 5.0
