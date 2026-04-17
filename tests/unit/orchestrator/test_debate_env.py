"""Unit tests for DebateEnv: real kernel, real rubric, fake client.

The client is the system boundary -- faking it is correct. Everything
inside (StaticSchedule, apply_action, DebateRubric, DebatePrompts,
rollout_to_member_rollouts) must be the real implementation.

Run from the fork venv (prime-rl .venv is empty on Darwin per linux-only
resolver env):

  cd ../verifiers && uv run pytest \\
    /path/to/prime-rl/tests/unit/orchestrator/test_debate_env.py --noconftest

The fork venv must have prime-rl installed editable (once per rebuild):

  cd ../verifiers && uv pip install -e ../prime-rl --no-deps

`--noconftest` skips prime-rl's root conftest which eagerly imports
prime_rl.trainer.world (torch/distributed). Orthogonal to this suite.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import verifiers as vf
from verifiers.clients import Client as _VFClient
from verifiers.utils.async_utils import maybe_retry
from verifiers.envs.debate_env import (
    DebateEnv,
    _consolidate_messages,
    _validate_judge_templates,
    load_environment,
)
from verifiers.envs.debate.prompts import DebatePrompts, build_context, resolve_prompts
from verifiers.envs.debate.fields import EnumScoring, FieldSpec
from verifiers.envs.debate_rubric import DebateRubric
from verifiers.envs.multi_actor_kernel import (
    KernelState,
    StaticSchedule,
    TurnSlot,
    apply_action,
)
from verifiers.errors import Error as VFError, KernelProtocolError
from verifiers.types import (
    Response,
    ResponseMessage,
    ResponseTokens,
    RolloutInput,
    State,
    TrajectoryStep,
    Usage,
)
from prime_rl.orchestrator.multi_actor_bridge import rollout_to_member_rollouts


# ---------------------------------------------------------------------------
# Fake client — system boundary stub
# ---------------------------------------------------------------------------


class FakeClient(_VFClient):
    """Concrete vf.Client subclass returning canned responses.

    Subclasses `verifiers.clients.Client` so `isinstance(x, Client)` passes
    the Environment's boundary check in `resolve_client`. `get_response` is
    overridden directly and the native-layer abstract methods raise
    NotImplementedError — they are never reached because the override
    bypasses the native conversion path.
    """

    def __init__(self, responses: list[Response]) -> None:
        import logging as _logging_mod

        # Skip vf.Client.__init__ (wants a concrete SDK client or ClientConfig).
        # Keep shape-compatible fields in case introspection touches them.
        self.logger = _logging_mod.getLogger(f"{__name__}.FakeClient")
        self._config = None
        self._client = None
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self, prompt, model, sampling_args=None, tools=None, **kwargs
    ) -> Response:
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "sampling_args": sampling_args,
        })
        if not self._responses:
            raise RuntimeError("FakeClient exhausted — more calls than expected")
        return self._responses.pop(0)

    def setup_client(self, config):
        raise NotImplementedError("FakeClient has no native SDK client")

    async def to_native_tool(self, tool):
        raise NotImplementedError("FakeClient does not support tool calls")

    async def to_native_prompt(self, messages):
        raise NotImplementedError("FakeClient bypasses native conversion")

    async def get_native_response(self, *args, **kwargs):
        raise NotImplementedError("FakeClient overrides get_response directly")

    async def raise_from_native_response(self, response):
        raise NotImplementedError("FakeClient overrides get_response directly")

    async def from_native_response(self, response):
        raise NotImplementedError("FakeClient overrides get_response directly")

    async def close(self) -> None:
        return None


def _make_response(content: str, token_ids: list[int] | None = None) -> Response:
    """Build a canned Response with optional real token data."""
    tokens = None
    if token_ids is not None:
        tokens = ResponseTokens(
            prompt_ids=[1, 2, 3],
            prompt_mask=[0, 0, 0],
            completion_ids=token_ids,
            completion_mask=[1] * len(token_ids),
            completion_logprobs=[-0.5] * len(token_ids),
            routed_experts=None,
        )
    return Response(
        id="fake-resp",
        created=0,
        model="test-model",
        usage=Usage(
            prompt_tokens=3,
            reasoning_tokens=0,
            completion_tokens=len(token_ids) if token_ids else len(content),
            total_tokens=3 + (len(token_ids) if token_ids else len(content)),
        ),
        message=ResponseMessage(
            content=content,
            finish_reason="stop",
            is_truncated=False,
            tokens=tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


TWO_TURN_SLOTS = (
    TurnSlot(slot_id=0, actors=("A",), phase="opening"),
    TurnSlot(slot_id=1, actors=("B",), phase="opening"),
    TurnSlot(slot_id=2, actors=("A",), phase="rebuttal"),
    TurnSlot(slot_id=3, actors=("B",), phase="rebuttal"),
)

import jinja2
import jinja2.sandbox

_je = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)

DEBATE_PROMPTS = DebatePrompts(
    system={
        "prover": _je.from_string("You argue for the answer."),
        "verifier": _je.from_string("You argue against."),
    },
    user={
        "prover": {
            "opening": _je.from_string("Make your opening argument."),
            "rebuttal": _je.from_string("Rebut."),
        },
        "verifier": {
            "opening": _je.from_string("Make your opening argument."),
            "rebuttal": _je.from_string("Rebut."),
        },
    },
    question={
        "prover": _je.from_string("Question: {{ task_prompt }}"),
        "verifier": _je.from_string("Question: {{ task_prompt }}"),
    },
    fields={},
    think_visibility={},
    think_tag="thinking",
    prefill={},
    opponent_wrap=None,
    judges={},
    source_ref="test",
)

ROLE_FOR_ACTOR = {"A": "prover", "B": "verifier"}


def _make_env(
    responses: list[Response],
    schedule_slots: tuple[TurnSlot, ...] = TWO_TURN_SLOTS,
    prompts: DebatePrompts = DEBATE_PROMPTS,
    role_for_actor: dict[str, str] | None = None,
    truth_role: str = "prover",
    members: list[str] | None = None,
    actor_overrides: dict | None = None,
) -> tuple[DebateEnv, FakeClient]:
    members = members or ["A", "B"]
    if role_for_actor is None:
        role_for_actor = ROLE_FOR_ACTOR
    rubric = DebateRubric(truth_role=truth_role, members=members, prompts=prompts)
    client = FakeClient(responses)
    env = DebateEnv(
        schedule=StaticSchedule(schedule_slots),
        prompts=prompts,
        members=members,
        role_for_actor=role_for_actor,
        actor_overrides=actor_overrides,
        rubric=rubric,
        dataset=lambda: None,
    )
    return env, client


def _run(coro):
    return asyncio.run(coro)


def _rollout_input(example_id: int = 1) -> RolloutInput:
    return RolloutInput(
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        example_id=example_id,
        task="debate_test",
        answer="4",
    )


# ---------------------------------------------------------------------------
# 1. rollout() produces tagged trajectory
# ---------------------------------------------------------------------------


def test_rollout_tags_every_step_with_member_id():
    """Every TrajectoryStep.extras must contain member_id, role_id, phase."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    trajectory = state["trajectory"]
    assert len(trajectory) == 4

    for step in trajectory:
        extras = step["extras"]
        assert "member_id" in extras, f"Step missing member_id: {step}"
        assert "role_id" in extras, f"Step missing role_id: {step}"
        assert "phase" in extras, f"Step missing phase: {step}"

    assert trajectory[0]["extras"]["member_id"] == "A"
    assert trajectory[0]["extras"]["role_id"] == "prover"
    assert trajectory[0]["extras"]["phase"] == "opening"

    assert trajectory[1]["extras"]["member_id"] == "B"
    assert trajectory[1]["extras"]["role_id"] == "verifier"
    assert trajectory[1]["extras"]["phase"] == "opening"

    assert trajectory[2]["extras"]["member_id"] == "A"
    assert trajectory[2]["extras"]["phase"] == "rebuttal"

    assert trajectory[3]["extras"]["member_id"] == "B"
    assert trajectory[3]["extras"]["phase"] == "rebuttal"


def test_rollout_trajectory_has_real_prompt_and_completion():
    """Steps must have non-empty prompt and completion messages."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    for step in state["trajectory"]:
        assert len(step["prompt"]) > 0, "Step has empty prompt"
        assert len(step["completion"]) > 0, "Step has empty completion"


# ---------------------------------------------------------------------------
# 2. Stop conditions
# ---------------------------------------------------------------------------


def test_debate_complete_fires_when_schedule_exhausted():
    """After all slots are consumed, is_completed should be True."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    assert state["is_completed"] is True
    assert state["stop_condition"] == "debate_complete"


def test_has_error_stop():
    """If error is set on state, has_error should return True."""
    env, _ = _make_env([])
    state = State()
    state["error"] = Exception("test error")
    state["is_completed"] = False
    result = _run(env.has_error(state))
    assert result is True


def test_debate_complete_false_when_slots_remain():
    """debate_complete should return False when schedule still has slots."""
    env, _ = _make_env([])
    state = State()
    state["_kernel"] = KernelState(slot_index=0)
    state["is_completed"] = False
    result = _run(env.debate_complete(state))
    assert result is False


def test_debate_complete_true_when_all_slots_done():
    """debate_complete should return True when slot_index exceeds schedule."""
    env, _ = _make_env([])
    state = State()
    state["_kernel"] = KernelState(slot_index=len(TWO_TURN_SLOTS))
    state["is_completed"] = False
    result = _run(env.debate_complete(state))
    assert result is True


# ---------------------------------------------------------------------------
# 3. Actor resolver
# ---------------------------------------------------------------------------


def test_resolve_actor_defaults_to_none_none():
    """Self-play mode: no overrides -> (None, None) for all actors."""
    env, _ = _make_env([])
    assert env._resolve_actor("A") == (None, None)
    assert env._resolve_actor("B") == (None, None)


def test_resolve_actor_with_overrides():
    """Mode 3: fixed opponent gets a separate client/model."""
    opp_client = FakeClient([])
    env, _ = _make_env(
        [],
        actor_overrides={"B": (opp_client, "opponent-model")},
    )
    assert env._resolve_actor("A") == (None, None)
    client_b, model_b = env._resolve_actor("B")
    assert client_b is opp_client
    assert model_b == "opponent-model"


# ---------------------------------------------------------------------------
# 4. Rubric scoring (real DebateRubric)
# ---------------------------------------------------------------------------


def test_rubric_sets_episode_and_per_member_rewards():
    """DebateRubric.score_rollout must set state['reward'] and state['metrics']."""
    rubric = _rubric()  # SELFPLAY_PROMPTS + auto-attached FakeJudgeClient

    state = State()
    state["answer"] = "C"
    state["trajectory"] = [
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("hi"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={"member_id": "A", "role_id": "debater_a", "phase": "propose"},
        ),
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("bye"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={"member_id": "B", "role_id": "debater_b", "phase": "propose"},
        ),
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("judge decides"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={
                "member_id": "J", "role_id": "judge", "phase": "final",
                "fields": {"decision": "debater_a"},
            },
        ),
    ]
    state["reward"] = None
    state["metrics"] = None

    _run(rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert state["metrics"]["reward/A"] == 1.0
    assert state["metrics"]["reward/B"] == 0.0
    assert state["metrics"]["turns/A"] == 1.0
    assert state["metrics"]["turns/B"] == 1.0


def test_rubric_verifier_wins():
    """When truth_role loses (judge picks opponent), episode reward = 0.0."""
    rubric = _rubric()  # SELFPLAY_PROMPTS + auto-attached FakeJudgeClient
    state = State()
    state["answer"] = "C"
    state["trajectory"] = [
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("x"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={"member_id": "A", "role_id": "debater_a", "phase": "propose"},
        ),
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("y"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={"member_id": "B", "role_id": "debater_b", "phase": "propose"},
        ),
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("judge decides"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={
                "member_id": "J", "role_id": "judge", "phase": "final",
                "fields": {"decision": "debater_b"},
            },
        ),
    ]
    state["reward"] = None
    state["metrics"] = None

    _run(rubric.score_rollout(state))

    assert state["reward"] == 0.0
    assert state["metrics"]["reward/A"] == 0.0
    assert state["metrics"]["reward/B"] == 1.0


# ---------------------------------------------------------------------------
# 5. render_completion
# ---------------------------------------------------------------------------


def test_render_completion_produces_valid_completion():
    """After rollout, state['completion'] must be a non-empty message list."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    assert state["completion"] is not None
    assert len(state["completion"]) > 0


def test_render_completion_excludes_per_turn_prompts():
    """Completion must contain only completion messages, not per-turn prompts."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    completion = state["completion"]
    # Should have exactly 4 completion messages (one per slot)
    assert len(completion) == 4

    # No system messages should appear in completion
    for msg in completion:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        assert role != "system", f"System message leaked into completion: {content}"
        assert "You argue" not in content, f"Per-turn prompt leaked into completion: {content}"


# ---------------------------------------------------------------------------
# 6. Simultaneous slots
# ---------------------------------------------------------------------------


def test_simultaneous_slot_produces_responses_for_all_actors():
    """Simultaneous slot with both actors -> 2 steps added at once."""
    sim_slots = (
        TurnSlot(slot_id=0, actors=("A", "B"), phase="simultaneous"),
    )
    responses = [
        _make_response("A simultaneous"),
        _make_response("B simultaneous"),
    ]
    env, client = _make_env(responses, schedule_slots=sim_slots)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    trajectory = state["trajectory"]
    assert len(trajectory) == 2

    member_ids = [s["extras"]["member_id"] for s in trajectory]
    assert "A" in member_ids
    assert "B" in member_ids

    for step in trajectory:
        assert step["extras"]["phase"] == "simultaneous"


# ---------------------------------------------------------------------------
# 7. Full pipeline: rollout -> rubric -> rollout_to_member_rollouts
# ---------------------------------------------------------------------------


def test_full_pipeline_rollout_to_member_rollouts():
    """The critical pipeline test: rollout -> score -> bridge -> MemberRollout.

    Uses real kernel, real rubric, real bridge. Only the client is faked.
    """
    responses = [
        _make_response("A opens", token_ids=[10, 11]),
        _make_response("B opens", token_ids=[20, 21]),
        _make_response("A rebuts", token_ids=[12, 13, 14]),
        _make_response("B rebuts", token_ids=[22, 23]),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(example_id=42), client, "test-model"))

    # Inject judge decision into trajectory (winner = prover)
    state["trajectory"] = list(state["trajectory"]) + [
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("verdict"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={
                "member_id": "J", "role_id": "judge", "phase": "final",
                "fields": {"decision": "prover"},
            },
        ),
    ]
    _run(env.rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert state["metrics"]["reward/A"] == 1.0
    assert state["metrics"]["reward/B"] == 0.0

    # Build a RolloutOutput-like dict from state
    output = {
        "trajectory": state["trajectory"],
        "sampling_args": {"temperature": 0.7},
        "example_id": state["example_id"],
        "trajectory_id": state["trajectory_id"],
        "metrics": state["metrics"],
    }

    rollouts = rollout_to_member_rollouts(output, "debate_test")
    assert len(rollouts) == 3  # A, B, and judge J

    a = next(r for r in rollouts if r["member_id"] == "A")
    b = next(r for r in rollouts if r["member_id"] == "B")

    # Training fields
    assert a["example_id"] == 42
    assert a["task"] == "debate_test"
    assert a["reward"] == 1.0
    assert a["role_id"] == "prover"
    assert len(a["trajectory"]) == 2  # A has 2 turns

    assert b["reward"] == 0.0
    assert b["role_id"] == "verifier"
    assert len(b["trajectory"]) == 2

    # Verify tokens survived -- critical for interleave_rollout downstream
    for step in a["trajectory"]:
        assert step["tokens"] is not None, "Token data lost in pipeline"
    for step in b["trajectory"]:
        assert step["tokens"] is not None, "Token data lost in pipeline"


# ---------------------------------------------------------------------------
# 8. Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_includes_system_and_question():
    """_build_prompt must include system prompt and question."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    _run(env.rollout(_rollout_input(), client, "test-model"))

    # Check first call's prompt (A's opening)
    first_prompt = client.calls[0]["prompt"]
    prompt_texts = [
        m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        for m in first_prompt
    ]
    prompt_str = " ".join(str(t) for t in prompt_texts)

    assert "You argue for the answer" in prompt_str


def test_prompt_includes_opponent_utterances_in_rebuttal():
    """By rebuttal phase, actor should see opponent's previous utterance."""
    responses = [
        _make_response("A opens with argument"),
        _make_response("B opens with counter"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    _run(env.rollout(_rollout_input(), client, "test-model"))

    # A's rebuttal prompt (3rd call, index 2) should contain B's opening
    rebuttal_prompt = client.calls[2]["prompt"]
    prompt_texts = [
        m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        for m in rebuttal_prompt
    ]
    prompt_str = " ".join(str(t) for t in prompt_texts)

    assert "B opens with counter" in prompt_str


# ---------------------------------------------------------------------------
# 9. Kernel state progression
# ---------------------------------------------------------------------------


def test_kernel_state_advances_through_slots():
    """After rollout, kernel slot_index should equal number of slots."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    kernel: KernelState = state["_kernel"]
    assert kernel.slot_index == 4
    assert len(kernel.transcript) == 4

    # Transcript order matches slot order
    assert kernel.transcript[0].member_id == "A"
    assert kernel.transcript[1].member_id == "B"
    assert kernel.transcript[2].member_id == "A"
    assert kernel.transcript[3].member_id == "B"


# ---------------------------------------------------------------------------
# 10. Consolidate messages helper
# ---------------------------------------------------------------------------


def test_consolidate_merges_contiguous_user_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a1"},
    ]
    result = _consolidate_messages(msgs)
    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert "q1" in result[1]["content"]
    assert "q2" in result[1]["content"]
    assert result[2]["role"] == "assistant"


def test_consolidate_does_not_merge_system_messages():
    msgs = [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
    ]
    result = _consolidate_messages(msgs)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 11. Kernel is real (adversarial — would pass with a fake kernel)
# ---------------------------------------------------------------------------


def test_kernel_transcript_contains_response_content():
    """Verify kernel transcript captures the actual response content.

    If the kernel were stubbed/mocked, transcript utterances would be empty
    or generic. This test checks that the content from FakeClient responses
    actually flows through apply_action into the kernel transcript.
    """
    responses = [
        _make_response("Alpha argument here"),
        _make_response("Beta counterpoint"),
    ]
    slots = (
        TurnSlot(slot_id=0, actors=("A",), phase="opening"),
        TurnSlot(slot_id=1, actors=("B",), phase="opening"),
    )
    env, client = _make_env(responses, schedule_slots=slots)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    kernel: KernelState = state["_kernel"]
    assert kernel.transcript[0].raw_content == "Alpha argument here"
    assert kernel.transcript[1].raw_content == "Beta counterpoint"
    assert kernel.transcript[0].public_channel == "Alpha argument here"
    assert kernel.transcript[1].public_channel == "Beta counterpoint"
    assert kernel.transcript[0].private_channel is None
    assert kernel.transcript[1].private_channel is None
    assert kernel.transcript[0].phase == "opening"
    assert kernel.transcript[1].phase == "opening"


def test_kernel_rejects_wrong_actor():
    """apply_action raises KernelProtocolError (a vf.Error subclass) when
    a non-scheduled actor submits. The rollout-layer vf.Error boundary
    catches it distinctly from generic Python errors.
    """
    schedule = StaticSchedule((
        TurnSlot(slot_id=0, actors=("A",), phase="opening"),
    ))
    ks = KernelState(slot_index=0)
    with pytest.raises(KernelProtocolError, match="not scheduled"):
        apply_action(ks, schedule, "B", "wrong actor", 5)
    # vf.Error boundary must still catch it.
    with pytest.raises(VFError):
        apply_action(ks, schedule, "B", "wrong actor", 5)
    # But it must NOT be a ValueError anymore — score_group needs the
    # distinct Error branch.
    assert not issubclass(KernelProtocolError, ValueError)


def test_kernel_rejects_finished_episode_with_protocol_error():
    schedule = StaticSchedule((
        TurnSlot(slot_id=0, actors=("A",), phase="opening"),
    ))
    r1 = apply_action(KernelState(slot_index=0), schedule, "A", "done", 1)
    with pytest.raises(KernelProtocolError, match="No active slot"):
        apply_action(r1.new_state, schedule, "A", "late", 1)


def test_kernel_rejects_duplicate_submission_with_protocol_error():
    schedule = StaticSchedule((
        TurnSlot(slot_id=0, actors=("A", "B"), phase="sim"),
    ))
    r1 = apply_action(KernelState(slot_index=0), schedule, "A", "first", 1)
    with pytest.raises(KernelProtocolError, match="already submitted"):
        apply_action(r1.new_state, schedule, "A", "again", 1)


def _two_actor_schedule() -> StaticSchedule:
    return StaticSchedule((
        TurnSlot(slot_id=0, actors=("A",), phase="p"),
        TurnSlot(slot_id=1, actors=("B",), phase="p"),
    ))


def test_debate_env_requires_members():
    """DebateEnv needs explicit `members` list — replaces fragile _count_actors."""
    schedule = _two_actor_schedule()
    rubric = DebateRubric(truth_role="prover", members=["A", "B"], prompts=DEBATE_PROMPTS)
    with pytest.raises(ValueError, match="non-empty members"):
        DebateEnv(
            schedule=schedule,
            prompts=DEBATE_PROMPTS,
            members=[],
            rubric=rubric,
            dataset=lambda: None,
        )
    with pytest.raises(ValueError, match="duplicates"):
        DebateEnv(
            schedule=schedule,
            prompts=DEBATE_PROMPTS,
            members=["A", "A"],
            rubric=rubric,
            dataset=lambda: None,
        )
    env = DebateEnv(
        schedule=schedule,
        prompts=DEBATE_PROMPTS,
        members=["A", "B"],
        rubric=rubric,
        dataset=lambda: None,
    )
    assert env.members == ["A", "B"]


def test_debate_env_members_must_match_rubric_members():
    """Cross-check 1: silent drift between env.members and rubric.members
    would desync round_index (env) from reward attribution (rubric).
    """
    schedule = _two_actor_schedule()
    rubric = DebateRubric(truth_role="prover", members=["A", "B"], prompts=DEBATE_PROMPTS)

    # Different set of members entirely.
    with pytest.raises(ValueError, match="members != rubric.members"):
        DebateEnv(
            schedule=schedule,
            prompts=DEBATE_PROMPTS,
            members=["X", "Y"],
            rubric=rubric,
            dataset=lambda: None,
        )

    # Same set, different order — still a failure (order matters for
    # any downstream index-based attribution).
    schedule_reversed = StaticSchedule((
        TurnSlot(slot_id=0, actors=("B",), phase="p"),
        TurnSlot(slot_id=1, actors=("A",), phase="p"),
    ))
    with pytest.raises(ValueError, match="members != rubric.members"):
        DebateEnv(
            schedule=schedule_reversed,
            prompts=DEBATE_PROMPTS,
            members=["B", "A"],
            rubric=rubric,
            dataset=lambda: None,
        )


def test_debate_env_members_must_match_static_schedule_actors():
    """Cross-check 2: StaticSchedule's unique slot actors must equal members."""
    rubric = DebateRubric(truth_role="prover", members=["A", "B"], prompts=DEBATE_PROMPTS)

    # Member declared but never appears in schedule.
    schedule_missing_b = StaticSchedule((
        TurnSlot(slot_id=0, actors=("A",), phase="p"),
    ))
    with pytest.raises(ValueError, match="unique actors in StaticSchedule"):
        DebateEnv(
            schedule=schedule_missing_b,
            prompts=DEBATE_PROMPTS,
            members=["A", "B"],
            rubric=rubric,
            dataset=lambda: None,
        )

    # Actor appears in schedule but not declared as member.
    schedule_with_c = StaticSchedule((
        TurnSlot(slot_id=0, actors=("A",), phase="p"),
        TurnSlot(slot_id=1, actors=("B",), phase="p"),
        TurnSlot(slot_id=2, actors=("C",), phase="p"),
    ))
    with pytest.raises(ValueError, match="unique actors in StaticSchedule"):
        DebateEnv(
            schedule=schedule_with_c,
            prompts=DEBATE_PROMPTS,
            members=["A", "B"],
            rubric=rubric,
            dataset=lambda: None,
        )


def test_debate_env_skips_schedule_cross_check_for_dynamic_program():
    """Dynamic SlotProgram implementations are exempt from cross-check 2
    (actor set may be data-dependent).
    """
    class DynamicProgram:
        def current_slot(self, state):
            return None

    rubric = DebateRubric(truth_role="prover", members=["A", "B"], prompts=DEBATE_PROMPTS)
    env = DebateEnv(
        schedule=DynamicProgram(),
        prompts=DEBATE_PROMPTS,
        members=["A", "B"],
        rubric=rubric,
        dataset=lambda: None,
    )
    assert env.members == ["A", "B"]


# ---------------------------------------------------------------------------
# 12. Think stripping in _format_history
# ---------------------------------------------------------------------------


def _make_think_prompts(think_visibility: dict[str, str]) -> DebatePrompts:
    """Build DebatePrompts with specified think visibility."""
    return DebatePrompts(
        system={
            "prover": _je.from_string("Prover system."),
            "verifier": _je.from_string("Verifier system."),
        },
        user={},
        question={
            "prover": _je.from_string("Q"),
            "verifier": _je.from_string("Q"),
        },
        fields={},
        think_visibility=think_visibility,
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="test",
    )


def test_format_history_strips_thinking_for_opponents_private():
    """With private visibility, opponent's thinking is stripped."""
    prompts = _make_think_prompts({"prover": "private", "verifier": "private"})
    env, _ = _make_env([], prompts=prompts)

    # Simulate a kernel state with one utterance containing thinking
    ks = KernelState(slot_index=1)
    from verifiers.envs.multi_actor_kernel import Utterance

    ks = KernelState(
        slot_index=1,
        transcript=(
            Utterance(
                member_id="A",
                slot_id=0,
                raw_content="<thinking>my secret reasoning</thinking>The answer is 4.",
                public_channel="The answer is 4.",
                private_channel="my secret reasoning",
                phase="opening",
                token_count=10,
            ),
        ),
    )

    # B sees A's utterance with thinking stripped
    history = env._format_history(ks, "B")
    assert len(history) == 1
    assert "secret reasoning" not in history[0]["content"]
    assert "The answer is 4." in history[0]["content"]


def test_format_history_preserves_own_thinking():
    """Own utterances are shown verbatim (no stripping)."""
    prompts = _make_think_prompts({"prover": "private"})
    env, _ = _make_env([], prompts=prompts)

    from verifiers.envs.multi_actor_kernel import Utterance

    ks = KernelState(
        slot_index=1,
        transcript=(
            Utterance(
                member_id="A",
                slot_id=0,
                raw_content="<thinking>my reasoning</thinking>The answer is 4.",
                public_channel="The answer is 4.",
                private_channel="my reasoning",
                phase="opening",
                token_count=10,
            ),
        ),
    )

    # A sees own utterance with thinking preserved
    history = env._format_history(ks, "A")
    assert len(history) == 1
    assert "my reasoning" in history[0]["content"]


def test_format_history_open_visibility_preserves_for_all():
    """With open visibility, everyone sees thinking."""
    prompts = _make_think_prompts({"prover": "open", "verifier": "open"})
    env, _ = _make_env([], prompts=prompts)

    from verifiers.envs.multi_actor_kernel import Utterance

    ks = KernelState(
        slot_index=1,
        transcript=(
            Utterance(
                member_id="A",
                slot_id=0,
                raw_content="<thinking>visible reasoning</thinking>Answer.",
                public_channel="Answer.",
                private_channel="visible reasoning",
                phase="opening",
                token_count=10,
            ),
        ),
    )

    history = env._format_history(ks, "B")
    assert "visible reasoning" in history[0]["content"]


def test_format_history_attributes_both_debaters_distinctly():
    """F1 regression: when a viewer (e.g. judge) sees a multi-speaker
    transcript, the rendered opponent blocks MUST carry distinct speaker
    labels. Pre-fix both A and B rendered as identical "OPPONENT" blocks
    and the judge had to infer identity from transcript order.

    Wires a wrap template that uses {{ role_id | upper }} (matching the
    production packs), renders a 2-speaker transcript from a third-party
    viewpoint (actor "J" / role "judge"), and asserts that BOTH roles
    appear as distinct labels in the rendered history."""
    from verifiers.envs.multi_actor_kernel import Utterance

    labeled = DebatePrompts(
        system={
            "prover": _je.from_string("P system."),
            "verifier": _je.from_string("V system."),
            "judge": _je.from_string("J system."),
        },
        user={},
        question={
            "prover": _je.from_string("Q"),
            "verifier": _je.from_string("Q"),
            "judge": _je.from_string("Q"),
        },
        fields={},
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap={
            "debater": _je.from_string(
                "═══ {{ role_id | upper }} [phase={{ phase }}] ═══\n"
                "{{ text }}\n"
                "═══ END {{ role_id | upper }} ═══"
            ),
        },
        judges={},
        source_ref="test",
    )
    env, _ = _make_env(
        [],
        prompts=labeled,
        role_for_actor={"A": "prover", "B": "verifier", "J": "judge"},
        members=["A", "B", "J"],
    )

    ks = KernelState(
        slot_index=2,
        transcript=(
            Utterance(
                member_id="A", slot_id=0,
                raw_content="Prover opens: answer is 4.",
                public_channel="Prover opens: answer is 4.",
                private_channel=None,
                phase="opening", token_count=5,
            ),
            Utterance(
                member_id="B", slot_id=1,
                raw_content="Verifier challenges: show the work.",
                public_channel="Verifier challenges: show the work.",
                private_channel=None,
                phase="opening", token_count=5,
            ),
        ),
    )

    # Render history from the judge's viewpoint: both utterances are
    # "opponents" to J, so both should be wrapped with their role label.
    history = env._format_history(ks, "J")
    assert len(history) == 2

    a_block = history[0]["content"]
    b_block = history[1]["content"]

    # Speaker labels present and distinct.
    assert "PROVER" in a_block, f"A's block must carry PROVER label, got: {a_block!r}"
    assert "VERIFIER" in b_block, f"B's block must carry VERIFIER label, got: {b_block!r}"
    assert "PROVER" not in b_block, f"B's block must NOT carry PROVER label, got: {b_block!r}"
    assert "VERIFIER" not in a_block, f"A's block must NOT carry VERIFIER label, got: {a_block!r}"

    # Phase marker preserved alongside the role label.
    assert "[phase=opening]" in a_block
    assert "[phase=opening]" in b_block

    # Content still present inside the wrapped block.
    assert "answer is 4" in a_block
    assert "show the work" in b_block


def test_format_history_fallback_prefixes_role_id_when_no_template():
    """F1 fallback: a pack with no opponent_wrap template still attributes
    the speaker via a [role_id] prefix. Bare passthrough (the pre-fix
    behavior) would leave judges and peer debaters guessing from order."""
    from verifiers.envs.multi_actor_kernel import Utterance

    # DEBATE_PROMPTS is the fixture used across most env tests — it has
    # opponent_wrap=None, so this exercises the fallback branch.
    env, _ = _make_env([])

    ks = KernelState(
        slot_index=1,
        transcript=(
            Utterance(
                member_id="A", slot_id=0,
                raw_content="Prover says foo.",
                public_channel="Prover says foo.",
                private_channel=None,
                phase="opening", token_count=3,
            ),
        ),
    )

    # B (verifier) sees A's (prover's) utterance. Fallback should prefix
    # the content with [prover] so attribution is explicit even without a
    # custom wrap template.
    history = env._format_history(ks, "B")
    assert len(history) == 1
    assert history[0]["content"] == "[prover] Prover says foo."


def test_parse_channels_contract():
    """parse_channels is the single think-handling primitive.

    Happy path: 0 or 1 closed block. Malformed markup (unclosed, stray
    closer, multiple blocks, nested) is a KernelProtocolError. Replaces
    the legacy strip_think/redact_think pair — under the channel-split
    architecture, unclosed openers are no longer silently tolerated.
    """
    from verifiers.envs.multi_actor_kernel import parse_channels
    from verifiers.errors import KernelProtocolError

    # Happy path: one closed block.
    pub, priv = parse_channels("<thinking>reason</thinking>payload", tag="thinking")
    assert pub == "payload"
    assert priv == "reason"

    # No block → passthrough (stripped).
    assert parse_channels("  hello  ", tag="thinking") == ("hello", None)

    # Empty body → private None.
    assert parse_channels("<thinking>   </thinking>x", tag="thinking") == ("x", None)

    # think/thinking aliasing — either name matches both.
    assert parse_channels("<think>r</think>x", tag="thinking") == ("x", "r")
    assert parse_channels("<thinking>r</thinking>x", tag="think") == ("x", "r")

    # Unclosed opener → protocol error.
    with pytest.raises(KernelProtocolError, match="unbalanced"):
        parse_channels("<thinking>leak<answer>A</answer>", tag="thinking")

    # Stray closer → protocol error.
    with pytest.raises(KernelProtocolError, match="unbalanced"):
        parse_channels("payload</thinking>", tag="thinking")

    # Multiple blocks → protocol error.
    with pytest.raises(KernelProtocolError, match="multiple"):
        parse_channels(
            "<thinking>a</thinking>mid<thinking>b</thinking>",
            tag="thinking",
        )

    # Nested tags → protocol error.
    with pytest.raises(KernelProtocolError, match="nested"):
        parse_channels(
            "<thinking>outer<thinking>inner</thinking></thinking>",
            tag="thinking",
        )


def test_format_history_private_visibility_uses_public_channel():
    """Integration: private visibility → opponent view reads public_channel
    (think stripped at commit). Author's own view reads raw_content.

    Replaces the legacy redact_think integration test. Under the channel
    architecture the private channel is simply not rendered to opponents
    with private/disabled visibility, so leakage is structurally
    impossible — no aggressive EOF-strip heuristic required.
    """
    from verifiers.envs.multi_actor_kernel import Utterance

    prompts = _make_think_prompts({"prover": "private", "verifier": "private"})
    env, _ = _make_env([], prompts=prompts)

    ks = KernelState(
        slot_index=1,
        transcript=(
            Utterance(
                member_id="A",
                slot_id=0,
                raw_content="<thinking>my-secret-reasoning-token</thinking>Answer: A",
                public_channel="Answer: A",
                private_channel="my-secret-reasoning-token",
                phase="opening",
                token_count=10,
            ),
        ),
    )

    opp_history = env._format_history(ks, "B")
    assert len(opp_history) == 1
    opp_content = opp_history[0]["content"]
    assert "my-secret-reasoning-token" not in opp_content
    assert "<thinking>" not in opp_content
    assert "Answer: A" in opp_content

    # Author's own view gets raw_content — think intact for KV-cache coherence.
    own_history = env._format_history(ks, "A")
    assert len(own_history) == 1
    assert "my-secret-reasoning-token" in own_history[0]["content"]


def test_apply_action_rejects_malformed_think_markup():
    """Malformed think markup in raw model output → KernelProtocolError
    at commit. The rollout-layer vf.Error boundary catches it and records
    state['error'] rather than silently propagating malformed content.
    """
    from verifiers.envs.multi_actor_kernel import (
        KernelState as _KS, StaticSchedule as _Sch, TurnSlot as _TS, apply_action as _ap
    )
    from verifiers.errors import KernelProtocolError

    schedule = _Sch((_TS(slot_id=0, actors=("A",), phase="p"),))
    with pytest.raises(KernelProtocolError):
        _ap(
            _KS(slot_index=0), schedule, "A",
            "<thinking>leak<answer>A</answer>",
            10,
            think_tag="thinking",
        )


# ---------------------------------------------------------------------------
# 13. Field extraction
# ---------------------------------------------------------------------------


def _make_field_prompts() -> DebatePrompts:
    """Build DebatePrompts with field specs for extraction testing."""
    from verifiers.envs.debate.fields import FieldSpec

    return DebatePrompts(
        system={
            "prover": _je.from_string("System."),
            "verifier": _je.from_string("System."),
        },
        user={
            "prover": {"opening": _je.from_string("Argue.")},
        },
        question={
            "prover": _je.from_string("Q"),
            "verifier": _je.from_string("Q"),
        },
        fields={
            "prover": {
                "opening": {
                    "answer": FieldSpec(type=str, description="your answer"),
                },
            },
        },
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="test",
    )


def test_extract_fields_populates_extras():
    """Field extraction results stored in TrajectoryStep.extras['fields']."""
    prompts = _make_field_prompts()
    responses = [
        _make_response("I argue <answer>42</answer>"),
        _make_response("B disagrees"),
    ]
    slots = (
        TurnSlot(slot_id=0, actors=("A",), phase="opening"),
        TurnSlot(slot_id=1, actors=("B",), phase="opening"),
    )
    env, client = _make_env(responses, schedule_slots=slots, prompts=prompts)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    trajectory = state["trajectory"]
    assert len(trajectory) == 2

    # A's step should have extracted fields
    a_extras = trajectory[0]["extras"]
    assert "fields" in a_extras
    assert a_extras["fields"]["answer"] == "42"

    # B's step should NOT have fields (no field spec for verifier.opening)
    b_extras = trajectory[1]["extras"]
    assert "fields" not in b_extras


def test_extract_fields_duplicate_tag_is_per_step_failure():
    """Model emitting duplicate <answer> tags → step still appends to
    trajectory with fields=None, rollout continues, no terminal state['error'].
    Rubric's failed_members path handles the per-member signal structurally.

    See G4 cross-boundary fix: parsing ambiguity must not terminate the whole
    rollout nor leave the malformed attempt absent from the trajectory (which
    would cause the rubric to silently grade a stale earlier commit)."""
    prompts = _make_field_prompts()
    responses = [
        _make_response("<answer>A</answer> ... later <answer>B</answer>"),
        _make_response("B opens"),
    ]
    slots = (
        TurnSlot(slot_id=0, actors=("A",), phase="opening"),
        TurnSlot(slot_id=1, actors=("B",), phase="opening"),
    )
    env, client = _make_env(responses, schedule_slots=slots, prompts=prompts)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    # No terminal error — a per-step parse failure must not kill the rollout.
    assert state.get("error") is None

    # Both actors' steps must be in the trajectory — the failed step is still
    # appended so the rubric sees it as the LATEST step for member A.
    trajectory = state["trajectory"]
    assert len(trajectory) == 2
    assert trajectory[0]["extras"]["member_id"] == "A"
    assert trajectory[1]["extras"]["member_id"] == "B"

    # A's step carries no parsed fields — the structural signal the rubric's
    # failed_members path keys off to emit extraction_failed/A=1.0.
    a_extras = trajectory[0]["extras"]
    assert "fields" not in a_extras or a_extras["fields"] is None


def test_extract_fields_strips_thinking_before_parsing():
    """Field extraction strips think blocks before parsing XML fields."""
    prompts = _make_field_prompts()
    prompts_with_think = DebatePrompts(
        system=prompts.system,
        user=prompts.user,
        question=prompts.question,
        fields=prompts.fields,
        think_visibility={"prover": "private"},
        think_tag="thinking",
        prefill=prompts.prefill,
        opponent_wrap=prompts.opponent_wrap,
        judges=prompts.judges,
        source_ref="test",
    )
    responses = [
        _make_response("<thinking>let me reason</thinking><answer>correct</answer>"),
        _make_response("B says"),
    ]
    slots = (
        TurnSlot(slot_id=0, actors=("A",), phase="opening"),
        TurnSlot(slot_id=1, actors=("B",), phase="opening"),
    )
    env, client = _make_env(responses, schedule_slots=slots, prompts=prompts_with_think)
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    a_extras = state["trajectory"][0]["extras"]
    assert a_extras["fields"]["answer"] == "correct"


def test_round_index_is_round_not_slot():
    """round_index should count debate rounds, not raw slot indices.

    A 2-round debate with 2 actors has 4 slots but only 2 rounds.
    Slots 0,1 -> round 0; slots 2,3 -> round 1.

    round_index is rendered via the USER (instruction) block, not system —
    system prompts are phase-independent and must not reference per-turn state.
    """
    round_tracking_prompts = DebatePrompts(
        system={
            "prover": _je.from_string("Prover system."),
            "verifier": _je.from_string("Verifier system."),
        },
        user={
            "prover": {
                "opening": _je.from_string("Round {{ round_index }}/{{ num_rounds }} — Go."),
                "rebuttal": _je.from_string("Round {{ round_index }}/{{ num_rounds }} — Rebut."),
            },
            "verifier": {
                "opening": _je.from_string("Round {{ round_index }}/{{ num_rounds }} — Go."),
                "rebuttal": _je.from_string("Round {{ round_index }}/{{ num_rounds }} — Rebut."),
            },
        },
        question={
            "prover": _je.from_string("Q"),
            "verifier": _je.from_string("Q"),
        },
        fields={},
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="test",
    )
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses, prompts=round_tracking_prompts)
    _run(env.rollout(_rollout_input(), client, "test-model"))

    def _instruction_text(call_idx: int) -> str:
        # Instruction is rendered into the user message (last message in prompt).
        msgs = client.calls[call_idx]["prompt"]
        parts = [
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in msgs
        ]
        return "\n".join(parts)

    # Slot 0 (A opening) -> round 0, num_rounds 2
    assert "Round 0/2" in _instruction_text(0)
    # Slot 1 (B opening) -> round 0, num_rounds 2
    assert "Round 0/2" in _instruction_text(1)
    # Slot 2 (A rebuttal) -> round 1, num_rounds 2
    assert "Round 1/2" in _instruction_text(2)
    # Slot 3 (B rebuttal) -> round 1, num_rounds 2
    assert "Round 1/2" in _instruction_text(3)


def test_build_prompt_uses_debate_prompts():
    """_build_prompt renders via DebatePrompts, not static strings."""
    responses = [
        _make_response("A opens"),
        _make_response("B opens"),
        _make_response("A rebuts"),
        _make_response("B rebuts"),
    ]
    env, client = _make_env(responses)
    _run(env.rollout(_rollout_input(), client, "test-model"))

    first_prompt = client.calls[0]["prompt"]
    prompt_str = " ".join(
        m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        for m in first_prompt
    )
    # System prompt rendered from DebatePrompts template
    assert "You argue for the answer" in prompt_str
    # Question template rendered with actual question text, not the answer
    assert "Question: What is 2+2?" in prompt_str


# ---------------------------------------------------------------------------
# 15. Rubric W/G/M tests (real DebatePrompts, real FieldSpec, no mocks)
# ---------------------------------------------------------------------------


SELFPLAY_PROMPTS = resolve_prompts("selfplay")
DEFAULT_PROMPTS = resolve_prompts("default")


def _rubric(
    truth_role: str = "debater_a",
    members: list[str] | None = None,
    prompts: DebatePrompts | None = None,
    judge_client: "FakeJudgeClient | None" = None,
) -> DebateRubric:
    """Build a DebateRubric for rubric tests.

    If the pack declares LLM-judge templates (_grader/_matcher) and no
    judge_client is passed, we auto-attach a FakeJudgeClient with an
    empty verdict queue. Rubric tests that exercise MCQ-scored answer
    fields short-circuit via classify_enum before reaching the LLM
    grader, so the empty queue is never consumed.
    """
    pack = prompts or SELFPLAY_PROMPTS
    if judge_client is None and pack.judges:
        judge_client = FakeJudgeClient(verdicts=[])
    return DebateRubric(
        truth_role=truth_role,
        members=members or ["A", "B"],
        prompts=pack,
        judge_client=judge_client,
        judge_model="fake-model",
    )


def _state_with_trajectory(
    trajectory: list[TrajectoryStep],
    answer: str = "C",
) -> State:
    state = State()
    state["prompt"] = [{"role": "user", "content": "What is 2+2?"}]
    state["answer"] = answer
    state["trajectory"] = trajectory
    state["reward"] = None
    state["metrics"] = None
    return state


def _debater_step(
    member_id: str,
    role_id: str,
    phase: str = "propose",
    answer: str | None = None,
) -> TrajectoryStep:
    extras: dict = {"member_id": member_id, "role_id": role_id, "phase": phase}
    if answer is not None:
        extras["fields"] = {"answer": answer}
    return TrajectoryStep(
        prompt=[], completion=[], response=_make_response("arg"),
        tokens=None, reward=None, advantage=None, is_truncated=False,
        trajectory_id="t", extras=extras,
    )


def _judge_step(decision: str) -> TrajectoryStep:
    return TrajectoryStep(
        prompt=[], completion=[], response=_make_response("verdict"),
        tokens=None, reward=None, advantage=None, is_truncated=False,
        trajectory_id="t", extras={
            "member_id": "J", "role_id": "judge", "phase": "final",
            "fields": {"decision": decision},
        },
    )


def test_w_truth_wins():
    """W: judge picks truth_role -> episode reward=1.0, per-member correct."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert state["metrics"]["reward/A"] == 1.0
    assert state["metrics"]["reward/B"] == 0.0


def test_w_truth_loses():
    """W: judge picks opponent -> episode reward=0.0, per-member B=1.0 A=0.0."""
    rubric = _rubric(truth_role="debater_a")
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_b"),
    ])
    _run(rubric.score_rollout(state))

    assert state["reward"] == 0.0
    assert state["metrics"]["reward/A"] == 0.0
    assert state["metrics"]["reward/B"] == 1.0


def test_w_tie():
    """W: judge says 'tie' -> reward=0.0 (tie != any truth_role)."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("tie"),
    ])
    _run(rubric.score_rollout(state))

    assert state["reward"] == 0.0
    # Tie matches neither debater_a nor debater_b
    assert state["metrics"]["reward/A"] == 0.0
    assert state["metrics"]["reward/B"] == 0.0


def test_w_no_judge_raises_when_judge_declared():
    """Protocol YAML declares a judge but trajectory has no judge step
    (e.g. early termination before judge ran). Must RAISE — silently
    falling through to answer-grading would install fake training signal."""
    rubric = _rubric()  # uses SELFPLAY_PROMPTS which declares judge
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
    ])
    with pytest.raises(RuntimeError, match="[Jj]udge"):
        _run(rubric.score_rollout(state))


def _open_ended_prompts() -> DebatePrompts:
    """Minimal prompt pack that declares _grader/_matcher judges AND a
    non-EnumScoring 'answer' FieldSpec for both debaters. This is the
    synthetic case that MUST trigger DebateRubric's eager judge_client
    gate: both conditions hold (judges declared, open-ended answer routes
    through the LLM path), so constructing without a judge_client must
    raise ValueError at __init__ time."""
    from verifiers.envs.debate.prompts import JudgeTemplate

    grader = JudgeTemplate(
        system="Grade: CORRECT or INCORRECT.",
        user=_je.from_string("Target: {answer}\nResponse: {response}"),
        user_format_string="Target: {answer}\nResponse: {response}",
        positive="CORRECT",
        negative="INCORRECT",
    )
    matcher = JudgeTemplate(
        system="Compare: SAME or DIFFERENT.",
        user=_je.from_string("A: {answer}\nB: {response}"),
        user_format_string="A: {answer}\nB: {response}",
        positive="SAME",
        negative="DIFFERENT",
    )
    return DebatePrompts(
        system={
            "debater_a": _je.from_string("Argue."),
            "debater_b": _je.from_string("Argue."),
        },
        user={
            "debater_a": {"propose": _je.from_string("State your answer.")},
            "debater_b": {"propose": _je.from_string("State your answer.")},
        },
        question={
            "debater_a": _je.from_string("Q"),
            "debater_b": _je.from_string("Q"),
        },
        fields={
            "debater_a": {
                "propose": {
                    # No scoring: spec exists but routes to LLM grader.
                    "answer": FieldSpec(
                        type=str,
                        description="free-text answer",
                    ),
                },
            },
            "debater_b": {
                "propose": {
                    "answer": FieldSpec(
                        type=str,
                        description="free-text answer",
                    ),
                },
            },
        },
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={"grader": grader, "matcher": matcher},
        source_ref="open-ended-test",
    )


def _judgeless_prompts() -> DebatePrompts:
    """Minimal prompt pack that declares debater answer fields but no
    judge key at all. Used to verify the G-fallback still fires for
    genuinely judgeless protocols."""
    return DebatePrompts(
        system={
            "debater_a": _je.from_string("Argue."),
            "debater_b": _je.from_string("Argue."),
        },
        user={
            "debater_a": {"propose": _je.from_string("State your answer.")},
            "debater_b": {"propose": _je.from_string("State your answer.")},
        },
        question={
            "debater_a": _je.from_string("Q"),
            "debater_b": _je.from_string("Q"),
        },
        fields={
            "debater_a": {
                "propose": {
                    "answer": FieldSpec(
                        type=str,
                        description="your answer",
                        scoring=EnumScoring(values=("A", "B", "C", "D")),
                    ),
                },
            },
            "debater_b": {
                "propose": {
                    "answer": FieldSpec(
                        type=str,
                        description="your answer",
                        scoring=EnumScoring(values=("A", "B", "C", "D")),
                    ),
                },
            },
        },
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="judgeless-test",
    )


def test_w_no_judge_fallback_for_judgeless_pack():
    """Genuinely judgeless pack (no 'judge' key in prompts.fields) →
    G-fallback via truth-role answer vs ground truth still works."""
    prompts = _judgeless_prompts()
    rubric = DebateRubric(
        truth_role="debater_a", members=["A", "B"], prompts=prompts
    )
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
    ])
    # Ground truth "C", debater_a answered "C" → reward=1.0 via G-fallback.
    _run(rubric.score_rollout(state))

    assert state["reward"] == 1.0


def test_g_accuracy_mcq():
    """G: MCQ grading via classify_enum — A correct, B wrong."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    assert state["metrics"]["accuracy/A"] == 1.0
    assert state["metrics"]["accuracy/B"] == 0.0


def test_g_missing_answer():
    """G: debater whose LATEST step has no answer field -> extraction_failed=1.0,
    accuracy/B absent (must not conflate "wrong" with "unparseable")."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b"),  # no answer field
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    assert state["metrics"]["accuracy/A"] == 1.0
    assert state["metrics"]["extraction_failed/A"] == 0.0
    assert "accuracy/B" not in state["metrics"]
    assert state["metrics"]["extraction_failed/B"] == 1.0


def test_g_stale_commit_not_graded_when_latest_fails():
    """G4 cross-boundary: if a member had a valid EARLIER step and then their
    LATEST step has no answer field (parse failure → env appends step with
    fields=None), the rubric must NOT grade the stale earlier commit as the
    final answer. Expected: accuracy/A absent, extraction_failed/A=1.0,
    num_commits/A=1 (the earlier valid commit is still in the commit sequence
    but is NOT treated as the canonical answer).

    Mirrors the worker-parsing cross-boundary bug: parse() raised → env layer
    appends the failed step with fields=None → rubric's failed_members path
    must key off the LATEST step, not the last-known-good one."""
    rubric = _rubric()
    # phases: propose (valid commit) → critique (parse failure, fields=None).
    # Truth is C; A's stale commit is B → if graded, would give accuracy/A=0.0.
    # With the fix, accuracy/A must be ABSENT entirely.
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", phase="propose", answer="B"),  # stale
        _debater_step("B", "debater_b", phase="propose", answer="D"),
        _debater_step("A", "debater_a", phase="critique"),  # LATEST, no fields
        _debater_step("B", "debater_b", phase="critique", answer="D"),
        _judge_step("debater_a"),
    ], answer="C")
    _run(rubric.score_rollout(state))

    metrics = state["metrics"]
    assert "accuracy/A" not in metrics, (
        f"stale 'propose' commit 'B' must not be graded; "
        f"got accuracy/A={metrics.get('accuracy/A')}"
    )
    assert metrics["extraction_failed/A"] == 1.0
    # B is parseable on its latest step → accuracy fires normally (D != C)
    assert metrics["accuracy/B"] == 0.0
    assert metrics["extraction_failed/B"] == 0.0
    # Commit sequence still contains A's valid earlier commit for diagnostics,
    # but the flip/final_correct metrics must be skipped since the latest
    # step is unparseable (the rubric's rubric.py:313-315 guard).
    assert state["commits"]["A"] == ["B"]
    assert "final_correct/A" not in metrics, (
        "flip diagnostics must be skipped when latest step is unparseable"
    )


def test_m_agreement_same():
    """M: both debaters answer 'C' -> agreement=1.0."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="C"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    assert state["metrics"]["agreement"] == 1.0


def test_m_agreement_different():
    """M: A answers C, B answers B -> agreement=0.0."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    assert state["metrics"]["agreement"] == 0.0


class FakeJudgeClient(_VFClient):
    """Concrete vf.Client subclass that returns canned verdicts.

    Subclasses `verifiers.clients.Client` so `isinstance(x, Client)` passes
    DebateRubric's F-B Part 1 boundary check. We override `get_response`
    directly (the public high-level entry point) and stub out the native-
    layer abstract methods with NotImplementedError — they are never called
    because our override bypasses the native conversion path.

    NOT a MagicMock. The queue of canned verdicts is explicit and the
    `.calls` log supports per-call assertion. Can also be configured to
    raise a specific exception per call (for propagation tests).
    """

    def __init__(
        self,
        verdicts: list[str],
        *,
        usage: Usage | None = None,
        exc_to_raise: Exception | None = None,
    ) -> None:
        import logging as _logging_mod

        # Skip vf.Client.__init__ (which wants a concrete SDK client or
        # ClientConfig). We don't need either — get_response is overridden.
        # Keep shape-compatible fields in case introspection touches them.
        self.logger = _logging_mod.getLogger(f"{__name__}.FakeJudgeClient")
        self._config = None
        self._client = None
        self._verdicts: list[str] = list(verdicts)
        self._usage = usage
        self._exc_to_raise = exc_to_raise
        self.calls: list[dict] = []

    async def get_response(
        self,
        prompt,
        model,
        sampling_args,
        tools=None,
        **kwargs,
    ) -> Response:
        self.calls.append(
            {"prompt": prompt, "model": model, "sampling_args": sampling_args}
        )
        if self._exc_to_raise is not None:
            raise self._exc_to_raise
        if not self._verdicts:
            raise RuntimeError(
                "FakeJudgeClient exhausted — more calls than expected"
            )
        content = self._verdicts.pop(0)
        return Response(
            id="fake-judge",
            created=0,
            model=model,
            usage=self._usage,
            message=ResponseMessage(
                content=content,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            ),
        )

    # The vf.Client native-layer hooks are never reached because we
    # override `get_response` directly. Keep them as loud NotImplemented
    # so any accidental native-path call fails fast during tests.
    def setup_client(self, config):
        raise NotImplementedError("FakeJudgeClient has no native SDK client")

    async def to_native_tool(self, tool):
        raise NotImplementedError("FakeJudgeClient does not support tool calls")

    async def to_native_prompt(self, messages):
        raise NotImplementedError("FakeJudgeClient bypasses native conversion")

    async def get_native_response(self, *args, **kwargs):
        raise NotImplementedError("FakeJudgeClient overrides get_response directly")

    async def raise_from_native_response(self, response):
        raise NotImplementedError("FakeJudgeClient overrides get_response directly")

    async def from_native_response(self, response):
        raise NotImplementedError("FakeJudgeClient overrides get_response directly")

    async def close(self) -> None:
        return None


def _open_ended_rubric(
    verdicts: list[str],
    *,
    usage: Usage | None = None,
    exc_to_raise: Exception | None = None,
) -> tuple[DebateRubric, FakeJudgeClient]:
    """Build a DebateRubric with a FakeJudgeClient canned with the given
    verdicts. Uses the default pack (which has _grader/_matcher blocks
    shipped with CORRECT/INCORRECT and SAME/DIFFERENT tokens)."""
    client = FakeJudgeClient(verdicts, usage=usage, exc_to_raise=exc_to_raise)
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B"],
        prompts=DEFAULT_PROMPTS,
        judge_client=client,
        judge_model="fake-model",
    )
    return rubric, client


def test_grade_recognizes_positive_verdict():
    """Grader returns exact positive token 'CORRECT' -> _grade returns True."""
    rubric, _ = _open_ended_rubric(verdicts=["CORRECT"])
    state = _state_with_trajectory([])
    result = _run(rubric._grade("some answer", "target", "Q", spec=None, state=state))
    assert result is True


def test_grade_recognizes_negative_verdict_not_substring():
    """REGRESSION GUARD for the 'correct' in 'incorrect' == True bug.

    Before the fix, verdict 'INCORRECT' was classified as positive because
    'correct'.lower() appears inside 'incorrect'.lower(). After the fix,
    _resolve_verdict does exact-token match on the normalized first word,
    so 'INCORRECT' is recognized as the negative token and _grade returns
    False.
    """
    rubric, _ = _open_ended_rubric(verdicts=["INCORRECT"])
    state = _state_with_trajectory([])
    result = _run(rubric._grade("some answer", "target", "Q", spec=None, state=state))
    assert result is False


def test_grade_recognizes_verdict_with_trailing_punctuation():
    """'CORRECT.' normalizes to 'CORRECT' via _normalize_verdict_token."""
    rubric, _ = _open_ended_rubric(verdicts=["CORRECT."])
    state = _state_with_trajectory([])
    assert _run(rubric._grade("a", "t", "Q", spec=None, state=state)) is True


def test_grade_raises_on_unknown_verdict():
    """Grader returns a non-token verdict -> RuntimeError (no silent False)."""
    rubric, _ = _open_ended_rubric(verdicts=["maybe"])
    state = _state_with_trajectory([])
    with pytest.raises(RuntimeError, match="unrecognized verdict token"):
        _run(rubric._grade("a", "t", "Q", spec=None, state=state))


def test_grade_raises_on_empty_verdict():
    """Empty string verdict -> normalizes to None -> RuntimeError."""
    rubric, _ = _open_ended_rubric(verdicts=[""])
    state = _state_with_trajectory([])
    with pytest.raises(RuntimeError, match="unrecognized verdict token"):
        _run(rubric._grade("a", "t", "Q", spec=None, state=state))


def test_match_recognizes_positive_verdict():
    """Matcher returns 'SAME' -> _match returns True."""
    rubric, _ = _open_ended_rubric(verdicts=["SAME"])
    state = _state_with_trajectory([])
    result = _run(rubric._match("a", "b", "Q", spec=None, state=state))
    assert result is True


def test_match_recognizes_negative_verdict_not_substring():
    """Matcher returns 'DIFFERENT' -> _match returns False. Symmetric
    guard to the grader substring regression."""
    rubric, _ = _open_ended_rubric(verdicts=["DIFFERENT"])
    state = _state_with_trajectory([])
    result = _run(rubric._match("a", "b", "Q", spec=None, state=state))
    assert result is False


def test_match_raises_on_unknown_verdict():
    """Matcher returns a non-token verdict -> RuntimeError."""
    rubric, _ = _open_ended_rubric(verdicts=["probably"])
    state = _state_with_trajectory([])
    with pytest.raises(RuntimeError, match="unrecognized verdict token"):
        _run(rubric._match("a", "b", "Q", spec=None, state=state))


def test_default_pack_does_not_leak_ground_truth():
    """REGRESSION GUARD for Finding 2: default.yaml question template
    no longer renders '{{ answer }}' (ground truth) into the debater's
    question block. Previous default.yaml leaked GT as 'Your assigned
    position: <answer>' which collapses debate to echo-rhetoric."""
    prompts = resolve_prompts("default")
    ctx = build_context(
        task_prompt="What is 2+2?",
        viewer_role="debater_a",
        phase="propose",
        round_index=0,
        num_rounds=1,
        answer="UNIQUE_GROUND_TRUTH_TOKEN_42",
    )
    rendered_a = prompts.render_question("debater_a", ctx)
    rendered_b = prompts.render_question("debater_b", ctx)
    assert rendered_a == "What is 2+2?"
    assert rendered_b == "What is 2+2?"
    assert "UNIQUE_GROUND_TRUTH_TOKEN_42" not in rendered_a
    assert "UNIQUE_GROUND_TRUTH_TOKEN_42" not in rendered_b


def test_missing_fields_graceful():
    """Empty trajectory against a judgeless pack -> no crash, reward=0.0,
    no accuracy/agreement metrics. (A judge-declared pack with an empty
    trajectory would correctly raise under the G7.3 contract — that path
    is covered by test_w_no_judge_raises_when_judge_declared.)"""
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B"],
        prompts=_judgeless_prompts(),
    )
    state = _state_with_trajectory([])
    _run(rubric.score_rollout(state))

    assert state["reward"] == 0.0
    assert "accuracy/A" not in state["metrics"]
    assert "agreement" not in state["metrics"]


def test_yaml_without_answer_fields():
    """default.yaml has no debater answer fields -> G/M skipped, W still works."""
    rubric = _rubric(prompts=DEFAULT_PROMPTS)
    state = _state_with_trajectory([
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("arg"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={"member_id": "A", "role_id": "debater_a", "phase": "propose"},
        ),
        TrajectoryStep(
            prompt=[], completion=[], response=_make_response("arg"),
            tokens=None, reward=None, advantage=None, is_truncated=False,
            trajectory_id="t", extras={"member_id": "B", "role_id": "debater_b", "phase": "propose"},
        ),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    # W works: judge picked truth_role
    assert state["reward"] == 1.0
    # G/M skipped: no answer fields extracted, no answer_specs
    assert "accuracy/A" not in state["metrics"]
    assert "agreement" not in state["metrics"]


# ---------------------------------------------------------------------------
# Flip diagnostics (initial/final/num_commits/num_unique_commits)
# ---------------------------------------------------------------------------


def test_flip_hold_correct():
    """Policy commits correct answer at propose and holds at critique:
    num_commits=2, num_unique=1, initial_correct=1, final_correct=1."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", phase="propose", answer="C"),
        _debater_step("B", "debater_b", phase="propose", answer="B"),
        _debater_step("A", "debater_a", phase="critique", answer="C"),
        _debater_step("B", "debater_b", phase="critique", answer="B"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    m = state["metrics"]
    assert m["num_commits/A"] == 2.0
    assert m["num_unique_commits/A"] == 1.0
    assert m["initial_correct/A"] == 1.0
    assert m["final_correct/A"] == 1.0
    assert state["commits"]["A"] == ["C", "C"]


def test_flip_hold_incorrect():
    """Policy commits wrong answer and holds: both correctness = 0, unique = 1."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", phase="propose", answer="B"),
        _debater_step("B", "debater_b", phase="propose", answer="C"),
        _debater_step("A", "debater_a", phase="critique", answer="B"),
        _debater_step("B", "debater_b", phase="critique", answer="C"),
        _judge_step("debater_b"),
    ])
    _run(rubric.score_rollout(state))

    m = state["metrics"]
    assert m["num_commits/A"] == 2.0
    assert m["num_unique_commits/A"] == 1.0
    assert m["initial_correct/A"] == 0.0
    assert m["final_correct/A"] == 0.0


def test_flip_earned():
    """Policy commits wrong, then updates to correct (earned flip)."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", phase="propose", answer="B"),     # wrong
        _debater_step("B", "debater_b", phase="propose", answer="C"),
        _debater_step("A", "debater_a", phase="critique", answer="C"),    # updated → right
        _debater_step("B", "debater_b", phase="critique", answer="C"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    m = state["metrics"]
    assert m["num_commits/A"] == 2.0
    assert m["num_unique_commits/A"] == 2.0
    assert m["initial_correct/A"] == 0.0  # wrong at propose
    assert m["final_correct/A"] == 1.0    # right at critique
    assert state["commits"]["A"] == ["B", "C"]


def test_flip_unearned():
    """Policy commits correct, then capitulates to wrong (unearned flip)."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", phase="propose", answer="C"),     # right
        _debater_step("B", "debater_b", phase="propose", answer="B"),
        _debater_step("A", "debater_a", phase="critique", answer="B"),    # capitulated
        _debater_step("B", "debater_b", phase="critique", answer="B"),
        _judge_step("debater_b"),
    ])
    _run(rubric.score_rollout(state))

    m = state["metrics"]
    assert m["num_commits/A"] == 2.0
    assert m["num_unique_commits/A"] == 2.0
    assert m["initial_correct/A"] == 1.0
    assert m["final_correct/A"] == 0.0
    assert state["commits"]["A"] == ["C", "B"]


def test_flip_wobble_correct():
    """Three-round: right → wrong → right. num_unique=2, both endpoints correct."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", phase="propose", answer="C"),
        _debater_step("B", "debater_b", phase="propose", answer="B"),
        _debater_step("A", "debater_a", phase="critique", answer="B"),    # wobble out
        _debater_step("B", "debater_b", phase="critique", answer="B"),
        _debater_step("A", "debater_a", phase="critique", answer="C"),    # wobble back
        _debater_step("B", "debater_b", phase="critique", answer="B"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))

    m = state["metrics"]
    assert m["num_commits/A"] == 3.0
    assert m["num_unique_commits/A"] == 2.0
    assert m["initial_correct/A"] == 1.0
    assert m["final_correct/A"] == 1.0
    assert state["commits"]["A"] == ["C", "B", "C"]


def test_flip_empty_trajectory_emits_zero_commits():
    """Empty trajectory against a judgeless pack -> num_commits/A = 0,
    no initial/final_correct (nothing to grade), state['commits'] is
    an empty dict. (Uses a judgeless pack so the W contract does not
    raise on empty-traj; flip-diag behavior is independent of judging.)"""
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B"],
        prompts=_judgeless_prompts(),
    )
    state = _state_with_trajectory([])
    _run(rubric.score_rollout(state))

    m = state["metrics"]
    assert m["num_commits/A"] == 0.0
    assert m["num_commits/B"] == 0.0
    assert m["num_unique_commits/A"] == 0.0
    assert m["num_unique_commits/B"] == 0.0
    # No commits -> no initial/final correctness metric
    assert "initial_correct/A" not in m
    assert "final_correct/A" not in m
    assert state["commits"] == {}


# ---------------------------------------------------------------------------
# F-B: eager judge_client validation (fail at load time, not score time)
# ---------------------------------------------------------------------------


_SCHEDULE_SLOTS = [
    {"slot_id": 0, "actors": ["A", "B"], "phase": "propose"},
    {"slot_id": 1, "actors": ["J"], "phase": "final"},
]


def test_debate_rubric_init_raises_when_open_ended_without_judge_client():
    """Direct DebateRubric construction with a genuinely open-ended pack
    (_grader/_matcher declared AND at least one 'answer' FieldSpec whose
    scoring is non-enum → routes to the LLM grader path) and no
    judge_client must raise ValueError at __init__ time. We use a
    synthetic pack because the real default/selfplay packs either score
    via classify_enum (selfplay) or don't declare debater answer specs at
    all (default) — neither triggers the updated gate."""
    with pytest.raises(ValueError, match="judge_client"):
        DebateRubric(
            truth_role="debater_a",
            members=["A", "B"],
            prompts=_open_ended_prompts(),
            judge_client=None,
        )


def test_debate_rubric_init_accepts_selfplay_pack_without_judge_client():
    """Golden regression for Finding 1: selfplay.yaml declares _grader /
    _matcher templates as dead-code fallback, but every 'answer' field is
    EnumScoring → _grade takes the classify_enum fast path and never calls
    the LLM grader. Constructing a rubric against this pack WITHOUT a
    judge_client must succeed. Previously the eager gate raised
    unnecessarily, breaking the built-in pack."""
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B", "J"],
        prompts=SELFPLAY_PROMPTS,
        judge_client=None,
    )
    assert rubric.judge_client is None
    # And the rubric is functional: a clean rollout scores without touching
    # the LLM grader. If the fast path weren't hit, _grade would trip the
    # "Judge client not configured" runtime error.
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ])
    _run(rubric.score_rollout(state))
    assert state["reward"] == 1.0
    assert state["metrics"]["accuracy/A"] == 1.0


def test_debate_rubric_init_ok_when_judgeless_pack_has_no_judge_client():
    """Construction with a pack that declares NO judge templates must
    succeed with judge_client=None. Judgeless contracts (e.g. MCQ-only
    evals) don't need an LLM grader."""
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B"],
        prompts=_judgeless_prompts(),
        judge_client=None,
    )
    assert rubric.judge_client is None


def test_load_environment_raises_when_open_ended_pack_has_no_judge_client():
    """load_environment factory must eagerly validate judge_client against
    the combination of prompts.judges AND at least one 'answer' FieldSpec
    whose scoring is non-enum. We pass the synthetic open-ended pack
    directly so the gate fires at the load_environment mirror site."""
    with pytest.raises(ValueError, match="judge_client"):
        load_environment(
            schedule_slots=_SCHEDULE_SLOTS,
            members=["A", "B", "J"],
            truth_role="debater_a",
            prompts=_open_ended_prompts(),
            role_for_actor={"A": "debater_a", "B": "debater_b", "J": "judge"},
            eval_dataset=lambda: None,
        )


def test_load_environment_accepts_selfplay_pack_without_judge_client():
    """Golden regression mirror for Finding 1 at the load_environment site:
    selfplay ships _grader/_matcher as dead fallback but scores all answer
    fields via classify_enum. The eager gate must NOT fire and the factory
    must return a functional env."""
    env = load_environment(
        schedule_slots=_SCHEDULE_SLOTS,
        members=["A", "B", "J"],
        truth_role="debater_a",
        prompts_ref="selfplay",
        role_for_actor={"A": "debater_a", "B": "debater_b", "J": "judge"},
        eval_dataset=lambda: None,
    )
    assert env.rubric.judge_client is None


def test_load_environment_ok_when_pack_has_no_judge_templates():
    """load_environment with an inline judgeless DebatePrompts must
    construct cleanly without a judge_client."""
    env = load_environment(
        schedule_slots=_SCHEDULE_SLOTS,
        members=["A", "B", "J"],
        truth_role="debater_a",
        prompts=_judgeless_prompts(),
        role_for_actor={"A": "debater_a", "B": "debater_b", "J": "judge"},
        eval_dataset=lambda: None,
    )
    assert env.rubric.judge_client is None


def test_load_environment_ok_when_judge_client_provided():
    """load_environment with an open-ended pack AND a wired judge_client
    constructs successfully — the validation fires ONLY on the missing
    judge_client path."""
    client = FakeJudgeClient(verdicts=[])
    env = load_environment(
        schedule_slots=_SCHEDULE_SLOTS,
        members=["A", "B", "J"],
        truth_role="debater_a",
        prompts_ref="selfplay",
        role_for_actor={"A": "debater_a", "B": "debater_b", "J": "judge"},
        judge_client=client,
        judge_model="fake-model",
        eval_dataset=lambda: None,
    )
    assert env.rubric.judge_client is client


# ---------------------------------------------------------------------------
# F2: errored-rollout short-circuit (don't crash score_group's asyncio.gather)
# ---------------------------------------------------------------------------


def test_rubric_short_circuits_on_prompt_too_long():
    """When rollout loop set state['prompt_too_long']=True, score_rollout
    must NOT run the G7.3 invariant (which would raise RuntimeError on a
    truncated trajectory without a judge step) and must NOT crash
    asyncio.gather in score_group. Instead, emit reward=0.0 with
    errored_rollout=1.0 and error_type='prompt_too_long'."""
    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        # No B step, no judge step — rollout was truncated early.
    ])
    state["prompt_too_long"] = True
    _run(rubric.score_rollout(state))

    assert state["reward"] == 0.0
    assert state["metrics"] == {"errored_rollout": 1.0}
    assert state["error_info"] == {
        "error_type": "prompt_too_long",
        "error_phase": "rollout",
    }
    assert state["commits"] == {}


def test_rubric_short_circuits_on_state_error():
    """When rollout loop caught a vf.Error and stashed it as state['error'],
    score_rollout must short-circuit and emit error_type=<exception class
    name>. The G7.3 invariant is preserved for healthy rollouts — only
    already-errored rollouts take this path."""

    class _SimulatedEnvError(Exception):
        pass

    rubric = _rubric()
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
    ])
    state["error"] = _SimulatedEnvError("boom")
    _run(rubric.score_rollout(state))

    assert state["reward"] == 0.0
    assert state["metrics"] == {"errored_rollout": 1.0}
    assert state["error_info"] == {
        "error_type": "_SimulatedEnvError",
        "error_phase": "rollout",
    }
    assert state["commits"] == {}


def test_rubric_does_not_short_circuit_on_clean_rollout():
    """A rollout with no error flags and a judges-declaring pack but no
    judge step in trajectory STILL raises per the G7.3 invariant. The
    F2 short-circuit is strictly gated on state.error/prompt_too_long."""
    rubric = _rubric()  # SELFPLAY_PROMPTS declares judge
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        # No judge step, no error flags — this is a "normal" rollout
        # where the judge should have run but didn't. G7.3 must fire.
    ])
    with pytest.raises(RuntimeError, match="[Jj]udge"):
        _run(rubric.score_rollout(state))


def test_rubric_short_circuit_does_not_propagate_through_gather():
    """score_group runs score_rollout for N states via asyncio.gather.
    Before F2, one errored rollout would raise RuntimeError from its
    score_rollout and kill the whole batch. After F2, errored rollouts
    short-circuit cleanly and the batch completes."""
    rubric = _rubric()

    # Mix: one errored, one clean-with-judge.
    errored_state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
    ])
    errored_state["prompt_too_long"] = True

    clean_state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ])

    _run(rubric.score_group([errored_state, clean_state]))

    # Errored rollout: short-circuited.
    assert errored_state["reward"] == 0.0
    assert errored_state["metrics"]["errored_rollout"] == 1.0
    # Clean rollout: scored normally.
    assert clean_state["reward"] == 1.0
    assert clean_state["metrics"]["reward/A"] == 1.0


# ---------------------------------------------------------------------------
# Round 7: scoring-time vf.Error must be captured on state["error"] so the
# maybe_retry chain in verifiers/utils/async_utils.py can find it via
# reraise_error_from_state and trigger tenacity retry on retryable subclasses
# (vf.InfraError, vf.InvalidModelResponseError). The previous round-6 fix
# left state["error"] unset and silently suppressed retries.
#
# score_group itself must NOT catch anything — it uses bare gather. Schema
# violations (KeyError) and programming bugs propagate loud. vf.Error
# isolation lives at the score_rollout layer.
# ---------------------------------------------------------------------------


def _open_ended_rubric_with_state(
    client: _VFClient,
) -> tuple[DebateRubric, State]:
    """Build a DebateRubric against the synthetic open-ended pack and a
    state whose scoring will exercise _grade through the LLM path EXACTLY
    ONCE per attempt. The state has:
      - A judge step (winning_role set → W path skips the grader)
      - One debater_a step with an answer (G path fires once)
      - No debater_b answer (M path requires 2+ debater answers → skipped)
    So the grader is called exactly once per score_rollout invocation,
    which makes call-count assertions in retry tests deterministic."""
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B", "J"],
        prompts=_open_ended_prompts(),
        judge_client=client,
        judge_model="test-model",
    )
    state = _state_with_trajectory(
        [
            _debater_step("A", "debater_a", answer="foo"),
            _judge_step("debater_a"),
        ],
        answer="baz",
    )
    return rubric, state


def _open_ended_state_for_retry() -> State:
    """Fresh test state for retry-loop tests. Same shape as the one in
    _open_ended_rubric_with_state, but constructed inside an attempt so
    each retry gets a clean state object."""
    return _state_with_trajectory(
        [
            _debater_step("A", "debater_a", answer="foo"),
            _judge_step("debater_a"),
        ],
        answer="baz",
    )


def test_score_rollout_stores_vf_error_for_retry_discovery():
    """Round 7 Finding 1: a scoring-time vf.InvalidModelResponseError must
    be captured onto state['error'] so that maybe_retry can find it via
    reraise_error_from_state and trigger tenacity retry. Round 6 left
    state['error'] unset and only wrote metrics, which is invisible to the
    retry layer."""
    client = FakeJudgeClient(
        verdicts=[],
        exc_to_raise=vf.InvalidModelResponseError("transient judge outage"),
    )
    rubric, state = _open_ended_rubric_with_state(client)

    _run(rubric.score_rollout(state))

    err = state.get("error")
    assert err is not None, "state['error'] must be set so maybe_retry can find it"
    assert isinstance(err, vf.InvalidModelResponseError)
    assert state["reward"] == 0.0
    assert state["metrics"]["errored_rollout"] == 1.0
    assert state["error_info"]["error_type"] == "InvalidModelResponseError"
    assert state["error_info"]["error_phase"] == "scoring"
    assert state["commits"] == {}


def test_score_rollout_stores_non_retryable_vf_error_too():
    """Round 7 Finding 1: vf.Error subclasses that are NOT in maybe_retry's
    retryable tuple (e.g. a custom vf.Error subclass, or vf.OverlongPromptError)
    must STILL land on state['error']. The retry decision belongs to the
    caller (maybe_retry walks state['error'] and matches against its
    error_types tuple) — score_rollout's job is to record, not to filter."""

    class _PermanentTestError(vf.Error):
        pass

    client = FakeJudgeClient(
        verdicts=[],
        exc_to_raise=_PermanentTestError("not retryable"),
    )
    rubric, state = _open_ended_rubric_with_state(client)

    _run(rubric.score_rollout(state))

    err = state.get("error")
    assert isinstance(err, _PermanentTestError)
    assert state["metrics"]["errored_rollout"] == 1.0
    # Only one call — score_rollout itself does not retry, it just records.
    assert len(client.calls) == 1


def test_score_group_propagates_key_error_for_missing_answer():
    """Round 7 Finding 2: missing state['answer'] is the explicit dataset
    schema fail-fast contract in score_rollout (lines 377-381). The
    KeyError it raises MUST propagate through score_group — round 6's
    return_exceptions=True belt-and-braces silently swallowed it. Bare
    gather must let it through."""
    rubric = _rubric()  # selfplay pack
    state = State()
    state["prompt"] = [{"role": "user", "content": "Q?"}]
    # Intentionally NO state["answer"] set.
    state["trajectory"] = [
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ]
    state["reward"] = None
    state["metrics"] = None

    with pytest.raises(KeyError, match="state is missing 'answer'"):
        _run(rubric.score_group([state]))


def test_score_group_propagates_programming_bug_attribute_error():
    """Round 7 Finding 2: non-vf.Error exceptions (programming bugs) must
    propagate loud through score_group. We swap the rubric's outcome_fn
    for a buggy callable that raises AttributeError, then score a normal
    selfplay state. The error must escape gather."""
    rubric = _rubric()

    def buggy_outcome(_state: State) -> str | None:
        raise AttributeError("simulated programming bug in outcome_fn")

    rubric.outcome_fn = buggy_outcome  # type: ignore[method-assign]
    state = _state_with_trajectory([
        _debater_step("A", "debater_a", answer="C"),
        _debater_step("B", "debater_b", answer="B"),
        _judge_step("debater_a"),
    ])

    with pytest.raises(AttributeError, match="simulated programming bug"):
        _run(rubric.score_group([state]))


def test_score_rollout_state_error_is_visible_to_reraise_contract():
    """Round 7 Finding 1, contract test: after score_rollout records a
    retryable vf.InvalidModelResponseError on state['error'], the error
    must have the shape maybe_retry's nested reraise_error_from_state
    walks for — an instance of a retryable type on state['error']. If
    the shape were wrong (state['error'] unset, or a non-matching type),
    the retry chain would silently no-op and no retry would fire."""
    client = FakeJudgeClient(
        verdicts=[],
        exc_to_raise=vf.InvalidModelResponseError("transient"),
    )
    rubric, state = _open_ended_rubric_with_state(client)

    _run(rubric.score_rollout(state))

    # maybe_retry's default retryable tuple.
    retryable = (vf.InfraError, vf.InvalidModelResponseError)
    err = state.get("error")
    assert err is not None, "state['error'] must be set for retry discovery"
    assert isinstance(err, retryable), (
        f"state['error'] must be a retryable type; got {type(err).__name__}"
    )
    assert str(err) == "transient"


class _FlakeyClient(_VFClient):
    """Client that raises InvalidModelResponseError on the first `fail_n`
    calls, then returns 'CORRECT' thereafter. Used for retry-loop tests."""

    def __init__(self, fail_n: int) -> None:
        import logging as _logging_mod

        self.logger = _logging_mod.getLogger(f"{__name__}._FlakeyClient")
        self._config = None
        self._client = None
        self._fail_n = fail_n
        self.call_count = 0
        self.calls: list[dict] = []

    async def get_response(
        self, prompt, model, sampling_args, tools=None, **kwargs
    ) -> Response:
        self.call_count += 1
        self.calls.append({"prompt": prompt, "model": model})
        if self.call_count <= self._fail_n:
            raise vf.InvalidModelResponseError(f"transient #{self.call_count}")
        return Response(
            id="ok",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content="CORRECT",
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            ),
        )

    def setup_client(self, config):
        raise NotImplementedError

    async def to_native_tool(self, tool):
        raise NotImplementedError

    async def to_native_prompt(self, messages):
        raise NotImplementedError

    async def get_native_response(self, *args, **kwargs):
        raise NotImplementedError

    async def raise_from_native_response(self, response):
        raise NotImplementedError

    async def from_native_response(self, response):
        raise NotImplementedError

    async def close(self) -> None:
        return None


@pytest.fixture
def _instant_retry_waits(monkeypatch):
    """Pin tenacity's retry wait to zero so retry-loop tests don't sleep.

    `maybe_retry` hard-codes `tc.wait_exponential_jitter(initial=initial,
    max=max_wait)` inside its AsyncRetrying constructor. Passing
    `initial=0.0` still uses jitter (0-1s per retry by default), so the
    only clean way to kill the wait is to patch `wait_exponential_jitter`
    itself to return `tc.wait_none()`. Scoped to the test via monkeypatch.
    """
    import tenacity as tc

    monkeypatch.setattr(
        tc, "wait_exponential_jitter", lambda **_: tc.wait_none()
    )


def test_score_rollout_retry_loop_end_to_end(_instant_retry_waits):
    """Round 7 Finding 1, end-to-end: drive score_group through the REAL
    `maybe_retry` from verifiers.utils.async_utils. Confirms 2 transient
    failures + 1 success = exactly 3 grader calls and the final state is
    clean (no error, reward=1.0, accuracy/A=1.0)."""
    client = _FlakeyClient(fail_n=2)
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B", "J"],
        prompts=_open_ended_prompts(),
        judge_client=client,
        judge_model="test-model",
    )

    async def attempt() -> list[State]:
        state = _open_ended_state_for_retry()
        await rubric.score_group([state])
        return [state]

    wrapped = maybe_retry(attempt, max_retries=3)
    results = _run(wrapped())

    # 2 transient failures + 1 success = 3 grader calls.
    assert client.call_count == 3, (
        f"expected 3 grader calls (fail, fail, succeed), got {client.call_count}"
    )
    assert len(results) == 1
    final = results[0]
    assert final.get("error") is None
    metrics = final.get("metrics") or {}
    assert metrics.get("errored_rollout", 0.0) == 0.0
    # Judge picked debater_a, the truth role → W path → reward=1.0.
    assert final["reward"] == 1.0
    # G-path grader returned CORRECT for the truth member's answer.
    assert metrics.get("accuracy/A") == 1.0


def test_score_rollout_retry_loop_exhausts_retries_on_permanent_failure(
    _instant_retry_waits,
):
    """Round 7 Finding 1: when every attempt fails, `maybe_retry` exhausts
    retries and returns the last result via `return_last_result`. The
    terminal state must carry both state['error'] AND the metrics shape
    downstream consumers filter on (errored_rollout=1.0,
    error_phase='scoring')."""
    client = FakeJudgeClient(
        verdicts=[],
        exc_to_raise=vf.InvalidModelResponseError("permanent flake"),
    )
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B", "J"],
        prompts=_open_ended_prompts(),
        judge_client=client,
        judge_model="test-model",
    )

    async def attempt() -> list[State]:
        state = _open_ended_state_for_retry()
        await rubric.score_group([state])
        return [state]

    wrapped = maybe_retry(attempt, max_retries=2)
    results = _run(wrapped())

    # max_retries=2 → 3 total attempts → 3 grader calls (each fails).
    assert len(client.calls) == 3
    assert len(results) == 1
    final = results[0]
    err = final.get("error")
    assert isinstance(err, vf.InvalidModelResponseError)
    assert final["metrics"]["errored_rollout"] == 1.0
    assert final["error_info"]["error_phase"] == "scoring"
    assert final["error_info"]["error_type"] == "InvalidModelResponseError"


# ---------------------------------------------------------------------------
# Phase 1E v7 regression tests: the composition contract.
#
# DebateRubric composes two JudgeRubric instances (grader + matcher) via
# HybridMathRubric pattern. These tests enforce the invariants that would
# regress if someone re-introduced the _call_judge side-channel or the
# judge_usage string-in-float-dict mistake.
# ---------------------------------------------------------------------------


def test_debate_rubric_composes_grader_and_matcher():
    """v7 composition contract: when the pack declares both _grader and
    _matcher templates AND a judge_client is passed, DebateRubric
    instantiates BOTH as JudgeRubric attributes (not lazy singletons,
    not private _call_judge wrappers). judge_prompt on each child must
    carry the YAML's user_format_string verbatim."""
    client = FakeJudgeClient(verdicts=[])
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B"],
        prompts=_open_ended_prompts(),
        judge_client=client,
        judge_model="test-model",
    )
    assert rubric.grader_rubric is not None
    assert rubric.matcher_rubric is not None
    assert rubric.grader_rubric.judge_prompt == (
        "Target: {answer}\nResponse: {response}"
    )
    assert rubric.matcher_rubric.judge_prompt == (
        "A: {answer}\nB: {response}"
    )
    # Both must share the SAME judge_client instance — a round of
    # premature optimization that stored a second copy would double
    # the connection pool.
    assert rubric.grader_rubric.judge_client is client
    assert rubric.matcher_rubric.judge_client is client


def test_score_rollout_uses_composed_grader():
    """v7 score path: a single open-ended grade call routes through
    grader_rubric.judge (not _call_judge). The FakeJudgeClient sees
    exactly one get_response invocation and the verdict flows into
    accuracy/A via _resolve_verdict."""
    client = FakeJudgeClient(verdicts=["CORRECT"])
    rubric, state = _open_ended_rubric_with_state(client)
    _run(rubric.score_rollout(state))

    # Exactly one grader call — the rollout has one debater_a answer
    # and the judge step carries the W-path verdict (so G fires only
    # for the per-member accuracy branch on debater_a).
    assert len(client.calls) == 1
    # accuracy/A=1.0 because the FakeJudgeClient canned "CORRECT".
    assert state["metrics"]["accuracy/A"] == 1.0
    assert state["metrics"]["extraction_failed/A"] == 0.0


def test_score_rollout_uses_composed_matcher_for_open_ended():
    """v7 matcher contract: matcher is reformulated as an asymmetric
    grader (A in `answer` kwarg, B in completion.content). When two
    debaters commit open-ended answers, matcher_rubric.judge fires
    once with A as the pseudo-target and B as the response, and the
    verdict drives the agreement metric."""
    # Two grader calls + one matcher call per rollout:
    #   - accuracy/A (G branch, open-ended → grader)
    #   - accuracy/B (G branch, open-ended → grader)
    #   - agreement (M branch, open-ended → matcher)
    client = FakeJudgeClient(verdicts=["CORRECT", "CORRECT", "SAME"])
    rubric = DebateRubric(
        truth_role="debater_a",
        members=["A", "B", "J"],
        prompts=_open_ended_prompts(),
        judge_client=client,
        judge_model="test-model",
    )
    state = _state_with_trajectory(
        [
            _debater_step("A", "debater_a", answer="forty-two"),
            _debater_step("B", "debater_b", answer="42"),
            _judge_step("debater_a"),
        ],
        answer="42",
    )
    _run(rubric.score_rollout(state))

    # Three judge calls: two graders (A, B) + one matcher (A vs B).
    assert len(client.calls) == 3
    assert state["metrics"]["agreement"] == 1.0


def test_score_rollout_captures_vf_error_from_grader():
    """v7 silent-0 prevention (CRITICAL): when the composed grader raises
    a vf.Error, score_rollout must set state["error"] AND state["metrics"]
    AND state["error_info"] with the correct taxonomy so maybe_retry can
    discover the error via reraise_error_from_state. This is the round-7
    fix that the whole v7 cutover was built to enforce. If grader_rubric
    stops propagating the exception OR score_rollout stops writing
    state["error"], maybe_retry silent-0s and the retry chain is broken."""
    client = FakeJudgeClient(
        verdicts=[],
        exc_to_raise=vf.InvalidModelResponseError("grader backend blew up"),
    )
    rubric, state = _open_ended_rubric_with_state(client)
    _run(rubric.score_rollout(state))

    err = state.get("error")
    assert isinstance(err, vf.InvalidModelResponseError), (
        "score_rollout must capture vf.Error onto state['error'] so "
        "maybe_retry.reraise_error_from_state can discover it."
    )
    assert state["reward"] == 0.0
    assert state["metrics"]["errored_rollout"] == 1.0
    assert state["error_info"]["error_type"] == "InvalidModelResponseError"
    assert state["error_info"]["error_phase"] == "scoring"
    # Exactly one call — score_rollout records, it does not retry.
    assert len(client.calls) == 1


def test_debate_rubric_rejects_open_ended_without_judge_client():
    """v7 eager fail-loud: when the pack declares LLM-judge templates
    AND has at least one answer field routing through the LLM grader
    path, constructing DebateRubric without a judge_client must raise
    ValueError at __init__ time. Deferring the check to score time
    would silently burn rollout budget before surfacing the config
    mistake."""
    with pytest.raises(ValueError, match="judge_client"):
        DebateRubric(
            truth_role="debater_a",
            members=["A", "B"],
            prompts=_open_ended_prompts(),
            judge_client=None,
        )


def test_load_environment_rejects_colliding_verdict_tokens():
    """Phase 1D verdict-collision guard: if a judge template's positive
    or negative verdict token collides with any EnumScoring answer
    value (case-insensitive), load_environment must raise ValueError.
    Otherwise transcript greps for the verdict silently misattribute
    judge output to a debater commit."""
    from verifiers.envs.debate.prompts import JudgeTemplate
    from verifiers.envs.debate.fields import EnumScoring

    # Build a synthetic pack where the grader's positive token "A"
    # collides with the MCQ answer enum {A, B, C, D}.
    base = _open_ended_prompts()
    colliding_grader = JudgeTemplate(
        system="Grade",
        user=_je.from_string("q: {question} a: {answer} r: {response}"),
        user_format_string="q: {question} a: {answer} r: {response}",
        positive="A",
        negative="B",
    )
    mcq_field = FieldSpec(
        type=str,
        description="MCQ answer",
        scoring=EnumScoring(values=("A", "B", "C", "D")),
    )
    pack = DebatePrompts(
        system=base.system,
        user=base.user,
        question=base.question,
        fields={
            "debater_a": {"propose": {"answer": mcq_field}},
            "debater_b": {"propose": {"answer": mcq_field}},
        },
        think_visibility=base.think_visibility,
        think_tag=base.think_tag,
        prefill=base.prefill,
        opponent_wrap=base.opponent_wrap,
        judges={"grader": colliding_grader},
        source_ref="colliding-test",
    )
    with pytest.raises(ValueError, match="collides with answer enum"):
        _validate_judge_templates(pack)


def test_wrap_opponent_respects_viewer_role():
    """Phase 1D F4: opponent_wrap templates must receive viewer_role in
    their render context, so packs can switch framing based on who is
    consuming the opponent turn (debater vs judge). A template that
    references {{ viewer_role }} should see the caller's value, not
    silently fall back to an empty string."""
    pack = DebatePrompts(
        system={
            "debater_a": _je.from_string("Argue."),
            "debater_b": _je.from_string("Argue."),
        },
        user={
            "debater_a": {"propose": _je.from_string("State your answer.")},
            "debater_b": {"propose": _je.from_string("State your answer.")},
        },
        question={
            "debater_a": _je.from_string("Q"),
            "debater_b": _je.from_string("Q"),
        },
        fields={},
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap={
            "debater": _je.from_string(
                "[viewer={{ viewer_role }}] {{ role_id }}: {{ text }}"
            ),
        },
        judges={},
        source_ref="viewer-role-test",
    )
    rendered = pack.wrap_opponent(
        "propose",
        "my argument",
        member_id="B",
        role_id="debater_b",
        viewer_role="debater_a",
    )
    assert "viewer=debater_a" in rendered, (
        "wrap_opponent must thread viewer_role through to the opponent_wrap "
        "template render context."
    )
    assert "debater_b: my argument" in rendered
