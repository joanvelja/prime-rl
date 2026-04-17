"""Unit tests for MultiAgentEnv: generic multi-actor rollout loop.

System boundary is the Client (faked). The kernel, schedule, and
rollout loop are exercised for real. Subclass the abstract base with a
minimal EchoEnv that just echoes member_id + slot.phase — enough to
verify ordering, atomic commit, monotonic prompt invariant, stop
conditions, actor_override routing, and lineage-scoped prefix cache.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest

import verifiers as vf
from verifiers.clients import Client as _VFClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_actor_kernel import (
    StaticSchedule,
    TurnSlot,
)
from verifiers.errors import Error as VFError, OverlongPromptError
from verifiers.types import (
    Response,
    ResponseMessage,
    ResponseTokens,
    RolloutInput,
    State,
    Usage,
)


# ---------------------------------------------------------------------------
# Fake client — system boundary stub
# ---------------------------------------------------------------------------


class FakeClient(_VFClient):
    def __init__(self, responses: list[Response]) -> None:
        self.logger = logging.getLogger(f"{__name__}.FakeClient")
        self._config = None
        self._client = None
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self, prompt, model, sampling_args=None, tools=None, **kwargs
    ) -> Response:
        self.calls.append({"prompt": prompt, "model": model})
        if not self._responses:
            raise RuntimeError("FakeClient exhausted")
        return self._responses.pop(0)

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


class RaisingClient(FakeClient):
    def __init__(self, exc: Exception) -> None:
        super().__init__([])
        self._exc = exc

    async def get_response(self, prompt, model, sampling_args=None, tools=None, **kwargs):
        self.calls.append({"prompt": prompt, "model": model})
        raise self._exc


def _make_response(content: str) -> Response:
    return Response(
        id="r",
        created=0,
        model="m",
        usage=Usage(
            prompt_tokens=1, reasoning_tokens=0,
            completion_tokens=len(content), total_tokens=1 + len(content),
        ),
        message=ResponseMessage(
            content=content, finish_reason="stop",
            is_truncated=False, tokens=None,
        ),
    )


# ---------------------------------------------------------------------------
# Trivial subclass: EchoEnv
# ---------------------------------------------------------------------------


class EchoRubric(vf.Rubric):
    """Minimal rubric — scoring is not exercised here."""


class EchoEnv(MultiAgentEnv):
    """Build prompt by appending per-turn [user, assistant-prefill] pairs.

    Satisfies the monotonic extension invariant: for member A, slot_N+1's
    prompt is slot_N's prompt + two new messages at the tail.
    """

    async def build_prompt(self, state, member_id, slot):
        """Monotonic-extending prompt: each prior turn appended as
        (user=opponent OR instruction) + (assistant=self). The instruction
        for the CURRENT turn is baked into the trailing user message,
        which stays in place on the next call because we append our own
        assistant response (committed as an Utterance) before the NEXT
        instruction is appended."""
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": f"You are {member_id}"},
            {"role": "user", "content": "Q: echo"},
        ]
        # Walk transcript: for member A, appearances alternate
        # instruction → own-assistant, opponent → relayed user.
        seen_self_turns = 0
        for utt in state["_kernel"].transcript:
            if utt.member_id == member_id:
                # Insert the instruction that preceded this self-turn,
                # then the self-assistant message committed at that turn.
                msgs.append({"role": "user", "content": f"Your turn ({utt.phase})"})
                msgs.append({"role": "assistant", "content": utt.raw_content})
                seen_self_turns += 1
            else:
                msgs.append({"role": "user", "content": f"[{utt.member_id}] {utt.public_channel}"})
        # Current turn's instruction appended at the tail.
        msgs.append({"role": "user", "content": f"Your turn ({slot.phase})"})
        return msgs

    async def render_completion(self, state):
        out: list[dict[str, Any]] = []
        for step in state["trajectory"]:
            out.extend(step["completion"])
        state["completion"] = out


def _make_env(
    responses,
    slots=None,
    members=("A", "B"),
    actor_overrides=None,
):
    slots = slots or (
        TurnSlot(slot_id=0, actors=("A",), phase="p1"),
        TurnSlot(slot_id=1, actors=("B",), phase="p1"),
    )
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=list(members),
        actor_overrides=actor_overrides,
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    return env, FakeClient(responses)


def _run(coro):
    return asyncio.run(coro)


def _rollout_input() -> RolloutInput:
    return RolloutInput(
        prompt=[{"role": "user", "content": "echo"}],
        example_id=1,
        task="echo_test",
        answer="",
    )


# ---------------------------------------------------------------------------
# 1. Init validation
# ---------------------------------------------------------------------------


def test_init_rejects_empty_members():
    with pytest.raises(ValueError, match="non-empty"):
        EchoEnv(
            schedule=StaticSchedule(()),
            members=[],
            rubric=EchoRubric(),
            dataset=lambda: None,
        )


def test_init_rejects_duplicate_members():
    with pytest.raises(ValueError, match="duplicates"):
        EchoEnv(
            schedule=StaticSchedule(()),
            members=["A", "A"],
            rubric=EchoRubric(),
            dataset=lambda: None,
        )


def test_init_rejects_stray_actor_overrides():
    with pytest.raises(ValueError, match="not in members"):
        EchoEnv(
            schedule=StaticSchedule(()),
            members=["A", "B"],
            actor_overrides={"C": (None, "other-model")},
            rubric=EchoRubric(),
            dataset=lambda: None,
        )


# ---------------------------------------------------------------------------
# 2. Rollout loop — sequential + tagging
# ---------------------------------------------------------------------------


def test_rollout_sequential_tags_trajectory_in_order():
    env, client = _make_env([_make_response("A!"), _make_response("B!")])
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert len(state["trajectory"]) == 2
    assert state["trajectory"][0]["extras"]["member_id"] == "A"
    assert state["trajectory"][1]["extras"]["member_id"] == "B"
    assert state["is_completed"] is True
    assert state["stop_condition"] == "schedule_exhausted"


# ---------------------------------------------------------------------------
# 3. Stop conditions — priority order
# ---------------------------------------------------------------------------


def test_error_stop_fires_before_exhausted():
    slots = (TurnSlot(slot_id=0, actors=("A",), phase="p"),)
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    client = RaisingClient(VFError("boom"))
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert state["stop_condition"] == "has_error"
    assert isinstance(state["error"], VFError)


def test_prompt_too_long_sets_truncated():
    slots = (TurnSlot(slot_id=0, actors=("A",), phase="p"),)
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    client = RaisingClient(OverlongPromptError("too long"))
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert state["prompt_too_long"] is True
    assert state["is_truncated"] is True
    assert state["stop_condition"] == "prompt_too_long"


def test_schedule_exhausted_fires_at_end():
    env, client = _make_env([_make_response("A!"), _make_response("B!")])
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert state["stop_condition"] == "schedule_exhausted"


# ---------------------------------------------------------------------------
# 4. Simultaneous slot — atomic commit
# ---------------------------------------------------------------------------


def test_simultaneous_slot_commits_both_atomically():
    slots = (TurnSlot(slot_id=0, actors=("A", "B"), phase="p"),)
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A", "B"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    client = FakeClient([_make_response("A says"), _make_response("B says")])
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert len(state["trajectory"]) == 2
    # Both commits landed; kernel advanced past slot 0.
    assert state["_kernel"].slot_index == 1
    ids = [s["extras"]["member_id"] for s in state["trajectory"]]
    assert ids == ["A", "B"]


def test_simultaneous_slot_rolls_back_on_error():
    """If any actor in a simultaneous slot raises, NO partial commits land."""
    slots = (TurnSlot(slot_id=0, actors=("A", "B"), phase="p"),)
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A", "B"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    # First call succeeds; second raises. gather raises on first exception;
    # since we stage responses THEN commit, the kernel is untouched.
    class MixedClient(FakeClient):
        def __init__(self):
            super().__init__([])
            self._i = 0

        async def get_response(self, prompt, model, sampling_args=None, tools=None, **kwargs):
            self._i += 1
            if self._i == 1:
                return _make_response("ok")
            raise VFError("second fails")

    state = _run(env.rollout(_rollout_input(), MixedClient(), "m"))
    # Error captured, no trajectory commits.
    assert state["stop_condition"] == "has_error"
    assert state["trajectory"] == []
    assert state["_kernel"].slot_index == 0


def test_simultaneous_slot_cancels_peer_on_first_failure():
    """TaskGroup must cancel the surviving actor when one raises.

    The slow actor sleeps past the fast actor's failure; if it completes
    anyway, both the error semantics AND the shared usage tracker are
    compromised (late completion leaks accounting). With TaskGroup the
    slow actor is cancelled before it can reach the completion line.
    """
    slots = (TurnSlot(slot_id=0, actors=("A", "B"), phase="p"),)
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A", "B"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )

    class CancelProbeClient(FakeClient):
        def __init__(self):
            super().__init__([])
            self.completed: list[str] = []

        async def get_response(self, prompt, model, sampling_args=None, tools=None, **kwargs):
            if not self.calls:
                self.calls.append({"prompt": prompt, "model": model})
                raise VFError("A fails fast")
            self.calls.append({"prompt": prompt, "model": model})
            await asyncio.sleep(0.5)
            self.completed.append("B")  # should never run

    client = CancelProbeClient()
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert state["stop_condition"] == "has_error"
    assert state["trajectory"] == []
    assert state["_kernel"].slot_index == 0
    assert client.completed == [], "peer actor completed after first failure"


def test_simultaneous_slot_rolls_back_on_post_commit_hook_failure():
    """If on_step_committed raises mid-slot, NOTHING is published.

    extract_fields / _build_step / on_step_committed run after the
    kernel has been folded into a LOCAL buffer. Invariant: raising in
    any of them leaves state["_kernel"] and state["trajectory"] at
    their pre-slot snapshots.
    """
    slots = (TurnSlot(slot_id=0, actors=("A", "B", "C"), phase="p"),)

    class HookFailEnv(EchoEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.committed_calls = 0

        async def on_step_committed(self, state, utt, fields):
            self.committed_calls += 1
            if self.committed_calls == 2:
                raise VFError("hook boom on actor 2")

    env = HookFailEnv(
        schedule=StaticSchedule(slots),
        members=["A", "B", "C"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    client = FakeClient([
        _make_response("A says"),
        _make_response("B says"),
        _make_response("C says"),
    ])
    state = _run(env.rollout(_rollout_input(), client, "m"))
    assert state["stop_condition"] == "has_error"
    assert isinstance(state["error"], VFError)
    assert state["_kernel"].slot_index == 0
    assert state["trajectory"] == []


# ---------------------------------------------------------------------------
# 5. Monotonic prompt invariant — test-time checker
# ---------------------------------------------------------------------------


async def _build_all_prompts(env: MultiAgentEnv, state: State):
    """Re-run build_prompt across the same kernel states a rollout produced."""
    # Not applicable after rollout (kernel moved on). Instead, we walk
    # synthetic kernel states for each (member, slot) pair and verify
    # extension.


def _assert_monotonic(prev: list[dict], curr: list[dict]):
    assert len(curr) >= len(prev), f"curr len {len(curr)} < prev {len(prev)}"
    for i, (p, c) in enumerate(zip(prev, curr)):
        assert p == c, f"message {i} diverged: {p!r} vs {c!r}"


def test_build_prompt_monotonic_across_slots():
    """For each member, build_prompt(slot_N+1) extends build_prompt(slot_N)."""
    slots = (
        TurnSlot(slot_id=0, actors=("A",), phase="p1"),
        TurnSlot(slot_id=1, actors=("B",), phase="p1"),
        TurnSlot(slot_id=2, actors=("A",), phase="p2"),
        TurnSlot(slot_id=3, actors=("B",), phase="p2"),
    )
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A", "B"],
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    responses = [_make_response(f"r{i}") for i in range(4)]
    client = FakeClient(responses)
    state = _run(env.rollout(_rollout_input(), client, "m"))

    # Extract per-member build_prompt sequences from recorded trajectory.
    per_member: dict[str, list[list[dict]]] = {"A": [], "B": []}
    for step in state["trajectory"]:
        mid = step["extras"]["member_id"]
        per_member[mid].append(list(step["prompt"]))

    for mid, seq in per_member.items():
        for prev, curr in zip(seq, seq[1:]):
            _assert_monotonic(prev, curr)


# ---------------------------------------------------------------------------
# 6. actor_overrides routing
# ---------------------------------------------------------------------------


def test_actor_overrides_routes_per_member():
    """Member B uses override client; A falls through to default."""
    default = FakeClient([_make_response("A!")])
    override = FakeClient([_make_response("B!")])
    slots = (
        TurnSlot(slot_id=0, actors=("A",), phase="p"),
        TurnSlot(slot_id=1, actors=("B",), phase="p"),
    )
    env = EchoEnv(
        schedule=StaticSchedule(slots),
        members=["A", "B"],
        actor_overrides={"B": (override, "override-model")},
        rubric=EchoRubric(),
        dataset=lambda: None,
    )
    state = _run(env.rollout(_rollout_input(), default, "default-model"))
    assert len(state["trajectory"]) == 2
    assert len(default.calls) == 1
    assert len(override.calls) == 1
    assert override.calls[0]["model"] == "override-model"


# ---------------------------------------------------------------------------
# 7. Lineage-scoped prefix cache
# ---------------------------------------------------------------------------


class _LineageProbeClient(OpenAIChatCompletionsTokenClient):
    """Captures which trajectory step was prefix-matched."""

    def __init__(self):
        class _Stub:
            base_url = "http://x/v1"
            def with_options(self, **kw): return self
        super().__init__(_Stub())

    async def to_native_prompt(self, messages):
        return messages, {}

    async def tokenize(self, messages, tools, model, extra_kwargs={}, **kwargs):
        return [0]


def _step(member_id: str, prompt, completion, prompt_ids, completion_ids):
    return {
        "prompt": prompt,
        "completion": completion,
        "tokens": {
            "prompt_ids": prompt_ids, "completion_ids": completion_ids,
            "is_truncated": False,
        },
        "response": None,
        "extras": {"member_id": member_id},
    }


@pytest.mark.asyncio
async def test_lineage_filter_isolates_per_member_cache():
    """A's new prompt should only match A's prior step, not B's."""
    client = _LineageProbeClient()

    # Both steps share the same (user, assistant) prefix, but only A's
    # step is the correct continuation for A's next turn. Without lineage
    # filter, the scan would hit B's step first (reversed order) because
    # of equal prefix length.
    shared_prompt = [{"role": "user", "content": "u1"}]
    shared_completion = [{"role": "assistant", "content": "a1"}]

    trajectory = [
        _step("A", shared_prompt, shared_completion,
              prompt_ids=[1], completion_ids=[2]),
        _step("B", shared_prompt, shared_completion,
              prompt_ids=[100], completion_ids=[200]),
    ]
    state = cast(State, {"model": "m", "trajectory": trajectory})

    a_new = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u-next"},
    ]

    # With lineage=A, prefix should come from A's step (prompt_ids=[1],
    # completion_ids=[2]).
    ids_a = await client.get_prompt_ids(state, a_new, oai_tools=None, lineage_key="A")
    assert ids_a is not None
    assert ids_a[:2] == [1, 2], f"lineage A mispicked: {ids_a}"

    # With lineage=B, should come from B's step.
    ids_b = await client.get_prompt_ids(state, a_new, oai_tools=None, lineage_key="B")
    assert ids_b is not None
    assert ids_b[:2] == [100, 200], f"lineage B mispicked: {ids_b}"


@pytest.mark.asyncio
async def test_lineage_none_preserves_default_behavior():
    """lineage_key=None keeps the original latest-match behavior."""
    client = _LineageProbeClient()
    shared_prompt = [{"role": "user", "content": "u1"}]
    shared_completion = [{"role": "assistant", "content": "a1"}]
    trajectory = [
        _step("A", shared_prompt, shared_completion,
              prompt_ids=[1], completion_ids=[2]),
        _step("B", shared_prompt, shared_completion,
              prompt_ids=[100], completion_ids=[200]),
    ]
    state = cast(State, {"model": "m", "trajectory": trajectory})
    new = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u-next"},
    ]
    ids = await client.get_prompt_ids(state, new, oai_tools=None)
    # Scan is reversed; B is last so its ids win on tie.
    assert ids is not None
    assert ids[:2] == [100, 200]
