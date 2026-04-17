"""Contract + error-boundary tests for MultiAgentRubric base class."""

from __future__ import annotations

import asyncio

import pytest

import verifiers as vf
from verifiers.errors import KernelProtocolError
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric


def _run(coro):
    return asyncio.run(coro)


class _StructuredRubric(MultiAgentRubric):
    """Minimal subclass that writes member_rewards (+ flat metrics)."""

    def __init__(self, members: list[str]):
        super().__init__()
        self.members = members

    async def score_rollout(self, state) -> None:
        state["member_rewards"] = {m: 1.0 for m in self.members}
        state["metrics"] = {f"reward/{m}": 1.0 for m in self.members}


class _RaisingRubric(MultiAgentRubric):
    """Raises KernelProtocolError on the configured call-indexes."""

    def __init__(self, members: list[str], fail_on: set[int]):
        super().__init__()
        self.members = members
        self.fail_on = fail_on
        self._i = -1

    async def score_rollout(self, state) -> None:
        self._i += 1
        if self._i in self.fail_on:
            raise KernelProtocolError(f"boom-{self._i}")
        state["member_rewards"] = {m: 1.0 for m in self.members}


def test_multi_agent_rubric_contract_writes_member_rewards():
    rubric = _StructuredRubric(members=["alice", "bob"])
    state: dict = {}
    _run(rubric.score_rollout(state))
    assert state["member_rewards"] == {"alice": 1.0, "bob": 1.0}


def test_score_group_error_boundary_isolates_failures():
    """One rollout raises; other rollouts still score + default member_rewards populated."""
    rubric = _RaisingRubric(members=["alice", "bob"], fail_on={1})
    states: list[dict] = [{}, {}, {}]
    _run(rubric.score_group(states))

    failed = [s for s in states if s.get("error") is not None]
    succeeded = [s for s in states if s.get("error") is None]
    assert len(failed) == 1
    assert len(succeeded) == 2

    f = failed[0]
    assert isinstance(f["error"], KernelProtocolError)
    assert f["member_rewards"] == {"alice": 0.0, "bob": 0.0}

    for s in succeeded:
        assert s["member_rewards"] == {"alice": 1.0, "bob": 1.0}


def test_score_group_non_vf_error_propagates():
    """Non-vf.Error must NOT be swallowed — programming bugs escape loud."""

    class _BuggyRubric(MultiAgentRubric):
        members = ["a"]

        async def score_rollout(self, state) -> None:
            raise AttributeError("programming bug")

    rubric = _BuggyRubric()
    with pytest.raises(AttributeError, match="programming bug"):
        _run(rubric.score_group([{}]))


def test_score_group_all_succeed():
    rubric = _StructuredRubric(members=["alice", "bob"])
    states = [{}, {}]
    _run(rubric.score_group(states))
    for s in states:
        assert "error" not in s
        assert s["member_rewards"] == {"alice": 1.0, "bob": 1.0}


def test_multi_agent_rubric_is_subclass_of_rubric():
    """Type invariant: stays a Rubric so existing orchestrator code keeps working."""
    assert issubclass(MultiAgentRubric, vf.Rubric)
