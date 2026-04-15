from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from verifiers.envs.multi_actor_env import MultiActorEnv
from verifiers.types import (
    EpisodeResult,
    EpisodeSpec,
    EpisodeStart,
    Member,
    MemberResult,
    TurnReq,
    TurnResp,
)

__all__ = [
    "PolicyHandle",
    "run_episode",
    "run_episode_group",
]


@dataclass(frozen=True)
class PolicyHandle:
    policy_slot: str
    client: Any
    model_name: str
    adapter_name: str | None = None


class PolicyBinder(Protocol):
    async def resolve(self, episode: EpisodeSpec, member: Member, request: TurnReq) -> PolicyHandle: ...


class TurnSampler(Protocol):
    async def __call__(self, request: TurnReq, policy: PolicyHandle) -> TurnResp: ...


def _member_map(members: list[Member]) -> dict[str, Member]:
    member_by_id: dict[str, Member] = {}
    for member in members:
        if member.member_id in member_by_id:
            raise ValueError(f"Duplicate member_id in session: {member.member_id}")
        member_by_id[member.member_id] = member
    return member_by_id


def _validate_ready_turns(requests: list[TurnReq]) -> None:
    member_ids: set[str] = set()
    turn_ids: set[str] = set()
    for req in requests:
        if req.member_id in member_ids:
            raise ValueError(f"Duplicate member_id in ready_turns: {req.member_id}")
        member_ids.add(req.member_id)
        if req.turn_id in turn_ids:
            raise ValueError(f"Duplicate turn_id in ready_turns: {req.turn_id}")
        turn_ids.add(req.turn_id)


def _validate_episode_result(
    episode: EpisodeSpec, output: EpisodeResult, member_by_id: dict[str, Member]
) -> None:
    if output.episode_id != episode.episode_id:
        raise ValueError("Episode result episode_id does not match the started sample")
    if output.base_example_id != episode.base_example_id:
        raise ValueError("Episode result base_example_id does not match the started sample")

    output_member_by_id = {m.member_id: m for m in output.members}
    expected_ids = set(member_by_id)
    actual_ids = set(output_member_by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Episode result member ids do not match session members: "
            f"expected {sorted(expected_ids)}, got {sorted(actual_ids)}"
        )

    for member_id, member in member_by_id.items():
        output_member = output_member_by_id[member_id]
        if output_member.role_id != member.role_id:
            raise ValueError(f"Episode result role_id does not match member {member_id}")
        if output_member.seat_id != member.seat_id:
            raise ValueError(f"Episode result seat_id does not match member {member_id}")


async def _resolve_and_sample(
    episode: EpisodeSpec,
    member_by_id: dict[str, Member],
    request: TurnReq,
    policy_binder: PolicyBinder,
    sample_member: TurnSampler,
) -> TurnResp:
    if request.episode_id != episode.episode_id:
        raise ValueError("Request episode_id does not match the active episode")
    if request.member_id not in member_by_id:
        raise ValueError(f"Unknown member_id in request: {request.member_id}")

    member = member_by_id[request.member_id]
    policy = await policy_binder.resolve(episode, member, request)
    return await sample_member(request, policy)


async def _run_episode_from_start(
    env: MultiActorEnv,
    start: EpisodeStart,
    policy_binder: PolicyBinder,
    sample_member: TurnSampler,
) -> EpisodeResult:
    episode = start.episode
    episode_id = episode.episode_id
    member_by_id = _member_map(start.members)

    ready_turns = start.ready_turns
    while ready_turns:
        _validate_ready_turns(ready_turns)

        tasks = [
            asyncio.create_task(
                _resolve_and_sample(episode, member_by_id, req, policy_binder, sample_member)
            )
            for req in ready_turns
        ]
        try:
            responses = list(await asyncio.gather(*tasks))
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for response, request in zip(responses, ready_turns, strict=True):
            if response.episode_id != episode_id:
                raise ValueError("Response episode_id does not match the active episode")
            if response.turn_id != request.turn_id:
                raise ValueError(f"Mismatched turn_id in response: expected {request.turn_id}, got {response.turn_id}")
            if response.member_id != request.member_id:
                raise ValueError("Response member_id does not match the originating request")

        ready_turns = await env.submit_ready_turns(responses)

        for new_request in ready_turns:
            if new_request.episode_id != episode_id:
                raise ValueError("Environment returned a request for a different episode")

    output = await env.finalize_episode(episode_id)
    _validate_episode_result(episode, output, member_by_id)
    return output


async def run_episode(
    env: MultiActorEnv,
    example: dict[str, Any],
    sample_index: int,
    policy_binder: PolicyBinder,
    sample_member: TurnSampler,
) -> EpisodeResult:
    start = env.start_episode(example, sample_index)
    return await _run_episode_from_start(env, start, policy_binder, sample_member)


async def run_episode_group(
    env: MultiActorEnv,
    example: dict[str, Any],
    rollouts_per_example: int,
    policy_binder: PolicyBinder,
    sample_member: TurnSampler,
    *,
    sample_index_offset: int = 0,
    concurrent: bool = False,
) -> list[EpisodeResult]:
    if concurrent:
        starts = [
            env.start_episode(example, sample_index_offset + i)
            for i in range(rollouts_per_example)
        ]
        episode_ids = [start.episode.episode_id for start in starts]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Duplicate episode_id in sample group")
        return list(await asyncio.gather(*(
            _run_episode_from_start(env, start, policy_binder, sample_member)
            for start in starts
        )))

    seen_ids: set[str] = set()
    outputs: list[EpisodeResult] = []
    for i in range(rollouts_per_example):
        start = env.start_episode(example, sample_index_offset + i)
        eid = start.episode.episode_id
        if eid in seen_ids:
            raise ValueError("Duplicate episode_id in sample group")
        seen_ids.add(eid)
        outputs.append(
            await _run_episode_from_start(env, start, policy_binder, sample_member)
        )
    return outputs
