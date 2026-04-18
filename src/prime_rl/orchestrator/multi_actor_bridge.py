"""Bridge: split a multi-actor RolloutOutput into per-member training rollouts.

The env writes a single merged trajectory tagged with ``extras["member_id"]``
and an episode-level ``MARScore`` that covers every member. This bridge
projects the episode into ``MemberRollout`` records — one per member — that
the training pipeline consumes.

There is exactly one entry point and one source of truth: ``output["mar_score"]``.
Bridge raises ``KeyError`` if absent — that means the env didn't run a
``MultiAgentRubric`` and you're calling the wrong code path.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict

from verifiers.types import MARScore, TrajectoryStep


class MemberRollout(TypedDict):
    """RolloutOutput-compatible dict with multi-actor metadata.

    The training data path reads: trajectory, sampling_args, example_id, error.
    Everything else is for logging/metadata.
    """

    # Training-path fields (read by pretokenize → interleave → TrainingSample).
    example_id: int | str
    task: str
    trajectory: list[TrajectoryStep]
    sampling_args: dict[str, Any]
    error: dict[str, Any] | None
    reward: float

    # Multi-actor metadata
    episode_id: str
    member_id: str
    role_id: str


def rollout_to_member_rollouts(
    output: dict[str, Any],
    env_name: str,
) -> list[MemberRollout]:
    """Split a multi-actor RolloutOutput into one MemberRollout per member.

    Reads ``output["mar_score"]`` (the typed payload written by
    ``MultiAgentRubric``) for per-member rewards and roles, and groups
    ``output["trajectory"]`` by ``extras["member_id"]`` to assemble each
    member's step list.

    Args:
        output: RolloutOutput dict with required keys: trajectory, sampling_args,
            example_id, mar_score. Optional: error, trajectory_id.
        env_name: Environment name — becomes ``task``.

    Raises:
        KeyError: if ``mar_score`` is missing — the env did not run a
            ``MultiAgentRubric``.
    """
    mar_raw = output["mar_score"]
    mar = (
        mar_raw if isinstance(mar_raw, MARScore) else MARScore.model_validate(mar_raw)
    )

    sampling_args = output["sampling_args"]
    temperature = sampling_args["temperature"]
    example_id = output["example_id"]
    episode_id = output.get("trajectory_id", "")
    rollout_error = output.get("error")
    trajectory: list[TrajectoryStep] = output.get("trajectory", [])

    # Group steps by member_id (preserving temporal order).
    steps_by_member: dict[str, list[TrajectoryStep]] = defaultdict(list)
    for step in trajectory:
        extras = step.get("extras", {})
        mid = extras.get("member_id")
        if mid is None:
            raise ValueError(
                f"TrajectoryStep missing extras['member_id']: {step!r}"
            )
        steps_by_member[mid].append(step)

    return [
        MemberRollout(
            example_id=example_id,
            task=env_name,
            trajectory=steps_by_member.get(m.member_id, []),
            sampling_args={"temperature": temperature},
            error=rollout_error,
            reward=m.reward,
            episode_id=episode_id,
            member_id=m.member_id,
            role_id=m.role_id,
        )
        for m in mar.members
    ]
