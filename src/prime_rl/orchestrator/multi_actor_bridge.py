"""Bridge from multi-actor results to training-compatible rollout dicts.

Two entry points:
- episodes_to_member_rollouts: from MultiActorEnv's EpisodeResult protocol
- rollout_to_member_rollouts: from DebateEnv's tagged RolloutOutput
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict

from verifiers.types import EpisodeResult, MemberResult, TrajectoryStep


class MemberRollout(TypedDict):
    """RolloutOutput-compatible dict with multi-actor metadata.

    The training data path reads: trajectory, sampling_args, example_id, error.
    Everything else is for logging/metadata.
    """

    # Training-path fields (read by pretokenize → interleave → TrainingSample)
    # NOTE: int-only. EpisodeResult.base_example_id upstream is int | str,
    # but both the dataset normalization in
    # verifiers.envs.environment::Environment._ensure_example_id and the
    # int-keyed example store in prime_rl.orchestrator.buffer.Buffer
    # require int. Widening here would silently break non-locally on the
    # first str id. Full int | str propagation is tracked as a follow-up
    # task (dataset + buffer + bridge together).
    example_id: int
    task: str
    trajectory: list[TrajectoryStep]
    sampling_args: dict[str, Any]
    error: None
    reward: float | None

    # Multi-actor metadata
    episode_id: str
    member_id: str
    role_id: str


def episodes_to_member_rollouts(
    results: list[EpisodeResult],
    env_name: str,
    temperature: float,
) -> list[MemberRollout]:
    """Flatten EpisodeResults into one MemberRollout per member.

    Args:
        results: Completed episodes from a multi-actor env.
        env_name: Environment name — becomes ``task`` (NOT role_id).
        temperature: Sampling temperature injected as a bridge parameter.
            Neither TurnResp, PolicyHandle, nor EpisodeResult carry this.
    """
    rollouts: list[MemberRollout] = []
    for episode in results:
        example_id = _validated_example_id(episode)
        for member in episode.members:
            rollouts.append(_member_to_rollout(
                member=member,
                example_id=example_id,
                episode_id=episode.episode_id,
                env_name=env_name,
                temperature=temperature,
            ))
    return rollouts


def _validated_example_id(episode: EpisodeResult) -> int:
    """Enforce int example_id — dataset normalization and buffer both require int.

    EpisodeResult.base_example_id is typed int | str upstream, but
    downstream keying in the buffer uses int. Until the dataset/buffer
    layers are widened, the bridge is the right place to fail loud on
    a str id rather than let it propagate and blow up at buffer-insert
    time with a confusing stack trace.
    """
    raw = episode.base_example_id
    if not isinstance(raw, int):
        raise TypeError(
            f"base_example_id must be int (buffer keys by int), got "
            f"{type(raw).__name__}: {raw!r}"
        )
    return raw


def _member_to_rollout(
    member: MemberResult,
    example_id: int,
    episode_id: str,
    env_name: str,
    temperature: float,
) -> MemberRollout:
    return MemberRollout(
        example_id=example_id,
        task=env_name,
        trajectory=member.trajectory,
        sampling_args={"temperature": temperature},
        error=None,
        reward=member.reward,
        episode_id=episode_id,
        member_id=member.member_id,
        role_id=member.role_id,
    )


# ---------------------------------------------------------------------------
# RolloutOutput bridge (DebateEnv — tagged merged trajectory)
# ---------------------------------------------------------------------------


def rollout_to_member_rollouts(
    output: dict[str, Any],
    env_name: str,
) -> list[MemberRollout]:
    """Split a DebateEnv RolloutOutput into one MemberRollout per member.

    DebateEnv produces a single merged trajectory where each TrajectoryStep
    is tagged with ``extras["member_id"]`` and ``extras["role_id"]``.
    This function groups steps by member and builds MemberRollouts that
    are identical in shape to those from ``episodes_to_member_rollouts``.

    Args:
        output: RolloutOutput dict with keys: trajectory, sampling_args,
            example_id, and optionally metrics and trajectory_id.
        env_name: Environment name — becomes ``task``.
    """
    trajectory: list[TrajectoryStep] = output.get("trajectory", [])
    if not trajectory:
        return []

    sampling_args = output.get("sampling_args")
    if sampling_args is None:
        raise ValueError(
            "RolloutOutput missing 'sampling_args' — required for interleave_rollout"
        )

    temperature = sampling_args["temperature"]
    example_id = output["example_id"]
    episode_id = output.get("trajectory_id", "")
    metrics = output.get("metrics", {})

    # Group steps by member_id, preserving temporal order
    member_steps: dict[str, list[TrajectoryStep]] = defaultdict(list)
    member_role: dict[str, str] = {}

    for step in trajectory:
        extras = step.get("extras", {})
        mid = extras.get("member_id")
        if mid is None:
            raise ValueError(
                f"TrajectoryStep missing extras['member_id']: {step!r}"
            )
        member_steps[mid].append(step)
        if mid not in member_role:
            member_role[mid] = extras.get("role_id", "")

    rollouts: list[MemberRollout] = []
    for mid, steps in member_steps.items():
        reward = metrics.get(f"reward/{mid}")
        rollouts.append(MemberRollout(
            example_id=example_id,
            task=env_name,
            trajectory=steps,
            sampling_args={"temperature": temperature},
            error=None,
            reward=reward,
            episode_id=episode_id,
            member_id=mid,
            role_id=member_role[mid],
        ))

    return rollouts
