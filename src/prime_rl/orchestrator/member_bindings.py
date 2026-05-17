from __future__ import annotations

import hashlib
from collections.abc import Mapping

from prime_rl.configs.orchestrator import MultiAgentConfig, MultiAgentMemberBindingConfig

LINEAGE_HEADER = "X-Verifiers-Lineage-Key"
ROLLOUT_ID_HEADER = "X-Prime-RL-Rollout-ID"
DEFAULT_TARGET_HEADER = "X-Prime-RL-Default-Target"


def _overlay(
    base: MultiAgentMemberBindingConfig,
    override: MultiAgentMemberBindingConfig,
) -> MultiAgentMemberBindingConfig:
    return MultiAgentMemberBindingConfig(
        target=override.target if override.target is not None else base.target,
        model=override.model if override.model is not None else base.model,
        trainable=override.trainable,
    )


def _stable_one_of(candidates: list[str], *, seed: int, rollout_id: object) -> str:
    payload = f"{seed}:{rollout_id}".encode()
    digest = hashlib.sha256(payload).digest()
    idx = int.from_bytes(digest[:8], "big") % len(candidates)
    return candidates[idx]


def rollout_id_for_binding(rollout: Mapping) -> object:
    return rollout.get("trajectory_id") or rollout.get("example_id") or ""


def resolve_member_binding(
    config: MultiAgentConfig,
    *,
    member_id: str | None,
    rollout_id: object,
) -> MultiAgentMemberBindingConfig:
    binding = config.default_binding
    if member_id is not None and member_id in config.member_bindings:
        binding = _overlay(binding, config.member_bindings[member_id])

    if config.one_of is not None and member_id in config.one_of.candidates:
        selected = _stable_one_of(
            config.one_of.candidates,
            seed=config.one_of.seed,
            rollout_id=rollout_id,
        )
        policy_binding = config.one_of.selected if member_id == selected else config.one_of.unselected
        binding = _overlay(binding, policy_binding)

    return binding


def is_trainable_member(config: MultiAgentConfig, rollout: Mapping, member_id: str) -> bool:
    binding = resolve_member_binding(
        config,
        member_id=member_id,
        rollout_id=rollout_id_for_binding(rollout),
    )
    return binding.trainable
