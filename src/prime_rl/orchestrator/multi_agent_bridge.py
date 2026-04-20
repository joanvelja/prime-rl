"""Compatibility shim for the verifiers-owned multi-agent bridge."""

from __future__ import annotations

from verifiers import rollout_to_member_rollouts
from verifiers.types import MemberRollout

__all__ = ["MemberRollout", "rollout_to_member_rollouts"]
