"""Shared LoRA identity helpers."""

from __future__ import annotations


def versioned_lora_name(base_name: str, step: int) -> str:
    """Return the request-routable LoRA name for a policy version."""
    if not base_name:
        raise ValueError("LoRA base name must be non-empty")
    if step < 0:
        raise ValueError(f"LoRA step must be non-negative, got {step}")
    return f"{base_name}__v{step:08d}"


def versioned_lora_int_id(step: int) -> int:
    """Return the vLLM LoRA integer id for a policy version.

    NCCL LoRA is single-run only, so the policy step is enough to make ids
    unique within a serving process. Include the run index here before
    relaxing that single-run invariant.
    """
    if step < 0:
        raise ValueError(f"LoRA step must be non-negative, got {step}")
    return step + 1


def versioned_lora_adapter(base_name: str, step: int) -> dict[str, int | str]:
    """Metadata shared by trainer, orchestrator, and inference for NCCL LoRA."""
    return {
        "base_lora_name": base_name,
        "lora_name": versioned_lora_name(base_name, step),
        "lora_int_id": versioned_lora_int_id(step),
        "adapter_version": step,
    }
