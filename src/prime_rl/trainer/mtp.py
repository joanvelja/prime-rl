from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL


def _roll_left(t: Tensor) -> Tensor:
    """Shift tensor one position left along dim=-1, zero-pad the last position."""
    rolled = torch.roll(t, shifts=-1, dims=-1)
    rolled[..., -1] = 0
    return rolled


def compute_mtp_mask(loss_mask: Tensor, num_steps: int = 1) -> Tensor:
    """Compute MTP loss mask: position t is valid iff loss_mask[t+1..t+K+1] are all valid."""
    mask = loss_mask.float()
    result = torch.ones_like(mask)
    shifted = mask
    for _ in range(num_steps + 1):
        shifted = _roll_left(shifted)
        result = result * shifted
    return result


def compute_mtp_token_losses(
    model: PreTrainedModelPrimeRL,
    hidden_states: Tensor,
    input_ids: Tensor,
    position_ids: Tensor | None,
    chunk_size: int = 512,
) -> tuple[Tensor, int]:
    """Compute per-token MTP CE losses averaged across prediction steps.

    Returns (token_loss, num_steps) where token_loss is [B, S].
    """
    B, S, _H = hidden_states.shape
    V = model.lm_head.weight.shape[0]

    num_steps = model.mtp_num_prediction_steps
    shared = model.mtp_shared_weights
    layers = list(model.mtp_layers.values() if isinstance(model.mtp_layers, nn.ModuleDict) else model.mtp_layers)

    h = hidden_states.detach()
    detached_weight = model.lm_head.weight.detach()
    total_loss: Tensor | None = None

    current_ids = input_ids
    current_pos = position_ids

    for step in range(num_steps):
        shifted_ids = _roll_left(current_ids)
        labels = _roll_left(shifted_ids)
        shifted_pos = _roll_left(current_pos) if current_pos is not None else None

        with torch.no_grad():
            embeds = model.mtp_embed_tokens(shifted_ids)

        pos_emb = None
        rotary = model.mtp_rotary_emb
        if rotary is not None and shifted_pos is not None:
            with torch.no_grad():
                pos_emb = rotary(h, shifted_pos)

        layer = layers[0] if shared else layers[step]
        mtp_out = model.mtp_layer_forward(layer, h, embeds, shifted_pos, pos_emb)

        chunks = []
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            logits = F.linear(mtp_out[:, start:end, :], detached_weight)
            chunks.append(
                F.cross_entropy(
                    logits.reshape(-1, V),
                    labels[:, start:end].reshape(-1),
                    reduction="none",
                ).reshape(B, end - start)
            )
            del logits
        step_loss = torch.cat(chunks, dim=1)
        total_loss = step_loss if total_loss is None else total_loss + step_loss

        h = mtp_out
        current_ids = shifted_ids
        current_pos = shifted_pos

    if num_steps > 1:
        total_loss = total_loss / num_steps

    return total_loss, num_steps


def setup_mtp_training(model: nn.Module, mtp_config: object) -> None:
    """Validate model has MTP support and store config for use during forward."""
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    logger = get_logger()

    if not isinstance(model, PreTrainedModelPrimeRL):
        raise TypeError(
            f"MTP training requires a PreTrainedModelPrimeRL subclass, got {type(model).__name__}. "
            "Create a custom model implementation with the MTP interface."
        )

    if model.mtp_layers is None:
        raise ValueError(f"MTP training enabled but {type(model).__name__}.mtp_layers returned None.")

    model._mtp_config = mtp_config

    num_steps = model.mtp_num_prediction_steps
    shared = model.mtp_shared_weights
    num_params = sum(p.numel() for p in model.mtp_layers.parameters())
    logger.info(
        f"MTP training enabled: {len(model.mtp_layers)} layer(s), {num_steps} prediction step(s)"
        f"{' (shared weights)' if shared else ''}, {num_params:,} parameters"
    )
