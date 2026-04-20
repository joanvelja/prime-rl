from collections.abc import Sequence

import torch


def build_cu_seqlens(sequence_lengths: Sequence[int], device: torch.device | None = None) -> tuple[torch.Tensor, int]:
    """Build FlashAttention-style cumulative sequence lengths from explicit lengths."""
    if len(sequence_lengths) == 0:
        raise ValueError("sequence_lengths must be non-empty")
    lengths = torch.tensor(list(sequence_lengths), dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = lengths.cumsum(dim=0)
    return cu_seqlens, int(lengths.max().item())


def infer_sequence_lengths_from_position_ids(position_ids: torch.Tensor) -> list[int]:
    """Infer packed sequence lengths from position-id resets.

    This is a fallback for existing callers that do not pass explicit packing metadata.
    It treats a `0` following a non-zero position as a new sequence boundary only if the
    next token is `1`, which avoids misclassifying trailing zero padding as a boundary.
    """
    flat_position_ids = position_ids.view(-1)
    if flat_position_ids.numel() == 0:
        return []

    boundaries = [0]
    for i in range(1, flat_position_ids.numel()):
        curr = int(flat_position_ids[i].item())
        prev = int(flat_position_ids[i - 1].item())
        next_val = int(flat_position_ids[i + 1].item()) if i + 1 < flat_position_ids.numel() else None
        if curr == 0 and prev != 0 and next_val == 1:
            boundaries.append(i)

    lengths = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else flat_position_ids.numel()
        lengths.append(end - start)
    return lengths


def infer_cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Fallback cumulative sequence lengths derived from position-id resets."""
    return build_cu_seqlens(
        infer_sequence_lengths_from_position_ids(position_ids),
        device=position_ids.device,
    )
