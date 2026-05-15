"""Minimal vLLM repro: Triton top-k/top-p assumes contiguous logits rows.

Run on a CUDA machine with vLLM installed:

    python repro_vllm_noncontiguous_topk_topp.py

Expected on affected versions:

    FAIL: non-contiguous Triton output differs from contiguous Triton.

This does not require a model. It constructs a legal non-contiguous logits view
with shape [batch, vocab] and stride [padded_vocab, 1], then compares vLLM's
Triton top-k/top-p filter on the non-contiguous view against the same logical
values after `.contiguous()`.

The script checks both:

- top-k=1 on rows with one dominant token, where exactly one token should
  survive per row.
- top-p=0.95 on descending logits, where the contiguous and non-contiguous calls
  should keep the same token set per row.

Any mismatch means the wrapper accepted a layout that the kernel did not handle.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


def build_topk_logits(
    batch: int = 16,
    vocab: int = 1024,
    pad: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Return a [batch, vocab] logits view with padded physical row stride."""
    if pad <= 0:
        raise ValueError("pad must be positive")

    backing = torch.full((batch, vocab + pad), -1000.0, device=device, dtype=torch.float32)
    logits = backing[:, :vocab]
    assert logits.shape == (batch, vocab)
    assert logits.stride() == (vocab + pad, 1)
    assert not logits.is_contiguous()

    # Make top-k reference behavior simple and stable: each row should keep one
    # clearly dominant token and mask the rest to -inf.
    rows = torch.arange(batch, device=device)
    top_ids = (17 * rows + 123) % vocab
    logits[rows, top_ids] = 20.0 + rows.float() / 100.0
    return logits


def build_topp_logits(
    batch: int = 16,
    vocab: int = 1024,
    pad: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Return descending logits in a non-contiguous padded-row view."""
    if pad <= 0:
        raise ValueError("pad must be positive")

    backing = torch.full((batch, vocab + pad), -1000.0, device=device, dtype=torch.float32)
    logits = backing[:, :vocab]
    assert logits.shape == (batch, vocab)
    assert logits.stride() == (vocab + pad, 1)
    assert not logits.is_contiguous()

    rows = torch.arange(batch, device=device)
    base = torch.linspace(10.0, -10.0, vocab, device=device)
    logits.copy_(base[None, :] + rows[:, None] / 1000.0)
    return logits


def finite_count_per_row(x: torch.Tensor) -> list[int]:
    return torch.isfinite(x).sum(dim=-1).detach().cpu().tolist()


def check_case(name: str, logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> bool:
    contiguous_triton = apply_top_k_top_p_triton(logits.contiguous(), k, p)

    # Rebuild because the Triton filter mutates the input tensor.
    if name == "top-k":
        noncontig_input = build_topk_logits(
            batch=logits.shape[0], vocab=logits.shape[1], device=str(logits.device)
        )
    elif name == "top-p":
        noncontig_input = build_topp_logits(
            batch=logits.shape[0], vocab=logits.shape[1], device=str(logits.device)
        )
    else:
        raise ValueError(f"unknown case: {name}")
    noncontig_triton = apply_top_k_top_p_triton(noncontig_input, k, p)

    contiguous_mask = torch.isfinite(contiguous_triton)
    noncontig_mask = torch.isfinite(noncontig_triton)

    print(f"\n[{name}]")
    print("input shape:", tuple(logits.shape))
    print("input stride:", tuple(logits.stride()))
    print("contiguous Triton finite counts:", finite_count_per_row(contiguous_triton))
    print("non-contiguous Triton finite counts:", finite_count_per_row(noncontig_triton))

    mismatched_rows = torch.nonzero(
        contiguous_mask.ne(noncontig_mask).any(dim=-1), as_tuple=False
    ).flatten()
    if mismatched_rows.numel() == 0:
        print("PASS: non-contiguous Triton matched contiguous Triton.")
        return False

    print("mismatched rows:", mismatched_rows.detach().cpu().tolist())
    row = int(mismatched_rows[0].item())
    ref_finite = torch.nonzero(contiguous_mask[row], as_tuple=False).flatten()
    bad_finite = torch.nonzero(noncontig_mask[row], as_tuple=False).flatten()
    print(f"row {row} contiguous finite token ids:", ref_finite.detach().cpu().tolist()[:32])
    print(f"row {row} non-contig finite token ids:", bad_finite.detach().cpu().tolist()[:32])
    return True


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for vLLM's Triton top-k/top-p kernel.")

    torch.manual_seed(0)
    topk_logits = build_topk_logits()
    batch = topk_logits.shape[0]
    topk_failed = check_case(
        "top-k",
        topk_logits,
        torch.ones((batch,), device=topk_logits.device, dtype=torch.int32),
        None,
    )

    topp_logits = build_topp_logits()
    topp_failed = check_case(
        "top-p",
        topp_logits,
        None,
        torch.full((batch,), 0.95, device=topp_logits.device, dtype=torch.float32),
    )

    if topk_failed or topp_failed:
        raise SystemExit(
            "\nFAIL: non-contiguous Triton output differs from contiguous Triton. "
            "The kernel is treating row stride as vocab size."
        )

    print("\nPASS: both top-k and top-p matched for non-contiguous logits.")


if __name__ == "__main__":
    main()
