# OLMo 3 Instruct SFT Pipeline — Verbatim Reference

> Extracted 2026-04-16 from arXiv:2512.13961 (OLMo 3 paper), Appendix A.6.1.
> Full paper text fetched via alphaxiv MCP `get_paper_content` (872K chars, 16K lines).

---

## Source: Appendix A.6.1 — Supervised Finetuning Details (verbatim)

> "Using OLMo-core infrastructure for SFT Training. Relative to pretraining, this
> involves a substantially smaller batch size, different data packing, and masking.
> **This leads to an 8× faster training speed than open-instruct**, dramatically
> improving our iteration speed. We use between 1 and 8 8×H100 nodes, or 1 to 4
> 8×B200 nodes to train our 7B reasoner and instruct models. We use 32 8×H100
> nodes to train our 32B thinking model. As a consequence of using olmo-core, our
> batch size is now measured in tokens instead of instances, and we train with
> document packing instead of padding. We train all of our 7B SFT models with a
> batch size of 1M tokens and 32B SFT models with a batch size of 4M tokens, for
> two epochs, with packing, and a 32,768 sequence length."

## Table 47 — Training Hyperparameters (verbatim)

| Parameter | 7B Think SFT | 32B Think SFT | 7B Instruct SFT |
|---|---|---|---|
| Total Tokens | 45.4B | 45.2B | 3.4B |
| Learning Rate | 5.0 × 10⁻⁵ | 1.0 × 10⁻⁴ souped with 5.0 × 10⁻⁵ | 8.0 × 10⁻⁵ |
| Num. GPUs | 64 | 256 | 8-64 |
| Max Sequence Length | 32K | 32K | 32K |
| Batch Size | 1M tokens | 4M tokens | 1M tokens (implied) |
| Epochs | 2 | 2 | 2 |
| Packing | document packing | document packing | document packing |

## What the paper explicitly states about SFT

- **Framework**: OLMo-core (NOT open-instruct). 8× faster than open-instruct.
- **Batch size**: measured in tokens, not instances.
- **Packing**: document packing (not padding). Sequences packed together.
- **Masking**: mentioned as a feature ("different data packing, and masking") but the
  specific masking algorithm is NOT documented — no detail on incremental-prefix vs
  full-sequence-once, no detail on loss-on-completion-only vs full-sequence loss.
- **LR**: 8e-5 for 7B Instruct SFT. 5e-5 for 7B Think SFT. 1e-4 souped with 5e-5
  for 32B Think SFT.
- **"souped with"**: refers to model souping — averaging checkpoints trained at
  different learning rates.
- **Epochs**: 2 for all variants.
- **Sequence length**: 32K (32,768) for all variants.

## What the paper does NOT state about SFT

These are NOT documented anywhere in the 872K-char paper:

1. **Loss function**: standard NLL/cross-entropy assumed but not stated
2. **Loss masking algorithm**: "masking" is listed as a feature but the specific
   approach (incremental prefix tokenization vs full-sequence-once vs
   template-specific) is not documented
3. **Optimizer for SFT**: not stated (pretraining uses AdamW with β₁=0.9, β₂=0.95)
4. **LR schedule for SFT**: not stated (pretraining uses WSD — warmup-stable-decay)
5. **Warmup ratio for SFT**: not stated
6. **Weight decay for SFT**: not stated
7. **System prompt handling during SFT**: completely absent — not mentioned in any
   SFT-related section. System prompts appear only in tool-spec context and eval prompts.
8. **Chat template used during SFT**: not stated. We know from code inspection
   (open-instruct's `dataset_transformation.py`) that the `olmo` chat template has
   `{% if not loop.last %}<|im_end|>{% else %}{{ eos_token }}{% endif %}` branching.
9. **Per-message vs per-completion loss masking**: not stated.

## Comparison with our setup

| Parameter | OLMo 3 Instruct SFT | Our plan (prime-rl) |
|---|---|---|
| Framework | OLMo-core | prime-rl |
| LR | **8e-5** | 5e-6 (Tülu 3 canonical for Llama-3.1-8B) |
| Max seq length | **32K** | 4096 |
| Batch size | ~1M tokens | ~64K tokens (128 samples × ~500 tok) |
| Epochs | 2 | 2 |
| Packing | document packing | `cat` packing |
| Total tokens | 3.4B | ~1.7B |
| GPUs | 8-64 | 8 (single node) |
| Masking | undocumented | `build_incremental_token_mask` (prefix-monotonic) |
| System prompts | undocumented | 711-prompt pool, profile-weighted injection |

### LR discrepancy note

OLMo 3 Instruct SFT uses 8e-5 (16× higher than Tülu 3's 5e-6 for Llama-3.1-8B).
This is because OLMo-core starts from OLMo 3's **midtrained** checkpoint (which has
been through extensive midtraining + long-context extension), not from a vanilla
base. The midtrained model has already converged to a stable loss landscape, so it
can tolerate a higher SFT learning rate. Our bases haven't been midtrained by us,
so the lower Tülu 3 LR (5e-6) is appropriate — but this should be treated as a
starting point, not gospel. The smoke run (Rung 3) will validate.

### Seq length discrepancy note

OLMo 3 trains at 32K because Dolci-Instruct-SFT includes samples extended to 32K
context (OpenThoughts3+ Science, which was context-extended from 8K to 32K). Our
Dolci subset is filtered for debate relevance — median response is ~500 tokens.
4096 seq_len is sufficient and saves ~8× compute per step. If we later need
longer context (e.g., for multi-turn debate transcripts during RL), the base models'
native long-context capability survives SFT at 4096.

---

## Open questions for our pipeline

1. **Should we match OLMo 3's LR (8e-5) instead of Tülu 3's (5e-6)?**
   The answer depends on the base model. OLMo-3-1025-7B was midtrained before SFT,
   so 8e-5 was appropriate for it. Our 7 bases are NOT midtrained (except possibly
   Qwen3 and Marin which have midtraining phases). Tülu 3's 5e-6 was for Llama-3.1-8B
   base, which is closer to our situation. **Resolve via smoke run: try 5e-6 on marin,
   check loss curves. If loss doesn't descend fast enough, try 2e-5 or 5e-5.**

2. **Should we use 32K seq_len instead of 4096?**
   No for initial sweep. Our Dolci data is mostly <1K tokens per sample. 32K would
   waste 87.5% of each sequence on padding/packing overhead. 4096 is 4× headroom
   over median sample length. Revisit if we add long-form debate transcripts.

3. **What masking algorithm does OLMo-core actually use?**
   Not documented in the paper. From code inspection of open-instruct's
   `dataset_transformation.py`, the `mask_labels` function uses incremental prefix
   rendering (same family as prime-rl's `build_incremental_token_mask`) but without
   the strict assert. OLMo-core's native SFT path may use a different approach —
   possibly full-sequence rendering with template-delimiter scanning. We don't know.

4. **Tokenization verification coverage gap**
   Our 350+90 tests covered 4 of 12 HF configs (wildchat, persona-math, precise-if,
   flan) × single-turn only. NOT tested: evol-codealpaca (code fences), dolci-python-algo
   (bare Python), openthoughts-sci (LaTeX), persona-algebra (LaTeX), openmathinstruct-2,
   openassistant (multi-turn dropped), tablegpt (JSON/SQL), sciriff (science QA).
   **Should run a broader verification sweep before Rung 3.**
