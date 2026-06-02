# Parity diagnostic — gemma4-grouped-mm-off

**VERDICT: FORWARD-DIVERGENCE**

Driving threshold(s):
- step-0 clean (mean=0.0153 <= 0.1) but tail-shaped drift: growth_ratio=89.45 >= growth_ratio_flag=3.0; late_mean=0.8676 > late_mean_elevated=0.05; late catastrophe_frac=0.061241 > catastrophe_frac_significant=0.001; median stayed low (p50=0.0000) -> catastrophe-TAIL signature; LOCALIZE before concluding inherent (often a fixable discrete bug, cf. gemma-4 root_size reload clobber)

## Per-step signature (loss_mask=True tokens)

| step | seq | tok | null | mean | p50 | p99 | max | tail% | cat_frac | flipT | flipI | IR_mean | is_masked% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 774 | 0 | 0.0153 | 0.0000 | 0.2389 | 2.7769 | 0.778 | 0.000000 | 0 | 0 | 1.009 | 0.0103 |
| 1 | 20 | 1481 | 0 | 0.0068 | 0.0000 | 0.1485 | 1.6473 | 0.726 | 0.000000 | 0 | 0 | 1.001 | 0.0047 |
| 2 | 20 | 2569 | 0 | 0.0051 | 0.0000 | 0.1000 | 4.9409 | 0.829 | 0.000000 | 0 | 0 | 1.002 | 0.0031 |
| 3 | 24 | 1996 | 0 | 0.8489 | 0.0000 | 15.6645 | 32.4300 | 0.246 | 0.065631 | 39 | 0 | 0.926 | 0.0772 |
| 4 | 22 | 3002 | 0 | 0.5447 | 0.0000 | 15.0169 | 85.7692 | 0.454 | 0.033644 | 27 | 0 | 1.047 | 0.0420 |
| 5 | 24 | 1570 | 0 | 1.4513 | 0.0000 | 20.3334 | 210.8301 | 0.246 | 0.099363 | 54 | 0 | 1.031 | 0.0968 |
| 6 | 22 | 1972 | 0 | 0.9131 | 0.0000 | 16.6884 | 22.8504 | 0.214 | 0.068458 | 43 | 0 | 0.937 | 0.0781 |

## Aggregate features

- step-0 mean mismatch_kl: 0.0153
- overall mean / p50 / p99 / max: 0.5570 / 0.0000 / 14.0943 / 210.8301
- early mean (steps<=1): 0.0097
- late mean (steps>=3): 0.8676
- growth ratio (late/early): 89.45
- late catastrophe fraction: 0.061241
- catastrophe sign split (total): trainer-collapsed=163, inference-collapsed=0
- overall top-1% tail mass fraction: 0.385
- importance_ratio mean (should be ~1): 0.9933
- null/non-finite mismatch_kl tokens (divergence signal): 0

## Reference contrast

Reference: **olmo3_7b_ft** (verdict: CLEAN)

| feature | target | reference | delta (tgt-ref) |
|---|---|---|---|
| step-0 mean | 0.0153 | 0.0006 | 0.0147 |
| overall mean | 0.5570 | 0.0006 | 0.5564 |
| overall p50 | 0.0000 | 0.0000 | 0.0000 |
| overall p99 | 14.0943 | 0.0154 | 14.0789 |
| overall max | 210.8301 | 0.5131 | 210.3170 |
| late mean | 0.8676 | 0.0006 | 0.8669 |
| growth ratio | 89.45 | 1.01 | 88.44 |
| late cat_frac | 0.061241 | 0.000000 | 0.061241 |

