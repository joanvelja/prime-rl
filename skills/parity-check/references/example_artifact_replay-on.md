# Parity diagnostic — gemma4-replay-on

**VERDICT: ARTIFACT**

Driving threshold(s):
- overall p50(median) mismatch_kl=12.4821 > artifact_median=1.0 -> bulk of tokens mismatched (uniform elevation), not a tail

## Per-step signature (loss_mask=True tokens)

| step | seq | tok | null | mean | p50 | p99 | max | tail% | cat_frac | flipT | flipI | IR_mean | is_masked% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 1059 | 0 | 12.2461 | 12.4821 | 29.6128 | 38.0708 | 0.027 | 0.805477 | 644 | 0 | 0.085 | 0.4542 |
| 1 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | n/a | n/a |

## Aggregate features

- step-0 mean mismatch_kl: 12.2461
- overall mean / p50 / p99 / max: 12.2461 / 12.4821 / 29.6128 / 38.0708
- early mean (steps<=1): 12.2461
- late mean (steps>=3): n/a
- growth ratio (late/early): n/a
- late catastrophe fraction: 0.805477
- catastrophe sign split (total): trainer-collapsed=644, inference-collapsed=0
- overall top-1% tail mass fraction: 0.027
- importance_ratio mean (should be ~1): 0.0848
  - WARNING: |IR_mean - 1| = 0.9152 > ir_center_tol=0.25 (IR off-center)
- null/non-finite mismatch_kl tokens (divergence signal): 0

