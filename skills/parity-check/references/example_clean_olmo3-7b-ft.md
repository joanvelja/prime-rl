# Parity diagnostic — olmo3_7b_ft

**VERDICT: CLEAN**

Driving threshold(s):
- step-0 clean (mean=0.0006 <= 0.1), bounded growth (ratio=1.01 < 3.0), late_mean=0.0006 <= 0.05, ~no catastrophe (late_cat_frac=0.000000 <= 0.001)

## Per-step signature (loss_mask=True tokens)

| step | seq | tok | null | mean | p50 | p99 | max | tail% | cat_frac | flipT | flipI | IR_mean | is_masked% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 985 | 0 | 0.0006 | 0.0000 | 0.0128 | 0.0962 | 0.573 | 0.000000 | 0 | 0 | 0.999 | 0.0000 |
| 1 | 16 | 2148 | 0 | 0.0007 | 0.0000 | 0.0143 | 0.1737 | 0.623 | 0.000000 | 0 | 0 | 1.000 | 0.0005 |
| 2 | 16 | 2955 | 0 | 0.0006 | 0.0000 | 0.0158 | 0.0912 | 0.539 | 0.000000 | 0 | 0 | 0.999 | 0.0000 |
| 3 | 18 | 2703 | 0 | 0.0005 | 0.0000 | 0.0140 | 0.0877 | 0.530 | 0.000000 | 0 | 0 | 0.999 | 0.0000 |
| 4 | 18 | 2710 | 0 | 0.0006 | 0.0000 | 0.0161 | 0.0998 | 0.490 | 0.000000 | 0 | 0 | 1.000 | 0.0000 |
| 5 | 22 | 3744 | 0 | 0.0006 | 0.0000 | 0.0144 | 0.2879 | 0.649 | 0.000000 | 0 | 0 | 1.000 | 0.0000 |
| 6 | 16 | 3248 | 0 | 0.0007 | 0.0000 | 0.0172 | 0.1093 | 0.474 | 0.000000 | 0 | 0 | 1.000 | 0.0000 |
| 7 | 22 | 3206 | 0 | 0.0006 | 0.0000 | 0.0170 | 0.0585 | 0.474 | 0.000000 | 0 | 0 | 1.001 | 0.0000 |
| 8 | 20 | 2997 | 0 | 0.0008 | 0.0000 | 0.0150 | 0.5131 | 0.565 | 0.000000 | 0 | 0 | 1.001 | 0.0000 |
| 9 | 16 | 2607 | 0 | 0.0006 | 0.0000 | 0.0145 | 0.1024 | 0.527 | 0.000000 | 0 | 0 | 1.000 | 0.0000 |
| 10 | 20 | 3297 | 0 | 0.0006 | 0.0000 | 0.0151 | 0.1318 | 0.566 | 0.000000 | 0 | 0 | 1.000 | 0.0000 |
| 11 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 | n/a | n/a |

## Aggregate features

- step-0 mean mismatch_kl: 0.0006
- overall mean / p50 / p99 / max: 0.0006 / 0.0000 / 0.0154 / 0.5131
- early mean (steps<=1): 0.0006
- late mean (steps>=3): 0.0006
- growth ratio (late/early): 1.01
- late catastrophe fraction: 0.000000
- catastrophe sign split (total): trainer-collapsed=0, inference-collapsed=0
- overall top-1% tail mass fraction: 0.548
- importance_ratio mean (should be ~1): 1.0000
- null/non-finite mismatch_kl tokens (divergence signal): 0

