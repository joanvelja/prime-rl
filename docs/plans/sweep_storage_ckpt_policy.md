# SFT Sweep Storage & Checkpoint Policy

**Date**: 2026-04-19
**Context**: 3-base SFT sweep (marin-8b, rnj-1, Olmo-3-7B) on Isambard-AI Phase 2,
32 GPUs (8 nodes × 4 GH200), eval every 2500 steps.

## Measured ground truth (2026-04-19)

### Per-checkpoint layout (marin-8b-base example, step_500)

prime-rl SFT writes two parallel dirs per ckpt:

| path | contents | size |
|---|---|---:|
| `<run>/weights/step_N/` | HF-format safetensors + tokenizer + config | **15 GB** |
| `<run>/checkpoints/step_N/trainer/` | FSDP DCP (`.distcp`) optimizer/scheduler/dataloader state | **90 GB** |
| total per ckpt | | **~105 GB** |

Measured via `du -sh` on existing step_500 from 2026-04-17 smoke run
(`sft-smoke-marin`).

### Quota (`/lus/lfs1aip2`, project `brics.a6r`)

- Current user usage: **4.18 TiB** (`lfs quota -h -u joanv.a6r`)
- Project quota: **50 TiB soft / 55 TiB hard** (docs/03-storage.md)
- Free headroom: ~46 TiB

Cluster-total Lustre capacity is 20.3 PiB, but that is NOT the relevant
ceiling — the project quota is.

## Checkpoint cadence for the sweep

`configs/sft/isambard_fullrun_32gpu.toml`:

```toml
[ckpt]
interval = 2500            # save every 2500 steps → 100 min/ckpt at 2.4s/step
keep_last = 1              # always retain most-recent for resume (full state)
keep_interval = 2500       # retain EVERY periodic ckpt (matches eval cadence)
```

### Saved ckpts per base

| base | total steps | ckpts saved | disk (worst case) |
|---|---:|---|---:|
| marin | 13,188 | step_{2500, 5000, 7500, 10000, 12500, 13188} = **6** | 6 × 105 GB = 630 GB |
| rnj-1 | 13,188 | same = **6** | 630 GB |
| olmo3 | 19,782 | step_{2500, 5000, 7500, 10000, 12500, 15000, 17500, 19782} = **8** | 840 GB |
| **sweep total** | | **20 ckpts** | **~2.1 TB** (4% of quota) |

## Why we don't set `weights_only=true`

The prime-rl config flag `[ckpt].weights_only = true`
(`src/prime_rl/configs/trainer.py:573`) is **global** — when set, NO training
state is saved on any ckpt, including the rolling-last-1 that `keep_last`
would normally keep. This breaks mid-training resume entirely (job-kill,
node-fail recovery impossible).

prime-rl's ckpt system is all-or-nothing here. There is no per-ckpt flag to
distinguish "keep full state for latest, weights-only for periodic." To add
that would require a code change to `src/prime_rl/trainer/ckpt.py` and its
config, which is out-of-scope for this sweep.

## Workaround: post-training pruning

Script: `scripts/prune_ckpt_training_state.py`.

After training completes successfully, prune all `checkpoints/step_N/trainer/`
dirs except the final step. Weights under `weights/step_N/` are untouched, so
eval sweeps still work.

Safety rails in the script:
- Refuses to run if a `job_*.log` in the run dir has mtime < 15 min (training
  may still be writing).
- Dry-run by default — pass `--execute` to actually delete.
- Only touches `checkpoints/`, never `weights/`.

### Storage savings from pruning (per base)

| base | pre-prune | post-prune | saved |
|---|---:|---:|---:|
| marin (6 ckpts) | 6 × 105 = 630 GB | 6 × 15 + 1 × 90 = 180 GB | **450 GB** |
| rnj-1 (6 ckpts) | 630 GB | 180 GB | 450 GB |
| olmo3 (8 ckpts) | 840 GB | 8 × 15 + 1 × 90 = 210 GB | 630 GB |
| **sweep** | **2.1 TB** | **570 GB** | **~1.5 TB** |

Usage:
```bash
# dry run
uv run python scripts/prune_ckpt_training_state.py \
    --run-dir /lus/lfs1aip2/scratch/a6r/joanv.a6r/outputs/sft-baseline-marin

# execute (after confirming no training activity)
uv run python scripts/prune_ckpt_training_state.py \
    --run-dir /lus/lfs1aip2/scratch/a6r/joanv.a6r/outputs/sft-baseline-marin \
    --execute
```

## Eval workflow (new scripts)

- `scripts/evals/eval_all_ckpts.py` — iterates every `weights/step_N/` under a
  run's ckpt dir, calls `scripts/evals/run_all.py` per ckpt, writes per-step
  rollups. Idempotent (skips already-evaluated ckpts).
- `scripts/evals/aggregate_curves.py` — (to be written) reads all
  `step_N/rollup.json` + training log, emits a CSV suitable for
  plotting training curves (MFU, loss, eval metrics vs step).

## Operational checklist before launching a sweep

1. `lfs quota -h -u $USER /lus/lfs1aip2` — confirm < 45 TiB used
2. Dry-run the compose: `uv run sft @ baseline.toml @ overrides/<base>.toml
   @ isambard_fullrun_32gpu.toml --dry-run` — verify `ckpt.interval=2500`,
   `reshard_after_forward=true`, `num_nodes=8`.
3. Launch, monitor via `tail -F <output>/logs/trainer.log`.
4. After each base finishes, `prune_ckpt_training_state.py` (dry run first).
5. After all 3 finish, kick off `eval_all_ckpts.py` per base (single-GPU
   allocation; 20 ckpts × ~40 min each = ~13 h sequential, or faster via
   SLURM array).
6. Aggregate with `aggregate_curves.py`.
