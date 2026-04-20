# zebra-puzzles-v2 diagnostic probe — Isambard-AI P2 (a6r)

Purpose: pick the Goldilocks `grid_size` bucket of `joanvelja/zebra-puzzles-v2`
for RLVR stress-testing debate training in prime-rl, using
`Qwen/Qwen3-4B-Instruct-2507` (bf16, sm_90) as the base policy.

## Hardware reminders (from `/Users/joanvelja/isambard/docs/`)

- GH200 = **sm_90 Hopper** → **bf16 only**, no fp8.
- aarch64 compute nodes, x86_64 login nodes.
- HF cache → `$PROJECTDIR` (not `$HOME` 50 GiB cap, not `$LOCALDIR` tmpfs).
- Max 24h walltime; `clifton auth` every 12h for SSH cert.
- Allocation: 50,000 GPU-h, expires 2026-06-20. Probe costs 1 GPU-h.

## Pre-stage on login (no GPU-hours)

```bash
export HF_HOME=$PROJECTDIR/hf

hf download joanvelja/zebra-puzzles-v2 \
  data/train_canonical-00000-of-00002.parquet \
  --repo-type dataset \
  --local-dir $PROJECTDIR/datasets/zebra-puzzles-v2

hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir $HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507
```

## Submit

The sbatch uses the existing prime-rl `.venv` (`uv sync`'d). If anything is
missing, `uv sync` on a compute node first:

```bash
srun --gpus=1 --time=00:15:00 --pty bash -l
cd $PROJECTDIR/prime-rl    # wherever prime-rl is cloned
module load PrgEnv-gnu
uv sync
exit
```

Then:

```bash
cd $PROJECTDIR/prime-rl
sbatch tmp/zebra_probe/probe.sbatch
squeue -u $USER
```

Budget: 2 GPUs × 30 min = 1 GPU-h = 0.002% of allocation.

## Interpreting results

For each grid (`3x3`, `4x4`), reports:

- `pass@1`, `pass@8` — Chen et al. 2021 unbiased estimators
- `solve_none`, `solve_all` — `1 - solve_none - solve_all` = fraction of
  prompts with non-zero GRPO gradient (prime-rl `effective_batch_size`)
- `parse_rate` — `Answer: X` format compliance

**Goldilocks** iff `pass@1 ∈ [0.35, 0.75]` AND `grpo_effective ≥ 0.30`.

If both pass: pick the smaller for speed. If neither: try 5x5 (edit
`GRIDS` in the script) or change model size.

## Files

- `probe_pass_at_n.py` — vLLM pass@{1,8} probe
- `probe.sbatch` — Slurm wrapper (workq, 2 GPUs, 30 min)
- `sample_3x3.json` — format sanity
- `data/*.parquet` — **local Mac copy only**; pre-stage to Isambard separately.
