---
name: train-sft
description: Launch and tune prime-rl supervised fine-tuning runs. Use when asked to configure or run `uv run sft`, prepare SFT datasets, choose batch or sequence settings, resume training, or debug trainer memory and performance issues.
---

# Train SFT

## Goal
Run reproducible SFT jobs that are compatible with later RL fine-tuning.

## Preferred Workflow
1. Start from the closest `examples/*/sft.toml` or `configs/debug/sft/train.toml`.
2. Prefer the `sft` entrypoint for single-node training:

```bash
uv run sft @ path/to/sft.toml
```

It launches `torchrun` internally based on `deployment.num_gpus`.

3. Use `--dry-run` before long jobs or SLURM submissions:

```bash
uv run sft @ path/to/sft.toml --dry-run
```

4. For multi-node non-SLURM setups, use manual `torchrun` as shown in `docs/deployment.md`.

## Dataset Requirements
- Use datasets in `messages` format or prompt-completion format.
- If `messages` is present, it takes precedence over `prompt` and `completion`.
- The tokenizer chat template must satisfy the prefix property for correct loss masking. Do not assume every instruct model already does.
- Keep role-based loss masking defaults unless the user has a clear reason to train on user, system, or tool tokens.

## Hard Constraints
1. `data.batch_size` must be divisible by `data.micro_batch_size`.
2. `data.batch_size` must be at least `data.micro_batch_size`.
3. If `model.cp > 1`, keep `data.micro_batch_size = 1`.

## First Knobs To Change
1. `model.name`
2. `data.name`, `data.subsets`, `data.splits`
3. `data.batch_size`, `data.micro_batch_size`, `data.seq_len`
4. `optim.lr`
5. `max_steps` and `output_dir`
6. Optional `model.lora`

## Memory And Scale
1. Reduce `data.micro_batch_size` before reducing global `data.batch_size`.
2. Reduce `data.seq_len` if the trainer OOMs.
3. Then enable memory savers as needed:
- `model.ac.freq = 1`
- `model.ac_offloading.max_inflight_activations = 5`
- `model.optim_cpu_offload = true`
- `model.lora`
4. For single-node multi-GPU training, set `deployment.type = "single_node"` and increase `deployment.num_gpus`.
5. For SLURM, keep the config in the repo and validate with `--dry-run` first.

## Diagnosis
- Noisy or flat loss: verify the dataset format and chat template before retuning the optimizer.
- Trainer OOM: reduce `data.seq_len` or `data.micro_batch_size` first.
- Slow steps without OOM: check whether activation offloading or CPU optimizer offload is the trade-off you want, or whether the global batch is simply too large for the hardware.

## Resume Pattern

```bash
uv run sft @ path/to/sft.toml --max-steps 20 --ckpt
uv run sft @ path/to/sft.toml --max-steps 40 --ckpt.resume-step 20
```

## Related Skills
- Read `entrypoints` for launcher behavior.
- Read `config` for TOML composition and CLI overrides.
- Read `monitor-run` when diagnosing live trainer logs.

## Deliverable
Return:
1. The example or template the run started from.
2. Dataset and model choices.
3. Config deltas applied.
4. The exact launch command.
5. The expected checkpoint or output path.
