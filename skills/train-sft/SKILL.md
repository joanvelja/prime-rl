---
name: train-sft
description: Launch and tune prime-rl supervised fine-tuning runs. Use when asked to configure or run `uv run sft`, prepare SFT datasets, choose batch or sequence settings, resume training, or debug trainer memory and performance issues.
---

# Train SFT

## Goal
Run reproducible SFT jobs that are compatible with later RL fine-tuning.

## Default Workflow
1. Start from the closest `examples/*/sft.toml` or `configs/debug/sft/train.toml`.
2. Launch with:

```bash
uv run sft @ path/to/sft.toml
```

## Deployment Choice
- For multi-node or SLURM-specific setups, follow `docs/deployment.md`.

## Dataset Requirements
- Use datasets in `messages` format or prompt-completion format.
- If `messages` is present, it takes precedence over `prompt` and `completion`.
- The tokenizer chat template must satisfy the prefix property for correct loss masking. Do not assume every instruct model already does.
- Keep role-based loss masking defaults unless the user has a clear reason to train on user, system, or tool tokens.

## First Knobs To Change
1. `model.name`
2. `data.name`, `data.subsets`, `data.splits`
3. `data.batch_size`, `data.micro_batch_size`, `data.seq_len`
4. `optim.lr`
5. `max_steps` and `output_dir`
6. `model.lora` when hardware makes full-model SFT impractical

## Optimization
1. `optim.lr` is usually the first optimization knob to touch.
2. Keep the data format and chat template stable before changing optimization settings.
3. Do not change loss masking defaults unless the user explicitly wants a different training target.

## Diagnosis
- Noisy or flat loss: verify the dataset format and chat template before retuning the optimizer.
- Trainer OOM: follow `docs/memory_usage.md`.
- Slow steps without OOM: check whether the batch size, sequence length, or deployment choice fits the available hardware.

## Resume Pattern

```bash
uv run sft @ path/to/sft.toml --max-steps 20 --ckpt
uv run sft @ path/to/sft.toml --max-steps 40 --ckpt.resume-step 20
```

## Related Skills
- Read `entrypoints` for launcher behavior.
- Read `config` for TOML composition and CLI overrides.
- Read `monitor-run` when diagnosing live trainer logs.
- Read `docs/memory_usage.md` for trainer memory tradeoffs.
- Read `docs/deployment.md` for multi-node and SLURM setups.

## Deliverable
Return:
1. The example or template the run started from.
2. Dataset and model choices.
3. Config deltas applied.
4. The exact launch command.
5. The expected checkpoint or output path.
