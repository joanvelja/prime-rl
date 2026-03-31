---
name: train-with-environments
description: Train models with prime-rl on verifiers environments. Use when asked to choose between SFT warmup and direct RL, adapt an example config to a new environment, set up baseline evals, or plan environment-backed training runs end to end.
---

# Train With Environments

## Goal
Turn a working environment into a reproducible SFT and/or RL run with the smallest possible delta from an existing example.

## Preferred Starting Point
1. Do not build configs from scratch unless the user explicitly asks.
2. Start from the closest example in `examples/`:
- `reverse_text` for tiny single-turn SFT+RL smoke tests
- `wordle` for multi-turn SFT+RL
- `alphabet_sort` for LoRA-first RL without SFT warmup
- `wiki_search` for tool use and difficulty filtering
- `hendrycks_sanity` for RL algorithm sanity checks
3. Prefer the high-level entrypoints:

```bash
uv run sft @ path/to/sft.toml
uv run rl @ path/to/rl.toml
```

## First-Run Protocol
1. Install the environment and verify the Python package if needed:

```bash
prime env install owner/my-env
uv run python -c "import my_env"
```

2. Get a baseline against a local inference server before training:

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
uv run vf-eval my-env -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 -n 20
```

3. If the base model already follows the environment format well, consider going straight to RL.
4. If the model fails format, tool-calling, or multi-turn conventions, do a short SFT warmup first.

## Choose The Training Path
- Prefer SFT first when the base model does not understand the response format, tool schema, or conversational structure.
- Prefer direct RL when the base model already behaves structurally correctly and mainly needs reward-driven improvement.
- Prefer LoRA when hardware or turnaround time makes full-model training impractical.

## Keep Deltas Small
1. Change model, environment IDs or args, `output_dir`, and `max_steps` first.
2. Only change RL-specific knobs when the closest example is clearly not enough for the environment or hardware.
3. Only change SFT-specific knobs when the base model clearly needs format or tool-use warmup.
4. Read `train-rl` before changing RL batching, sampling, or optimization settings.
5. Read `train-sft` before changing SFT dataset, memory, or deployment settings.

## Failure Triage
- Flat reward or no learning: inspect baseline samples first. The problem is often environment format or reward quality, not just trainer hyperparameters.
- Trainer OOM: reduce sequence length and follow `train-rl`, `train-sft`, or `docs/memory_usage.md` instead of guessing.
- High `time/wait_for_batch`: inference or environment is the bottleneck.
- High `time/wait_for_ckpt`: trainer is the bottleneck.

## Related Skills
- Read `train-rl` for RL-specific tuning.
- Read `train-sft` for SFT-specific tuning.
- Read `entrypoints` for command coverage.
- Read `config` for TOML composition and CLI overrides.
- Read `monitor-run` when diagnosing live runs.

## Deliverable
Return:
1. Which example the run was based on.
2. Config deltas applied.
3. Why SFT, RL, or both were chosen.
4. The exact launch command.
5. The first metrics or logs to watch.
