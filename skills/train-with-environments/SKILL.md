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

4. Run `--dry-run` before long jobs or SLURM submissions:

```bash
uv run sft @ path/to/sft.toml --dry-run
uv run rl @ path/to/rl.toml --dry-run
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
- Prefer LoRA when VRAM is tight, when training on one or a few envs, and when the batch size is small


## RL Rules Of Thumb
1. Keep `orchestrator.batch_size` divisible by `orchestrator.rollouts_per_example`.
2. Keep `orchestrator.max_concurrent` and `orchestrator.max_inflight_rollouts` at least `rollouts_per_example`.
3. Simple smoke-test starting point: `rollouts_per_example = 8`, `batch_size = 128`, `oversampling_factor = 2.0`.
4. Common stable starting point: `rollouts_per_example = 16`, `batch_size = 512`, `oversampling_factor = 2.0`.
5. Reduce `orchestrator.sampling.max_tokens` before touching more exotic knobs when rollouts are too slow or memory-heavy. Mainly prefer this if the truncation rate is low.
6. A good initial `orchestrator.sampling.temperature` is around `0.7-1.0`. Change this depending on the entropy, a good heuristic is to keep entropy around `0.3-0.5`. Increase `orchestrator.sampling.temperature` to increase entropy and vice versa.
7. Keep `max_async_level` low on first runs. Use `0` for fully synchronous debugging or `1` for the usual async path. `nccl` weight broadcast requires `max_async_level = 1`.
8. Turn on difficulty filtering only after rewards are meaningful enough to separate easy and hard cases.

## SFT Rules Of Thumb
1. Use datasets in `messages` or prompt-completion format.
2. If `messages` is present, it takes precedence over `prompt` and `completion`.
3. Start from the closest example `sft.toml` and first change only model, dataset, batch size, sequence length, and output directory.

## Failure Triage
- Flat reward or no learning: inspect baseline samples first. The problem is often environment format or reward quality, not just trainer hyperparameters.
- Trainer OOM: reduce sequence length or batch pressure, then enable activation checkpointing, activation offloading, CPU optimizer offload, or LoRA.
- High `time/wait_for_batch`: inference or environment is the bottleneck.
- High `time/wait_for_ckpt`: trainer is the bottleneck.

## Related Skills
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
