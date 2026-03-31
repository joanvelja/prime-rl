---
name: train-rl
description: Launch and tune prime-rl reinforcement learning runs. Use when asked to configure or run `uv run rl`, adapt `rl.toml`, choose rollout, trainer, or inference settings, resume RL runs, or diagnose rollout and trainer bottlenecks.
---

# Train RL

## Goal
Launch stable RL jobs that match the user's hardware, environment, and rollout cost budget.

## Default Workflow
1. Start from the closest `examples/*/rl.toml`.
2. Prefer a single unified `rl.toml` for local or single-node runs.
3. Use `bash scripts/tmux.sh` for local multi-process workflows when the user wants separate panes for logs and components.
4. Validate the resolved config before long jobs:

```bash
uv run rl @ path/to/rl.toml --dry-run
```

## Deployment Choice
- Single node: use `deployment.type = "single_node"` and launch with `uv run rl @ ...`.
- Larger runs: use `deployment.type = "multi_node"` or start from an example `slurm_rl.toml`.
- Multi-node RL requires a shared filesystem and exactly one orchestrator per run.

## First Knobs To Change
1. `output_dir`, `model.name`, `max_steps`, `seq_len`
2. `[[orchestrator.env]]` IDs, names, and args
3. `orchestrator.batch_size`, `orchestrator.rollouts_per_example`, `orchestrator.sampling.max_tokens`
4. `deployment.*` and `inference.parallel.*`
5. `trainer.optim.lr` and optional `trainer.model.lora`

## Hard Constraints
1. Set exactly one of `orchestrator.batch_size` or `orchestrator.token_batch_size`.
2. `orchestrator.batch_size` must be divisible by `orchestrator.rollouts_per_example`.
3. If `orchestrator.token_batch_size` is used, set `orchestrator.max_inflight_rollouts`.
4. If `orchestrator.max_concurrent` is set, it must be at least `orchestrator.rollouts_per_example`.
5. `weight_broadcast.type = "nccl"` requires `max_async_level = 1`.
6. NCCL weight broadcast does not support LoRA.

## Rollout Geometry
1. `orchestrator.rollouts_per_example` controls group size for per-example advantages and is one of the most important RL knobs.
2. `orchestrator.batch_size` is total rollout samples per step, not number of prompt groups.
3. Good starting points:
- Simple smoke tests: `rollouts_per_example = 8`, `batch_size = 128`
- Common single-node or moderate runs: `rollouts_per_example = 8` or `16`, `batch_size = 512`
- Large cluster runs: `rollouts_per_example = 16`, then scale `batch_size` to `1024-4096` only after throughput is stable
4. `orchestrator.oversampling_factor = 2.0` is a common starting point for rollout-mode batching.
5. If rollouts are long or expensive, lower `orchestrator.sampling.max_tokens` before shrinking `rollouts_per_example`.

## Sampling And Exploration
1. Training sampling defaults to `orchestrator.sampling.temperature = 1.0` if neither `orchestrator.sampling.temperature` nor `orchestrator.sampling.temp_scheduler` is set.
2. Start with `orchestrator.sampling.temperature` around `0.7-1.0`.
3. Lower `orchestrator.sampling.temperature` if outputs are noisy, malformed, or mostly low reward.
4. Raise `orchestrator.sampling.temperature` cautiously only when rollouts collapse to near-identical completions and advantages vanish.
5. Use `orchestrator.sampling.temp_scheduler` instead of constant temperature when you want exploration to decay over training.
6. Leave `orchestrator.sampling.repetition_penalty = 1.0` unless the model gets stuck in repetition loops.
7. Use `orchestrator.sampling.min_tokens` only if the model is ending generations too early.

## Async And Off-Policy
1. Use `max_async_level = 0` for synchronous debugging.
2. Use `max_async_level = 1` for the standard async path and whenever NCCL broadcast is enabled.
3. Increase `max_off_policy_steps` only after measuring throughput pressure. The default is `8`; larger example runs often use `16`.
4. Leave `strict_async_level = false` unless you need tightly controlled staleness for an experiment.
5. If eval quality regresses while throughput looks good, reduce async aggressiveness before touching optimizer knobs.

## Optimizer, Loss, And Distillation
1. `trainer.optim.lr` is usually `1e-6` for large full-model runs and `3e-6` to `1e-5` for smaller or LoRA-heavy jobs.
2. Start with `trainer.optim.max_norm = 1.0` and treat it as a safety knob, not a first-line tuning lever.
3. Treat `weight_decay` as secondary regularization. Existing examples span `0.0` to `0.1`.
4. Leave `trainer.loss.adv_tau = 1.0` and `trainer.loss.kl_tau = 1e-3` unless you are intentionally rebalancing reward learning versus KL regularization.
5. Use `trainer.loss.teacher_tau > 0` only with a configured teacher model or teacher server. Set `trainer.loss.adv_tau = 0` for pure distillation, where you can also set `verification.enabled = true` to save compute.
6. Leave `trainer.loss.ipo_mask_low` and `trainer.loss.ipo_mask_high` at defaults unless you are intentionally changing token masking behavior.

## Difficulty Filtering And Advantage Shaping
1. Turn on `orchestrator.buffer.online_difficulty_filtering = true` only after rewards are trustworthy.
2. `orchestrator.buffer.easy_threshold = 1.0` and `orchestrator.buffer.hard_threshold = 0.0` are common starting points for binary-ish rewards.
3. Use `orchestrator.buffer.easy_fraction` and `orchestrator.buffer.hard_fraction` to bleed some filtered examples back into normal sampling instead of fully excluding them.
4. `advantage.length_shaping_alpha = 0.33` is the recommended GR^3 starting value and requires `online_difficulty_filtering`.

## Throughput And Stability
- Lower `orchestrator.sampling.max_tokens` first when rollouts are too expensive.
- Keep `max_async_level` low on first runs. Use `0` for synchronous debugging and `1` for the normal async path.
- Use `orchestrator.oversampling_factor` or `orchestrator.max_inflight_rollouts` to keep the trainer fed without letting rollout cost explode.
- Turn on `orchestrator.buffer.easy_threshold`, `hard_threshold`, or `online_difficulty_filtering` only after rewards are already meaningful.

## Memory Tuning
1. Reduce `seq_len` first.
2. Reduce `orchestrator.batch_size` or `orchestrator.token_batch_size` next.
3. Then enable memory savers as needed:
- `trainer.model.ac.freq = 1`
- `trainer.model.ac_offloading.max_inflight_activations = 5`
- `trainer.model.optim_cpu_offload = true`
- `trainer.model.lora`

## Diagnosis
- High `time/wait_for_batch`: inference or environment is the bottleneck.
- High `time/wait_for_ckpt`: trainer is the bottleneck.
- Flat reward: inspect rollout samples and reward quality before assuming optimizer issues.
- Unstable reward swings: lower `trainer.optim.lr`, lower `orchestrator.sampling.temperature`, or reduce async aggressiveness before changing advanced loss knobs.
- Trainer OOM: shrink `seq_len` and trainer batch pressure before adding more advanced features.

## Resume Pattern

```bash
uv run rl @ path/to/rl.toml --max-steps 10 --ckpt
uv run rl @ path/to/rl.toml --max-steps 20 --ckpt.resume-step 10
```

The inference server is stateless. On resume, the orchestrator will reload the correct weights into inference.

## Related Skills
- Read `entrypoints` for launcher behavior.
- Read `config` for TOML composition and CLI overrides.
- Read `monitor-run` for log paths and runtime bottleneck analysis.

## Deliverable
Return:
1. The example or template the run started from.
2. Config deltas applied.
3. Why each major RL knob was changed.
4. The exact launch command.
5. Which trainer and orchestrator metrics to watch first.
