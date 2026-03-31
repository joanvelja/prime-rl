---
name: train-rl
description: Launch and tune prime-rl reinforcement learning runs. Use when asked to configure or run `uv run rl`, adapt `rl.toml`, choose rollout, trainer, or inference settings, resume RL runs, or diagnose rollout and trainer bottlenecks.
---

# Train RL

## Goal
Launch stable RL jobs that match the user's hardware, environment, and rollout cost budget.

## Default Workflow
1. Start from the closest `examples/*/rl.toml`.
2. Launch with:

```bash
uv run rl @ path/to/rl.toml
```

## Deployment Choice
- Larger runs: use `deployment.type = "multi_node"` or start from an example `slurm_rl.toml`.
- Multi-node RL requires a shared filesystem and exactly one orchestrator per run.

## Sampling
1. Touch `orchestrator.sampling.temperature` only when entropy is not around `0.2-0.5` which is a good range to be in
2. `orchestrator.oversampling_factor` is the main throughput knob if the trainer is starving.

## Optimization
1. `trainer.optim.lr` is usually the first optimization knob to touch.
2. Do not change `trainer.loss.*` unless the user explicitly asks for distillation or objective changes.
3. Use `orchestrator.buffer.*` only when reward quality is already good enough to support difficulty-based sampling.

## Diagnosis
- High `time/wait_for_batch`: inference or environment is the bottleneck.
- High `time/wait_for_ckpt`: trainer is the bottleneck.
- Flat reward: inspect rollout samples and reward quality before assuming optimizer issues.
- Unstable reward swings: lower `trainer.optim.lr` or `orchestrator.sampling.temperature` before changing advanced loss knobs.
- Trainer OOM: follow `docs/memory_usage.md`.

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
- Read `docs/memory_usage.md` for trainer memory tradeoffs.
- Read `docs/on_policy_distillation.md` for distillation-specific tuning.

## Deliverable
Return:
1. The example or template the run started from.
2. Config deltas applied.
3. Why each major RL knob was changed.
4. The exact launch command.
5. Which trainer and orchestrator metrics to watch first.
