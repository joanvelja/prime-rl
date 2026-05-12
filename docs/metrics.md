# Metrics

## W&B

For most runs we recommend logging metrics to [W&B](https://wandb.ai). Before enabling W&B, make sure that you have an account and are logged in.

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

### SFT

Logging to W&B is disabled by default. Enable the default configuration with `--wandb`

```bash
uv run sft ... --wandb
```

This will log to the `prime-rl` project with a random run name. You can specify which project and name to log to 

```bash
uv run sft ... --wandb.project my-project --wandb.name my-run
```

The same settings also work for multi-node training with `torchrun`. Note, that we only log global metrics from the master rank (e.g. the all-reduced loss)

```bash
uv run torchrun --nproc-per-node 8 ...  --wandb
```

### RL

For RL training, both the trainer and orchestrator log to W&B as separate runs. Again, logging to W&B is disabled by default. Enable the default configuration with `--wandb`

```bash
uv run rl ... --wandb
```

This will log to the `prime-rl` project with a random run name. The trainer run is suffixed with `-trainer` and the orchestrator run is suffixed with `-orchestrator`. You can specify which project and name to log to using the same flags as for SFT.

```bash
uv run rl ... --wandb.project my-project --wandb.name my-run
```

For the RL trainer, we support logging samples (e.g. prompt, completion, reward, advantage for selected rollouts) and distributions (e.g. reward, advantage, entropy distributions) as W&B tables using the `wandb.log-extras` subconfig. If W&B is setup, this is enabled by default and will log for the RL trainer and orchestrator every 10 steps.

Useful RL monitoring metrics include:

| Metric | Run | Description |
|---|---|---|
| `time/wait_for_batch` | trainer | Trainer wall time spent waiting for rollout batches. High values usually mean rollout generation or env execution is the bottleneck. |
| `time/forward_backward` | trainer | Trainer compute time for the accepted token batch. |
| `mismatch_kl/mean` | trainer | Drift between trainer logprobs and rollout-time inference logprobs. |
| `optim/grad_norm` | trainer | Gradient norm before clipping. Compare against the configured `max_norm`. |
| `scheduler/async_level` | orchestrator | Current trainer/inference async level. With NCCL broadcast this should stay at 1. |
| `scheduler/inflight_rollouts` | orchestrator | Number of rollout groups currently in flight. |
| `scheduler/cancelled_rollouts` | orchestrator | Stale rollout groups cancelled instead of used. Spikes near `max_inflight_rollouts` mean queue geometry or eval cancellation is wasting generated samples. |
| `off_policy_level/all/mean` | orchestrator | Average policy staleness of consumed rollout groups. |
| `is_truncated/all/mean` | orchestrator | Fraction of completions hitting the completion cap. |

You can configure this on the trainer and orchestrator separately. For example, to only log samples on the orchestrator every 50 steps, but not distribution on either

```bash
uv run rl  ... \
  --no-trainer.wandb.log-extras.distributions \
  --orchestrator.wandb.log-extras.interval 50
```
