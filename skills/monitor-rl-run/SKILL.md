---
name: monitor-rl-run
description: How to monitor an ongoing RL training run — find output directories, check logs, diagnose performance, and inspect SLURM jobs. Use when asked to check on a run, debug training issues, or investigate performance.
---

# Monitor an RL Run

## Find the output directory

The output directory is set in the config (`output_dir`). To find it:

- **Local run**: check the resolved config at `{output_dir}/configs/rl.toml`
- **SLURM run**: check `squeue -u $USER` to find the job, then look at the sbatch script or the config dir

## Log files

### Local runs

```
{output_dir}/logs/
├── rl.log                    # main launcher log
├── inference.stdout          # vLLM inference server
├── orchestrator.stdout       # orchestrator process
├── trainer.stdout            # torchrun wrapper output
├── trainer/
│   ├── rank_0.log            # per-rank trainer logs (rank 0 is most useful)
│   └── ...
└── envs/
    ├── train/{env_name}/
    │   ├── env_server.log
    │   └── env_worker_{id}.log
    └── eval/{env_name}/
        └── ...
```

### SLURM runs

```
{output_dir}/slurm/
├── latest_train_node_rank_{N}.log        # trainer nodes
├── latest_orchestrator.log               # orchestrator
├── latest_infer_node_rank_{N}.log        # inference nodes
├── latest_router_replica_{N}.log         # vllm-router (if multi-node inference)
└── job_{SLURM_JOB_ID}_*.log             # permanent copies
```

Env server logs are still under `{output_dir}/logs/envs/`.

## SLURM: check node allocation

```bash
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
```

In a multi-node RL run, nodes are split between trainer and inference:
- **Trainer nodes**: run torchrun (rank 0 also runs the orchestrator)
- **Inference nodes**: run vLLM (first node of each replica also runs vllm-router)

The node assignment is visible in the SLURM logs and the generated sbatch script at `{output_dir}/rl.sbatch`.

## Performance: trainer

Check `{output_dir}/logs/trainer/rank_0.log` or the SLURM trainer log.

Key metrics per step:
- `time/step` — total step time
- `time/wait_for_batch` — time waiting for the orchestrator to deliver a batch
- `time/forward_backward` — forward/backward pass time
- `time/broadcast_weights` — time broadcasting weights to inference servers
- `time/save_ckpt` — checkpoint save time

High `wait_for_batch` means the orchestrator is the bottleneck (slow rollouts, slow envs, or too few inference replicas).

## Performance: orchestrator

Check `{output_dir}/logs/orchestrator.log` or the SLURM orchestrator log.

Key metrics per step:
- `time/step` — total orchestrator step time
- `time/generate_completions` — rollout generation time
- `time/wait_for_ckpt` — time waiting for trainer checkpoint
- `time/update_weights` — weight update time
- `scheduler/async_level` — current async level
- `empty_rollouts/all` — fraction of empty rollouts
- `errored_rollouts/all` — fraction of errored rollouts

High `wait_for_ckpt` means the trainer is the bottleneck. The orchestrator logs when it pauses/resumes:
```
"Orchestrator paused: waiting for trainer process to complete checkpoint ..."
"Orchestrator resumed: checkpoint ... ready (after ...s)"
```

## Performance: env servers

Check `{output_dir}/logs/envs/train/{env_name}/env_worker_{id}.log`.

Key things to look for:
- **Event loop lag**: workers log lag stats periodically. A warning is emitted when median > 0.5s or p90 > 1.0s or max > 5.0s — this means the worker is overloaded.
- **Active task distribution**: check if tasks are evenly distributed across workers per-env and across envs. Uneven distribution suggests some workers/envs are slower.

## Quick health check

```bash
# Tail the most important logs
tail -F {output_dir}/logs/trainer/rank_0.log        # training progress
tail -F {output_dir}/logs/orchestrator.log           # rollout generation
tail -F {output_dir}/logs/inference.stdout           # inference server health

# Check for errors across all logs
grep -r "ERROR\|Exception\|Traceback" {output_dir}/logs/ --include="*.log"

# Check inference server health
curl http://{infer_host}:{port}/health
```

## Key files

- `src/prime_rl/entrypoints/rl.py` — local and SLURM launch logic, log path setup
- `src/prime_rl/templates/multi_node_rl.sbatch.j2` — multi-node SLURM template with node roles
- `src/prime_rl/orchestrator/scheduler.py` — async scheduling, wait_for_ckpt logging
- `src/prime_rl/trainer/rl/train.py` — training loop, wait_for_batch logging
- `src/prime_rl/orchestrator/env_server/event_loop_lag.py` — event loop lag monitoring
