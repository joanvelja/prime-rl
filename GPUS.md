# GPU Utilization Tracking: 8-node OmniMath2 OLMo3 RLVR

Purpose: keep utilization experiments falsifiable. Do not treat a created tmux
window, generated launcher, or latest-state GPU snapshot as evidence of GPU
utilization.

## Current Jobs

As of 2026-05-13 09:46 UTC:

| field | value |
|---|---|
| `1e-6` refill train job | `4570549`, timed out after 6h with stable checkpoints `25,50,75` |
| `1e-6` refill eval job | `4582655`, running on a separate 8-node allocation |
| `3e-6` refill train job | `4574276`, timed out at Slurm level after reaching `step_100` in RL logs |
| `3e-6` refill eval job | `4582691`, running on a separate 8-node allocation |
| live interactive allocation | `4574749` / `joanv_cc_8node`; user may be working on these GPUs |
| monitor pane | `joanv_cc_8node:4 eval-watch` |
| comparison report | `outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md` |
| refill caveat | `1e-6` used refill v1/full-candidate-batch; `3e-6` used optimized candidate-batched refill |

The live allocation is shared with the user. Do not run cleanup, `srun`, or GPU
probes against it without explicit confirmation. Read-only log inspection and
`squeue` are fine.

## Historical Allocation Snapshot

As of 2026-05-12 14:11 UTC:

| field | value |
|---|---|
| allocation | `4555723` |
| nodes | `nid010685`, `nid010752`, `nid010753`, `nid010756`, `nid010757`, `nid010758`, `nid010765`, `nid010768` |
| GPUs | 8 nodes x 4 GH200 = 32 GPUs |
| best-known topology | 28 inference GPUs / 4 trainer GPUs via `multi_node` |
| current default config | `configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4.toml` |
| default batch shape | `batch_size = 256`, `max_inflight_rollouts = 768`, `rollouts_per_example = 8` |
| exploratory larger batch | `batch_size = 512`, `max_inflight_rollouts = 1024`; viable but slower and KV-pressure-limited |

## Telemetry Rules

`GPUSTAT_DIR` files from the allocation wrapper are live latest-state snapshots
only. They are useful while the run is alive, but they are not historical
utilization logs and cannot support after-the-fact utilization claims.

Historical GPU utilization requires an explicit sampler such as
`scripts/monitoring/gpu_telemetry_loop.sh`, which writes per-host CSV rows:

```text
ts,host,index,name,util_pct,mem_used_mib,mem_total_mib,power_w,temperature_c
```

W&B/log metrics are the training time series. For these runs, track at least:
`perf/mfu`, trainer tokens/sec if present, `time/step`,
`time/wait_for_batch`, `time/forward_backward`, `time/generate_completions`,
`time/update_weights`, inference queue/decode metrics, off-policy level,
rollout buffer length/staleness if present, pool fractions, reward
distribution, truncation, and eval pass@k.

## Experiment Record Template

| field | value |
|---|---|
| label | BOTEC / smoke / canary |
| hypothesis | one sentence |
| config | temp config path plus relevant diff |
| launch command | exact command |
| allocation | job id, node list, node count, GPUs/node |
| topology | train GPUs, inference GPUs, host mapping |
| run dir | output path |
| W&B | trainer/orchestrator run ids or URLs |
| telemetry | CSV dir, sampling interval, start/end UTC |
| live-only snapshots | `GPUSTAT_DIR` path, marked non-historical |
| validity | failed before launcher / failed before serving / reached N steps |
| comparison window | e.g. steps 2-5, excluding step 0/1 pipeline fill |
| trainer metrics | mean MFU, tokens/sec, step time, wait_for_batch, fwd/bwd |
| orchestrator metrics | generate time, update_weights, inflight, off-policy mean/max |
| inference pressure | requests running/waiting, decode TPS, prompt TPS if available |
| GPU metrics | mean/p50/p95 utilization, memory, power by role/node |
| bottleneck call | trainer-bound / rollout-bound / eval-stalled / inconclusive |
| decision | promote / retry with change / discard |

## 2026-05-12 24i/8t Smoke Plan

Status: superseded as the default path. It remains here as historical context
for the first 8-node smoke. Follow-up 28i/4t runs matched or beat 24i/8t wall
time with higher trainer MFU because trainer FSDP stayed single-node.

Temp configs:

- `tmp/rl_olmo3_dpo_default_24i8t_bs256_smoke_20260512.toml`
- `tmp/rl_olmo3_dpo_maxrl_24i8t_bs256_smoke_20260512.toml`

Static target:

```toml
[deployment]
type = "multi_node"
gpus_per_node = 4
num_train_nodes = 2
num_infer_nodes = 1
num_infer_replicas = 6
nodes_per_fsdp_group = 2
```

This should render 8 Slurm nodes: 6 one-node inference replicas and 2 trainer
nodes. With `tp = 1`, resolved inference should be `parallel.dp = 4`,
`data_parallel_size_local = 4`, and `api_server_count = 4`, giving 24 vLLM
GPU workers. Trainer world size should be 8.

Preflight checks before any throughput claim:

```bash
uv run --no-sync rl @ tmp/rl_olmo3_dpo_default_24i8t_bs256_smoke_20260512.toml
uv run --no-sync rl @ tmp/rl_olmo3_dpo_maxrl_24i8t_bs256_smoke_20260512.toml
bash -n outputs/omni_math2_rlvr_canary/default_24i8t_bs256_smoke_20260512/rl.sbatch
bash -n outputs/omni_math2_rlvr_canary/maxrl_24i8t_bs256_smoke_20260512/rl.sbatch
```

Inspect generated values:

```bash
rg -n 'num_train_workers|batch_size|max_inflight_rollouts|rollouts_per_example|easy_threshold|hard_threshold|dp_rank_count' \
  outputs/omni_math2_rlvr_canary/*_24i8t_bs256_smoke_20260512/configs/orchestrator.toml
rg -n 'dp_replicate|max_steps|seq_len' \
  outputs/omni_math2_rlvr_canary/*_24i8t_bs256_smoke_20260512/configs/trainer.toml
rg -n 'api_server_count|data_parallel_size_local|\\[parallel\\]|tp =|dp =' \
  outputs/omni_math2_rlvr_canary/*_24i8t_bs256_smoke_20260512/configs/inference.toml
rg -n 'SBATCH --nodes|NUM_TRAIN_NODES|NUM_INFER_NODES|NODES_PER_INFER_REPLICA|NUM_INFER_REPLICAS|INFERENCE_DP_LOCAL' \
  outputs/omni_math2_rlvr_canary/*_24i8t_bs256_smoke_20260512/rl.sbatch
```

Visible telemetry launch shape:

```bash
tmux new-window -t joanv_cc_8node -n gpu-telemetry \
  'srun --overlap --jobid=4555723 --nodes=8 --ntasks=8 --ntasks-per-node=1 tmp/gpu_telemetry_loop.sh outputs/omni_math2_rlvr_canary/gpu_telemetry_24i8t_20260512 5 7200; exec bash'
```

Visible run launch shape from inside the current allocation:

```bash
tmux new-window -t joanv_cc_8node -n smoke-default \
  'bash outputs/omni_math2_rlvr_canary/default_24i8t_bs256_smoke_20260512/rl.sbatch; exec bash'
```

The temp configs set `dry_run = true` so `uv run rl @ <config>` only generates
resolved configs and the launcher. Run the generated `rl.sbatch` with `bash`
inside the current allocation; do not submit a fresh nested `sbatch` unless
that is explicitly intended.

## Result Log

### 2026-05-12 24i/8t Smoke

Static preflight passed, but live multi-node `srun` failed before any smoke
could launch.

Dry-runs:

- `uv run --no-sync rl @ tmp/rl_olmo3_dpo_default_24i8t_bs256_smoke_20260512.toml`
- `uv run --no-sync rl @ tmp/rl_olmo3_dpo_maxrl_24i8t_bs256_smoke_20260512.toml`

Generated config checks:

- `rl.sbatch` syntax passed for both generated scripts.
- `rl.sbatch` rendered `#SBATCH --nodes=8`, `NUM_TRAIN_NODES=2`,
  `NUM_INFER_NODES=6`, `NODES_PER_INFER_REPLICA=1`,
  `NUM_INFER_REPLICAS=6`.
- `orchestrator.toml` resolved `num_train_workers = 8`,
  `batch_size = 256`, `max_inflight_rollouts = 768`,
  `rollouts_per_example = 8`, `max_off_policy_steps = 8`,
  `dp_rank_count = 4`, `easy_threshold = 1.0`,
  `online_difficulty_filtering = true`, and no `hard_threshold`.
- `inference.toml` resolved `api_server_count = 4`,
  `data_parallel_size_local = 4`, `tp = 1`, `dp = 4`.
- `trainer.toml` resolved `max_steps = 5`, `dp_replicate = 1`,
  `ckpt.interval = 1`, and `lr = 1e-6`.

Live blocker:

- Historical CSV telemetry attempt:
  `srun --overlap --jobid=4555723 --nodes=8 --ntasks=8 --ntasks-per-node=1 ...`
  failed with `srun: error: task N launch failed: Error configuring interconnect`.
- Minimal probes showed one-node `srun` works, including on non-head hosts, but
  two-node and eight-node `srun` fail with the same interconnect error across
  `--mpi=none`, `pmi2`, `pmix`, and `cray_shasta`.

Verdict: `multi_node` route is statically plausible but not launchable from
this allocation as currently configured. Do not call this utilization-tested.

Update, 2026-05-12 10:49 UTC: the "not launchable" verdict above is
superseded for this allocation. Minimal `srun` probes showed that multi-node
launches work with `--network=no_vni,disable_rdzv_get` (`no_vni` alone also
worked; `disable_rdzv_get` alone did not). The generated multi-node launcher
now defaults `PRIME_RL_SRUN_NETWORK` to that working value.

Runtime/template fixes verified before the next smoke:

- `src/prime_rl/templates/multi_node_rl.sbatch.j2` sets node-local compile
  caches: `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `XDG_CACHE_HOME`, and `VLLM_RPC_BASE_PATH` under `/tmp`.
- The synthesis env names `INDUCTOR_CACHE_DIR` and
  `VLLM_TORCH_COMPILE_CACHE_DIR` were wrong for this stack; the diagnosis was
  right, the variable names were not.
- Isambard NCCL/FI knobs are now explicit and overridable in
  `configure_prime_rl_runtime`, including `NCCL_NET="AWS Libfabric"`,
  `NCCL_DEBUG=INFO`, `NCCL_NET_GDR_LEVEL=PHB`, `NCCL_MIN_NCHANNELS=4`, and the
  documented CXI rendezvous settings.
- NCCL probe with repo env loaded `NCCL version 2.30.4+cuda13.2`, not the old
  CUDA 12 stack. Two-node busbw was ~139 GB/s at 1 GiB; eight-node busbw was
  ~85 GB/s at 1 GiB.
- `vllm-router` is now available on aarch64 via the vendored
  `vendor/wheels/aarch64/vllm_router-0.1.22-cp38-abi3-linux_aarch64.whl`.
  `uv sync --extra disagg --locked --dry-run` would replace the ad hoc wheel
  from `tmp/vllm-router-aarch64/dist` with the vendored wheel.
- Checked against `PrimeIntellect-ai/router@v0.1.22`: the local wheel's Python
  shim matches the tag, the `vllm-router` entry point is
  `vllm_router.launch_router:main`, and the launcher flags used by the template
  (`--worker-urls`, `--policy consistent_hash`, `--host`, `--port`,
  `--intra-node-data-parallel-size`, `--worker-startup-timeout-secs`) parse
  cleanly.
- Upstream `vllm_router.version.__version__` is stale at `0.1.12` even in tag
  `v0.1.22`; use wheel metadata / `uv.lock` for package version checks.
- Fresh `bash -n` passed for
  `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_smoke_20260512/rl.sbatch`.
- Fresh generated-config check for Default 24i/8t still resolves
  `batch_size=256`, `max_inflight_rollouts=768`, `rollouts_per_example=8`,
  `max_off_policy_steps=8`, `num_train_workers=8`,
  `api_server_count=4`, `data_parallel_size_local=4`, `dp_rank_count=4`,
  `easy_threshold=1.0`, `online_difficulty_filtering=true`, and no
  `hard_threshold`.

### 2026-05-12 28i/4t Fallback Smoke

Pending. Temp configs:

- `tmp/rl_olmo3_dpo_default_28i4t_bs256_smoke_20260512.toml`
- `tmp/rl_olmo3_dpo_maxrl_28i4t_bs256_smoke_20260512.toml`

Rationale: `gpu_layout` can use one-node `srun` steps, and one-node `srun`
works in allocation `4555723`. This fallback should use 7 full inference nodes
and 1 full trainer node.

Pre-launch smoke-specific overrides:

- `max_steps = 5`
- `ckpt.interval = 1`, to force post-step weight broadcast paths during the smoke
- `orchestrator.eval.interval = 5`
- `orchestrator.eval.eval_base_model = false`, to avoid spending smoke budget on base-model eval
- `orchestrator.wandb.log_extras.interval = 1`, to make the short run inspectable

### 2026-05-12 11:38 UTC: bs512/fsasync4 Concurrency Probe

Verdict: `discard at bf16 KV`. This run did not crash, but it saturated the
vLLM KV cache and built large preemption counts. Continuing to 25 steps would
mostly measure an overloaded queue.

Run:

- Config: `tmp/rl_olmo3_dpo_default_24i8t_bs512_fsasync4_25step_20260512.toml`
- Output: `outputs/omni_math2_rlvr_canary/default_24i8t_bs512_fsasync4_25step_20260512`
- W&B: `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/2aaaf6116f8c4b1b8cf85c5f547b7ccf`
- Topology: 24 inference GPUs / 8 trainer GPUs.
- Batch: `batch_size=512`, `rollouts_per_example=8`,
  `max_inflight_rollouts=1536`, `max_async_level=4`,
  `weight_broadcast.type="filesystem"`, `gpu_memory_utilization=0.95`.
- Checkpoint: `step_5` exists; trainer DCP checkpoint is about 41 GiB.
- Stop: manually interrupted visible tmux pane during step 6 after step 5 was
  complete.

Observed trainer/orchestrator:

| step | orchestrator time (s) | trainer time (s) | trainer MFU | notes |
|---:|---:|---:|---:|---|
| 0 | 209.6 | 344.3 | 0.0% | cold fill |
| 1 | 188.1 | 202.4 | 6.2% | off-policy 1 |
| 2 | 79.4 | 85.8 | 10.0% | off-policy 2 |
| 3 | 47.6 | 42.6 | 12.7% | off-policy 2 |
| 4 | 116.5 | 100.2 | 12.6% | off-policy 4 |
| 5 | 150.8 | 144.6 | 12.0% | off-policy 5 |

vLLM pressure from `monitor/vllm_metrics_8node.tsv`:

- KV average commonly `0.96-0.998`; per-engine max often `0.999+`.
- Per-node cumulative preemptions reached roughly `2.5k-3.6k` by step 6.
- Waiting queues were nonzero/high during rollout phases.
- `error_sum=0`; this was a utilization/concurrency failure, not a fatal
  runtime failure.

Filesystem broadcast findings:

- `FileSystemWeightUpdateWorker` loaded correctly.
- Inference still pauses during weight updates. Observed pauses were usually
  about 3-4 seconds, so filesystem broadcast is not "no pause" in this
  implementation. It may still be faster than NCCL broadcast, but the gain
  must be measured rather than assumed.

Decision:

- Do not run `512/1536` at bf16 KV for the main 25-step smoke.
- Next useful tests are either:
  - `256/768` with the same fs/async4 settings for the longer smoke; or
  - an intermediate `384/1152`; or
  - `512/1536` only after enabling `kv_cache_dtype="fp8_e5m2"` and rechecking
    reward/truncation/preemption behavior.

### 2026-05-12 11:43 UTC: 24i/8t bs256/fsasync4 25-Step Smoke

Status: `stopped after step 16`; enough evidence to reject the simple
`async4 + filesystem broadcast => ~30% trainer MFU` prediction.

Run:

- Config: `tmp/rl_olmo3_dpo_default_24i8t_bs256_fsasync4_25step_20260512.toml`
- Output: `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_fsasync4_25step_20260512`
- W&B: `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/5181b162216b4cd7817ed64eb3371442`
- GPU CSV telemetry:
  `outputs/omni_math2_rlvr_canary/gpu_telemetry_bs256_fsasync4_25step_20260512`
- vLLM metrics:
  `outputs/omni_math2_rlvr_canary/default_24i8t_bs256_fsasync4_25step_20260512/monitor/vllm_metrics_8node.tsv`

Resolved config:

- Topology: 6 one-node inference replicas plus 2 trainer nodes.
- Trainer world size: 8.
- Inference: 6 routers, each with 4 DP/API workers; 24 total inference GPU
  workers.
- `batch_size=256`, `rollouts_per_example=8`, so 32 problem groups/step.
- `max_inflight_rollouts=768`, `max_async_level=4`,
  `max_off_policy_steps=8`.
- `weight_broadcast.type="filesystem"`, `gpu_memory_utilization=0.95`.
- Eval: 32 examples x 4 rollouts every 10 steps.
- Checkpoint interval: 5, keep last 2, keep interval 25.
- Filtering: `easy_threshold=1.0`, no `hard_threshold`,
  `online_difficulty_filtering=true`.

Early pressure through step 1:

- Running requests: roughly 70-135 per inference node.
- Waiting requests: 0 through most of step 0; by late step 1 the worst nodes
  showed waiting up to 4.
- KV average rose from about `0.07` to roughly `0.84-0.89`; per-engine max
  touched about `0.99`.
- Preemptions: 0 through step 0; by late step 1, cumulative preemptions were
  small but nonzero on a few nodes, max observed 8.

Initial read: `256/768` is far healthier than `512/1536`, but it is already
close enough to the KV ceiling on long-tail rollouts that a blind concurrency
bump is not justified. Let it establish a stable 25-step baseline first.

First steps:

| step | orchestrator time (s) | trainer time (s) | trainer MFU | notes |
|---:|---:|---:|---:|---|
| 0 | 122.1 | 241.3 | 0.0% | cold fill |
| 1 | 126.8 | 128.7 | 4.8% | minor preemptions late in step |

Final readout:

- Trainer MFU through step 10 rose only to `12.9%`; this did not become a
  healthy 8-GPU trainer.
- Orchestrator max off-policy reached `8` by step 9.
- Step-10 `32x4` eval took `363.88s` and forced a stale-backlog sawtooth:
  post-eval step 11 took `370.50s`, then steps 12-15 were near-instant stale
  flushes.
- Bottleneck call: still rollout/eval-backlog bound; filesystem broadcast works
  but is not enough.
- Decision: do not promote this exact shape. Keep `256/768` as the safe bf16 KV
  concurrency point, but fix eval/backlog/refill instrumentation before a long
  burn.

### 2026-05-12 12:12 UTC: 28i/4t bs256/fsasync4 Prefix-Off Probe

Status: `completed 25 train steps; final eval wedged and was cancelled`.

Run:

- Config:
  `tmp/rl_olmo3_dpo_default_28i4t_bs256_fsasync4_prefixoff_25step_20260512.toml`
- Output:
  `outputs/omni_math2_rlvr_canary/default_28i4t_bs256_fsasync4_prefixoff_25step_20260512`
- W&B:
  `https://wandb.ai/jvelja-private/omni-math2-rlvr/runs/eea5e138b12341ff94c6a326e52adf20`
- GPU CSV:
  `outputs/omni_math2_rlvr_canary/gpu_telemetry_28i4t_bs256_fsasync4_prefixoff_25step_20260512`
- vLLM metrics:
  `outputs/omni_math2_rlvr_canary/default_28i4t_bs256_fsasync4_prefixoff_25step_20260512/monitor/vllm_metrics_8node.tsv`

Resolved config:

- Topology: 7 inference nodes, 1 trainer node.
- Trainer world size: 4.
- Inference: 7 routers, each with 4 DP/API workers; 28 total inference GPU
  workers.
- `batch_size=256`, `rollouts_per_example=8`, so 32 problem groups/step.
- `max_inflight_rollouts=768`, `max_async_level=4`,
  `max_off_policy_steps=8`.
- `weight_broadcast.type="filesystem"`, `gpu_memory_utilization=0.95`.
- `enable_prefix_caching=false`.
- Eval: 32 examples x 4 rollouts every 10 steps.

GPU/trainer readout:

- Trainer peak memory: `80.8 GiB`, so FSDP4 fits on GH200 with this shape.
- Per-trainer-GPU MFU warmed to `26.4%` at steps 11-12, then mostly stayed in
  the `20-23%` band, with one `28.0%` point at step 22.
- Inference memory at `gpu_memory_utilization=0.95` was close to the edge but
  did not show the bs512 death spiral. Preemptions were present but not
  thousands-per-node.
- Interval evals: `345.34s` and `318.00s` for `32x4`.
- Staleness remained bad: max off-policy hit `8` after evals, and post-eval
  steps flushed stale data.

Bottleneck call:

- 28i/4t improves per-trainer-GPU MFU versus 24i/8t, but cluster trainer-FLOP
  contribution is not obviously better because trainer GPU count halves.
- Eval/backlog interaction is still the dominant bad shape.
- Final eval wedged after the progress stream reached about `87/128`; vLLM
  metrics showed all endpoints healthy but `running_sum=0`, `waiting_sum=0`,
  and zero KV usage. The run step was cancelled after trainer completion and
  final checkpoint write.

Decision:

- `28i/4t` is viable to retest, but not proven superior.
- Do not use final-eval wall time for throughput laws.
- Next utilization run should either disable eval while measuring train
  throughput, or run a clean saturated eval-only/interval eval with explicit
  cancellation and no stale backlog.

Follow-up comparison at 2026-05-12 13:14 UTC changes the topology read:

- On a fair pre-eval window, 28i/4t beats 24i/8t on orchestrator wall-clock:
  `48.89s` mean / `47.70s` median for 28i/4t steps 3-12 versus `60.80s`
  mean / `61.22s` median for 24i/8t steps 3-10.
- Trainer time is effectively tied: `51.51s` mean for 28i/4t versus `50.75s`
  mean for 24i/8t.
- Trainer MFU is much higher on 28i/4t: `21.57%` mean versus `11.07%` mean.
- The 28i/4t "steps 3-22 excluding 13" summary includes stale-flush steps
  14-16; excluding those, 28i/4t still has mean orchestrator time `55.48s`
  and median `51.82s`.
- Conclusion: for training-loop utilization at `bs=256`, 28i/4t fs+async4
  prefix-off is the current best-known tested shape. The earlier "not proven
  superior" note was too conservative after doing the apples-to-apples parse.

## 2026-05-12 28i/4t `bs512` Inflight-1024 Result

Run:

- Config:
  `tmp/rl_olmo3_dpo_default_28i4t_bs512_inflight1024_fsasync4_prefixoff_12step_20260512.toml`
- Output:
  `outputs/omni_math2_rlvr_canary/default_28i4t_bs512_inflight1024_fsasync4_prefixoff_12step_20260512`
- W&B:
  `c9373ff8a0ae42dba4833f21abbd9dad`
- Telemetry:
  `outputs/omni_math2_rlvr_canary/gpu_telemetry_28i4t_bs512_i1024_fsasync4_prefixoff_12step_20260512`

Result: completed 12 train/orchestrator steps with no eval. After trainer and
orchestrator completed, leftover telemetry/run Slurm steps were cancelled.

| window | orchestrator time | trainer time | trainer MFU | trainer tok/s | max off-policy |
|---|---:|---:|---:|---:|---:|
| steps 2-11 | mean `95.31s`, median `98.38s` | mean `94.16s`, median `94.53s` | mean `27.22%`, median `27.75%` | mean `15,314` | `5` |
| steps 4-11 | mean `100.55s`, median `101.97s` | mean `92.39s`, median `94.53s` | mean `28.62%`, median `28.15%` | mean `16,110` | `5` |

Late trainer steps reached `32.5%` and `32.3%` MFU on steps 10 and 11, but
orchestrator wall-clock remained around `80-119s` after warmup.

Inference pressure:

| metric | value |
|---|---:|
| running requests, mean / max | `836.5` / `1021` |
| waiting requests, mean / max | `34.9` / `130` |
| KV avg, mean / p95 | `0.744` / `0.938` |
| max KV observed | `1.000` |
| cumulative preemptions | `5242` |

Historical GPU CSV summary, dropping obvious pre-start idle samples:

| role | util mean | util median | p90 util | mean memory | max memory |
|---|---:|---:|---:|---:|---:|
| inference | `75.3%` | `100.0%` | `100.0%` | `85.4 GiB` | `94.9 GiB` |
| trainer | `47.3%` | `15.0%` | `100.0%` | `66.5 GiB` | `83.5 GiB` |
| all GPUs | `72.0%` | `100.0%` | `100.0%` | `83.2 GiB` | `94.9 GiB` |

Decision:

- `bs512` at `inflight=1024` is viable on `28i/4t`, unlike the earlier
  `24i/8t bs512 inflight=1536` overload.
- It is not the current default for throughput. It raises trainer MFU, but
  inference/KV pressure pushes wall-clock to roughly `95-101s` steady steps,
  materially slower than the `28i/4t bs256` result.
- The run was uncompiled on the trainer (`compile=None` in logs). The next fair
  max-performance probe should add `[trainer.model.compile]` and disable W&B
  sample-table logging before comparing.

## Scaling / Extrapolation Laws

Use rollout units explicitly:

```text
problems_per_step = batch_size / rollouts_per_example
train_rollouts = steps * batch_size
train_problem_groups = steps * batch_size / rollouts_per_example
```

For a fixed topology/config family:

```text
T_total(S, B, E, R) =
  T_startup
  + T_step0(B)
  + (S - 1) * median_steady_step_time(B)
  + floor(S / eval_interval) * T_eval(E, R)
  + floor(S / ckpt_interval) * T_ckpt
```

For online eval:

```text
eval_rollouts = num_examples * eval_rollouts_per_example
T_eval(E, R) ~= fixed_startup
  + generated_tokens / effective_decode_tok_s
  + tail_penalty
```

Fit this from saturated evals on the same topology and sampling settings. Do
not linearly scale current `32x4` wall-clock to `100x8`; `32x4` underfills
24-28 inference GPUs and is tail-dominated.

Observed correction on 2026-05-12:

- Current `32x4` evals took `318-364s` for only 128 rollouts.
- Earlier clean 14i/2t interval `100x8` evals often took `~360-450s` for 800
  rollouts.
- Final evals can take `~1700-1850s`; that is a separate pathological regime,
  not the central interval-eval estimate.

Worst-case problem-level binomial standard error is approximately:

```text
SE_worst ~= 0.5 / sqrt(num_examples)
```

So:

- 32 examples: about 8.8 percentage points.
- 100 examples: about 5.0 percentage points.
- 600 examples: about 2.0 percentage points.

## 2026-05-12 Offline Eval Utilization Correction

The 8-node direct-backend offline eval route was invalid for quality and
throughput accounting:

- Eval driver and one vLLM shard both ran on `nid010685`.
- `nid010685` vLLM DP coordinator died with `RuntimeError: cancelled`.
- The corresponding eval shard logged repeated `APIConnectionError` messages
  while the aggregate progress bar continued.

Current clean route:

```text
driver-only: nid010685
vLLM nodes:  nid010752 nid010753 nid010756 nid010757 nid010758 nid010765 nid010768
GPUs used:   28 / 32
```

Observed shortly after generation started:

| host class | GPU state |
|---|---|
| `nid010685` | 4 GPUs idle, no vLLM residency |
| 7 vLLM nodes | all 28 GPUs at 100% util, about 88.5 GiB used/GPU |

Conclusion: for offline eval in the current tmux-on-compute-node setup, use
the 7-node clean route for apples-to-apples comparisons. A true 32/32 eval
needs the driver moved off the inference nodes or a more robust remote driver
path; otherwise "all 32 GPUs" can silently turn into partial failed shards.

## 2026-05-13 DAPO Refill MFU Correction

The overnight `1e-6` and `3e-6` runs were both DAPO-style drop/refill runs.
The difference is implementation version, not refill on/off:

| run | implementation | candidate behavior |
|---|---|---|
| `4570549` / `1e-6` | refill v1 from `0470684cb` | full `batch_size=256` candidate batches per refill round |
| `4574276` / `3e-6` | optimized refill from `c8a5b8307` | smaller top-up candidate batches with `candidate_groups_per_round=32` |

Trainer MFU from parsed trainer logs:

| run | window | mean step time | mean trainer MFU | mean trainer tok/s |
|---|---:|---:|---:|---:|
| non-refill `28i/4t bs256` | steps `25-74` | `49.6s` | `28.70%` | `16,156` |
| `1e-6` refill v1 | steps `25-74` | `230.0s` | `10.82%` | `6,119` |
| `3e-6` refill v2 | steps `25-74` | `191.6s` | `12.40%` | `6,990` |
| `3e-6` refill v2 | steps `75-99` | `194.0s` | `11.90%` | `6,710` |

Refill accounting from per-step `train_filter_metrics.json`:

| run | window | candidate groups | accepted groups | filtered groups | rounds | prompts / accepted group | unconditioned reward | conditioned reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-6` refill v1 | steps `25-74` | `65.28` | `32.00` | `25.40` | `2.04` | `2.04` | `0.248` | `0.408` |
| `3e-6` refill v2 | steps `25-74` | `55.26` | `32.00` | `21.80` | `2.44` | `1.73` | `0.258` | `0.418` |
| `3e-6` refill v2 | steps `75-99` | `57.44` | `32.00` | `24.16` | `2.64` | `1.79` | `0.242` | `0.413` |

Conclusion: the optimized refill path reduced candidate waste relative to v1,
but current refill is still a throughput loss versus the non-refill 28i/4t
baseline. Quality must be judged by the running offline evals, not accepted
train reward alone.

## 2026-05-13 Routed 8-Node Offline Eval Relaunch

The direct-backend DAPO eval route was killed and relaunched with
`vllm-router` enabled on all 8 nodes.

Current jobs:

| job | arm | route | status at relaunch check |
|---:|---|---|---|
| `4583877` | `1e-6` refill | 8 nodes × 4 DP-aware vLLM workers through router | running `step_25` |
| `4583883` | `3e-6` refill | 8 nodes × 4 DP-aware vLLM workers through router | running `step_25` |

Both router logs showed:

- 8 out of 8 unique hosts healthy.
- each host expanded to ranks `0..3`.
- total routed worker set size: 32 DP-aware endpoints.

This is the first DAPO offline eval route in this sequence that should be
trusted for throughput/load-balance conclusions if it completes without backend
errors.

Update: jobs `4583877` and `4583883` did not complete. They proved router
startup and 32 DP-aware worker expansion, but then failed at the baseline
runner readiness layer because `vllm-router /v1/models` returns 500. The fix is
to treat provisioned router generation URLs as router-health endpoints for
external baseline readiness. Relaunch required before drawing throughput
conclusions.

## 2026-05-13 Routed Eval Step-Split Correction

The corrected 8-node routed eval path did use all 32 DP-aware workers, but a
serial `600x8` eval over checkpoints `25,50,75,100` cannot fit in a single
6-hour allocation at observed throughput.

Observed on the `1e-6` routed eval before cancellation:

| signal | value |
|---|---:|
| active workers | 32 DP-aware endpoints |
| active requests | 63-64 |
| queued requests | 0 |
| generated token throughput | ~2,120 tok/s across 32 GPUs |
| step-25 rollout rows after ~25 min | 228 / 4,800 |

Interpretation: the router was no longer the bottleneck; the workload was
long-tail decode-bound at full-context `600x8`.

Action taken:

| arm | checkpoint | job | output suffix |
|---|---:|---:|---|
| `1e-6` refill | 25 | `4584726` | `offline_eval_600x8_8node_router_step25` |
| `1e-6` refill | 50 | `4584727` | `offline_eval_600x8_8node_router_step50` |
| `1e-6` refill | 75 | `4584733` | `offline_eval_600x8_8node_router_step75` |
| `1e-6` refill | 100 | `4584739` | `offline_eval_600x8_8node_router_step100` |
| `3e-6` refill | 25 | `4584740` | `offline_eval_600x8_8node_router_step25` |
| `3e-6` refill | 50 | `4584741` | `offline_eval_600x8_8node_router_step50` |
| `3e-6` refill | 75 | `4584743` | `offline_eval_600x8_8node_router_step75` |
| `3e-6` refill | 100 | `4584744` | `offline_eval_600x8_8node_router_step100` |

Each job requests 8 nodes for 8 hours and evaluates exactly one checkpoint.
As of `2026-05-13 11:29 UTC`, all eight were pending on priority. `squeue
--start` estimated the first two starts around `13:43 UTC` and the final
group around `16:06-16:10 UTC`.

Update at `2026-05-13 11:42 UTC`: the first split job, `4584726`, failed in
43 seconds because the stale-cleanup backport self-killed the remote cleanup
task. The fix is in `src/prime_rl/baselines/provision.py`: cleanup now excludes
its own process group before killing stale vLLM/prime-rl processes.

Resubmitted jobs:

| arm | checkpoint | job | status at `11:40 UTC` |
|---|---:|---:|---|
| `1e-6` refill | 100 | `4585067` | running; router/backends starting cleanly |
| `1e-6` refill | 25 | `4585068` | running; router/backends starting cleanly |
| `1e-6` refill | 50 | `4585069` | running; router/backends starting cleanly |
| `1e-6` refill | 75 | `4585070` | pending on resources |
| `3e-6` refill | 100 | `4585071` | pending on priority |
| `3e-6` refill | 25 | `4585072` | pending on priority |
| `3e-6` refill | 50 | `4585073` | pending on priority |
| `3e-6` refill | 75 | `4585074` | pending on priority |

The running shard logs show `nccl_net=AWS Libfabric`, 8 backend hosts, and
vLLM router startup. Monitor status is written to
`outputs/omni_math2_rlvr_canary/postrun_eval_monitor_20260513_stepsplit.md`.

Update at `2026-05-13 11:50 UTC`: `1e-6` step `100` was a bad target; the run
has no stable `step_100` broadcast and stopped at `step_85`. Job `4585067`
failed after readiness with no matching stable checkpoint. Job `4585068`
(`step_25`) also failed after readiness with the same message, but the local
checkpoint discovery path sees `step_25`, so it was retried as a likely
compute-side filesystem visibility miss.

Current corrected `1e-6` targets:

| checkpoint | job | note |
|---:|---:|---|
| 25 | `4585323` | retry after false missing-checkpoint failure |
| 50 | `4585069` | running; reached pause/update/resume and partial rollouts |
| 75 | `4585070` | valid, pending/running by scheduler state |
| 85 | `4585324` | actual final stable broadcast for the `1e-6` run |

`3e-6` targets remain `25`, `50`, `75`, and `100`; all four stable broadcasts
exist under `lr3e6_28i4t_refill_shared_submit_20260512_2155/run_default/broadcasts`.
