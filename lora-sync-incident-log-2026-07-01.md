# LoRA Sync Incident Log

Date: 2026-07-01
Author: Codex
Repo: `/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main`
Branch at write time: `chore/nccl-2.30-ofi-1.20-stack`
HEAD at write time: `8df936912`

The repo had a large dirty tree at write time. This log intentionally describes
both committed changes and active uncommitted changes.

This is a reconstruction of the LoRA sync work from the recent Isambard debugging
push. It covers both sync transports:

- `weight_broadcast.type = "nccl"`: Prime-RL custom NCCL LoRA tensor receive and
  resident vLLM adapter registration.
- `weight_broadcast.type = "filesystem"`: trainer writes adapter dirs to a shared
  filesystem, inference servers load them through vLLM's LoRA loader.

It is deliberately historical. Do not treat it as stable product docs.

## Executive Read

The LoRA sync problem split into two different failure families.

NCCL:

- The original failures looked like a dirty-state NCCL collective entering while
  vLLM had live/frozen decode state.
- We fixed the most suspicious race by moving the NCCL LoRA receive inline onto
  the worker busy-loop thread, hardening the pause/update/resume semantics, and
  refusing normal resume after a post-`NCCL_READY` failure.
- Small and medium NCCL canaries then passed.
- The 16-node production-shaped NCCL run still failed at the first production
  adapter update. It did not fail in `/pause`. It entered `/update_lora`, then all
  12 inference peers hit `ReadTimeout` after roughly the 720 s update timeout.
- That means NCCL LoRA is still not production-servable. The current unproven
  boundary is inside worker-side `receive_lora_update`: receive, materialization,
  adapter construction, or adapter activation.

Filesystem:

- The original FS concern was Lustre fanout: many inference replicas reading the
  same multi-GB adapter file at once.
- We hardened FS by versioning adapter names, extending load timeouts, bounding
  fanout, adding Lustre striping defaults, adding node-local staging in
  `/dev/shm`, verifying staged adapter files, unloading old adapter versions, and
  adding diagnostics for RSS/staged bytes/loaded adapter ids.
- The 16-node FS canary loaded adapters through step 6 and died during the step 7
  load with a node OOM. That is progress compared with NCCL, but it does not yet
  prove memory is plateauing.
- The 4-node Qwen1.5B FS storm proxy survived many hot adapter swaps, but exposed
  a different availability issue: bursts of `No available workers` during churn.

My current view:

- FS is the nearer production path, provided the new diagnostics prove old
  adapters and staged dirs are actually released.
- NCCL remains worth pursuing, but only with phase-level instrumentation and a
  dedicated stress harness. Another blind 16-node RL canary is a bad use of time.

## Baseline Architecture Before This Debugging Push

### Shared Control Plane

Both sync paths use the trainer/orchestrator broadcast directory convention:

```text
outputs/<run>/broadcasts/step_N/
  STABLE
  adapter_model.safetensors
  adapter_config.json
  NCCL_READY        # NCCL path only
```

`STABLE` tells the orchestrator that a checkpoint or adapter is ready. The
watcher then calls into the inference pool to update the serving policy.

Important files:

- `src/prime_rl/orchestrator/watcher.py`
- `src/prime_rl/utils/client.py`
- `src/prime_rl/inference/vllm/server.py`
- `src/prime_rl/inference/vllm/worker/nccl.py`
- `src/prime_rl/inference/vllm/worker/filesystem.py`
- `src/prime_rl/trainer/rl/broadcast/nccl.py`
- `src/prime_rl/trainer/rl/broadcast/filesystem.py`

### Original NCCL LoRA Flow

The custom NCCL LoRA path was:

```text
trainer writes STABLE
watcher sees STABLE
watcher calls on_version_pending()
orchestrator POST /pause to every inference API server
orchestrator touches NCCL_READY
trainer rank enters rooted NCCL broadcast
orchestrator POST /update_lora to every inference API server
each vLLM worker runs receive_lora_update
worker mutates vLLM LoRA manager
server registers a LoRARequest
orchestrator POST /resume
policy.version advances
```

The dangerous bit is the combination of:

- one-shot cross-role collective;
- hot vLLM engines with active decode/prefill state;
- vLLM LoRA private manager mutation;
- async-RL policy staleness;
- no safe retry after the marker releases the trainer broadcaster.

### Original FS LoRA Flow

The filesystem path was:

```text
trainer writes adapter files to shared FS
trainer touches STABLE
watcher sees STABLE
orchestrator POST /load_lora_adapter to every inference API server
each vLLM server reads the adapter path
vLLM loads the adapter via its LoRA loader
policy.version advances
```

The old FS path avoided the NCCL collective but made the shared filesystem the
broadcast primitive. With 12 inference replicas and multi-GB adapters, that is a
read storm.

## Root-Cause Model We Built

### Old NCCL Failure Model

Observed old failure shapes:

1. A worker threw `RuntimeError: NCCL error: unhandled cuda error` during
   `receive_lora_update`.
2. Other peers stayed HTTP-alive but stopped making engine progress.
3. The orchestrator saw `AdminGatherError` during `/update_lora` or `/resume`.
4. Logs showed vLLM shared-memory broadcast block warnings and stuck running
   requests.

The important correction was that the watcher was downstream. The watcher was
not "flaky"; it was aggregating a distributed update failure.

### Pause Semantics Correction

We originally suspected `/pause` meant "drain in-flight requests." That was
not true for `mode="keep"`.

Actual semantics:

- `mode="wait"`: step the engine until running work drains.
- `mode="keep"`: freeze work so it can be resumed later.
- `mode="abort"`: kill in-flight work.

The code and comments contradicted each other:

- client-side comments claimed `/pause` drained;
- server-side code used `pause_generation(mode="keep", clear_cache=False)`;
- logs showed running requests frozen in place.

We then refined the model:

- async RL intentionally keeps fresh within-lag rollouts alive;
- `on_version_pending()` only drops stale groups past `max_off_policy_steps`;
- fresh groups are supposed to resume and remain usable;
- therefore aborting every sync is too wasteful and may reintroduce KV cleanup
  races;
- but freezing live work is not necessarily a clean device state for a custom
  LoRA NCCL receive.

### Async RL Versioning Correction

The old design effectively reused one adapter identity. That is wrong for an
async policy-lag setup.

The better semantic model is:

```text
adapter at step N has unique name and id
new rollout groups route to newest adapter
old groups keep the adapter/model name they started with
retire adapters only after no in-flight group can reference them
```

This affects both FS and NCCL. It is independent of transport.

## Committed Base Changes

### CUDA/vLLM Hygiene

Commit `cf1ef043c`:

- changed the aarch64 vLLM pin away from the wrong CUDA-suffixed wheel;
- intent: stop mixing a CUDA 12.9 wheel with the CUDA 13 runtime.

Commit `8df936912`:

- adopted NCCL `2.30.7` and `aws-ofi-nccl 1.20.0` fabric stack;
- added `scripts/env/nccl-ofi-stack.sh`;
- updated Isambard fabric env wiring.

These were hygiene, not a logical fix for LoRA sync. They remove avoidable ABI
and fabric-stack noise.

### Blocking Inline NCCL LoRA Receive

Commit `3b13ef877`:

- moved LoRA receive to run inline through vLLM `collective_rpc`;
- removed the prior daemon-thread style overlap;
- added stricter client behavior around post-marker failure;
- kept the invariant that once `NCCL_READY` is touched, a failed update is
  terminal for that inference pool;
- added/update tests around client failure behavior and worker receive.

This addressed the most plausible old "collective overlaps live decode" race.

### Drain/Pause Semantics Fix For 65k Debate Runs

Commit `ceab2d3db`:

- made `/pause` accept and honor explicit `mode` and `clear_cache` parameters;
- updated client-side calls so pause semantics are visible in logs;
- changed configs toward 65k model length and production debate settings;
- added tests for admin route semantics and client fanout behavior.

The key effect: the logs now say things like:

```text
Updating policy in-flight to v1 (pause mode=keep, clear_cache=False)
```

That made later failures easier to classify.

## Current Uncommitted Code Changes

This section describes the dirty working tree as inspected on 2026-07-01.

### Versioned LoRA Helpers

New file:

- `src/prime_rl/utils/lora.py`

Conceptual API:

```text
versioned_lora_name(base_name, step) -> "<base>__v000000NN"
versioned_lora_int_id(step) -> unique positive int
versioned_lora_adapter(base_name, step) -> name/id/version metadata
```

Used by:

- `watcher.py`
- `orchestrator.py`
- `client.py`
- `trainer/rl/broadcast/nccl.py`

Purpose:

- do not mutate one stable adapter identity;
- let old and new adapter versions coexist during `target_lag`;
- allow explicit retirement.

### Dispatcher Pins Model Name At Group Start

Changed:

- `src/prime_rl/orchestrator/types.py`
- `src/prime_rl/orchestrator/dispatcher.py`

New field:

```python
GroupState.model_name_at_start: str
```

Why:

- a rollout group must keep using the adapter/model name it started with;
- if `policy.model_name` changes mid-group, the group should not silently hop to
  the new adapter.

Dispatcher also exposes:

```python
active_policy_versions() -> set[int]
```

Used by the watcher to decide when adapter versions can be retired.

### Watcher Tracks And Retires Live LoRA Versions

Changed:

- `src/prime_rl/orchestrator/watcher.py`

New state:

```python
self.live_lora_steps: set[int]
```

New behavior:

- after successful update, switch `policy.model_name` to the versioned adapter;
- ask observers for active policy versions;
- retire versions no longer referenced;
- for NCCL, call `/remove_lora_adapter`;
- for FS, call `/unload_lora_adapter`.

This makes `max_loras` meaningful. With `target_lag=3`, the immediate invariant
became:

```text
target_lag = 3
max_off_policy_steps = 3
max_loras >= 4
max_cpu_loras >= 4
```

Four live versions means current plus up to three lagged versions.

### Client Admin Path

Changed:

- `src/prime_rl/utils/client.py`

Key changes:

- added `remove_lora_adapter()` and `unload_lora_adapter()` to the
  `InferencePool` protocol;
- filesystem LoRA loads now use `versioned_lora_name(lora_name, step)`;
- NCCL LoRA updates now send `versioned_lora_adapter(lora_name, step)`;
- `_pause_engines()` now accepts `mode`, `clear_cache`, and `timeout_s`;
- NCCL LoRA uses:

```text
mode="keep"
clear_cache=False
timeout_s=120
```

- normal full-weight update still uses drain-like semantics;
- `LORA_LOAD_READ_TIMEOUT_S = 900`;
- `LORA_LOAD_TOTAL_TIMEOUT_S = 1200`;
- `LORA_LOAD_MAX_CONCURRENCY = 4`;
- `/unload_lora_adapter` is now the Prime-RL wrapper route, not raw
  `/v1/unload_lora_adapter`;
- 404 on unload is tolerated as "already absent."

Important constraint:

- after `NCCL_READY` is released, `/update_lora` failure must not be retried as
  if it were an ordinary HTTP failure;
- normal `/resume` is skipped after a failed LoRA update.

### NCCL Receiver Header And Removal Path

Changed:

- `src/prime_rl/inference/vllm/worker/nccl.py`
- `src/prime_rl/trainer/rl/broadcast/nccl.py`

Header now carries and validates:

```text
lora_name
lora_int_id
adapter_version
peft_config
num_chunks
```

Trainer-side NCCL headers use the versioned adapter helper.

Worker-side receive path still does:

```text
receive header
receive chunks
copy/reshape tensors
build LoRAModel.from_lora_tensors(...)
remove existing same id
add adapter
activate adapter
```

New worker method:

```python
remove_lora_adapter(lora_int_id: int) -> dict
```

Purpose:

- explicit cleanup of resident adapter versions;
- lets watcher retire unused NCCL adapter versions.

Current unresolved NCCL problem:

- the worker logs are still not granular enough for production-scale failure;
- we need phase timestamps around receive vs materialize vs add/activate.

### FS Node-Local Staging

New file:

- `src/prime_rl/inference/lora_staging.py`

The staging design:

```text
source adapter dir on shared FS
validate required files
compute manifest: source path, lora name, file sizes
copy to /dev/shm/prime_rl_lora_<job>_<host>/<safe-name>-<digest>
copy into a private temp dir
write manifest
rename temp dir into place
load vLLM adapter from staged local path
cleanup staged path on unload
```

Required files:

```text
adapter_model.safetensors
adapter_config.json
```

Safety features:

- per-adapter lock file under the node-local stage root;
- timeout while waiting for another local process to finish staging;
- size verification after copy;
- manifest verification before reuse;
- corrupt staged copies are replaced;
- partial source dirs fail instead of loading.

Default stage root:

```text
/dev/shm/prime_rl_lora_<SLURM_JOB_ID>_<hostname>
```

Override:

```text
PRIME_RL_LORA_STAGE_ROOT=/some/node/local/path
```

Disable:

```text
PRIME_RL_LORA_STAGE_ENABLED=0
```

### FS vLLM Server Wrapper

Changed:

- `src/prime_rl/inference/vllm/server.py`

`/load_lora_adapter` now:

1. receives vLLM `LoadLoRAAdapterRequest`;
2. calls `maybe_stage_lora_adapter(source_path, lora_name)`;
3. rewrites `lora_path` to local staged path if staging is enabled;
4. calls vLLM's public LoRA loader;
5. records the staged path by `lora_name`;
6. logs diagnostics.

New `/unload_lora_adapter` wrapper:

1. removes worker-side LoRA by int id when registered;
2. calls vLLM unload;
3. removes the staged path from app state;
4. deletes staged files;
5. logs diagnostics.

New `/remove_lora_adapter` wrapper:

- for NCCL-resident adapters;
- calls worker `remove_lora_adapter` via `collective_rpc`;
- removes public model entry.

### FS Diagnostics

New file:

- `src/prime_rl/inference/vllm/worker/diagnostics.py`

Diagnostics collect:

- API server PID and RSS;
- number/names of public LoRA requests;
- staged path count and total staged bytes;
- stage root total bytes;
- worker RSS;
- worker registered LoRA ids;
- worker active LoRA ids;
- GPU allocated/reserved bytes where available;
- vLLM LoRA slot/cache shape where available.

Server logs emit one JSON blob per load/unload with marker:

```text
prime_rl_lora_diagnostics {...}
```

The old FS production canary did not yet have these diagnostics. The replacement
pending canary was launched to get them.

### FS Lustre Striping

Changed:

- `packages/prime-rl-configs/src/prime_rl/configs/trainer.py`
- `src/prime_rl/trainer/rl/broadcast/filesystem.py`

New config fields:

```python
stripe_enabled: bool = True
stripe_count: int = 16
stripe_size: str = "1M"
```

Trainer master now tries:

```bash
lfs setstripe -c <stripe_count> -S <stripe_size> <broadcast_dir>
```

before adapter files are created. If `lfs` is missing or fails, it logs a
warning and continues.

Purpose:

- reduce single-OST read storms for new adapter artifacts;
- must happen before files are created, because Lustre stripe layout is fixed at
  file creation.

### Full Rollout Transcript Sidecars

Changed:

- `src/prime_rl/orchestrator/*`
- `scripts/docent/ingest_prime_rollouts.py`
- `scripts/docent/create_prime_analysis_plan.py`

Config:

```toml
[orchestrator]
save_full_rollouts = true
```

This is adjacent to LoRA sync rather than a sync fix, but it was part of the
science relaunch prep. It preserves richer rollout material for Docent analysis.

## Stress And Monitor Scripts Added

New directory:

- `scripts/stress/`

### `scripts/stress/lora_stage_stress.py`

Purpose:

- test node-local staging correctness;
- compare shared-FS read storm vs local staged reads;
- smoke corrupt/partial staged adapter handling.

It tests:

- staging a real adapter;
- immediate stage reuse;
- corrupt staged file repair;
- partial adapter source rejection;
- concurrent readers from original Lustre file;
- concurrent readers from staged `/dev/shm` file.

Observed earlier benchmark result, using the 9.4 GB adapter artifact:

```text
Lustre direct K=1:  p50 3.83s, max 3.83s, agg 2.5 GB/s
Lustre direct K=12: p50 9.02s, max 9.12s, agg 12.4 GB/s
/dev/shm K=1:       p50 1.21s, max 1.21s, agg 7.8 GB/s
/dev/shm K=12:      p50 1.40s, max 1.41s, agg 80.3 GB/s
```

Conclusion:

- staging location matters more than loader API choice for the read storm;
- `/dev/shm` kills the shared-FS storm on one node;
- production still needs multi-node confirmation and memory plateau checks.

### `scripts/stress/pause_keep_live_probe.py`

Purpose:

- boot a small local vLLM server;
- flood it with long generations;
- repeatedly call `/pause?mode=keep&clear_cache=false` and `/resume`;
- check whether `keep` is fast and preserves in-flight requests.

Limitation:

- no cross-node NCCL collective;
- does not reproduce production NCCL transport behavior.

### `scripts/stress/nccl_lora_live_stress.py`

Purpose:

- boot local vLLM NCCL worker-extension servers;
- initialize a trainer-plus-inference NCCL group;
- start generations against one adapter version;
- run the real Prime-RL `update_lora_adapter` client path;
- broadcast versioned LoRA tensors;
- verify old-version generations complete.

Limitation:

- single-node;
- can use multiple local server processes, but it is not 12-node Slingshot.

### `scripts/stress/monitor_lora_canary.py`

Purpose:

- cheap terminal monitor for Slurm canaries;
- prints `squeue`/`sacct` state;
- tails orchestrator memory and LoRA events;
- parses new `prime_rl_lora_diagnostics` blobs.

It specifically reports:

- API RSS;
- stage bytes;
- public LoRA names;
- worker registered ids;
- worker max RSS;
- worker max GPU reserved.

## Config Changes For Science/Canary Runs

### Production-shaped FS canary

Config:

- `configs/calibration/gpqa_openended_debate_20step_bs512_g16_seq4_fs_sync_ab.toml`

Important values:

```toml
max_steps = 20
weight_broadcast.type = "filesystem"
target_lag = 3
max_off_policy_steps = 3
max_inflight_rollouts = 1200
save_full_rollouts = true
max_completion_tokens = 16384
gpu_memory_utilization = 0.9
max_loras = 4
max_cpu_loras = 4
max_model_len = 65536
max_num_seqs = 128
```

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/sync_ab_fs_seq4_20step
```

Replacement diagnostic output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/sync_ab_fs_seq4_20step_fs_hardened_20260701_1345
```

Replacement job:

```text
5455616 fs-seq4-20-r2
```

State at inspection:

```text
PENDING, reason Priority
```

### Production-shaped NCCL canary

Config:

- `configs/calibration/gpqa_openended_debate_20step_bs512_g16_seq4_nccl_sync_ab.toml`

Important values:

```toml
max_steps = 20
weight_broadcast.type = "nccl"
target_lag = 3
max_off_policy_steps = 3
max_inflight_rollouts = 1200
save_full_rollouts = true
max_completion_tokens = 16384
gpu_memory_utilization = 0.9
max_loras = 4
max_cpu_loras = 4
max_model_len = 65536
max_num_seqs = 128
```

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/sync_ab_nccl_seq4_20step
```

Job:

```text
5453015 sync-ab-nccl-seq4-20
```

### Small FS storm proxy

Config:

- `configs/calibration/qwen15b_math_fs_lora_storm_2t2i.toml`

Important values:

```toml
max_steps = 30
weight_broadcast.type = "filesystem"
trainer/inference split = 2 + 2 nodes
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
max_inflight_rollouts = 512
target_lag = 3
max_off_policy_steps = 3
lora name = "qwen15b-r256"
max_loras = 4
max_cpu_loras = 4
max_model_len = 8192
max_num_seqs = 128
```

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/qwen15b_math_fs_lora_storm_2t2i
```

## Empirical Run Log

### Original overnight science runs

The eight long science runs failed for a mixture of:

- overlong debate prompt failures before score generation;
- NCCL LoRA update failures;
- vLLM engine wedging after failed admin update.

The overlong prompt family was separate from LoRA sync. The seq/prompt fix was:

- move Qwen3.5 debate serving to `max_model_len = 65536`;
- keep 16k completion budget for train debate;
- tune inference capacity knobs (`max_inflight_rollouts`, `max_num_seqs`) rather
  than reducing trainer batch size as the first lever.

The LoRA failure family stayed open.

### Small NCCL canary, 3 inference servers

Output:

```text
outputs/isambard/calibration/nccl_lora_sync_4node_bs64_g4_simul_20260630T1235Z
```

Job:

```text
5437120
```

Shape:

- 4 nodes total;
- 1 trainer side plus 3 inference servers;
- inference world size 12;
- batch size 64;
- group size 4;
- Qwen3.5-A3B;
- 65k serving;
- NCCL LoRA.

Important log evidence:

```text
16:16:18 Initializing NCCL broadcast: 3 servers, inference_world_size=12
16:33:32 Updating weights to step 1
16:36:34 step 1 received 61520 tensors (19 chunks) in 8.09s, committed in 3.98-7.10s
16:45:12 step 4 received 61520 tensors (19 chunks) in 7.39s, committed in 4.75-5.70s
16:52:26 Orchestrator step loop done
16:52:28 Orchestrator finished
```

Conclusion:

- the inline NCCL receive path can work for Qwen3.5-A3B at 3 inference servers;
- it does not prove 12 inference servers and production load are safe.

### Larger NCCL canary, short max steps

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/nccl_lora_canary_bs64_g4_seq4
```

Job:

```text
5448456
```

Observed:

```text
01:59:27 Student inference pool ready
02:06:11 Step 0 success
02:08:32 Updating weights to step 1 (pause mode=keep, clear_cache=False)
02:09:41 Updating weights to step 2 (pause mode=keep, clear_cache=False)
02:09:52 Orchestrator step loop done
02:09:55 Orchestrator finished
```

Conclusion:

- another NCCL path passed, but it was not a full production-shaped 20-step,
  high-inflight test;
- passing this did not falsify the later production NCCL failure.

### Production-shaped NCCL canary

Job:

```text
5453015 sync-ab-nccl-seq4-20
```

Slurm:

```text
FAILED, exit 15:0, elapsed 01:41:52, 16 nodes
```

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/sync_ab_nccl_seq4_20step
```

Important orchestrator log:

```text
10:10:12 Step 0 success, trainable 512/512, error 0.1%
10:10:58 Step 1 success
10:11:50 Step 2 success
10:13:50 Step 3 success
10:14:52 Updating weights to step 1
10:14:52 Updating policy in-flight to v1 (pause mode=keep, clear_cache=False)
10:27:03 update LoRA adapter failed on 12/12 inference peer(s): ReadTimeout
10:27:03 Skipping /resume after failed LoRA update because NCCL_READY was released
```

Important inference-side evidence:

```text
node_1.log: RPC call to receive_lora_update timed out
job log: EngineCore later emitted shared-memory broadcast block warnings
job log: EngineDeadError / execute_model timeout after orchestrator had already failed
```

Interpretation:

- this was not a pause-drain timeout;
- the system got into `/update_lora`;
- all 12 inference API peers timed out;
- current evidence does not show one clean initiating `NCCL unhandled cuda error`;
- the likely failing boundary is worker-side receive/materialize/commit at full
  production scale.

Operational conclusion:

- NCCL LoRA is still unservable for the production shape;
- increasing the 720 s timeout would only hide the problem;
- we need phase-level worker logs before the next NCCL production canary.

### Production-shaped FS canary

Job:

```text
5453016 sync-ab-fs-seq4-20
```

Slurm:

```text
FAILED, exit 1:0, elapsed 00:46:20, 16 nodes
task 11 on nid011092: Out Of Memory
```

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/sync_ab_fs_seq4_20step
```

Important orchestrator log:

```text
10:15:11 Loaded LoRA adapter debate-r64__v00000001 on 12 inference server(s)
10:18:14 Loaded LoRA adapter debate-r64__v00000002 on 12 inference server(s)
10:21:51 Loaded LoRA adapter debate-r64__v00000003 on 12 inference server(s)
10:24:31 Loaded LoRA adapter debate-r64__v00000004 on 12 inference server(s)
10:26:56 Loaded LoRA adapter debate-r64__v00000005 on 12 inference server(s)
10:29:05 Loaded LoRA adapter debate-r64__v00000006 on 12 inference server(s)
10:30:46 Updating weights to step 7
10:31-ish task 11 on nid011092: Out Of Memory
```

Interpretation:

- FS versioned loading worked through multiple updates;
- the crash was node memory pressure during or just after another adapter load;
- this old run did not include new RSS/stage/loaded-id diagnostics, so it cannot
  prove whether the culprit was vLLM adapter retention, staged file retention,
  Python/HTTP memory, or unrelated node pressure;
- the replacement FS run exists to answer exactly that.

### Small Qwen1.5B FS storm proxy

Output:

```text
/scratch/a6r/joanv.a6r/outputs/isambard/calibration/qwen15b_math_fs_lora_storm_2t2i
```

Shape:

- 4-node allocation;
- 2 trainer nodes;
- 2 inference nodes;
- small Qwen1.5B model;
- LoRA rank 256;
- max steps 30;
- `max_loras = 4`, `max_cpu_loras = 4`;
- filesystem sync;
- high churn.

Observed:

```text
13:01:12 Loaded qwen15b-r256__v00000001
13:02:06 Loaded qwen15b-r256__v00000004
13:02:06 Retired version [1]
13:02:08 Loaded qwen15b-r256__v00000007
13:02:08 Retired version [4]
...
13:08:44 onward: many ModelError -> InternalServerError('No available workers')
```

Interpretation:

- versioned FS load and retirement logic can run through many swaps;
- memory did not obviously kill this run;
- availability during hot churn became the main signal;
- `No available workers` is not a corrupt-adapter error. It is serving capacity
  unavailable at admission time, likely because version routing and LoRA load
  churn reduced effective serving capacity while the dispatcher kept admitting.

This points to a smarter admission/backpressure problem, not another pure FS
atomicity problem.

## Adapter Size And Data-Type Findings

The measured Qwen3.5-A3B adapter artifact used in the local stress benchmark:

```text
adapter_model.safetensors: 9.41 GB
tensors: 61520
dtype: torch.float32
```

This corrected an earlier underestimate. The adapter is not bf16 on disk in
that observed broadcast artifact. It is fp32, which doubles file size and
memory movement relative to bf16.

Measured local ingestion proxy:

```text
Hijack/on-GPU materialize:        0.62 s
Naive tmpfs native path total:   13.32 s
  D2H:                            1.05 s
  safetensors serialize:          5.81 s
  deserialize:                    1.45 s
  H2D:                            5.08 s
Naive per-tensor pinning total:  35.79 s
```

Takeaway:

- the MoE adapter has many tiny tensors;
- naive per-tensor host staging is operation-count-bound;
- bf16 serialization is a real lever;
- packed/flattened transport is also a real lever;
- node-local staging solves shared-FS storm, but does not make native-path
  ingestion free.

## What We Did Not Prove

NCCL:

- We did not prove production-scale NCCL LoRA can complete reliably.
- We did not phase-separate the 720 s timeout into receive vs materialize vs
  adapter registration.
- We did not prove `keep` pause is a sufficient device quiescence boundary at
  12 inference servers under production inflight load.

Filesystem:

- We did not yet prove memory plateaus in the 16-node FS production shape.
- The OOM in job `5453016` could be:
  - retained vLLM LoRA adapters;
  - staged `/dev/shm` adapters;
  - duplicate load/unload queues;
  - Python process RSS from load churn;
  - node-local pressure from something else.
- The new diagnostic run is needed before claiming it is fixed.

Async admission:

- `No available workers` during the Qwen1.5B storm means count-based
  `max_inflight_rollouts` is not enough.
- The missing knob is probably capacity-aware admission/backpressure, not simply
  lowering training batch size.

## Current Next Actions

### FS

1. Let job `5455616` run.
2. Monitor:

```bash
uv run --no-sync python scripts/stress/monitor_lora_canary.py \
  --job-id 5455616 \
  --output-dir /scratch/a6r/joanv.a6r/outputs/isambard/calibration/sync_ab_fs_seq4_20step_fs_hardened_20260701_1345 \
  --interval-s 30 \
  --max-minutes 240
```

3. Acceptance criteria:

- multiple adapter loads and unloads succeed;
- public LoRA names stay within expected live window;
- worker registered ids stay within expected live window;
- `/dev/shm` staged bytes plateau;
- API/worker RSS plateaus after warmup;
- no node OOM;
- rollout error rate does not blow up during adapter churn.

### NCCL

Do not launch another full production NCCL run until the worker path is
instrumented.

Add per-rank logs around:

```text
receive_lora_update entry
header receive start/end
chunk receive start/end per chunk
total bytes/tensors/dtypes
host staging allocation/reuse
LoRAModel.from_lora_tensors start/end
adapter_manager.remove_adapter start/end
adapter_manager.add_adapter start/end
adapter_manager.activate_adapter start/end
collective return
```

Then run a dedicated NCCL stress job:

- no full RL loop;
- same Qwen3.5-A3B adapter tensor count/shape/dtype;
- 12 inference servers if allocation allows;
- active decode load optional second phase;
- success target: receive+commit under 30-60 s, no tail rank outliers.

If receive+commit is still hundreds of seconds or wedges, switch production to
FS and keep NCCL as a research branch.

## Current Files To Know

Planning/history:

- `vllm-syncer.md`
- `nccl-fix.md`
- `lora-sync-incident-log-2026-07-01.md`

Core LoRA sync code:

- `src/prime_rl/utils/client.py`
- `src/prime_rl/utils/lora.py`
- `src/prime_rl/inference/lora_staging.py`
- `src/prime_rl/inference/vllm/server.py`
- `src/prime_rl/inference/vllm/worker/nccl.py`
- `src/prime_rl/inference/vllm/worker/filesystem.py`
- `src/prime_rl/inference/vllm/worker/diagnostics.py`
- `src/prime_rl/trainer/rl/broadcast/nccl.py`
- `src/prime_rl/trainer/rl/broadcast/filesystem.py`

Async/versioning:

- `src/prime_rl/orchestrator/watcher.py`
- `src/prime_rl/orchestrator/dispatcher.py`
- `src/prime_rl/orchestrator/types.py`
- `src/prime_rl/orchestrator/orchestrator.py`
- `src/prime_rl/orchestrator/utils.py`

Configs:

- `configs/calibration/gpqa_openended_debate_20step_bs512_g16_seq4_fs_sync_ab.toml`
- `configs/calibration/gpqa_openended_debate_20step_bs512_g16_seq4_nccl_sync_ab.toml`
- `configs/calibration/qwen15b_math_fs_lora_storm_2t2i.toml`

Stress/monitor:

- `scripts/stress/lora_stage_stress.py`
- `scripts/stress/pause_keep_live_probe.py`
- `scripts/stress/nccl_lora_live_stress.py`
- `scripts/stress/monitor_lora_canary.py`

Tests:

- `tests/unit/inference/test_lora_staging.py`
- `tests/unit/inference/test_nccl_lora_receive.py`
- `tests/unit/inference/test_vllm_admin_routes.py`
- `tests/unit/train/rl/test_filesystem_broadcast.py`
- `tests/unit/train/rl/test_nccl_lifecycle.py`
- `tests/unit/utils/test_client.py`
- `tests/unit/orchestrator/test_watcher_visibility.py`

## Copying This File To A Local Laptop

The remote cannot copy directly into a laptop that is "out of ssh" unless the
laptop exposes an inbound service or you provide another push target.

The clean option is to pull from the laptop.

From your laptop, if your SSH alias for this machine is `isambard`:

```bash
scp isambard:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main/lora-sync-incident-log-2026-07-01.md ~/Desktop/
```

Or with `rsync`:

```bash
rsync -av isambard:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main/lora-sync-incident-log-2026-07-01.md ~/Desktop/
```

If you do not have an SSH alias, replace `isambard` with the login host you
normally use, for example:

```bash
scp joanv.a6r@<isambard-login-host>:/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main/lora-sync-incident-log-2026-07-01.md ~/Desktop/
```

Alternative options:

- commit and push the file, then pull it locally from Git;
- copy it through VS Code / Cursor remote file explorer;
- temporarily expose a local SSH receiver or tunnel, then `scp` from remote to
  laptop, but that is more moving parts than this needs.
