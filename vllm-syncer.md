# vLLM Syncer Plan

Date: 2026-06-29

Goal: replace Prime-RL's custom NCCL LoRA update choreography with a sync path that follows vLLM's native RL weight-update lifecycle where possible, keeps Prime-RL's multi-run/LoRA semantics where necessary, and fails fast with useful attribution when NCCL or engine state is bad.

This is an implementation plan, not a claim that the current pinned vLLM wheel is drop-in ready. The first milestone is proving the exact local API shape on the active aarch64 runtime, currently `vllm==0.22.0` with the CUDA-13 wheel.

## Current Failure Model

Recent Isambard debate runs failed in two related shapes:

1. Hard NCCL/CUDA failure during LoRA receive.
   - Example: `outputs/isambard/calibration/debate_qwen35-a3b__qwen9b-or__pcd4final/logs/inference/node_6.log:2051`.
   - Worker stack: `receive_lora_update -> _receive_lora_object -> communicator.broadcast(...) -> RuntimeError: NCCL error: unhandled cuda error`.
   - Server returned HTTP 500 for `/update_lora`.

2. Engine/worker utility path wedged while HTTP stayed alive.
   - Example: `outputs/isambard/calibration/debate_qwen35-a3b__qwen9b-or__sequential4/logs/inference/node_9.log:4635`.
   - Engine throughput dropped to zero with running requests still present, repeated "No available shared memory broadcast block found in 60 seconds", `/metrics` and `/health` still returned 200.
   - Orchestrator observed `ReadTimeout`/`ReadError` across peers and then `/resume` timeouts.

The watcher is not the root cause. It is the component that observes the failed update and then crashes the run. That crash is preferable to silently serving mixed or unknown weights.

## Existing Local Flow

Important files:

- `src/prime_rl/orchestrator/watcher.py`
- `src/prime_rl/utils/client.py`
- `src/prime_rl/inference/vllm/server.py`
- `src/prime_rl/inference/vllm/worker/nccl.py`
- `src/prime_rl/trainer/rl/broadcast/nccl.py`

Current LoRA update sequence:

1. Trainer decides a run is ready to update.
2. Trainer writes `broadcasts/step_N/STABLE` and waits for `broadcasts/step_N/NCCL_READY`.
3. Watcher sees `STABLE`.
4. Watcher calls observers' `on_version_pending(N)` to drain stale/off-policy rollouts.
5. Orchestrator calls `update_lora_adapter(...)`.
6. Orchestrator sends `/pause?mode=keep&clear_cache=false` to every inference peer.
7. Orchestrator touches `NCCL_READY`, releasing the trainer broadcaster.
8. Orchestrator POSTs `/update_lora` to every inference peer.
9. `/update_lora` calls `engine_client.collective_rpc("receive_lora_update", args=(step, header_expectation))`.
10. Worker receives custom header/chunks over Prime-RL's `PyNcclCommunicator` and commits a resident LoRA adapter.
11. Server registers a sentinel `LoRARequest` so future requests can route to the in-memory adapter.
12. Orchestrator always attempts `/resume` in a `finally`.
13. Watcher advances `policy.version` only after update succeeds.

This flow has three uncomfortable properties:

- `pause_generation(mode="keep")` freezes requests for later continuation. It is not a drain and not a clean quiescence barrier.
- `/update_lora` calls `collective_rpc(..., timeout=None)`. The outer HTTP read timeout tells the orchestrator a peer did not answer; it does not prove the engine-side utility RPC was cancelled.
- A failed NCCL collective is terminal for that communicator. Retrying `/update_lora` after a read timeout is unsafe, and resuming after a true NCCL failure is at best unproven.

Refined root cause:

- `on_version_pending` runs before pause, but it only drops train rollout groups past `max_off_policy_steps`; it intentionally keeps fresh within-lag rollouts running.
- The current `mode="keep"` pause freezes those fresh requests in-place. Logs showing 168-199 `Running` requests after pause are therefore expected, not proof that `on_version_pending` failed.
- A frozen engine is not a clean device state for a rooted NCCL receive: resident KV/cache state and pending scheduler/GPU work remain live, but the engine is no longer stepping toward quiescence.
- `mode="wait"` is the best default for weight mutation under this design: stale groups have already been cancelled, fresh within-lag groups finish on old weights and stay usable, and the engine reaches a stepped idle point before `NCCL_READY`.
- `mode="abort"` is not the safe default. It discards valid within-lag rollouts and can reintroduce the KV-connector cleanup race that `watcher.py` explicitly avoids when PD/NIXL is enabled.
- The eight debate configs inspected here all set `use_pd_kv_transfer = false`, so the NIXL-specific cleanup race is probably not the direct mechanism for these failures. It still constrains the general design because the same code path supports PD/NIXL configs.

## Current Proceedings

The investigation changes the order of work more than the destination.

Old tempting order:

1. jump to vLLM native lifecycle;
2. maybe bump to `0.23.0`;
3. then debug the existing path if needed.

New order:

1. Finish the `0.22.0` CUDA hygiene first. This is done for aarch64: the runtime should use the no-suffix CUDA-13 wheel, not `0.22.0+cu129`.
2. Fix the contract contradiction in the current path before adding new abstractions:
   - make `/pause` honor `mode` and `clear_cache` request parameters instead of hardcoding `mode="keep", clear_cache=False`;
   - stop saying `/pause` drains unless the selected mode is actually `wait`;
   - default weight mutation to `mode="wait", clear_cache=True`;
   - treat `mode="keep"` as an async-RL continuation mode, not as an idle barrier;
   - reserve `mode="abort"` for explicit emergency cancellation or configs that have proven abort-safe cleanup.
3. Make failed updates terminal for that inference pool:
   - if `NCCL_READY` has been touched and `/update_lora` or `/update_weights` fails or times out, do not normal-resume;
   - classify the pool as requiring restart/recovery;
   - keep current "no retry after started collective" behavior.
4. Add small observability around the existing path:
   - capability endpoint;
   - current pause mode and paused state;
   - per-peer update phase, elapsed time, and first exception;
   - explicit `collective_rpc(timeout=...)` rather than only relying on the outer HTTP timeout.
5. Add the manifest sidecar before any native lifecycle migration. This gives the orchestrator something to validate before releasing the trainer sender.
6. Only then wrap vLLM's lifecycle calls on `0.22.0` and measure whether they improve failure behavior for LoRA updates.
7. Keep `0.23.0` as a parallel compatibility spike, not the main fix path. Its sparse NCCL update surface is interesting, but the linked commit proves the bump has real Prime-RL patch/API churn.

This makes the next engineering move smaller and more falsifiable: repair the current pause/update semantics, then test whether native lifecycle integration buys anything beyond clearer phase boundaries.

Do not make `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` part of the weight-sync fix. PyTorch's expandable segments allocator mode is a fragmentation/OOM mitigation, not a scheduler-quiescence or NCCL-collective correctness mechanism. Prime-RL already sets inference to `expandable_segments:False`; the local production comment says `True` was observed to crash vLLM v1 mid-run with `api_server_count>1`. Trainer/SFT use `True`, but inference should keep `False` unless a separate A/B test shows allocator-fragmentation OOMs after the wait-based semantic fix lands.

## Upstream Direction

vLLM stable docs now document a native RL weight-transfer lifecycle:

1. `init_weight_transfer_engine(...)`
2. `start_weight_update(...)`
3. `update_weights(...)`
4. `finish_weight_update()`

Context7 retrieved the stable vLLM docs for:

- `training/weight_transfer/nccl`
- `examples/rl/rlhf_http_nccl`
- `examples/rl/rlhf_nccl_fsdp_ep`
- `training/async_rl`
- `examples/features/pause_resume`

Local `vllm==0.22.0` has the protocol declarations in `.venv/lib/python3.12/site-packages/vllm/engine/protocol.py`:

- `init_weight_transfer_engine`
- `start_weight_update`
- `update_weights`
- `finish_weight_update`

Local implementation availability is not yet proven. Grep found the protocol declarations and `vllm.distributed.weight_transfer.*` engine classes, but not obvious concrete async server implementations wired through the current Prime-RL server path. Treat native lifecycle support as a capability to probe, not an assumption.

## vLLM 0.23 Probe

The Prime-RL commit `3cbeb5d0f1fd810340027e1fbe73e694fbae3d62` is useful evidence, not a drop-in answer. It bumps vLLM to `0.23.0`, but also performs a compatibility pass:

- `pyproject.toml` moves the minimum and wheel URLs from `0.22.0` to `0.23.0`.
- `src/prime_rl/inference/patches.py` removes/adapts monkeypatches that were specific to the `0.22.0` internals.
- `src/prime_rl/inference/vllm/serving_tokens.py` updates a moved `get_max_tokens` import.
- `src/prime_rl/inference/server.py` sets `VLLM_USE_V2_MODEL_RUNNER=0` to keep Prime-RL's padded-input scrub effective.
- `src/prime_rl/inference/vllm/worker/nccl.py` keeps the Nemotron reload workaround because the relevant upstream fix was not in `0.23.0`.

So `0.23.0` is plausible, but it is not a one-line wheel bump for this branch.

What it may buy:

- vLLM release notes for `0.23.0` include sparse NCCL in-place update support. The current local `0.22.0` wheel exposes lifecycle methods on `EngineClient`, but `WeightTransferEngine` does not expose `trainer_send_sparse_weights` or `receive_sparse_weights`.
- Sparse in-place updates are directionally relevant to LoRA or partial adapter updates, but this still needs a mapping from Prime-RL's resident adapter registration to vLLM's update metadata. Do not assume sparse full-model update support automatically solves LoRA adapter residency.

What it does not buy by itself:

- It does not make `pause_generation(mode="keep")` a drain. vLLM's own protocol says `keep` freezes queued/in-flight requests until resume; `wait` drains; `abort` aborts.
- It does not remove Prime-RL's version-pinned sampler patches. `sampler_perf.py` and `flashinfer_sampler.py` still guard on `vllm.__version__ == "0.22.0"` in this branch.
- It does not prove the custom `/update_lora` collective is recoverable after an NCCL failure.

Decision rule:

- Baseline fix: keep `0.22.0`, eliminate the old aarch64 `+cu129` wheel, and fix the pause/update semantics.
- Separate spike: test `0.23.0` in a disposable branch/env, port the Prime-RL compatibility changes, then run server import, sampler fast-path validation, one tiny full-weight lifecycle smoke, and one tiny LoRA adapter update smoke before considering it for long Isambard runs.

## Design Constraints

Hard constraints:

- Preserve Prime-RL's policy-version invariant: `policy.version` advances only after inference has definitely adopted the new adapter.
- Do not serve unknown or mixed weight state.
- Do not retry a started NCCL receive after HTTP `ReadTimeout`.
- Do not make the trainer block indefinitely in a rooted NCCL send.
- Do not silently fall back from NCCL to filesystem without an explicit config choice.
- Do not touch `optimization_dtype` or `reduce_dtype`.
- Keep filesystem `STABLE` as the trainer-to-orchestrator notification unless the whole trainer/orchestrator control plane is redesigned.
- Keep raw failure evidence per peer: host, node rank, phase, request path, exception class, and elapsed time.

Soft constraints:

- Prefer vLLM's standard lifecycle over Prime-RL worker internals.
- Keep legacy NCCL/filesystem paths runnable behind a feature flag during rollout.
- Make small compatibility probes before changing long-running training configs.
- Minimize new global state in vLLM app state; use typed request models where practical.

## Target Architecture

Introduce a small "syncer" abstraction around inference weight updates:

```text
WeightWatcher
  -> InferencePool.update_weights(...)
    -> WeightSyncer.update(step_dir, step, lora_name)
      -> pause/drain
      -> start update phase
      -> release trainer sender
      -> receive/update weights
      -> register serving adapter
      -> finish update phase
      -> resume or mark pool dead
```

Suggested module layout:

- `src/prime_rl/orchestrator/weight_syncer.py`
  - phase orchestration and failure classification
- `src/prime_rl/utils/client.py`
  - HTTP client endpoints and `_gather_admin` helpers
- `src/prime_rl/inference/vllm/server.py`
  - thin endpoint wrappers around vLLM native APIs and any Prime-RL LoRA registration endpoint
- `src/prime_rl/inference/vllm/worker/lora_weight_transfer.py`
  - only if a custom vLLM `WeightTransferEngine` or worker method is needed
- `src/prime_rl/trainer/rl/broadcast/vllm_native.py`
  - trainer-side metadata and sender path, once native lifecycle is proven

Do not move the whole watcher into `utils/client.py`. The syncer should make phase sequencing explicit.

## Protocol Choice

Add a config switch rather than replacing legacy behavior immediately:

```toml
[weight_broadcast]
type = "nccl"
protocol = "legacy_lora"        # current behavior
# protocol = "vllm_lifecycle"   # native pause/start/update/finish where available
# protocol = "vllm_lora_engine" # custom LoRA WeightTransferEngine path
```

Possible names:

- `legacy_lora`
- `vllm_lifecycle`
- `vllm_lora_engine`

Do not overload `type = "nccl"` for this. Transport and lifecycle are different axes.

## Phase 0: Capability Probe

Objective: determine what the pinned vLLM wheel and Prime-RL server can actually do.

Add a temporary or permanent admin endpoint:

```http
GET /weight_sync_capabilities
```

Response shape:

```json
{
  "vllm_version": "0.22.0",
  "engine_client_type": "...",
  "has_pause_generation": true,
  "has_resume_generation": true,
  "has_init_weight_transfer_engine": true,
  "has_start_weight_update": true,
  "has_update_weights": true,
  "has_finish_weight_update": true,
  "has_collective_rpc_timeout": true,
  "supports_lora_registration": true,
  "supports_native_lora_weight_transfer": false
}
```

Also add a local script or command recipe:

```bash
uv run --active --no-sync python - <<'PY'
from vllm.engine.protocol import EngineClient
for name in [
    "init_weight_transfer_engine",
    "start_weight_update",
    "update_weights",
    "finish_weight_update",
    "pause_generation",
    "resume_generation",
]:
    print(name, hasattr(EngineClient, name))
PY
```

Acceptance:

- We know whether native methods exist on the live engine client object, not only on abstract protocol classes.
- We know whether calling `start_weight_update` on a tiny local engine succeeds, raises `NotImplementedError`, or fails due to missing backend config.
- We know whether vLLM's server already exposes HTTP endpoints for the native lifecycle in this pinned wheel. If not, Prime-RL must expose thin wrappers.

## Phase 1: Make Current Path Safer Before Migration

This is the smallest risk reduction, even if native migration takes longer.

Changes:

1. Add engine-side timeout to custom LoRA receive:

```python
await engine_client(request).collective_rpc(
    "receive_lora_update",
    timeout=UPDATE_WEIGHTS_TIMEOUT_S,
    args=(step, header_expectation),
)
```

Use a server constant, not the HTTP client's private value. If vLLM `collective_rpc(timeout=...)` is not honored on this engine type, log that in capabilities.

2. Stop unconditional resume after fatal update failure.

Current code resumes in `finally`. Replace with phase-aware behavior:

```text
pause failed -> do not update; best-effort resume only peers that acknowledged pause
preflight failed before NCCL_READY -> resume safe
receive returned 4xx before NCCL_READY -> resume safe
NCCL_READY touched and update failed/timeouts -> mark pool unhealthy; no normal resume
finish/register failed after receive success -> mark version ambiguous; no normal resume
```

3. Prefer `pause_generation(mode="wait", clear_cache=True)` for weight mutation.

Rationale:

- `on_version_pending` only cancels stale groups; fresh within-lag rollouts are supposed to finish and be used.
- `wait` keeps the engine stepping until those valid in-flight requests complete on old weights.
- `wait` lets stale-abort/KV-cleanup work settle under normal stepping where PD/NIXL is enabled.
- Once `wait` returns, the device is much closer to a clean quiescent state for the NCCL receive than `keep` provides.

Keep `mode="keep"` behind a config option only for workflows that explicitly accept resumed old-KV/new-weight continuation. Keep `mode="abort"` as an emergency/fault-handling option, not the default update path.

4. Add a real post-pause idle assertion.

After pause returns, poll per-peer metrics or a new admin endpoint until:

- no running requests, or
- timeout -> fail before touching `NCCL_READY`.

For `mode="wait"`, `/pause` should itself drain by stepping. The assertion exists to catch implementation drift, bugs, or engine-side hangs before the trainer sender is released.

Acceptance:

- A killed receiver before or during `/update_lora` yields one fatal, attributed sync failure and does not attempt to continue serving.
- If wait never reaches quiescence, trainer never sees `NCCL_READY`.
- Legacy behavior is still available via config.

## Phase 2: Metadata Sidecar

Native `update_weights` needs metadata before the receiver can post an update request. The current custom LoRA path sends header and chunk metadata over NCCL. That makes preflight weaker.

Change trainer broadcast publication so `STABLE` means:

```text
broadcasts/step_N/
  STABLE
  update_manifest.json
```

For LoRA, manifest should include:

```json
{
  "protocol_version": 1,
  "step": 20,
  "kind": "lora",
  "adapter": {
    "lora_name": "...",
    "lora_int_id": 1,
    "rank": 64,
    "alpha": 128,
    "peft_config": {}
  },
  "chunks": [
    {
      "names": ["..."],
      "dtype_names": ["bfloat16"],
      "shapes": [[64, 8192]],
      "packed": false
    }
  ],
  "trainer_world": {
    "src_rank": 0,
    "world_size": 49,
    "inference_world_size": 48
  }
}
```

For full weights, manifest should mirror vLLM `NCCLWeightTransferUpdateInfo`:

```json
{
  "kind": "full",
  "names": ["..."],
  "dtype_names": ["bfloat16"],
  "shapes": [[...]],
  "packed": true
}
```

Ordering:

1. Trainer computes manifest.
2. Trainer writes manifest atomically (`tmp` then rename).
3. Trainer writes `STABLE`.
4. Trainer waits for `NCCL_READY`.
5. Trainer sends exactly what manifest describes.

Acceptance:

- Orchestrator can validate adapter name, step, world size, and metadata before releasing trainer.
- A mismatch fails before `NCCL_READY`.
- Logs include manifest path and digest.

## Phase 3: Native Lifecycle Wrapper

Expose Prime-RL server endpoints that map directly onto vLLM native methods if the live engine supports them:

```http
POST /weight_sync/init
POST /weight_sync/start
POST /weight_sync/update
POST /weight_sync/finish
POST /weight_sync/register_lora
GET  /weight_sync/state
```

Server-side mapping:

```python
await engine_client(request).init_weight_transfer_engine(
    WeightTransferInitRequest(init_info=...)
)

await engine_client(request).pause_generation(mode="wait", clear_cache=True)

await engine_client(request).start_weight_update(is_checkpoint_format=True)

await engine_client(request).update_weights(
    WeightTransferUpdateRequest(update_info=...)
)

await engine_client(request).finish_weight_update()
```

`register_lora` is Prime-RL-specific and may still be needed:

- native full-weight update mutates base model weights and needs no `LoRARequest`.
- LoRA update must leave a resident adapter and register a stable `LoRARequest` name/id.

State machine per peer:

```text
idle
  -> paused
  -> update_started
  -> receiving
  -> received
  -> lora_registered
  -> finished
  -> resumed
```

Every transition logs:

- `sync_id`
- step
- phase
- peer base URL
- host/node rank if known
- elapsed seconds
- error class and message

Use one `sync_id` per watcher update. Include it in all endpoint bodies.

Acceptance:

- Repeated successful updates under load show all peers entering the same phases in order.
- On one-peer failure, peer phases identify the first bad peer and whether it died before or after `NCCL_READY`.
- No update relies only on HTTP read timeout for engine-side cancellation.

## Phase 4: LoRA Integration Choices

There are three paths. Do them in this order.

### Option A: Native Lifecycle Around Existing Receiver

Minimal migration:

```text
pause_generation(mode="wait", clear_cache=True)
start_weight_update(...)
touch NCCL_READY
collective_rpc("receive_lora_update", timeout=...)
register LoRARequest
finish_weight_update()
resume_generation()
```

Pros:

- Smallest code change.
- Keeps current LoRA commit path.
- Uses native pause/start/finish if available.

Cons:

- Still uses custom `PyNcclCommunicator` and custom chunk protocol.
- `start_weight_update` may not provide much protection if `receive_lora_update` is outside vLLM's native transfer engine.
- Not the final maintainable design.

Use this only as a stepping stone.

### Option B: Custom vLLM LoRA WeightTransferEngine

Better long-term path:

- Implement a Prime-RL LoRA transfer engine that conforms to vLLM's `WeightTransferEngine` abstraction.
- Reuse vLLM's init/start/update/finish lifecycle.
- Keep Prime-RL-specific LoRA metadata and commit semantics.
- Let trainer send through vLLM's trainer-side NCCL helpers where possible.

Receiver responsibilities:

1. Parse `LoRAWeightTransferUpdateInfo`.
2. Receive chunked tensors.
3. Validate adapter header: step, name, id, rank, alpha, expected PEFT config.
4. Stage tensors in bounded host/pinned buffers.
5. Commit into vLLM LoRA manager.
6. Return a receipt: tensor count, bytes, dtype set, adapter id, elapsed.

Pros:

- Aligns with vLLM's extension point.
- Keeps LoRA as a first-class update kind rather than pretending it is a full checkpoint.
- Easier to delete legacy custom receiver later.

Cons:

- More work.
- Need to understand vLLM's engine factory registration and whether plugins can register custom weight-transfer backends cleanly in `0.22.0`.

### Option C: Convert LoRA Update to Filesystem Adapter Load

Fallback, not preferred:

- Trainer writes adapter to filesystem.
- Orchestrator calls `/v1/load_lora_adapter` with bounded concurrency.

Pros:

- Avoids NCCL update collective.
- Existing path already exists.

Cons:

- Prior evidence suggests parallel multi-GB adapter loads from Lustre can cause read storms and watcher death.
- Slower and less scalable.
- Does not solve long-term sync architecture.

Keep as emergency fallback only.

## Phase 5: Trainer Sender Migration

Current trainer sender:

- creates a custom `StatelessProcessGroup`;
- uses Prime-RL `broadcast_object` / `broadcast_state_dict`;
- waits for `NCCL_READY`;
- has a local CUDA event deadline for LoRA send completion;
- chunks LoRA at 512 MiB.

Native sender target:

- use `vllm.distributed.weight_transfer.nccl_engine.NCCLWeightTransferEngine.trainer_init`;
- use `NCCLWeightTransferEngine.trainer_send_weights`;
- preserve Prime-RL's `NCCL_READY` release mechanism until trainer/orchestrator control is unified;
- preserve or reimplement the deadline/abort behavior if vLLM trainer sender does not provide it.

Migration detail:

1. Add `VLLMNativeWeightBroadcastSender` alongside `NCCLWeightBroadcastSender`.
2. Teach `setup_weight_broadcast` to select it via `weight_broadcast.protocol`.
3. Build metadata from exactly the iterator the sender will transmit.
4. Write manifest before `STABLE`.
5. Wait for `NCCL_READY`.
6. Call native trainer send.
7. Mark post-send completion in a `SEND_DONE` marker for debugging, not for correctness.

Acceptance:

- Full-weight native smoke passes.
- LoRA native/custom engine smoke passes.
- Failed receiver causes trainer send to abort within bounded time.
- No trainer rank begins DTensor resolution before orchestrator pause, preserving the existing barrier invariant.

## Failure Semantics

Classify failures explicitly.

Recoverable before update starts:

- manifest missing or invalid;
- adapter header mismatch;
- peer unreachable before pause;
- pause rejected before any peer enters update;
- no `NCCL_READY` touched.

Action:

- do not advance version;
- resume only peers that were paused;
- leave trainer waiting or fail before releasing trainer depending on phase;
- watcher may retry next poll if no fatal peer state was observed.

Fatal after update starts:

- HTTP 500 from `/update_lora` or native `update_weights`;
- `ReadTimeout` after `NCCL_READY`;
- engine-side collective timeout;
- NCCL/CUDA error;
- `finish_weight_update` failure;
- LoRA registration failure after tensors committed.

Action:

- do not advance version;
- mark inference pool unhealthy;
- do not normal-resume;
- request orchestrator shutdown of the run;
- preserve first-failure peer and phase in logs.

Ambiguous:

- HTTP client disconnects while engine may still be in `collective_rpc`;
- peer process still answers `/health` but liveness probe hangs;
- trainer send timed out but some receivers may have committed.

Action:

- treat as fatal;
- do not resume;
- require engine restart.

NCCL rule: once a communicator has had an unhandled CUDA/NCCL error or a collective timeout, do not attempt to reuse that communicator for serving correctness. Abort/destroy/recreate via process restart unless vLLM exposes a documented reset path that is validated in our harness.

## Observability

Add structured logs and W&B metrics.

Metrics:

- `watcher/sync_phase`
- `watcher/sync_elapsed_seconds`
- `watcher/sync_failures_total`
- `watcher/sync_first_failure_peer`
- `watcher/sync_first_failure_phase`
- `watcher/sync_paused_peers`
- `watcher/sync_update_started_peers`
- `watcher/sync_finished_peers`
- `watcher/sync_resumed_peers`
- `watcher/sync_nccl_ready_touched`
- `watcher/sync_manifest_bytes`
- `watcher/sync_lora_bytes`
- `watcher/sync_lora_chunks`

Log fields:

```json
{
  "event": "weight_sync_phase",
  "sync_id": "...",
  "step": 20,
  "phase": "receiving",
  "peer": "http://nid010902:8000",
  "rank_offset": 24,
  "inference_world_size": 48,
  "elapsed_s": 12.34
}
```

Useful admin endpoint:

```http
GET /weight_sync/state
```

Response:

```json
{
  "status": "idle|paused|updating|failed",
  "sync_id": "...",
  "step": 20,
  "phase": "receiving",
  "last_error": null,
  "registered_lora_steps": [20],
  "engine_dead": false
}
```

## Test Plan

### Unit Tests

Target pure logic only.

- manifest serialization/deserialization;
- manifest validation against step and adapter name;
- phase-state transitions;
- failure classification;
- peer result aggregation;
- config parsing for `protocol`.

Do not unit-test vLLM internals with mocks so heavy that the test says nothing.

### Local CPU/No-GPU Checks

- import checks;
- endpoint schema tests if app can initialize without vLLM engine;
- capability probe unit around protocol attributes.

### Single-Node GPU Smoke

Allocation: 1 node / 4 GPUs.

Purpose:

- prove server starts;
- prove native lifecycle methods exist on live engine;
- run one update with tiny model if possible;
- run one LoRA registration if possible.

Acceptance:

- one update succeeds;
- generation before and after update works;
- logs show ordered phases.

### Two-Inference-Replica Smoke

Allocation: 3 nodes preferred:

- 1 trainer/broadcaster node;
- 2 inference nodes, each one vLLM server with TP=4.

Purpose:

- catch data-parallel pause/update coordination bugs;
- exercise rank offsets and `inference_world_size`;
- reproduce the class of failure closer to the recent runs without spending a 13-node allocation.

Acceptance:

- 20 repeated LoRA updates under active generation load;
- no wedged `/resume`;
- no peer has running requests at update release;
- p95 update latency stable;
- all peers report same adapter step.

### Fault Injection

Run on 3-node setup.

Faults:

- kill one inference worker after pause before `NCCL_READY`;
- kill one worker after `NCCL_READY`;
- delay trainer send beyond update timeout;
- corrupt manifest step;
- corrupt adapter name;
- make one peer return 500 on update;
- make one peer accept update but hang.

Acceptance:

- pre-release faults do not release trainer;
- post-release faults fail fast and attribute first bad peer;
- no case advances `policy.version` after failed update;
- fatal cases do not normal-resume.

### Scale Confidence

Allocation: 5 nodes, then 13 nodes.

5-node shape:

- 1 trainer;
- 4 inference nodes.

13-node shape:

- 1 trainer;
- 12 inference nodes.

Acceptance:

- 50 repeated updates under load at 5 nodes;
- 20 repeated updates under load at 13-node shape;
- no read-timeout cascade;
- sync failure, if induced, has precise first-peer attribution.

## Rollout Plan

0. Land the aarch64 CUDA-13 vLLM wheel hygiene fix.
1. Fix `/pause` parameter handling and the false drain comments.
2. Switch weight mutation to an explicit wait-and-clear pause mode behind config.
3. Make failed post-`NCCL_READY` updates skip normal resume and mark the pool restart-required.
4. Add capability/state endpoints and per-peer phase diagnostics.
5. Add an explicit engine-side `collective_rpc` timeout.
6. Add manifest sidecar validation before touching `NCCL_READY`.
7. Run 3-node fault-injection suite on the safer legacy path.
8. Run 5-node confidence suite on the safer legacy path.
9. Run 13-node reproduction shape on the safer legacy path.
10. Add vLLM lifecycle wrappers on `0.22.0`.
11. Prototype Option A on 1 node, then 3 nodes under load.
12. If Option A materially reduces failure rate, keep it as guarded experimental protocol.
13. Run the `0.23.0` compatibility spike in a disposable branch/env.
14. Implement Option B custom LoRA `WeightTransferEngine` only if Option A is inadequate and sparse/native transfer evidence justifies the extra surface.
15. Flip generated debate configs only after the 5-node suite passes.

## Suggested PR Slices

PR 1: Observability and capability probe.

- `GET /weight_sync_capabilities`
- `GET /weight_sync/state`
- structured sync logs
- no behavior change

PR 2: Safer legacy sync.

- engine-side collective timeout
- conditional resume
- `pause_mode` config with default still legacy until tested
- first-failure aggregation

PR 3: Manifest sidecar.

- trainer writes `update_manifest.json`
- orchestrator validates before touching `NCCL_READY`
- no native vLLM lifecycle yet

PR 4: vLLM lifecycle wrapper.

- `/weight_sync/start`
- `/weight_sync/update`
- `/weight_sync/finish`
- feature-flagged `protocol = "vllm_lifecycle"`

PR 5: LoRA native/custom transfer engine.

- custom LoRA update info
- receiver commit path integrated with vLLM lifecycle
- trainer sender migration

PR 6: Config migration and runbook.

- update generated debate configs only after validation
- update launch skill/runbook if the workflow changes

## Open Questions

1. Does the live `AsyncLLM` object in the pinned wheel implement `start_weight_update` and friends, or only the abstract protocol?
2. Does vLLM's native lifecycle support LoRA adapter updates, or only full model weight updates?
3. Does `pause_generation(mode="wait", clear_cache=True)` on the live server actually return only after every in-flight request has reached a terminal state? Verify with metrics, not comments.
4. Does native `NCCLWeightTransferEngine.trainer_send_weights` provide a host-side completion/abort mechanism equivalent to Prime-RL's event deadline?
5. Can a custom weight-transfer backend be registered through vLLM plugin/config without monkeypatching?
6. Should fatal sync failure request process-level restart from the launcher, or simply crash the orchestrator and let Slurm tear down the whole job?

## Initial GPU Budget

Minimum useful path:

- 0 GPUs: static API inspection and endpoint schema work.
- 1 node / 4 GPUs: first live smoke.
- 3 nodes / 12 GPUs: recommended prototype allocation; one trainer plus two TP=4 inference replicas.
- 5 nodes / 20 GPUs: confidence before using real debate shapes.
- 13 nodes / 52 GPUs: final reproduction shape for the observed 12-peer failure pattern.

Do not start with 13 nodes. Start with 1 node to prove the API, then 3 nodes to test DP pause/update behavior.

## Non-Goals

- Do not solve overlong debate prompts in this syncer work.
- Do not redesign trainer/orchestrator transport.
- Do not replace all vLLM patches.
- Do not silently change numerical dtype knobs.
- Do not optimize throughput before correctness and failure attribution are fixed.
