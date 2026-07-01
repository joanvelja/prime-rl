# NCCL LoRA Fix Plan

Date: 2026-06-30

## Short Answer

SkyRL/veRL-style is not the only viable path, and it is mostly not "LoRA over
NCCL." The useful part is the semantic model:

1. Each policy update becomes a distinct adapter version.
2. New requests route to the newest adapter version.
3. Old in-flight requests may finish on the adapter version they started with.
4. Training explicitly filters or weights samples by policy staleness.
5. Old adapter versions are retired only after they are no longer needed.

That semantic model is elegant. The transport is a separate choice.

For this Qwen3.5-A3B setup, naive filesystem fanout is not obviously efficient:
local rank-64 adapter artifacts are about 20.2 GB each, so 12 inference replicas
independently reading one adapter from shared storage is roughly 240 GB of
shared-filesystem read traffic per update, before tail latency and contention.
The current custom NCCL path exists for a good reason: it can move the adapter
payload without making Lustre the broadcast primitive.

The best direction is therefore:

```text
versioned LoRA adapter semantics
+ efficient transport, probably NCCL at this scale
+ no in-place mutation of the active adapter id/name
```

## Current Problem

The current NCCL LoRA path updates a resident adapter in place:

- trainer broadcasts LoRA header and tensor chunks after `NCCL_READY`;
- inference workers receive the chunks via `receive_lora_update`;
- vLLM commits them into the LoRA manager;
- the server registers a stable `LoRARequest` name/id for future requests.

Relevant local code:

- `src/prime_rl/utils/client.py`: orchestrates `/pause`, `NCCL_READY`,
  `/update_lora`, `/resume`.
- `src/prime_rl/inference/vllm/server.py`: exposes `/pause`,
  `/load_lora_adapter`, and `/update_lora`.
- `src/prime_rl/inference/vllm/worker/nccl.py`: receives and commits LoRA
  tensors into vLLM.
- `src/prime_rl/trainer/rl/broadcast/nccl.py`: chunks and broadcasts LoRA
  state dicts.

The main issue is not that NCCL is inefficient. It is that a one-shot 49-rank
collective is being combined with in-place adapter mutation and ambiguous async
request semantics. Once `NCCL_READY` is touched, failure is a distributed
teardown event. Retrying a started receive is not safe.

## Complexity Model

Let:

- `R`: inference replicas, currently 12.
- `G`: inference GPUs, currently 48.
- `A`: adapter artifact size, currently about 20.2 GB for observed rank-64
  Qwen3.5-A3B adapters.
- `V`: number of live adapter versions retained.
- `L`: allowed policy lag / staleness window.

### Current In-Place NCCL

Transfer cost:

```text
O(A) broadcast payload through the NCCL collective
```

State cost:

```text
O(1) live adapter versions
```

Failure complexity:

```text
O(G) coupled failure surface: one bad participant poisons the collective
```

Pros:

- Efficient payload movement.
- Avoids many independent shared-filesystem reads.
- Already implemented.

Cons:

- Mutates the active adapter in place.
- Any post-`NCCL_READY` failure requires inference restart/reinit.
- Hard to reason about requests that were queued or partially generated across
  the update boundary.

### Naive Versioned Filesystem Load

This is the simple implementation:

```text
trainer writes adapter_vN to shared filesystem
all replicas POST /load_lora_adapter(adapter_vN)
each replica reads adapter_vN from shared storage
new requests route to adapter_vN
```

Transfer cost:

```text
O(R * A) shared-filesystem read traffic
```

State cost:

```text
O(V * A) across retained versions, depending on vLLM adapter cache behavior
```

Failure complexity:

```text
mostly per-replica and retryable before routing switches
```

Pros:

- Clean semantics.
- Avoids a one-shot NCCL receive for LoRA.
- Per-replica load failures can be attributed and retried before the new
  version is exposed.

Cons:

- At 20 GB adapters, naive shared filesystem fanout is expensive.
- Load tail latency may be dominated by the slowest replica.
- Adapter cache pressure increases with retained versions.

This is why "versioned filesystem load" is not automatically the right
implementation for our Qwen3.5-A3B runs, even though the semantic model is good.

### Versioned NCCL LoRA

Transfer cost:

```text
O(A) NCCL transfer, same payload class as current path
```

State cost:

```text
O(V * adapter_residency)
```

With bounded staleness, `V` should be small: usually current plus one or a few
lagged versions.

Failure complexity:

```text
NCCL transfer is still all-or-nothing, but request semantics are no longer
in-place mutation of an active adapter
```

Pros:

- Keeps efficient transport.
- Gives async RL a clean version boundary.
- Avoids stale queued requests silently switching weights under the same adapter
  name.

Cons:

- Requires adapter lifecycle management.
- Requires unique adapter names and ids.
- Still needs hard post-`NCCL_READY` failure handling.

This is the preferred end state.

## Important Correction: Waiting Requests

Do not require `waiting_requests == 0` after `/pause`.

After a scheduler pause, `running_requests == 0` and `waiting_requests > 0` can
be normal. A paused scheduler is specifically not moving waiting requests into
running requests, so waiting may never drain while paused.

The correctness question is not "is the queue empty?" It is:

- no worker is executing decode/prefill when the collective begins;
- queued requests are versioned, cancelled, or routed under an explicit policy;
- no request silently changes semantic policy version because the same adapter
  id/name was mutated underneath it.

## Recommended Patch Ladder

### P0: Make Current NCCL Path Observable And Bounded

Goal: get reliable evidence from the next canary and prevent misleading
success/failure modes.

Changes:

1. Add per-peer structured admin metrics for:
   - pause start/end/status;
   - update_lora start/end/status;
   - resume start/end/status;
   - whether `NCCL_READY` was released;
   - peer URL/hostname;
   - immediate unsmoothed running/waiting/KV values around pause/update.
2. Keep the invariant:
   - before `NCCL_READY`, admin failures are recoverable and resume may be safe;
   - after `NCCL_READY`, `/update_lora` failure or read timeout means no normal
     resume; inference must restart/reinitialize.
3. Add a configurable LoRA pause policy:
   - current: `wait, clear_cache=true`;
   - canary option: `keep, clear_cache=false`;
   - optional future: bounded wait then abort, only if NIXL/KV-connector cleanup
     is proven safe for the launch shape.

Tests/checks:

- Unit-test that post-`NCCL_READY` failure skips normal resume.
- Unit-test admin fanout attribution.
- Cluster canary with one LoRA update under realistic queue pressure.

### P1: Add Versioned Adapter Semantics

Goal: stop mutating the active adapter identity.

Changes:

1. Name adapters by policy step, for example:

   ```text
   prime_lora_step_000123
   ```

2. Use unique `lora_int_id` per live adapter version, not a permanent id.
3. Update `policy.model_name` only after every inference peer has registered
   the new adapter version.
4. Preserve existing rollout metadata:
   - `policy_version_at_start`;
   - `off_policy_steps`;
   - group/routing cache salt.
5. Add explicit cleanup:
   - keep adapters in `[current - L, current]`;
   - unload old adapters only when no in-flight group can still reference them;
   - fail loud if vLLM evicts an adapter still within the allowed lag window.

Tests/checks:

- Unit-test that new groups use the new adapter name after policy update.
- Unit-test that old groups retain the adapter/model name they were compiled
  with.
- Unit-test adapter retirement respects `max_off_policy_steps`.

### P2: Keep NCCL As The Efficient Transport, But Register New Adapter Versions

Goal: combine Prime-RL's efficient transfer path with versioned semantics.

Changes:

1. Extend the LoRA NCCL header to carry:
   - adapter version;
   - adapter name;
   - unique `lora_int_id`;
   - expected previous/current policy version;
   - chunk count and dtype/shape metadata.
2. Receiver commits the new adapter id/name without removing the previous live
   adapter unless capacity requires and retirement rules allow it.
3. Server registers the new `LoRARequest` under the versioned name.
4. Orchestrator switches `policy.model_name` to the versioned name only after
   all peers acknowledge registration.

Tests/checks:

- Single-node fake or smoke: two sequential NCCL LoRA updates produce two
  separately routable adapter names.
- Route request to old adapter and new adapter after update.
- Kill/fail one peer before `NCCL_READY`: no marker is released.
- Kill/fail one peer after `NCCL_READY`: run is marked restart-required.

### P3: Decide Whether Filesystem Adapter Load Is A Fallback Or A Main Path

Goal: avoid assuming either NCCL or filesystem is always best.

Decision criteria:

- If adapter size is small and update cadence is low, filesystem load may be
  simpler and robust enough.
- If adapter size is about 20 GB and update cadence is frequent, naive
  shared-filesystem fanout is probably too expensive.
- If we use filesystem loading, make it non-naive:
  - bounded fanout;
  - node-local staging/cache;
  - content-addressed adapter artifact;
  - load from local NVMe/tmpfs when possible;
  - verify all peers before switching routing.

Existing code already has bounded-concurrency filesystem loading in
`load_lora_adapter`, so this can remain a fallback or canary path.

### P4: Revisit vLLM Native Weight-Transfer Lifecycle

Goal: use upstream lifecycle where it actually fits.

Current observation:

- vLLM 0.22 exposes weight-transfer lifecycle methods such as
  `init_weight_transfer_engine`, `start_weight_update`, `update_weights`, and
  `finish_weight_update`.
- Native dynamic LoRA loading is path/load-oriented and supports `load_inplace`.
- Native full-weight transfer does not automatically solve Prime-RL's resident
  LoRA adapter registration and versioning needs.

Plan:

1. Keep current custom LoRA receive until versioned NCCL LoRA is proven or
   replaced.
2. Spike a custom LoRA `WeightTransferEngine` only if it reduces code surface
   rather than adding another abstraction layer.
3. Treat a vLLM version bump as a compatibility project, not as the first fix.

## What Not To Do

- Do not require `waiting_requests == 0` after pause.
- Do not retry a started NCCL LoRA receive after HTTP read timeout.
- Do not resume normally after a post-`NCCL_READY` collective failure.
- Do not silently fall back from NCCL to filesystem without an explicit config.
- Do not call versioned adapter loading "solved" until adapter load time and
  cache pressure are measured for 20 GB adapters.

## Success Criteria

Short canary:

- One realistic Qwen3.5-A3B LoRA update completes under active rollout load.
- W&B/logs show per-peer pause/update/resume phases.
- No post-marker retry occurs.
- Trainer observes `NCCL_READY` only after all peers are armed.

Versioned canary:

- `adapter_step_N` and `adapter_step_N+1` are both routable during the lag
  window.
- New groups use the new adapter name.
- Old groups complete or are dropped according to existing off-policy policy.
- Adapter retirement does not evict a still-referenced version.

Performance target:

- Sync time is dominated by payload transfer/commit, not unbounded pause/drain
  or shared-filesystem fanout.
- Update cost scales with `A`, not `R * A`, for large adapters.
- Failures are attributable to a peer and phase.

## Bottom Line

The elegant fix is not "use SkyRL's filesystem adapter load exactly." The
elegant fix is to separate adapter version semantics from adapter transport.

For these runs, the likely best design is:

```text
versioned LoRA adapters
+ bounded staleness
+ NCCL transfer for large adapter payloads
+ explicit no-retry post-NCCL_READY failure semantics
+ measurable filesystem-load fallback
```
