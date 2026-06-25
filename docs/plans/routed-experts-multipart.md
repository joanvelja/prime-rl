# routed_experts multipart transport (Issue #76)

Status: PLAN — not implemented, not committed. Assess complete; awaiting Joan's go.
Branch: `feat/routed-experts-multipart` (off `main` @ `14edc66de`, has the shipped mallopt OOM fix).
Pins: `verifiers@0d253754`, `renderers@2d8825e` — both ends pinned ⇒ single atomic cutover, no negotiation.

---

## 1. Problem recap

Today every rollout's bulk `routed_experts` tensor rides **inside** the control-plane msgpack body: vLLM base64-encodes it into the HTTP/JSON response (1.33× inflation), the renderer detaches it as a `memoryview` over the *full* HTTP body (pinning the entire 2.3–3.5 MB body until msgpack serialization), the env-worker re-packs it as a msgpack `bin` field on the shared 512-thread executor, and the orchestrator base64-decodes it back. This is a **throughput / peak-memory** problem, **not** the host-OOM (the `mallopt(M_MMAP_THRESHOLD,1MiB)` fix at `14edc66de` owns the OOM — that fix addresses glibc per-thread-arena fragmentation, a different mechanism). The cost here: −33% wasted wire bytes on the dominant payload (worst on the cross-node `tcp://` remote-env config), a ~512×(body + msgpack-blob) double-allocation peak during `packb`, GIL contention on multi-MB `packb` across 512 threads, and an HTTP-body pin. The goal: get the bulk tensor off the msgpack body onto raw uint8 ZMQ multipart frames — one decoder, one format — freeing peak-memory headroom (→ raise `max_num_seqs`) and cutting cross-node bytes. Honest prior per MEMORY.md decode profile (comm 35% + sampling 30%, model 7%): serialization is **off the wall-clock critical path**, so the E2E throughput delta is ≈0 on loopback; the real wins are peak memory and cross-node bandwidth.

---

## 2. Chosen design

**Base = Design C ("perf+safety max")** — the only design that closes the peak-memory loop end-to-end (renderer copy-out kills the body pin + byte-budget semaphore bounds the `O(workers×inflight)` materialization the mallopt fix doesn't touch). **Grafts** (per judge): A's function-level edit list as the PR1 checklist; A's renderer copy-out edit verbatim; B's explicit `encoding` field + `nbytes` self-consistency guard + frame-count-mismatch-raises. **Rejected** (YAGNI): B's sentinel-hole / `AttachmentSet` / path-agnostic-walk generality — one tensor type, both ends pinned, greenfield; targeted "walk the `routed_experts` sites" is the right altitude.

### Wire format — response leg only

Inbound rollout requests carry **no** `routed_experts` (`RunRolloutRequest`/`RunGroupRequest` have no expert field), so K attachment frames appear **only** on the worker→server→client response path (PUSH→PULL→ROUTER→DEALER). The request/dispatch leg stays exactly 3 frames. This halves the plumbing surface vs. the symmetric sketch in the maps.

ZMQ multipart, **response leg**, today 3 frames → now `3 + K`:

```
Frame 0: request_id            (worker→router: client_id, request_id; router→client: client_id stripped)
Frame 1: msgpack_control       full RunRolloutResponse/RunGroupResponse, each routed_experts site has
                               NO "data" key — carries {shape, dtype, start, frame, encoding}
Frame 2 .. 2+K-1: raw uint8 bytes, one frame per expert blob, ordered by `frame` ordinal
```

K is **variable** — the load-bearing fact the maps' "K=1" framing got wrong. `RunGroupResponse.outputs: list[CoercedRolloutOutput]` (`serve/types.py:80-81`) means a group reply nests **G outputs × Sᵢ steps**, each step optionally carrying experts. g16 debate ⇒ up to ~16+ blobs per message. ∴ the descriptor MUST carry an explicit per-blob ordinal, not a fixed slot.

### Descriptor schema (per `routed_experts` site, inside `msgpack_control`)

```python
{
    "frame":    int,    # 0-based ordinal → raw bytes live in frames[2 + frame]
    "shape":    [int, int, int],   # [step_seq_len, layers, topk]; step_seq_len may be 0
    "dtype":    "uint8",           # carried explicitly → parameterizes np.frombuffer (no hardcode)
    "start":    int,               # token offset; 0 first turn, prefix_len-1 for extensions — VERBATIM
    "encoding": "raw",             # explicit (b64-text and raw-bytes both land as msgpack bin) — fail-loud if != "raw"
}
```

- **Ordering key** = `frame` ordinal (explicit), NOT msgpack traversal position. Detach assigns `frame` in a single deterministic walk; hydrate looks up `frames[2 + frame]` by ordinal — so hydrate never re-derives order, it reads the index. This survives `model_dump`/`omit_defaults` round-trips and is robust to same-shape collisions in g16 debate (the silent-shear nightmare, Attack 7).
- **Encoding explicit** because msgpack `bin` is encoding-blind. Single `"raw"` value; assert-fail on anything else — fail-loud, no dual-decode.
- **`start` rides verbatim**; all downstream replay (`start == prefix_len-1` assert, boundary row-0 split, `align_routed_experts` dim-0 truncation) is byte-identical to today.

### Atomic cutover

One coordinated commit across `verifiers` + `renderers` + `src/prime_rl`. No version flag, no `if isinstance(data, str)` fallback, no dual decode. A mismatched pair fails loud (`np.frombuffer` rejects, `encoding != "raw"` raises). The b64 encode (`routed_experts.py:24`) and decode (`trajectories.py:253`) cease to exist in the same change set — exactly one wire format.

### Split into two landable PRs

- **PR1 (the bulk of the win, no vLLM change):** ZMQ raw frames + dedicated serializer executor + byte-budget semaphore + finite HWMs + renderer **copy-out** (decodes b64→raw once at the renderer, breaking the pin at its source). b64 stays on the *HTTP* hop and is decoded exactly once during the renderer copy-out, before bytes ever reach the worker walk. Delivers ~all the peak-mem + cross-node-byte win.
- **PR2 (isolated, gated on a vLLM response-format change):** HTTP binary trailer — drops the last b64 (`routed_experts.py:24`) and the HTTP-body pin entirely. Higher risk (touches vLLM serving handler + httpx parse); land after PR1 is measured.

PR1 and PR2 are independent: PR1's hydrator already consumes raw bytes; PR2 only changes *where* b64→raw happens.

---

## 3. File-by-file change list

Legend: **[V]** = `deps/verifiers`, **[R]** = `deps/renderers`, **[P]** = `src/prime_rl`. All paths under `/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-routed-experts/`.

### PR1

| # | File | Anchor (file:line, verified) | Change |
|---|---|---|---|
| 1 | **[R]** `deps/renderers/renderers/client.py` | `124-141` `strip_routed_experts_data` / `parse_generate_response` | Replace the pin `routed_data = memoryview(raw)[data_start:data_end]` (`:131`) with copy-out + decode: `routed_raw = pybase64.b64decode_as_bytearray(bytes(raw[data_start:data_end]))`. `raw` (full HTTP body) frees immediately after `stripped` is built. From here `payload[...]["routed_experts"]["data"]` is **raw uint8 bytes**, not b64/memoryview. (Graft A§3.B verbatim — the precise edit that breaks the pin AND converts b64→raw once.) |
| 2 | **[V]** `deps/verifiers/verifiers/serve/server/env_worker.py` | `226-248` serialize+send; `406-417` executor setup; `111-114` HWM | (a) Before packb, walk `response.model_dump(...)` → pop each `routed_experts["data"]` (raw bytes) into ordered `attachments: list[bytes]`, stamp `{"frame": i, "encoding": "raw", "dtype": ...}`, drop `"data"`. (b) `packb` the lightened body on a **dedicated serializer executor** (`run_in_executor(self._ser_executor, ...)`, 16 threads). (c) `send_multipart([client_id, request_id.encode(), control_bytes, *attachments])`. (d) Wrap (a)→(c) in the **byte-budget semaphore** acquire/`try-finally`-release. Error path (`:189-191`) stays 3-frame, zero attachments. |
| 3 | **[V]** `deps/verifiers/verifiers/serve/server/env_router.py` | `259-263` response recv (verified `if len(frames) != 3: continue`) | `!= 3` → `< 3`; unpack `client_id, request_id, response_bytes = frames[0], frames[1], frames[2]`; keep tail `frames[3:]`; `complete_request(request_id)` unchanged (keys on `frames[1]`); `await on_response(client_id, request_id, response_bytes, frames[3:])`. |
| 4 | **[V]** `deps/verifiers/verifiers/serve/server/zmq_env_server.py` | `43-50` `send_response` (verified 3-arg) | Add `*attachments` param; `frontend.send_multipart([client_id, request_id, response_bytes, *attachments])`. **Do NOT** touch the request-path `!= 3` guard at `:82` (request leg, never carries experts). |
| 5 | **[V]** `deps/verifiers/verifiers/serve/client/zmq_env_client.py` | `195-216` receive loop (verified `if len(msg) < 2`, unpack `msg[0], msg[1]`) | Keep `< 2` guard; unpack `request_id_bytes, control_bytes = msg[0], msg[1]`, `attachments = msg[2:]`. After `msgpack.unpackb(control_bytes)`, **rebind**: walk dict, set each `routed_experts["data"] = attachments[desc["frame"]]` (raw bytes), assert `len(attachments[frame]) == prod(shape)*itemsize` and `encoding == "raw"`, before `future.set_result`. |
| 6 | **[P]** `src/prime_rl/orchestrator/trajectories.py` | `252-257` hydrate (verified `pybase64.b64decode_as_bytearray(...["data"])`) | Drop the b64 decode. `data` is already raw bytes (rebound at client). `routed_experts = np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])`; `routed_experts_start = payload["start"]`. **Only** orchestrator edit. Replay/align (`:394-407`, `:486-505`) untouched. |
| 7 | **[V]** `deps/verifiers/verifiers/utils/thread_utils.py` | `45` `register_executor`, `80` `scale_executors` | Used by #2 to register a `ThreadPoolExecutor(max_workers=16, "vf-ser")` in `env_worker.run()` alongside `scale_executors(512)`. No edit to thread_utils itself if `register_executor` suffices. |
| 8 | **[V]** HWMs | `env_worker.py:112` SNDHWM=0; `env_router.py:132` RCVHWM=0 | `0 → finite` (message-count, e.g. a few thousand) on the response leg only. `send_multipart` is already `await`ed on the async socket → finite HWM = backpressure to the rollout coroutine (intended). Keep request-leg + stats (`:117` SNDHWM=100) HWMs as-is. |
| 9 | **[V]** new helper module | `deps/verifiers/verifiers/serve/server/_routed_frames.py` (new) | `detach_routed_attachments(payload) -> (control, list[bytes])` and `reattach_routed_attachments(control, attachments)` — the **one shared traversal** for producer (env_worker) + consumer (zmq_env_client). The only new abstraction. |

### PR2 (after PR1 measured)

| # | File | Anchor | Change |
|---|---|---|---|
| 10 | **[P]** `src/prime_rl/inference/vllm/routed_experts.py` | `22-27` (verified `pybase64.b64encode(memoryview(compact))`) | Emit raw bytes as a length-prefixed binary trailer instead of b64-in-JSON. |
| 11 | **[P]** `src/prime_rl/inference/vllm/serving_tokens.py` | `71-85` response builder | Build the multipart/binary-trailer HTTP response. |
| 12 | **[R]** `deps/renderers/renderers/client.py` | `136-141` `parse_generate_response`, `280-281` post | Parse the binary trailer; the copy-out from #1 becomes a plain slice (no b64 decode). Removes `ROUTED_EXPERTS_DATA_PREFIX` + the HTTP-body pin entirely. |

**Untouched** (confirmed encoding-agnostic raw-bytes contract): trainer boundary `src/prime_rl/transport/types.py:19-23` (`RoutedExperts(data: bytes, shape, dtype)`), `trainer/batch.py`, `trainer/rl/data.py:241-252` (`torch.frombuffer(data, dtype)`). `save_utils.py` has zero routed_experts references — no on-disk format depends on the b64 dict shape.

---

## 4. Must-fix guardrails (from red-team) → enforcement site

| ID | Guardrail | Enforced where |
|---|---|---|
| **G1** | Truncation: the only hydrate-side length check is `len(frame) == prod(shape)·itemsize`. **NEVER** cross-check frame length against `completion_ids`/`prompt_ids`/token counts — `align_routed_experts` slices/pads dim-0 and a token-count check would reject every truncated rollout. | `zmq_env_client.py` rebind (#5) + `_routed_frames.py` (#9). `align_routed_experts` (`trajectories.py:30-49`) stays the SOLE dim-0 arbiter, unchanged. |
| **G2** | Cancel-safety: byte-budget acquire/release in `try/finally` spanning extract→pack→send, releasing on `CancelledError`. The executor `packb` thread can't be cancelled and finishes regardless — account its bytes until it returns. A cancel storm must not leak credits → deadlock. | `env_worker.py` (#2). Verified cancel path: `:212-216` sends 3-frame shielded error (no attachments); success-send `:245-248` stays unshielded (dropping a cancelled response is correct). |
| **G3** | Group fan-out: detach + hydrate share **one** traversal impl; descriptor carries explicit per-blob `frame` ordinal so hydrate looks up by index, not re-derived order. Highest-risk silent-shear path (rollout 3's experts → rollout 7) — no assert catches it if shapes collide. | `_routed_frames.py` (#9) is the single shared walk. Test **T1** (distinct *contents* per rollout) is the proof. |
| **G4** | Frame guards: flip **exactly ONE** guard — `env_router.py:259` `!= 3` → `< 3` + forward `frames[3:]`. Keep `env_worker.py:338`, `zmq_env_server.py:82` (request leg), and health `== 2` strict. Blanket-flipping loosens request-leg validation for no reason and masks framing bugs. | `env_router.py` (#3). Add a justification comment on the request-leg guards that they stay strict. |
| **G5** | Encoding explicit: descriptor carries `dtype` → parameterize `np.frombuffer` (drop hardcoded `np.uint8` at `trajectories.py:254`); carries `encoding="raw"` → assert-fail on anything else. | descriptor schema (§2); `trajectories.py` (#6); rebind assert in `zmq_env_client.py` (#5). |
| **G6** | Boundary assert: keep `trajectories.py:394` `assert routed_experts_start == prefix_len - 1` **verbatim** — the only tripwire that catches a cross-wired `start` (Attack 1↔6 interaction). | `trajectories.py:394` unchanged. |

Non-issues confirmed by red-team (no guard needed): intra-message frame reordering (ZMQ multipart is atomic, in-order or not-at-all — incl. `tcp://`); multi-worker ordinal collision (ordinal scoped per-message, no global counter — **do not introduce one**); hydration-before-compaction (rebind at client receive is strictly before `prepare_step_tokens` → `np.frombuffer` → prefix-match at `:424-484`; **do not** move `np.frombuffer` to lazy/finalize or you break `routed_prefix_states` slicing at `:329-355`).

---

## 5. Phased TODO + parallelization structure

Per Joan's planning gate: this is the plan; **execution enters only after explicit approval.**

### Phase 0 — de-risk (DONE + one residual)
- [x] Wire-mechanic spike (blind N-frame relay on real socket types, K=3, byte-identical) — `tmp/routed_experts_spike/spike.py`.
- [ ] **T4 residual:** rerun spike over `tcp://` (not `ipc://`) before cutover. *Sequential, blocks Phase 3 sign-off; can run anytime.*

### Phase 1 — shared contract (DEPENDENCY ROOT, sequential)
- [ ] **TASK-A:** write `_routed_frames.py` (#9) — `detach`/`reattach` with `frame` ordinal + `encoding`/`dtype`/`nbytes`. *Everything downstream depends on this signature.* Land with its unit test (T-RT) red-first.

### Phase 2 — wire plumbing (CONCURRENT after TASK-A)
Three independent subagents, each owns disjoint files, all consume TASK-A's contract:
- [ ] **TASK-B (worker):** `env_worker.py` (#2) — detach + serializer executor + byte-budget semaphore (G2) + HWM (#8). [V]
- [ ] **TASK-C (router+server):** `env_router.py` (#3, G4) + `zmq_env_server.py` (#4). [V]
- [ ] **TASK-D (client+orch):** `zmq_env_client.py` rebind (#5, G1+G5) + `trajectories.py` hydrate (#6, G5). [V]+[P]

B/C/D touch disjoint files → safe concurrent. Coordinate only via the `_routed_frames.py` + `send_response`/`on_response` signatures (frozen at end of Phase 1).

### Phase 3 — producer-side pin fix (CONCURRENT with Phase 2)
- [ ] **TASK-E:** `renderers/client.py` copy-out (#1, graft A§3.B). [R] Independent of B/C/D (different repo, different hop). Can run parallel to Phase 2.

### Phase 4 — gatekeeper (sequential, ADVERSARIAL)
- [ ] **TASK-G (gatekeeper):** adversarial review of B+C+D+E — hunt for: any token-count cross-check (G1 violation), missing `try/finally` on the semaphore (G2), divergent detach/hydrate walks (G3), over-broad guard flips (G4), hardcoded `uint8` survivors (G5), a global ordinal counter, lazy-decode moved past prefix-match. Route flags back to the owning task — lead does NOT fix directly. Run `/codex:adversarial-review` on the diff.

### Phase 5 — validation (sequential, gated on green gatekeeper)
- [ ] **TASK-H:** run test plan (§6) — all pytest green incl. T1/T2/T3.
- [ ] **TASK-I:** A/B on one debate step (§6) — current `main` vs branch, same seed/config.

### Phase 6 — PR2 (separate, after PR1 lands + measured)
- [ ] HTTP binary trailer (#10–12). Own PR, own gatekeeper pass.

**Parallel groups:** `{TASK-A}` → `{TASK-B ‖ TASK-C ‖ TASK-D ‖ TASK-E}` → `{TASK-G}` → `{TASK-H → TASK-I}`. T4 (tcp spike) floats, must complete before Phase 5 sign-off.

---

## 6. Test plan + A/B protocol

### Tests (plain pytest functions, fixtures not classes — per AGENTS.md)

New file: `deps/verifiers/tests/test_routed_frames.py`
- **T-RT `test_detach_reattach_roundtrip`** — single-rollout (K=1) RunRolloutResponse with one routed_experts blob; `reattach(detach(body))` byte-identical `data` per site; `frame` ordinals dense `0..K-1`. (Red-first against the new API, per prove-it.)
- **T1 `test_group_multi_sidecar_distinct_content`** — **highest value.** G=16 RunGroupResponse, each rollout's RE array filled with its **own index value** (distinct *contents*, not just shapes — shapes collide in real debate); round-trip; assert each hydrated array carries its own index → no cross-wiring (G3).
- **T-ENC `test_encoding_guard`** — `encoding != "raw"` raises (G5, fail-loud).
- **T-NB `test_nbytes_mismatch_raises`** — frame length ≠ `prod(shape)·itemsize` raises; a dropped/short frame raises (catches a `<3`-router-that-should've-been `>=3`).
- **T-EMPTY `test_zero_len_step`** — `shape[0]==0` → 0-byte frame round-trips (msgpack + ZMQ allow empty frames).

Extend existing (do not add new files): `src/prime_rl/orchestrator/` trajectories tests
- **T2 `test_truncation_align_byte_identical`** — rollout where `routed_experts.shape[0] != len(completion_ids)`, both over and under; assert hydrate accepts (G1) and `align_routed_experts` produces finalized bytes **byte-identical to the pre-cutover b64 golden**. The strongest correctness proof — same bytes reach the trainer contract.
- Existing `interleave_rollout` / boundary-row / `start==prefix_len-1` tests must stay green unchanged.

Env_worker (only if isolable without heavy mocking, else skip per AGENTS.md):
- **T3 `test_cancel_no_credit_leak`** — cancel storm during pack; assert byte-budget credits return to full (G2).

Run: `cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-routed-experts && unset VIRTUAL_ENV && uv run --no-sync pytest deps/verifiers/tests/test_routed_frames.py src -k "routed or trajector or interleav or zmq" -q`

### A/B measurement (honest, per CLAUDE.md evidence standard)

Cutover is atomic ⇒ A/B is **two builds** (baseline = `main`@`14edc66de`+mallopt; treatment = branch), not a runtime flag. One identical debate config (`gpqa_openended_debate_50step_bs512_g16`, r64/4t12i, max_num_seqs raised), same seed, ≥20 steps.

Metrics (report **all three** memory sources — don't credit an RSS drop the cgroup can't see; the mallopt lesson):
- **Memory:** env_worker process RSS (`/proc/<pid>/status:VmHWM`) **AND** cgroup `memory.current`/`memory.peak` **AND** node `MemAvailable` (`/proc/meminfo`). Plus the existing `mem/orch_children_gb` metric (target ≤0.5 GB/step).
- **Peak in-flight bytes:** instrument `env_worker` send → log `len(control_bytes) + sum(len(a) for a in attachments)` vs baseline `len(response_bytes)`. Expect control_bytes MB→KB; total −25% on the expert portion (spike-exact 1.333×→1.0×).
- **Pack/decode CPU:** time the `packb` to_thread (expect ~38× cheaper on the routed portion: spike 0.46ms→0.012ms/blob) + the dropped orchestrator b64-decode pass.
- **Latency:** rollout p50/p95 wall-time, step time.
- **Throughput:** tokens/s.

Expected verdict: worker peak ~3.6 GB → ≤0.6 GB; expert wire bytes −25% exact; **throughput Δ within ±2% noise on loopback** (decode is comm+sampling-bound — state this up front so the A/B isn't read as a throughput claim). The win is peak-memory headroom + cross-node bytes (measurable on the `tcp://` remote-env config), not wall-time.

---

## 7. Spike result + residual risk

**Ran:** `tmp/routed_experts_spike/spike.py` under `uv run --no-sync python`, production socket types (worker PUSH→server PULL, server ROUTER→client DEALER), realistic scale (4.7 MB single, 9.8 MB / 3 turns).

**De-risked (proven):**
1. Blind N-frame relay works on the real topology — `frames[N:]` forwarded verbatim, order + per-frame bytes preserved (sha256 end-to-end, K=3 frame-counts `[4,4,6]`). No socket needs envelope special-casing; ROUTER only touches the leading identity frame.
2. Control envelope is O(K) descriptors, O(1) in payload bytes (149 B @ K=1/4.7 MB, 265 B @ K=3/9.8 MB → ~27 B/payload-MB). msgpack body collapses MB→KB.
3. Raw uint8 round-trips byte-identical; `np.frombuffer+reshape` == wire frame.
4. **Refuted MAP-1's "K=1" framing** — exercised K=3 to confirm variable K (group = G×S blobs), forcing the explicit `frame` ordinal.

**Residual risk (NOT de-risked by the spike — must be honored by impl / tested):**
- **`tcp://` not exercised** (spike used `ipc://`). ZMQ multipart is transport-agnostic + both ends pinned, but **rerun over tcp before cutover (T4)** — and note finite-SNDHWM under tcp means a slow/dead remote consumer stalls the worker rollout loop (intended backpressure, but verify).
- **G3 group fan-out at G=16 distinct-content** verified only at the *type* level (`outputs: list`) + *mechanic* level (spike K=3 order) — **not** at G=16 distinct-content. T1 is the must-fix that closes this (the one place silent shear can't be ruled out by reading alone).
- **align/truncation/boundary-row/start-assert** — spike proved wire mechanic only, NOT replay alignment. T2 + G1 + G6 own these.

---

## 8. Open questions for Joan

1. **Byte-budget default.** Design C proposes 512 MiB/worker (≈512 inflight × 1 MB amortized, conservative vs the unbounded ~2.3–3.6 GB peak). Derive it from `max_inflight` × measured-amortized-blob, or hardcode-with-comment for PR1 and tune in the A/B? My lean: derive from config, fail-loud if the key's missing (no magic 512). $_{70\%}$
2. **Finite HWM value.** Response-leg SNDHWM/RCVHWM `0 → N` (message-count). N = `num_workers × max_inflight_per_worker × small_factor`? Or a flat few-thousand for PR1, tune later? Backpressure semantics change (slow consumer stalls rollout) — acceptable, or keep HWM=0 for PR1 and add HWM in a fast-follow to isolate the variable?
3. **Serializer executor size.** 16 threads (packb is GIL-bound, >16 buys nothing). Confirm, or want it config-exposed?
4. **PR1 vs PR1+PR2 scope for the next run.** PR1 alone leaves b64 on the HTTP hop (decoded once at renderer). If the next debate run is loopback-local, PR1 captures ~all the peak-mem win and PR2's incremental gain is just the HTTP-hop b64 + a transient body alloc. Ship PR1 solo first, or block on both?
5. **Does the next science run actually need this before launching, or is it a fast-follow?** The mallopt fix already owns the OOM. This is headroom (→ raise `max_num_seqs` further) — is that on the critical path for the queued 50-step runs (5175049-52), or land it post-hoc?
