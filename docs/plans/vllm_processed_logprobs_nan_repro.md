# vLLM processed_logprobs NaN Repro Status

Last updated: 2026-05-14 19:30 UTC

## Short Version

We have a strong live-workload repro that localizes the failure to vLLM's Triton
top-k/top-p filtering path used by `processed_logprobs`, and a high-confidence
mechanism: non-contiguous logits are reaching a Triton kernel that indexes rows
as if the tensor were contiguous.

This is no longer "some mysterious top-p numerical edge". The live kernel debug
evidence points at tensor layout. The kernel does `LOGITS_ROW = LOGITS + row_id
* VOCAB_SIZE`; that is only valid for contiguous `[batch, vocab]` logits. A
same-file vLLM comment in the FlashInfer branch already says fp32 inference can
produce non-contiguous logits due to slicing in `logits_processor`, and that
branch calls `logits.contiguous()`. The native Triton branch did not.

Operationally, this bug is high impact: default live runs produce frequent
NaN-400 retries, while replacing only the top-k/top-p filter with vLLM's PyTorch
reference implementation completed a canary step cleanly.

## Current Claim

Precise claim we can defend:

> In vLLM 0.20.1, under the live async/multi-replica `processed_logprobs`
> workload, `TopKTopPSampler.forward_native` sometimes passes non-contiguous
> fp32 logits into `apply_top_k_top_p_triton`. That Triton kernel indexes rows
> with contiguous row-stride arithmetic (`row_id * VOCAB_SIZE`). Live debug
> captures show impossible row-stat mismatches consistent with reading one
> logical row using another row's physical memory. The resulting post-top-p
> logits can contain an all-`-inf` row; `logits.log_softmax(dim=-1,
> dtype=torch.float32)` then produces NaN logprobs, JSON rendering fails, and
> the orchestrator retry path absorbs the 400.

Do **not** overclaim:

- Not proven to be a specific Triton source line.
- Not proven to be shared scratch-buffer reuse.
- Not proven to be model-logit NaNs.
- Not proven to be deterministic on the captured logits.
- Not proven fixed by any upstream vLLM release.
- Not proven to require fp32 lm head in all workloads. Current evidence says
  fp32 lm head exposes the bad layout/path in this workload; it does not say the
  model is numerically invalid.

## Environment

- Repo: `$PRIME_RL_REPO`
- Allocation used: `4593289`
- vLLM: `0.20.1`
- Model: `allenai/Olmo-3-7B-Instruct-DPO`
- Inference mode:
  - `logprobs_mode = "processed_logprobs"`
  - bf16 model
  - fp32 lm head patch enabled
  - chunked prefill enabled
  - `max_num_seqs = 192`
  - `max_num_batched_tokens = 65536`
  - `performance_mode = "throughput"`
  - async scheduling enabled by vLLM
- Topology:
  - 7 inference nodes
  - 4 local DP workers per inference node
  - vLLM router in front of each replica
  - 1 trainer node

## Code Paths

Relevant vLLM path:

```text
vllm.v1.sample.sampler.Sampler.forward
  -> self.sample(...)
  -> self.topk_topp_sampler(logits, generators, k, p)
  -> vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native
  -> apply_top_k_top_p(logits, k, p)
       if HAS_TRITON and batch_size >= 8:
           apply_top_k_top_p_triton(logits, k, p)
       else:
           apply_top_k_top_p_pytorch(logits, k, p)
  -> logits.log_softmax(dim=-1, dtype=torch.float32)   # processed_logprobs
```

Important vLLM source facts:

- `processed_logprobs` structurally excludes the FlashInfer sampler path.
- `apply_top_k_top_p_triton` mutates the logits tensor in place and returns it.
- `_topk_topp_kernel` computes row pointers as `LOGITS + row_id * VOCAB_SIZE`.
  There is no stride argument.
- `apply_top_k_top_p_triton` asserts rank and dtype, but did not assert or
  enforce contiguity before local instrumentation.
- `TopKTopPSampler.forward_cuda` has a vLLM comment saying FlashInfer sampling
  expects contiguous logits and that fp32 inference can make logits
  non-contiguous because of a slicing operation in `logits_processor`; it calls
  `flashinfer_sample(logits.contiguous(), ...)`. The `processed_logprobs`
  native path cannot use FlashInfer and did not make logits contiguous.
- The PyTorch reference implementation also mutates in place, but preserves at
  least one token under top-p (`top_p_mask[:, -1] = False` after sorting).

## Local Instrumentation Added

Files:

- `src/prime_rl/inference/patches.py`
- `scripts/debug/repro_vllm_topk_topp_capture.py`

Env flags:

- `PRIME_TTP_CAPTURE_EMPTY=1`
  - Keep a pre-mask clone for empty-row diagnostics.
- `PRIME_TTP_CAPTURE_FULL_ROW=1`
  - Save a `.pt` payload when a finite pre-mask row becomes all `-inf`.
- `PRIME_TTP_CAPTURE_DIR=/path/to/dir`
  - Capture output directory.
- `PRIME_TTP_CAPTURE_LIMIT=1`
  - Limit saved captures per process.
- `PRIME_TTP_FORCE_PYTORCH=1`
  - Diagnostic only. Inside `TopKTopPSampler.forward_native`, call
    `apply_top_k_top_p_pytorch` instead of `apply_top_k_top_p`. This preserves
    processed-logprobs semantics better than `raw_logprobs`; it is not a final
    performance fix.
- `PRIME_TTP_FORCE_CONTIGUOUS=1`
  - PrimeRL-side env-gated mitigation. Inside the patched
    `TopKTopPSampler.forward_native`, call `logits.contiguous()` before
    `apply_top_k_top_p` when the input logits are non-contiguous. This preserves
    the logits values and changes only layout before vLLM's Triton kernel.
- `VLLM_TOPK_TOPP_DEBUG_CAPTURE=1`
  - Temporary local vLLM-site-package patch. Captures Triton per-row debug
    scalars when a finite pre-row becomes all `-inf`.
- `VLLM_TOPK_TOPP_FORCE_CONTIGUOUS=1`
  - Temporary local vLLM-site-package patch. Calls `logits.contiguous()` inside
    `apply_top_k_top_p_triton` if the input logits are non-contiguous. This is
    a layout precondition fix, not a distribution-altering clamp.

## Captured Bad Row

Capture file:

```text
tmp/canary/ttp-captures-4593289-rerun/ttp_empty_row_pid282731_call364_n0.pt
```

Payload summary:

```json
{
  "bad_idx": [14],
  "pre_shape": [32, 100278],
  "post_shape": [1, 100278],
  "pre_bad_finite": [100278],
  "pre_bad_nan": [false],
  "pre_bad_neginf": [0],
  "post_finite": [0],
  "post_all_neginf": [true],
  "post_nan": [false],
  "post_posinf": [false],
  "post_neginf_count": [100278],
  "k": null,
  "p_bad": [0.949999988079071],
  "pre_dtype": "torch.float32",
  "post_dtype": "torch.float32"
}
```

Interpretation:

- The row entering top-p was fully finite fp32.
- The row after top-p was truly all `-inf`.
- The NaN comes later from `log_softmax(all -inf)`, not from upstream logits.

## Replay Results

Replay command shape:

```bash
srun --overlap --jobid=4593289 --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --cpus-per-task=8 --mem=32G --exact bash -lc '
    cd $PRIME_RL_REPO &&
    VLLM_PLUGINS= uv run --no-sync python \
      scripts/debug/repro_vllm_topk_topp_capture.py \
      tmp/canary/ttp-captures-4593289-rerun/ttp_empty_row_pid282731_call364_n0.pt
  '
```

Result:

- Triton replay finite counts matched PyTorch replay exactly.
- Bad row finite count under replay:
  - Triton: `1`
  - PyTorch: `1`
- No all-`-inf` rows in standalone replay.

Interpretation:

- This is not a deterministic pure function of the saved logits batch and
  `p=0.95`.
- The failure requires some live-context condition not reproduced by the simple
  replay harness.

## Stress Harness Results

The replay script has concurrent-stream stress options. Be careful interpreting
them.

An early stress run appeared to reproduce empty rows, but that was a harness
bug: tensors were prepared on the default stream and consumed on side streams
without a readiness barrier. After adding `torch.cuda.synchronize()` before
side-stream kernel launch, both cached-buffer and uncached-buffer variants were
clean over 2,000 launches.

Current result:

- `--stress-streams 32 --stress-iters 2000`: zero failures.
- `--stress-streams 32 --stress-iters 2000 --stress-uncached-buffer`: zero
  failures.

Interpretation:

- Shared scratch-buffer reuse is still plausible but not validated.
- Local multi-stream replay is not enough to reproduce the live failure.

## Live Canaries

### Default Triton path

Default canary with capture enabled:

```bash
mkdir -p tmp/canary/ttp-captures-4593289-rerun
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-rerun \
PRIME_TTP_CAPTURE_LIMIT=1 \
bash outputs/canary-E8-empty-row-detail/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E8-empty-row-detail-rerun.log
```

Observed:

- Fresh NaN-400s occurred.
- Empty-row capture was produced.
- Capture confirmed finite pre-row -> all-`-inf` post-row.

### Forced PyTorch top-p/top-k filter

Diagnostic canary:

```bash
mkdir -p tmp/canary/ttp-captures-4593289-force-pytorch
PRIME_TTP_FORCE_PYTORCH=1 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-force-pytorch \
PRIME_TTP_CAPTURE_LIMIT=1 \
bash outputs/canary-E8-empty-row-detail/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E8-force-pytorch.log
```

Observed:

- Completed 1 training step.
- 749 successful `POST /v1/chat/completions` responses in the log.
- No `EMPTY_AFTER_TOPKP`.
- No `OUTPUT_NAN`.
- No `Out of range float`.
- No `BadRequest` / `ModelError` strings.
- No capture files in `tmp/canary/ttp-captures-4593289-force-pytorch`.
- Step summary:
  - `Step 0 | Time: 100.91s | Reward: 0.4507 | Seq. Length: 1835.2 tokens/sample`

Interpretation:

- Replacing the Triton top-p/top-k filter with the PyTorch reference path is a
  strong live discriminator.
- This points at the Triton top-p/top-k implementation path, not at
  `processed_logprobs` semantics by themselves.

### Default Triton path with `enforce_eager=true`

Diagnostic canary:

```bash
mkdir -p tmp/canary/ttp-captures-4593289-enforce-eager
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-enforce-eager \
PRIME_TTP_CAPTURE_LIMIT=1 \
bash outputs/canary-E9-enforce-eager/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E9-enforce-eager.log
```

Observed:

- vLLM confirmed `enforce_eager=True`.
- vLLM confirmed CUDAGraph/torch.compile disabled.
- Reproduced quickly after 93 successful `POST /v1/chat/completions` responses.
- Fresh capture:
  `tmp/canary/ttp-captures-4593289-enforce-eager/ttp_empty_row_pid184716_call202_n0.pt`
- Capture summary:

```json
{
  "bad_idx": [24],
  "pre_shape": [32, 100278],
  "post_shape": [1, 100278],
  "pre_bad_finite": [100278],
  "pre_bad_nan": [false],
  "pre_bad_neginf": [0],
  "post_finite": [0],
  "post_all_neginf": [true],
  "post_neginf_count": [100278],
  "p_bad": [0.949999988079071],
  "pre_dtype": "torch.float32",
  "post_dtype": "torch.float32"
}
```

Interpretation:

- The failure does not require CUDAGraph capture or torch.compile.
- The same finite pre-row -> all-`-inf` post-row signature survives in eager
  mode.

### Default Triton path with bf16 lm head

Diagnostic canary:

```bash
mkdir -p tmp/canary/ttp-captures-4593289-bf16-lm-head
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-bf16-lm-head \
PRIME_TTP_CAPTURE_LIMIT=1 \
bash outputs/canary-E10-bf16-lm-head/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E10-bf16-lm-head.log
```

Only intended inference config delta from the default E8 canary:

```toml
enable_fp32_lm_head = false
enforce_eager = false
```

Observed:

- Completed 1 training step.
- 734 successful `POST /v1/chat/completions` responses in the log.
- No `EMPTY_AFTER_TOPKP`.
- No `OUTPUT_NAN`.
- No `Out of range float`.
- No `ValueError`.
- No `BadRequest` / `ModelError` strings.
- No capture files in `tmp/canary/ttp-captures-4593289-bf16-lm-head`.
- Rollout generation completed `256/256`.
- Step summary:
  - `Step 0 | Time: 101.44s | Reward: 0.4817 | Seq. Length: 1922.8 tokens/sample`
- One unrelated-looking `Timeout during comparison` appeared during rollout
  comparison, but the step completed and did not coincide with NaN-400 or empty
  top-p evidence.

Interpretation:

- The non-fp32/bf16 lm-head path did not reproduce the sampler empty-row bug in
  this one-step canary.
- This does **not** prove bf16 lm-head is a safe mitigation. It says the fp32
  lm-head path is now a live suspect because changing only that knob eliminated
  the observed symptom over one in-distribution canary step.
- Plausible mechanism, still unproven: fp32 lm-head changes logit sharpness or
  tie/duplicate structure enough to expose a rare top-p Triton edge case.

### E11: fp32 lm head with fp32-vs-bf16 compare

Diagnostic canary:

```bash
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-fp32-compare \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E11-fp32-lm-head-compare/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E11-fp32-lm-head-compare.log
```

Observed:

- Reproduced heavily:
  - `Out of range float`: 306
- Capture:
  `tmp/canary/ttp-captures-4593289-fp32-compare/ttp_empty_row_pid190036_call243_n0.pt`
- Capture summary:
  - bad row: 24
  - pre-row finite count: `100278/100278`
  - post-row finite count: `0`
  - post-row all `-inf`
  - `p = 0.95`
  - `k = None`
  - PyTorch reference support for the bad row: `1`
  - top-1 logit: `26.34285`
  - top-2 logit: `5.32741`
  - top-1/top-2 gap: `21.01544`
- Standalone replay of the saved capture with
  `scripts/debug/repro_vllm_topk_topp_capture.py` kept one finite token under
  Triton and under PyTorch.
- Node-local fp32-vs-bf16 compare logs found no non-finite logits in either
  path. Both paths produced many support-1 rows.

Interpretation:

- fp32 lm head does not appear to be emitting NaN/Inf logits.
- Support-1/peaky rows alone are insufficient to reproduce the bug.
- The live failure still requires a live tensor/runtime condition not preserved
  by the saved contiguous CPU clone.

### E12: CUDA launch blocking

Diagnostic canary:

```bash
CUDA_LAUNCH_BLOCKING=1 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-E12-launch-blocking \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E12-fp32-triton-launch-blocking/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E12-fp32-triton-launch-blocking.log
```

Observed before cancellation:

- `Out of range float`: 124
- `ValueError`: 62
- `BadRequest`: 124
- Captures:
  - `tmp/canary/ttp-captures-4593289-E12-launch-blocking/ttp_empty_row_pid125678_call147_n0.pt`
    - PyTorch reference support: 473
    - softmax max probability: 0.157
    - post-row all `-inf`
  - `tmp/canary/ttp-captures-4593289-E12-launch-blocking/ttp_empty_row_pid192568_call216_n1.pt`
    - PyTorch reference support: 3
    - softmax max probability: 0.768
    - post-row all `-inf`

Interpretation:

- The failure is not just async launch visibility.
- The failure is not only the extreme support-1 / peaky-distribution edge case.

### E13: Triton kernel row-debug capture

Temporary vLLM patch:

- Added a debug output tensor to `_topk_topp_kernel`.
- For each row, stored:
  - pivot before guard
  - final pivot
  - max logit
  - min logit
  - duplicate logit
  - duplicate count
  - requested keep count
  - kept duplicate count
- Saved payloads when a finite pre-row became all `-inf`.

Diagnostic canary:

```bash
VLLM_TOPK_TOPP_DEBUG_CAPTURE=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/topk-topp-debug-4593289-E13 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_LIMIT=4 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-E13-kernel-debug \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E13-topk-topp-kernel-debug/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E13-topk-topp-kernel-debug.log
```

Observed:

- Reproduced quickly.
- Debug captures:
  - `tmp/canary/topk-topp-debug-4593289-E13/topk_topp_debug_pid195063_n0.pt`
  - `tmp/canary/topk-topp-debug-4593289-E13/topk_topp_debug_pid195063_n1.pt`
  - `tmp/canary/topk-topp-debug-4593289-E13/topk_topp_debug_pid128441_n2.pt`
  - `tmp/canary/topk-topp-debug-4593289-E13/topk_topp_debug_pid218762_n3.pt`

Key debug evidence:

```text
pid195063_n0: bad row 27, pre row max 22.00065, kernel max 29.69087
             29.69087 equals pre row 26 max.
pid195063_n1: bad row 27, pre row max 18.83059, kernel max 34.43157
             34.43157 equals pre row 26 max.
pid128441_n2: bad row 15, pre row max 22.02559, kernel max 34.97572
             34.97572 equals pre row 14 max.
pid218762_n3: bad row 6, pre row max 6.8408, kernel max 13.6518
             not a clean adjacent-row match, but still impossible for row 6.
```

Interpretation:

- The kernel's own row stats can disagree with the saved logical row.
- In three captures, the kernel max for the bad row exactly matched the
  preceding logical row's max.
- This is the first evidence that directly implicates layout/stride, not just
  top-p pivot math.
- The debug wrapper cloned `pre_logits` before saving, so the saved payload has
  contiguous stride and loses the original live stride. That explains why
  standalone replay on the saved tensor stays clean.

### E14: Force contiguous before Triton

Temporary vLLM patch inside `apply_top_k_top_p_triton`:

```python
original_stride = tuple(logits.stride())
original_is_contiguous = logits.is_contiguous()
if (
    os.environ.get("VLLM_TOPK_TOPP_FORCE_CONTIGUOUS", "0") == "1"
    and not original_is_contiguous
):
    logits = logits.contiguous()
```

Diagnostic canary:

```bash
VLLM_TOPK_TOPP_FORCE_CONTIGUOUS=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/topk-topp-debug-4593289-E14-contiguous \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_LIMIT=4 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-E14-contiguous \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E14-topk-topp-contiguous/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E14-topk-topp-contiguous.log
```

Observed:

- Completed one full training step in the same deployment-shaped canary.
- Orchestrator finished.
- Trainer finished.
- No failure signatures:

```json
{
  "Out of range float": 0,
  "ValueError": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "SAVED_EMPTY_ROW_CAPTURE": 0
}
```

- No debug or empty-row capture dirs were created:
  - `tmp/canary/topk-topp-debug-4593289-E14-contiguous`
  - `tmp/canary/ttp-captures-4593289-E14-contiguous`
- Step summaries:
  - Orchestrator: `Step 0 | Time: 98.02s | Reward: 0.4353 | Seq. Length: 2069.9 tokens/sample`
  - Trainer: `Step 0 | Time: 289.52s | Loss: -0.0036 | Entropy: 1.0338 | Mismatch KL: 0.3523`
- After both orchestrator and trainer finished, the Slurm step was cancelled to
  stop lingering inference health checks. The allocation remained alive.

Interpretation:

- This is the strongest causal discriminator so far.
- Same processed-logprobs semantics, same fp32 lm-head path, same Triton
  sampler path, same multi-replica live workload. The only intended change was
  enforcing the layout precondition before the Triton kernel.
- This does not "hide" bad probabilities. It prevents the Triton kernel from
  reading/writing the wrong physical rows.

### E15: Default Triton with stride metadata

Purpose:

- Confirm the final gap: whether failing live calls are non-contiguous.
- Same as E13-style default fp32/Triton debug, but after adding metadata capture
  for original/runtime stride and contiguity.

Diagnostic canary:

```bash
VLLM_TOPK_TOPP_DEBUG_CAPTURE=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/topk-topp-debug-4593289-E15-stride \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_LIMIT=4 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-E15-stride \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E15-topk-topp-stride-debug/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E15-topk-topp-stride-debug.log
```

Observed before cancellation:

```json
{
  "Out of range float": 479,
  "ValueError": 240,
  "BadRequest": 478,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "SAVED_EMPTY_ROW_CAPTURE": 0
}
```

Debug captures:

```text
tmp/canary/topk-topp-debug-4593289-E15-stride/topk_topp_debug_pid112630_n3.pt
tmp/canary/topk-topp-debug-4593289-E15-stride/topk_topp_debug_pid133534_n1.pt
tmp/canary/topk-topp-debug-4593289-E15-stride/topk_topp_debug_pid200342_n0.pt
tmp/canary/topk-topp-debug-4593289-E15-stride/topk_topp_debug_pid223830_n2.pt
```

Every debug capture had:

```json
{
  "original_stride": [100288, 1],
  "original_is_contiguous": false,
  "runtime_stride": [100288, 1],
  "runtime_is_contiguous": false
}
```

The vocabulary size passed to the Triton kernel was `100278`, so the kernel's
row pointer arithmetic used `row_id * 100278` on tensors whose real row stride
was `100288`. That is a 10-float row-stride error.

Bad-row examples:

```text
pid112630_n3: bad row 13, pre row max 23.97648, kernel max 34.80531
             previous row max 34.80531
pid200342_n0: bad row 14, pre row max 23.21280, kernel max 35.03107
             previous row max 35.03107
pid223830_n2: bad row 23, pre row max 21.14634, kernel max 34.78697
             previous row max 34.78697
pid133534_n1: bad row 10, pre row max 10.97761, kernel max 17.55555
             also impossible for the logical bad row
```

Interpretation:

- This validates the layout-root-cause hypothesis directly.
- The live bad tensors are not contiguous.
- The Triton kernel does not receive or use stride information.
- For stride `(100288, 1)` and vocab `100278`, row `i` starts `10*i` floats
  earlier than it should under the kernel's pointer math. That is exactly the
  kind of cross-row contamination seen in the debug max values.
- E14's `logits.contiguous()` fix removes this stride mismatch and completed
  cleanly.

### E16: PrimeRL env-gated contiguous mitigation

Purpose:

- Validate the durable PrimeRL patch path, not the temporary local
  site-package `VLLM_TOPK_TOPP_FORCE_CONTIGUOUS` patch.
- Same deployment-shaped fp32-lm-head / processed-logprobs canary.
- `VLLM_TOPK_TOPP_FORCE_CONTIGUOUS` was deliberately **not** set.

Diagnostic canary:

```bash
PRIME_TTP_FORCE_CONTIGUOUS=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/topk-topp-debug-4593289-E16-prime-contiguous \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_LIMIT=4 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-E16-prime-contiguous \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E16-prime-force-contiguous/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E16-prime-force-contiguous.log
```

Observed:

- Completed one full training step.
- Orchestrator finished.
- Trainer finished.
- No failure signatures in the combined log:

```json
{
  "Out of range float": 0,
  "ValueError": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "SAVED_EMPTY_ROW_CAPTURE": 0,
  "Orchestrator finished": 1,
  "RL trainer finished": 1
}
```

- No debug or empty-row capture dirs were created:
  - `tmp/canary/topk-topp-debug-4593289-E16-prime-contiguous`
  - `tmp/canary/ttp-captures-4593289-E16-prime-contiguous`
- The `FORCE_CONTIGUOUS` breadcrumbs are in per-process observer logs, not the
  combined tee. Across E16 observer logs:

```json
{
  "files": 14,
  "force_contiguous_markers": 32,
  "empty_after": 0,
  "output_nan": 0,
  "out_of_range": 0,
  "max_logged_calls_per_proc": 3100,
  "max_logged_rows_per_proc": 85062
}
```

- Example observer evidence:

```text
/tmp/processed_logprobs_observer_69888.log:
[ttp.forward_native] config ... force_contiguous=True
[ttp.forward_native] call=1 dtype=torch.float32 shape=(192, 100278)
[ttp.forward_native] FORCE_CONTIGUOUS shape=(192, 100278) stride=(100288, 1)
...
[ttp.forward_native] calls=3100 rows=83421 ... empty_from_nonempty=0 out_nan_rows=0
```

- Step summaries:
  - Orchestrator: `Step 0 | Time: 76.94s | Reward: 0.4363 | Seq. Length: 1845.1 tokens/sample`
  - Trainer: `Step 0 | Time: 236.58s | Loss: -0.0110 | Entropy: 0.9307 | Mismatch KL: 0.3212`
- After both orchestrator and trainer finished, Slurm step `4593289.183` was
  cancelled to stop lingering inference health checks. The allocation remained
  alive.

Interpretation:

- This validates the repo-side env-gated mitigation path.
- The patch actually saw non-contiguous fp32 logits with stride `(100288, 1)`
  and converted them before vLLM's Triton top-k/top-p kernel.
- Over this one-step canary, the conversion path saw no empty rows and no
  processed-logprob NaNs.
- This is not a probability clamp and does not alter the intended sampling
  distribution. It enforces the row-layout precondition that the Triton kernel
  already assumes.

### E19: PrimeRL contiguous repeat canary

Purpose:

- Repeat E16 once with a fresh output directory to reduce the chance that E16
  was a lucky clean one-step canary.
- Same env-gated PrimeRL mitigation: `PRIME_TTP_FORCE_CONTIGUOUS=1`.
- Same deployment-shaped fp32-lm-head / processed-logprobs config.

Diagnostic canary:

```bash
PRIME_TTP_FORCE_CONTIGUOUS=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE=1 \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/topk-topp-debug-4593289-E19-prime-contiguous \
VLLM_TOPK_TOPP_DEBUG_CAPTURE_LIMIT=4 \
PRIME_TTP_CAPTURE_EMPTY=1 \
PRIME_TTP_CAPTURE_FULL_ROW=1 \
PRIME_TTP_CAPTURE_DIR=$PRIME_RL_REPO/tmp/canary/ttp-captures-4593289-E19-prime-contiguous \
PRIME_TTP_CAPTURE_LIMIT=2 \
bash outputs/canary-E19-prime-force-contiguous-repeat/multi_node_rl.sh \
  2>&1 | tee tmp/canary/canary-E19-prime-force-contiguous-repeat.log
```

Observed:

```json
{
  "Out of range float": 0,
  "ValueError": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "SAVED_EMPTY_ROW_CAPTURE": 0,
  "Orchestrator finished": 1,
  "RL trainer finished": 1,
  "POST /v1/chat/completions": 732
}
```

- No debug or empty-row capture dirs were created:
  - `tmp/canary/topk-topp-debug-4593289-E19-prime-contiguous`
  - `tmp/canary/ttp-captures-4593289-E19-prime-contiguous`
- Across E19 observer logs:

```json
{
  "files": 14,
  "force_contiguous_markers": 32,
  "empty_after": 0,
  "output_nan": 0,
  "max_logged_calls_per_proc": 5200,
  "max_logged_rows_per_proc": 145365
}
```

- Step summaries:
  - Orchestrator: `Step 0 | Time: 100.69s | Reward: 0.4271 | Seq. Length: 1893.2 tokens/sample`
  - Trainer: `Step 0 | Time: 250.36s | Loss: -0.0089 | Entropy: 1.0142 | Mismatch KL: 0.3538`
- After both orchestrator and trainer finished, Slurm step `4593289.187` was
  cancelled to stop lingering inference health checks. The allocation remained
  alive.

Interpretation:

- E19 repeats E16's conclusion: the repo-side contiguous mitigation exercised
  non-contiguous logits and prevented the observed empty-row / NaN path over a
  full one-step canary.
- Current live evidence is now: default Triton path reproduces; PyTorch
  reference path passes; site-package contiguous path passes; PrimeRL
  contiguous path passes twice.

## Things Falsified So Far

- Upstream model logits are NaN before sampling: falsified by capture.
- The bad row starts as all `-inf`: falsified by capture.
- `raw_logprobs` is a viable training mitigation: falsified earlier by KL blowup.
- Exact captured logits deterministically empty under Triton: falsified by
  replay.
- Simple local concurrent-stream stress reproduces after proper barriers:
  falsified by current stress harness.
- `p` temporary dtype conversion lifetime bug for this capture: unlikely; saved
  `p` was already `torch.float32`, and `k` was `None`.
- The failure requires CUDAGraph or torch.compile: falsified by the
  `enforce_eager=true` canary.
- The failure is guaranteed under bf16/non-fp32 lm head: falsified by the E10
  one-step canary. This is not the same as proving bf16 lm head cures it.
- "This is just top-p deleting the whole support because the distribution is
  too peaky": falsified by E12 captures with PyTorch reference support 473 and
  3.
- "fp32 lm head emits invalid logits": not supported by E11 compare logs;
  logits were finite in fp32 and bf16 paths.
- "Making logits contiguous changes the sampling distribution": wrong framing.
  It changes layout, not values. The kernel already assumes contiguous row
  layout.

## Remaining Plausible Causes

Ranked current guesses:

1. `apply_top_k_top_p_triton` is being called with non-contiguous logits, but
   `_topk_topp_kernel` indexes rows as `LOGITS + row_id * VOCAB_SIZE` and has no
   stride arguments. Confidence: 98%.
2. The E14 contiguous canary was a lucky clean run despite not addressing the
   root cause. Confidence: 1%.
3. The local instrumentation perturbed execution enough to make E14 clean for a
   different reason. Confidence: 0.5%.
4. A separate Triton pivot/duplicate bug exists and is masked by contiguity in
   this one-step canary. Confidence: 0.4%.
5. Something else in vLLM mutates logits between the pre clone and post
   observation. Confidence: 0.1%.

## Next Isolation Steps

Run these in the existing allocation; do not request a new allocation if one is
available.

1. Repeat E16/E19-style canaries a few more times if we need a tighter upper
   bound
   - Same config, same `PRIME_TTP_FORCE_CONTIGUOUS=1`.
   - We now have two clean one-step repeats on this path. More repeats are only
     for confidence/throughput quantification, not for basic localization.

2. Prepare a minimal upstream patch
   - Conservative version:
     `if not logits.is_contiguous(): logits = logits.contiguous()` in
     `apply_top_k_top_p_triton`.
   - Better version:
     either assert contiguity loudly at the wrapper boundary or pass row stride
     into the kernel and use `LOGITS + row_id * stride0`.
   - Because the function returns the masked logits for `processed_logits` /
     `processed_logprobs`, a clone-on-non-contiguous path must be checked for
     semantic compatibility. For our observed processed-logprobs path, returning
     the contiguous masked tensor is fine because the caller rebinds `logits =
     apply_top_k_top_p(...)`.

3. Measure throughput cost
   - If most calls are already contiguous, the cost should be near zero.
   - If fp32 lm-head path routinely returns non-contiguous logits, the cost is
     one `[batch, vocab]` fp32 copy per sampler call. Still likely cheaper than
     NaN retries and data corruption risk, but measure it.

4. Check current upstream vLLM
   - Checked vLLM latest generated docs on 2026-05-14. The published
     `apply_top_k_top_p_triton` wrapper still asserts only rank and dtype before
     launching `_topk_topp_kernel`; it does not enforce contiguity or pass row
     stride.
   - The latest documented `TopKTopPSampler.forward_cuda` path still has the
     comment that FlashInfer sampling expects contiguous logits and that fp32
     inference can make logits non-contiguous due to `logits_processor` slicing.
     It calls `logits.contiguous()` there, but `processed_logprobs` takes the
     native path.
   - If this docs snapshot matches the current main branch, upgrade alone is
     unlikely to fix it. File issue/PR with E15 evidence.

## Upstream Escalation Position

Do not file upstream as:

> Triton pivot line X is wrong.

That is probably false now.

Reasonable upstream issue framing:

> vLLM 0.20.1 `processed_logprobs` live workload intermittently gets all-`-inf`
> rows from `apply_top_k_top_p_triton` despite finite fp32 input logits. Kernel
> debug captures show row-stat mismatches consistent with non-contiguous logits
> being read via contiguous row arithmetic. `TopKTopPSampler.forward_cuda`
> already makes logits contiguous for FlashInfer because fp32 inference can
> produce non-contiguous logits, but `processed_logprobs` excludes FlashInfer
> and the native Triton path does not enforce contiguity. Forcing
> `logits.contiguous()` before the Triton kernel eliminated the failure in a
> deployment-shaped one-step canary.

Attach:

- Environment/config above.
- The capture payload summary.
- Default Triton canary log summary.
- Forced PyTorch canary log summary.
- E13 debug row-mismatch summary.
- E14 force-contiguous canary summary.
- E15 stride metadata showing `(100288, 1)` non-contiguous logits versus
  `VOCAB_SIZE=100278`.
- E16 PrimeRL env-gated contiguous canary summary.
- E19 PrimeRL env-gated contiguous repeat summary.
- Statement that standalone replay is clean.

## Operational Guidance

For training correctness right now:

- Keep the orchestrator retry-on-400 path intact.
- Do not clamp NaNs or replace all-`-inf` rows with finite sentinels and call it
  fixed; that can silently corrupt rollouts.
- `raw_logprobs` avoids the symptom but breaks training semantics.
- `PRIME_TTP_FORCE_PYTORCH=1` is the best current diagnostic mitigation to test
  correctness-preserving behavior, but throughput cost needs measurement before
  using it as a production workaround.
- The likely production-quality fix is a layout fix at the vLLM Triton wrapper
  boundary, not a probability clamp. For our path, `logits.contiguous()` before
  the kernel is semantically clean; the remaining work is direct stride logging,
  repeat canaries, and throughput measurement.
