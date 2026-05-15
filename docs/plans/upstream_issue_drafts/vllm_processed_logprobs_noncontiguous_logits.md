# Draft vLLM Issue: `processed_logprobs` Triton top-k/top-p path assumes contiguous logits

## Title

`processed_logprobs` native Triton sampler can produce NaN logprobs when logits are non-contiguous

## Summary

We are seeing intermittent NaN logprobs from the V1 `processed_logprobs` path under a high-throughput live workload. The proximate failure is:

1. `TopKTopPSampler.forward_native` receives finite fp32 logits with shape `[batch, vocab]`.
2. The logits tensor is non-contiguous, with stride `(padded_vocab, 1)` where `padded_vocab > vocab`.
3. `apply_top_k_top_p_triton` launches `_topk_topp_kernel`.
4. `_topk_topp_kernel` indexes each row as `LOGITS + row_id * VOCAB_SIZE`.
5. That row pointer arithmetic is only correct for contiguous logits.
6. In live debug captures, the kernel reads row statistics from the wrong physical row and can mask one logical row to all `-inf`.
7. `processed_logprobs` then runs `logits.log_softmax(dim=-1, dtype=torch.float32)` on an all-`-inf` row, producing NaN logprobs.
8. JSON rendering fails with `ValueError: Out of range float values are not JSON compliant: nan`.

This looks like a layout precondition bug, not a probability-distribution edge case. Calling `logits.contiguous()` before the Triton top-k/top-p kernel eliminated the issue in repeated live canaries without changing logits values or sampling semantics.

## Environment

- vLLM: `0.20.1`
- Engine: V1
- Platform: CUDA
- dtype: bf16 model, fp32 logits
- `logprobs_mode`: `processed_logprobs`
- Sampling: `top_p=0.95`, no top-k
- `max_model_len`: 16k
- `max_num_seqs`: 192
- `max_num_batched_tokens`: 65536
- Chunked prefill: enabled
- Async scheduling: enabled
- Multi-replica/high-throughput serving workload

I also have a deterministic single-process repro for the kernel layout issue. It checks both top-k and top-p while avoiding any model dependency.

## Relevant Code Path

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

The Triton kernel computes row pointers as if logits are contiguous:

```python
LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
```

But the wrapper only checks rank/dtype. It does not enforce contiguity or pass `stride(0)` into the kernel.

The FlashInfer path already appears to know about this class of issue: its code/comment says FlashInfer sampling expects contiguous logits and fp32 inference can make logits non-contiguous due to slicing in the logits processor, so it calls `logits.contiguous()`. The `processed_logprobs` path cannot take the FlashInfer branch because it needs post-filter logits/logprobs, and the native path does not currently make logits contiguous.

## Evidence

### 0. Minimal deterministic repro

This script does not require a model. It constructs legal non-contiguous logits views with shape `[batch, vocab]` and stride `(vocab + pad, 1)`, then compares the Triton filter on those views against the same logical values after `.contiguous()`. It checks both:

- `top_k=1`, where exactly one token should remain finite per row.
- `top_p=0.95`, where contiguous and non-contiguous rows should keep the same token set.

```text
docs/plans/upstream_issue_drafts/repro_vllm_noncontiguous_topk_topp.py
```

Run:

```bash
python repro_vllm_noncontiguous_topk_topp.py
```

Observed on vLLM `0.20.1`:

```text
[top-k]
input shape: (16, 1024)
input stride: (1032, 1)
contiguous Triton finite counts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
non-contiguous Triton finite counts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 121]
mismatched rows: [15]
row 15 contiguous finite token ids: [378]
row 15 non-contig finite token ids: [378, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934]

[top-p]
input shape: (16, 1024)
input stride: (1032, 1)
contiguous Triton finite counts: [154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154]
non-contiguous Triton finite counts: [154, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
mismatched rows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
FAIL: non-contiguous Triton output differs from contiguous Triton. The kernel is treating row stride as vocab size.
```

This is not the whole production NaN repro. It is the smaller kernel-contract repro: the native Triton wrapper accepts a non-contiguous logits tensor, but the kernel indexes rows as if the stride is `vocab_size`.

### 1. Finite input row becomes all `-inf` after top-p/top-k

Captured inside `TopKTopPSampler.forward_native`, immediately before and after `apply_top_k_top_p`:

```json
{
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

So the model/logits processor did not emit NaNs. The all-`-inf` row appears after top-p/top-k filtering.

### 2. Kernel row-debug captures show wrong-row statistics

Temporary instrumentation recorded per-row kernel statistics and the original logical row statistics. Bad examples:

```text
bad row 13: logical pre-row max 23.97648, kernel max 34.80531
            previous logical row max 34.80531

bad row 14: logical pre-row max 23.21280, kernel max 35.03107
            previous logical row max 35.03107

bad row 23: logical pre-row max 21.14634, kernel max 34.78697
            previous logical row max 34.78697
```

This is consistent with the kernel reading the wrong physical offset for a logical row.

### 3. The bad live tensors are non-contiguous

The same captures recorded:

```json
{
  "original_stride": [100288, 1],
  "original_is_contiguous": false,
  "runtime_stride": [100288, 1],
  "runtime_is_contiguous": false,
  "vocab_size": 100278
}
```

The kernel used `row_id * 100278`, but the real row stride was `100288`, a 10-float stride mismatch.

### 4. Control canaries

Default native Triton path reproduced quickly:

```json
{
  "Out of range float": 479,
  "ValueError": 240,
  "BadRequest": 478
}
```

Replacing only the top-k/top-p filter with the PyTorch reference path completed the same live workload cleanly:

```json
{
  "Out of range float": 0,
  "ValueError": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0
}
```

Forcing `logits.contiguous()` before the Triton kernel completed repeated live workload runs cleanly:

```json
{
  "Out of range float": 0,
  "ValueError": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "workload_completed": true
}
```

The force-contiguous observer confirmed the workload was actually converting non-contiguous logits:

```text
force_contiguous=True
call=1 dtype=torch.float32 shape=(192, 100278)
FORCE_CONTIGUOUS shape=(192, 100278) stride=(100288, 1)
...
empty_from_nonempty=0 out_nan_rows=0
```

## What I Think Is Happening

The native Triton top-k/top-p kernel assumes contiguous row layout but the wrapper accepts non-contiguous logits. When `processed_logprobs` is active, vLLM cannot use the FlashInfer branch, so the native path sees logits that the FlashInfer path would have made contiguous.

This mismatch can cause top-p filtering to operate on/read/write the wrong physical row offsets, occasionally producing all-`-inf` logical rows. `log_softmax(all -inf)` then produces NaNs.

## Suggested Fix

Conservative wrapper fix:

```python
def apply_top_k_top_p_triton(logits, k, p):
    ...
    if not logits.is_contiguous():
        logits = logits.contiguous()
    ...
```

This is semantically clean for the observed caller because `forward_native` rebinds:

```python
logits = apply_top_k_top_p(logits, k, p)
```

Alternative fix:

- Assert contiguity at the wrapper boundary and fail loudly, or
- Pass `logits.stride(0)` into `_topk_topp_kernel` and compute row pointers as `LOGITS + row_id * stride0`.

The stride-aware kernel would avoid copies, but the wrapper-level contiguous fix is smaller and matches the FlashInfer path's existing defensive behavior.

## Notes / Non-Claims

- I am not claiming the top-p pivot logic itself is wrong.
- I am not claiming model logits contain NaN/Inf; our captures show finite fp32 input rows.
- I am not claiming the minimal script reproduces the exact production top-p NaN by itself. It reproduces the lower-level row-stride bug that explains the production captures.
- I am not claiming all fp32-logits users will hit this. The trigger seems to require non-contiguous logits plus the native Triton `processed_logprobs` path under load.
