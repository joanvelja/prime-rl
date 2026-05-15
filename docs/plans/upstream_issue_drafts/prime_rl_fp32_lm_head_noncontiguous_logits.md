# Draft PrimeRL Issue: fp32 lm-head path can return non-contiguous logits to vLLM sampler

## Title

fp32 lm-head path can expose non-contiguous logits to vLLM `processed_logprobs` sampler

## Summary

The fp32 lm-head path appears to produce or expose fp32 logits with a padded physical row stride. In our live workload, vLLM's native Triton top-k/top-p sampler received logits with:

```json
{
  "shape": [192, 100278],
  "stride": [100288, 1],
  "is_contiguous": false
}
```

That layout is legal PyTorch, but it is unsafe for vLLM's current native Triton top-k/top-p wrapper because the Triton kernel indexes rows as `row_id * VOCAB_SIZE`, not `row_id * stride0`.

The vLLM-side fix should be to enforce/handle layout at the sampler boundary. But PrimeRL should also guard its fp32 lm-head path if that path slices a padded vocabulary dimension and hands the resulting non-contiguous view to vLLM.

This caused frequent NaN-400 retries in `processed_logprobs` training/eval serving. It is not just a logging annoyance: the retry path is currently load-bearing for correctness because otherwise degenerate processed logprobs can silently corrupt rollouts.

## Why PrimeRL Is Involved

The fp32 lm-head path computes logits in fp32, gathers/slices them to the original vocabulary size, and returns them to vLLM's logits/sampling path. If the underlying projection produces a padded vocabulary width, slicing to `org_vocab_size` can produce a non-contiguous view:

```python
logits = torch.mm(flat_hidden, lm_head.weight.t(), out_dtype=torch.float32)
...
logits = logits[..., :org_vocab_size]
```

If the padded physical vocabulary is `100288` and `org_vocab_size` is `100278`, the resulting logical tensor can have:

```text
shape  = [batch, 100278]
stride = [100288, 1]
```

That is exactly the layout captured in the failing live calls.

## Observed Failure

Live sampling with:

- vLLM V1
- `logprobs_mode = "processed_logprobs"`
- bf16 model
- PrimeRL fp32 lm-head enabled
- chunked prefill
- high concurrency / multi-replica serving
- `top_p = 0.95`

produced frequent JSON 400s:

```text
ValueError: Out of range float values are not JSON compliant: nan
```

Instrumentation showed:

- input logits to top-p/top-k were finite fp32
- input logits were non-contiguous with stride `(100288, 1)`
- vLLM's Triton kernel used logical vocab size `100278` for row pointer arithmetic
- some rows became all `-inf` after top-p/top-k
- `processed_logprobs` then computed `log_softmax(all -inf)` and emitted NaNs

Representative failing metadata:

```json
{
  "pre_shape": [32, 100278],
  "pre_bad_finite": [100278],
  "pre_bad_nan": [false],
  "post_finite": [0],
  "post_all_neginf": [true],
  "p_bad": [0.949999988079071],
  "pre_dtype": "torch.float32",
  "post_dtype": "torch.float32",
  "original_stride": [100288, 1],
  "original_is_contiguous": false,
  "vocab_size": 100278
}
```

## Controls

Default fp32 lm-head + native Triton path reproduced:

```json
{
  "Out of range float": 479,
  "ValueError": 240,
  "BadRequest": 478
}
```

bf16/non-fp32 lm-head did not reproduce over one comparable canary. That does not prove bf16 is a safe mitigation, but it does implicate the fp32 lm-head path as the source/exposer of the non-contiguous layout in this workload.

Forcing PyTorch top-p/top-k filtering completed cleanly, which kept `processed_logprobs` semantics but bypassed the unsafe Triton layout assumption.

Forcing contiguity before vLLM's native Triton top-p/top-k completed repeated canaries cleanly:

```json
{
  "Out of range float": 0,
  "ValueError": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "Orchestrator finished": 1,
  "RL trainer finished": 1
}
```

Observer logs confirmed the mitigation did real work:

```text
force_contiguous=True
call=1 dtype=torch.float32 shape=(192, 100278)
FORCE_CONTIGUOUS shape=(192, 100278) stride=(100288, 1)
...
empty_from_nonempty=0 out_nan_rows=0
```

## Suggested PrimeRL Fix

Short-term:

- Keep the existing retry-on-400 path. It is protecting correctness.
- Add an env-gated mitigation around the sampler boundary:

```python
if force_contiguous and not logits.is_contiguous():
    logits = logits.contiguous()
```

This has now passed repeated live canaries and does not alter logits values.

Production fix options:

1. Make the fp32 lm-head path return contiguous logits after slicing:

```python
logits = logits[..., :org_vocab_size]
if not logits.is_contiguous():
    logits = logits.contiguous()
```

2. Keep the guard closer to the sampler boundary, because other logits processors could also create non-contiguous views.

3. Once vLLM has a layout-safe `apply_top_k_top_p_triton`, remove the local workaround or leave it as a cheap defensive assertion/guard.

The safest PrimeRL-local patch is probably option 2 while the upstream vLLM issue is open, because it protects any source of non-contiguous logits, not just this fp32 lm-head implementation.

## Suggested Validation

Use a live serving canary with `processed_logprobs`, top-p, fp32 lm-head, and deployment-like concurrency:

```text
logprobs_mode=processed_logprobs
top_p=0.95
fp32_lm_head=true
chunked_prefill=true
max_num_seqs≈production
max_num_batched_tokens≈production
```

Collect:

- count of JSON `Out of range float` errors
- count of NaN/Inf rows in returned processed logprobs
- count of finite pre-top-p rows that become all `-inf`
- input logits shape/stride/contiguity at the top-p/top-k boundary
- count of times the contiguous guard fires

Expected after mitigation:

```json
{
  "Out of range float": 0,
  "BadRequest": 0,
  "EMPTY_AFTER_TOPKP": 0,
  "OUTPUT_NAN": 0,
  "force_contiguous_markers": "> 0 if fp32 lm-head emits sliced views"
}
```

## Non-Claims

- This does not mean fp32 lm-head numerics are invalid. The captured fp32 logits were finite.
- This does not mean raw logprobs are a valid workaround. In our training use case, `raw_logprobs` changed the policy semantics and broke training metrics.
- This does not mean clamping NaNs is acceptable. Clamping hides corrupted rollouts. The layout needs to be fixed before top-p/top-k.
- This may also need a vLLM upstream fix, because vLLM's sampler should not assume contiguity without asserting/enforcing it.
