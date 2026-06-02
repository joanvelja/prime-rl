# Parity probes — train↔inference forward divergence

Short RL probes that check whether a model's **trainer-forward** logprobs match its
**vLLM-inference** logprobs of the same sampled tokens (`mismatch_kl`). Run one when you
port/integrate a new model, then classify the result with `scripts/parity_classify.py`.

**Read the [`parity-check` skill](../../skills/parity-check/SKILL.md) first** — it covers the
methodology (why the sampling must be truncation-free), the workflow, and how to read the
CLEAN / DISCRETE-BUG-suspect / FORWARD-DIVERGENCE / ARTIFACT verdict.

## Templates here

Proven tier (dense, single 1-train + 1-infer node):

| config | temp | notes |
|---|---|---|
| `parity_qwen3_4b_{ft,lora}.toml` | 0.7 | dense clean control — use as the `--reference` |
| `parity_olmo3_7b_{ft,lora}.toml` | 0.6 | dense clean control |
| `parity_gemma4_31b_{ft,lora}.toml` | 1.0 | gemma4 renderer, sdpa, `language_model_only`; FT uses `optim_cpu_offload` |

For MoE / VL / large models add `[inference.parallel].tp`, `language_model_only`,
`optim_cpu_offload` or `ep`, per the skill.

## Cluster-specifics

The `[slurm]` (`account`, `partition`) and `[deployment]` blocks are **Isambard examples** —
edit them for your cluster. To run in-allocation instead, drop `[slurm]` (see the
[`in-alloc-launch` skill](../../skills/in-alloc-launch/SKILL.md)).
