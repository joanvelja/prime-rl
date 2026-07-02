---
name: parity-check
description: Verify train<->inference forward parity when porting/integrating a new model into prime-rl. Runs a short alphabet-sort RL probe with token_export, then classifies the per-token mismatch_kl into CLEAN / DISCRETE-BUG-suspect / FORWARD-DIVERGENCE / ARTIFACT. Use after adding a new model arch (or attn/MoE/reload change) and before trusting it for real RL.
---

# Parity check — train↔inference forward divergence

prime-rl async RL only works if the **trainer-forward** logprobs match the **vLLM-inference** logprobs of the *same* sampled tokens. The per-token gap is `mismatch_kl` (a k3 KL). A new model port can pass smoke tests, look fine at step 0, and then silently corrupt RL a few steps in — the trainer assigns ~`e^-33` to tokens vLLM sampled at p≈0.98 (an argmax flip), so the policy gradient quietly drops the tokens inference was most confident about.

This skill runs a short, controlled RL probe and classifies the result, so you catch that divergence *before* it eats a real run.

> **Worked example — why this exists.** gemma-4-26B-A4B looked clean at steps 0–2, then `mismatch_kl` jumped ~200× at step 3. The signature *looked* like an irreducible bf16 kernel gap. It was a **discrete, fixable vLLM bug**: the Gemma4 router's `hidden_size**-0.5` constant is a non-persistent buffer that warm layerwise reload clobbered to `1.0` (a ~53× router-logit multiplier), so vLLM sampled from a corrupted policy. The fix was preserving one buffer on reload. **A `FORWARD-DIVERGENCE` verdict is a "go localize", not a "give up".**

## When to use

- You added/ported a model architecture (new `modeling_*.py`, MoE/router, attention, norms).
- You changed the weight-sync / vLLM reload path, an attention backend, or expert dispatch.
- You're deciding whether a model is safe to use for an experiment (e.g. before committing GPU-weeks).

## Methodology (load-bearing — don't change these blindly)

`mismatch_kl` compares trainer-forward vs vLLM-inference logprobs of the same tokens. The trainer replicates **temperature only** (the loss applies `completion_temperatures`; there is **no** top_p/top_k/penalty truncation). So the probe must be **truncation-free**, or you measure a sampling-config artifact instead of a real forward gap:

- Set each model's **recommended temperature** (trainer replicates it → fair comparison).
- Keep `top_p = 1.0`, `top_k = -1` (off), `min_p = 0`, no penalties — env defaults.
- `lr = 1e-6`: enough to sharpen the policy over ~12 steps and surface a gap that only appears once logits concentrate (the gemma probe blew up at step 3 with exactly this LR).
- `max_steps = 12`, `[trainer.experimental.token_export]` on, `alphabet-sort` env.
- Run **both regimes**: FullFT and LoRA rank-32. LoRA adds adapter-application (vLLM merge/scale vs trainer) as an extra divergence class — label the arms distinctly. Run `preflight-lora-smoke` before the first LoRA GPU run.

Recommended temperatures (raw HF cards): qwen3-4b 0.7 · gemma-4 1.0 · olmo3 0.6 · qwen3.5 0.7 instruct / 1.0 thinking · gpt-oss 1.0.

## Workflow

Templates live in `examples/parity/` (`parity_<model>_{ft,lora}.toml`). The `[slurm]` / `account` / `deployment` blocks in them are Isambard examples — adjust for your cluster (or drop `[slurm]` to run in-allocation; see `in-alloc-launch`).

**1. Make a config** for the new model — copy the closest template, set `[model].name`, the renderer, and `[orchestrator.train.sampling].temperature`. For MoE/VL/large models also set `[inference.parallel].tp`, `language_model_only`, and `optim_cpu_offload` / `ep` as needed.

**2. Validate (required — pydantic + merge gate):**
```bash
uv run rl @ examples/parity/parity_<model>_ft.toml --dry-run --no-wandb
# inspect outputs/parity/<run>/configs/{trainer,orchestrator,inference}.toml:
#   trainer.toml: lora present (LoRA) or absent (FullFT); model.ac.freq=1
#   orchestrator.toml: train.env alphabet-sort; train.sampling.temperature set
```

**3. Submit** (the `[slurm]` block makes `rl` self-submit a fresh job; omit it to run in-allocation):
```bash
uv run rl @ examples/parity/parity_<model>_ft.toml
```

**4. Classify** against a known-CLEAN reference (qwen3-4b / olmo3-7b are clean controls):
```bash
uv run python scripts/parity_classify.py \
  --exports outputs/parity/<model>-ft \
  --reference outputs/parity/olmo3-7b-ft \
  --out outputs/parity/<model>-ft/report.md
```
`--exports` accepts a run root or a `token_exports/` dir. Stdlib-only; no GPU. Every CONFIG threshold is a `--<knob>` CLI flag.

## Interpreting the verdict

| verdict | meaning | what to do |
|---|---|---|
| **CLEAN** | step-0 faithful, bounded growth, ~no catastrophe tail | ship it |
| **DISCRETE-BUG-suspect** | step-0 mean already > 0.1 — the static port disagrees before any RL drift | audit the forward statically (HF-reference parity test, layer-by-layer) — it's a port bug, not drift |
| **FORWARD-DIVERGENCE** | clean start, then tail-shaped growth + catastrophe tail (median stays low) | **LOCALIZE — see below.** Do NOT conclude "irreducible bf16 gap" without the protocol; this is often a fixable discrete bug |
| **ARTIFACT** | the *median* itself is high — bulk of tokens mismatched, not a tail | a broken capture/replay measurement, not a real parity gap — fix the export/replay path, not the model |

### Localizing a FORWARD-DIVERGENCE (the protocol that cracked gemma)

The signature alone can't tell "fixable discrete bug" from "irreducible kernel gap" — you have to localize. In rough cost order:

1. **Rule out the obvious discrete causes by A/B** — flip one suspect at a time (expert kernel, attention backend, weight-broadcast transport) and re-probe. Each is one config knob.
2. **Cold-load vs warm-reload A/B** — load the same checkpoint into a *fresh* vLLM (cold) and compare to the live update-route (warm) on the catastrophe sequence. If cold agrees with the trainer but warm doesn't → it's the **reload path** (the gemma root_size case: warm reload clobbered a non-persistent buffer).
3. **Suspect-tensor restore** — after a warm update, restore a suspected buffer/param in memory and re-score. If divergence collapses to the cold baseline, you've found it (gemma: restoring `root_size` → exactly cold).
4. **Per-layer hidden-state parity** — on one catastrophe sequence, dump per-decoder-layer hidden states from trainer-forward and vLLM-forward on identical weights; the first layer that diverges localizes it. (See the `mismatch_kl` debugging history for the harness shape.)

Only after this protocol comes up empty is "irreducible bf16 train↔inference kernel gap" a defensible conclusion — and even then it's usually *manageable* (lower LR / tighter async / matched kernels), not fatal.

## References

`references/` has real reports from this tool, one per verdict:
- `example_clean_olmo3-7b-ft.md` — CLEAN (flat `mismatch_kl` ~6e-4, 0 catastrophe tokens).
- `example_forward-divergence_gemma4-26b.md` — FORWARD-DIVERGENCE (clean → step-3 catastrophe jump; the root_size bug).
- `example_artifact_replay-on.md` — ARTIFACT (elevated median from a broken router-replay capture).

Thresholds are calibrated on these three labeled runs (`scripts/parity_classify.py` `CONFIG`).
