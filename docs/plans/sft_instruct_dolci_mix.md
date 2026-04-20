# SFT Instruct Dolci Mix — Decision Record

> **Status**: Rungs 0-2 complete. System prompt infra built. Ready for Rung 3 (SFT config + smoke test).
> **Date**: 2026-04-14 (started) → 2026-04-16 (current)
> **Worktree**: `worktree-sft-instruct`

---

## 1. Purpose

Build a clean SFT-only instruct checkpoint from 4–6 vanilla base models, to serve as
a **shared starting point** for two parallel RL branches:

- **Experimental**: debate RL
- **Control**: RLVR

Both branches fork from the same SFT checkpoint so that downstream differences are
attributable to the RL training method, not to pretraining or post-training contamination.

### Hard constraints on the SFT baseline

1. **No RLHF / DPO** — sycophancy emerges primarily from preference-based training.
   A sycophantic model is a broken debater (confirmed from the live-rollout findings
   memory: `project_live_rollout_findings.md`).
2. **No explicit safety / refusal SFT data** — safety-training teaches "when to refuse"
   behavior which confounds debate rollouts. Strip `domain == "Safety"`.
3. **No RLVR** — we want headroom; signal/noise in debate-vs-RLVR comparison
   requires starting from a point before either converges.
4. **Format-permissive** — debate RL needs the model to produce multi-paragraph
   argumentative prose. The SFT prior should not be rigid (`Answer: X` terminators,
   `\boxed{}`, bare one-word answers) in ways that lock the response length distribution.

### Candidate base models — 7 locked (2026-04-15 after full 2026 landscape search)

After a 4-agent research swarm covering (a) HF API direct crawling, (b) 2026 community-consensus blog/leaderboard synthesis, (c) Chinese ecosystem deep dive, and (d) named-model investigation, plus an additional swarm on small-MoE landscape with 8×H100 memory-footprint analysis, the final locked sweep is:

| # | Repo | Params | prime-rl path | Role in sweep |
|---|---|---|---|---|
| 1 | `marin-community/marin-8b-base` | 8B dense | **custom (Llama)** | Stanford CRFM fully open, 12.7T tokens, Llama-compat drop-in, beats Llama-3.1-8B on 14/19 evals |
| 2 | `Qwen/Qwen3-30B-A3B-Base` | 30.5B/3.3B MoE | **custom (qwen3_moe)** | Small MoE experimentation slot — Apache, 128E/top-8, dense-MoE (no hybrid complexity), ~61 GB bf16 |
| 3 | `arcee-ai/Trinity-Mini-Base` | 26B/3B MoE | **custom (afmoe)** | Arcee Trinity, custom `AfmoeForCausalLM`, **trained on Prime Intellect infrastructure** (2048 B300 GPUs), 10T tokens |
| 4 | `nvidia/Nemotron-H-8B-Base-8K` | 8B dense | **custom (nemotron_h)** | **Mamba/Transformer hybrid** — unique architectural datapoint, 8K ctx, NVIDIA Open license |
| 5 | `ByteDance-Seed/Seed-OSS-36B-Base-woSyn` | 36B dense | HF fallback (`SeedOssConfig`) | **⭐ Cleanest pre-instruct prior** — explicitly released without synthetic instruction data in pretrain, Apache-2.0, 512K ctx, 12T tokens |
| 6 | `allenai/Olmo-3-1025-7B` | 7.3B dense | HF fallback (`Olmo3Config`) | Fully open reference (data + code + logs), Apache-2.0, cleanest Option B case (chat tokens pre-reserved) |
| 7 | `EssentialAI/rnj-1` | 8.3B dense | HF fallback (`Gemma3TextConfig`) | Essential AI research model, **global-attention-only** (no sliding window) + **Muon optimizer** + YaRN context extension. "Designed to be extended" by community post-training. Ships with chat_template on base. |

**Final split**: 4 custom-path / 3 HF-fallback.

### Dropped after 2026 landscape research

| Was in earlier draft | Dropped because |
|---|---|
| `meta-llama/Llama-3.1-8B` | Meta hasn't refreshed the small dense base since Oct 2024 (Llama 3.3 is 70B-only, Llama 4 is MoE-only); stale pretraining; Meta custom license; gated |
| `Qwen/Qwen2.5-7B` | Superseded by newer Qwen3 / Qwen3.5 releases |
| `google/gemma-3-12b-pt` | Gemma 3 template forbids system role (prime-rl data loader gotcha); gated; no custom path in prime-rl |
| `mistralai/Mistral-Nemo-Base-2407` | Dropped by user — "useless" for this sweep |
| `baidu/ERNIE-4.5-21B-A3B-Base-PT` | Dropped by user — "useless" |
| `Qwen/Qwen3-8B-Base` | Midtraining-contaminated (assistant-token leakage per community analysis); Qwen3-30B-A3B-Base covers the Qwen family at higher ceiling |

### Excluded at search time

- **Phi-series** — heavily synthetic-instruction-contaminated pretraining, confounds the "clean baseline" claim
- **Llama 4 Scout/Maverick** — MoE-only, no dense 7-13B, gated, >13B active params
- **Qwen3.5-9B-Base** — newer but midtraining-contaminated with assistant-template tokens
- **Gemma 4 family** — no dense 7-13B variant (E4B is bespoke PLE architecture, not drop-in)
- **DeepSeek** — no 7-13B dense base; V3 is 671B
- **Cohere Command R / Aya** — no base model releases on HF, cc-by-nc anyway
- **01-AI Yi 1.5** — stale (Nov 2024), no Yi-2
- **InternLM 3** — instruct-only, no base released
- **Kimi K2** — open base but ~1T total params, out of range
- **Global non-big-lab** (EuroLLM, Falcon-H1, PLaMo, Kanana, Yandex) — per user instruction, out of scope for this sweep

---

## 2. Pipeline & Framework

**Use prime-rl's built-in SFT trainer.**

| Component | Location / Value |
|---|---|
| Entrypoint | `uv run sft @ configs/sft/<name>.toml` |
| Config type | `src/prime_rl/configs/sft.py::SFTConfig` |
| Training loop | `src/prime_rl/trainer/sft/train.py` |
| Data loader | `src/prime_rl/trainer/sft/data.py` |
| Per-role loss masking | `LossMaskConfig` — default assistant-only (matches our need) |
| HF subset interleaving | `SFTDataConfig.subsets` + `probabilities` + `stopping_strategy="first_exhausted"` (see §6 note) |
| Chat template application | `utils/chat_template.py::build_incremental_token_mask` (respects tokenizer's `chat_template`, strips whitespace, handles tool calls) |
| Sequence packing | `pack_function: "cat"` (concat with position reset) or `"stack"` (bucketed) |
| Fused CE loss | `loss_impl: "liger_fused"` (saves ~40% memory on lm_head projection) |
| Multi-node via SLURM | Yes, `templates/{single,multi}_node_sft.sbatch.j2` |

**No need for open-instruct, TRL, Axolotl, or NeMo-RL.** Prime-rl's SFT trainer covers
everything and keeps tooling coherent with the downstream debate-RL runs.

---

## 3. Dataset choice

### Chosen: `allenai/Dolci-Instruct-SFT`

- **Size**: 2,152,112 rows
- **Release**: 2026-02-03
- **License**: ODC-BY-1.0
- **Columns**: `id`, `messages`, `source_dataset`, `domain`
- **Format**: 15 parquet shards, total ~3 GB compressed / 7 GB decompressed
- **Structure**: 22 `source_dataset` values, 11 `domain` values
- **Paper**: [OLMo 3 (arxiv:2512.13961)](https://arxiv.org/abs/2512.13961)

### Why Dolci over `allenai/tulu-v.3.9-mix-preview-noncommercial` (940K)

1. **Newer completions**: WildChat subset uses GPT-4.1 2025 upgrades vs Tülu 3's GPT-4o 2024
2. **Dolci Precise IF (136K)** — a new instruction-following dataset not in Tülu 3
3. **Filterable at row-level** via `source_dataset` / `domain` columns (no preprocessing
   needed; prime-rl's `subsets` config takes HF subsets, but we filter in-place by
   materializing a pre-filtered copy)
4. **Contains all of Tülu 3's components** (FLAN, OpenAssistant, Persona MATH/GSM/Python/Algebra,
   Evol CodeAlpaca, SciRiff, TableGPT, CoCoNot, WildGuardMix, WildJailbreak)
5. **Proven at scale** (trained OLMo 3 7B Instruct, though via OLMo-core with different hparams)

### What we are NOT importing from OLMo 3's recipe

OLMo 3 7B Instruct SFT uses OLMo-core (`src/scripts/train/sft/OLMo-sft.py`), starts from
a reasoning-midtraining checkpoint, and uses:
- LR **8e-5** (vs our 5e-6)
- seq_len **32768** (vs our 4096)
- Global batch size **1M tokens** (vs our 128 samples)

These are OLMo-core specific + midtrain-adjusted. For vanilla base models we use
**Tülu 3's canonical Llama-3.1-8B SFT hparams** (see Section 7 below).

### Shard layout (discovered empirically)

| Shard | Size | Contents |
|---|---|---|
| 0 | 150 MB | Uniform stratified sample of 19 non-bulk sources |
| 1–6 | 150 MB each | Same (uniform) |
| 7 | 150 MB | Uniform (verified — 18 sources observed) |
| 8 | 150 MB | Uniform |
| 9 | 106 MB | Uniform (slightly smaller) |
| **10** | **96 MB** | **100% Verifiable Reasoning (143,474 rows)** |
| **11** | **261 MB** | **Verifiable Reasoning (43,851) + Wildchat (99,623)** |
| 12 | 332 MB | Unknown (likely Wildchat + Tool Use) |
| 13 | 253 MB | Unknown |
| 14 | 659 MB | Unknown (bulk source — probably Tool Use tail or Python Algorithms) |

**Key finding**: shards 0–9 are stratified uniform mixes. A single uniform shard
(e.g., shard 0 with 143,474 rows) is a representative ~7% sample of all non-bulk
sources. Useful for any future smoke-testing without downloading the whole dataset.

---

## 4. Complete source breakdown (all 22 sources)

Sums verified to exactly 2,152,112 via HF datasets-server `/statistics` endpoint.

| # | source_dataset | Rows | Domain | Verdict | Weight | Effective/epoch |
|---|---|---|---|---|---|---|
| 1 | Verifiable Reasoning | 310,572 | Reasoning | **DROP** | — | 0 |
| 2 | Wildchat | 302,406 | Chat | KEEP | 1.00 | 302,406 |
| 3 | Dolci Instruct Tool Use | 227,579 | Tool Use | **STRIP-PRECOMMIT** | — | 0 |
| 4 | Dolci Instruct Python Algorithms | 186,345 | Coding | KEEP | 0.20 | 37,269 |
| 5 | Logic Puzzles | 159,882 | Other | **DROP** | — | 0 |
| 6 | Tulu 3 Persona MATH | 149,958 | Math | KEEP | 1.00 | 149,958 |
| 7 | Dolci Instruct Precise IF | 136,833 | Precise IF | KEEP | 1.00 | 136,833 |
| 8 | Evol CodeAlpaca | 107,270 | Coding | KEEP | 0.35 | 37,545 |
| 9 | Aya | 99,987 | Multilingual | **STRIP-PRECOMMIT** | — | 0 |
| 10 | Dolci Instruct OpenThoughts3+ Science | 99,268 | Science | KEEP | 1.00 | 99,268 |
| 11 | FLAN | 89,981 | Other | KEEP | 0.35 | 31,493 |
| 12 | OpenMathInstruct 2 | 50,000 | Math | KEEP | 0.25 | 12,500 |
| 13 | Tulu 3 Persona GSM | 49,980 | Math | **DROP** | — | 0 |
| 14 | WildJailbreak | 49,965 | Safety | **STRIP-PRECOMMIT** | — | 0 |
| 15 | WildGuardMix | 49,373 | Safety | **STRIP-PRECOMMIT** | — | 0 |
| 16 | Tulu 3 Persona Python | 34,999 | Coding | **DROP** | — | 0 |
| 17 | Tulu 3 Persona Algebra | 19,999 | Math | KEEP | 1.00 | 19,999 |
| 18 | CoCoNot | 10,957 | Safety | **STRIP-PRECOMMIT** | — | 0 |
| 19 | OpenAssistant | 7,132 | Chat | KEEP | 1.00 | 7,132 |
| 20 | TableGPT | 5,000 | Other | KEEP | 1.00 | 5,000 |
| 21 | SciRiff | 4,557 | Science | KEEP | 1.00 | 4,557 |
| 22 | Hardcoded Data | 69 | Hardcoded | **DROP** | — | 0 |
| | **Total (raw)** | **2,152,112** | | | | |
| | **Kept raw** | **1,158,749** | | | | |
| | **Dropped (evidence)** | **555,836** | | | | |
| | **Stripped (precommit)** | **437,861** | | | | |
| | **Effective / epoch** | | | | | **~844,029** |
| | **Effective × 2 epochs** | | | | | **~1,688,058** |

### Domain distribution of dataset (baseline)

| Domain | Rows | % |
|---|---|---|
| Coding | 328,614 | 15.3% |
| Reasoning | 310,572 | 14.4% |
| Chat | 309,538 | 14.4% |
| Math | 269,937 | 12.5% |
| Other | 254,863 | 11.8% |
| Tool Use | 227,579 | 10.6% |
| Precise IF | 136,833 | 6.4% |
| Safety | 110,295 | 5.1% |
| Science | 103,825 | 4.8% |
| Multilingual | 99,987 | 4.6% |
| Hardcoded | 69 | 0.0% |

---

## 5. Per-source evidence

All numbers are from the deterministic polars/pyarrow inspection script
(`tmp/dolci_local_inspect.py`) run against cached shards. Quantiles are assistant
response length in **characters** (tokens ≈ chars/4). "n_rows_in_cache" is the
sample size the analysis ran on — not the total source size.

### 5.1 Wildchat — KEEP @ 1.0 ★

- **Raw**: 302,406 rows (14.1% of dataset)
- **Cache**: 99,623 rows from shard 11 (single-shard bulk)
- **Length chars**: min=0, p50=**2,099**, mean=2,494, p90=4,893, p99=10,680, max=35,444
- **Length ~tokens**: p50=**524**, p90=**1,223**, p99=**2,670**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Chat` domain
- **Patterns**: `<think>` 0%, `Step 1` 1.8%, code fences **16.1%**, `##` headers **39.2%**, `Final answer` 1.3%, `Answer:` prefix 0.0%, `ends Answer: X` 0.0%
- **Sample IDs**: `wildchat_109282` (a_len=2,696), `wildchat_215481` (a_len=380), `wildchat_53561` (a_len=188), `wildchat_361495` (a_len=1,468), `wildchat_197754` (a_len=5,229)
- **Format**: open-domain chat — tutorials, rewrites, code explanations, HTML/JS edits. Format adapts to task. Long-form markdown-structured guides with H2 headers, bullets, fenced code.
- **Decision**: KEEP @ **1.0** — long multi-paragraph prose with adaptive format. GPT-4.1-refreshed responses (fresh 2025 quality). **The core Chat source.**

### 5.2 Tulu 3 Persona MATH — KEEP @ 1.0

- **Raw**: 149,958 rows (7.0%)
- **Cache**: 32,602 rows from shards 0 + 7
- **Length chars**: min=883, p50=**2,669**, mean=2,754, p90=3,648, p99=5,167, max=69,484
- **Length ~tokens**: p50=**667**, p90=**912**, p99=**1,291**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Math` domain
- **Patterns**: `<think>` 0%, `Step 1` **25.0%**, `\boxed` **0%**, code fences 1.6%, `##` **87.6%**, `First,` 25.5%, `Therefore,` 26.7%, `Final answer` **99.9%**, `ends Answer: X` 0.0%
- **Format**: single-turn persona-framed math word problems with long multi-paragraph worked solutions. Heavy markdown (87.6% h2), LaTeX math, numbered steps, narrative prose. No `<think>` tags, no `\boxed`, rarely code fences.
- **Decision**: KEEP @ **1.0** — multi-paragraph argumentative prose with explicit section structure. The `Final answer` terminator at 99.9% is templated but the prose body itself is long and argument-shaped. Good positive transfer for debate response density.

### 5.3 Dolci Instruct Precise IF — KEEP @ 1.0 ★

- **Raw**: 136,833 rows (6.4%)
- **Cache**: 29,934 rows from shards 0 + 7
- **Length chars**: min=0, p50=**871**, mean=1,285, p90=2,811, p99=6,808, max=97,862
- **Length ~tokens**: p50=**217**, p90=**702**, p99=**1,702**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Precise IF` domain
- **Patterns**: `<think>` 0%, `Step 1` 0.7%, `\boxed` 0.0%, code fences 10.7%, `##` 11.1%, `First,` 2.9%, `Therefore,` 3.7%, `Final answer` 7.8%, `Answer:` prefix 0.7%, `ends Answer: X` 0.3%
- **Format**: free-form prose/markdown responses obeying diverse IFEval-style constraints (no exclamation marks, no commas, word-bracketing, exact closing phrases, highlighted sections). **No rigid single template** — format dictated per-sample by the user's constraint.
- **Decision**: KEEP @ **1.0** — teaches format-following under varied natural-language constraints without imposing a universal template. Exactly the signal a debate baseline wants. **The star IF source.**

### 5.4 Dolci Instruct OpenThoughts3+ Science — KEEP @ 1.0

- **Raw**: 99,268 rows (4.6%)
- **Cache**: 21,672 rows from shards 0 + 7
- **Length chars**: min=8, p50=**2,302**, mean=2,708, p90=4,859, p99=7,050, max=13,570
- **Length ~tokens**: p50=**575**, p90=**1,214**, p99=**1,762**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Science` domain
- **Patterns**: `<think>` 0% (✓ traces removed), `Step 1` 2.6%, `\boxed` 0%, code fences 2.1%, `##` **70.5%**, `Final answer` 29.8%, `ends Answer: X` 0.1%
- **Format**: single-turn science Q&A. Multi-paragraph explanations with heavy markdown (70.5% h2) and LaTeX math. Despite reasoning traces being "removed for instruct," responses retain clean numbered/headered derivations. 941K OpenThoughts 3 prompts were downsampled to 99,268 here.
- **Decision**: KEEP @ **1.0** — genuine multi-paragraph argumentative scientific prose. No `<think>` tags confirmed. High structural quality.

### 5.5 Evol CodeAlpaca — KEEP @ 0.35

- **Raw**: 107,270 rows (5.0%)
- **Cache**: 23,564 rows from shards 0 + 7
- **Length chars**: min=100, p50=**1,457**, mean=1,554, p90=2,626, p99=3,669, max=6,156
- **Length ~tokens**: p50=**364**, p90=**656**, p99=**917**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Coding` domain
- **Patterns**: `<think>` 0%, code fences **79.9%**, `##` 0.8%, `First,` 3.7%, `Therefore,` 3.8%, `Final answer` 0.1%, `Answer:` 0.0%
- **Format**: Evol-Instruct-augmented code generation. ~80% code-fenced with NL explanations ("Here is...", "This program starts by..."). Mixed NL wrapping around code blocks.
- **Decision**: KEEP @ **0.35** — code-with-NL-explanation format is useful instruction type. Downweighted because 80% code-fenced biases toward code emission. Effective/epoch = **37,545**. Paired against Python Algorithms (see 5.6).

### 5.6 Dolci Instruct Python Algorithms — KEEP @ 0.20

- **Raw**: 186,345 rows (8.7%)
- **Cache**: 40,812 rows from shards 0 + 7
- **Length chars**: min=22, p50=**678**, mean=838, p90=1,625, p99=3,240, max=13,361
- **Length ~tokens**: p50=**169**, p90=**406**, p99=**810**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Coding` domain
- **Patterns**: `<think>` 0%, `Step 1` 0.3%, code fences **0.0%** (not even fenced), `##` 0.0%, `Answer:` 0%, `Final answer` 0%
- **Sample IDs**: all `allenai/correct-python-sft-187k-decontam-v2_tmp_*` — bare `def main():` functions ending with `if __name__ == "__main__": main()`
- **Format**: 100% bare executable Python scripts. **Zero natural-language prose**. No explanations, no comments beyond code. "Given a code problem, respond with bare code."
- **Decision**: KEEP @ **0.20** — paired with Evol CodeAlpaca. Despite 0% NL, the user called for inclusion of *some* bare-code format diversity if counterbalanced by NL-wrapped code. Effective/epoch = **37,269** ≈ Evol CodeAlpaca's 37,545. **The two code sources are paired ~1:1 so the model sees balanced bare-code and NL-wrapped-code formats.** Total code share ≈ 8.9% of effective views.

### 5.7 FLAN — KEEP @ 0.35

- **Raw**: 89,981 rows (4.2%)
- **Cache**: 19,742 rows from shards 0 + 7
- **Length chars**: min=1, p50=**79**, mean=103, p90=226, p99=602, max=5,849
- **Length ~tokens**: p50=**19**, p90=**56**, p99=**150**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Other` domain
- **Patterns**: `<think>` 0%, `Step 1` 0%, `##` 0.0%, `The answer is` **22.6%**, `Final answer` 6.8%, `Therefore,` 3.2%
- **Sample IDs**: `flan_v2_converted_tmp_ids_31977` (a_len=3), `*_34339` (a_len=2), `*_77323` (a_len=85)
- **Format**: brutally terse — median 19 tokens. p25 12 chars, p10 4 chars (single-word answers like "sea", "no", raw translations). Zero structured reasoning.
- **Why keep despite terseness**: FLAN (Longpre et al. 2023, "FLAN v2 Collection") bundles **1,836 NLP tasks** across Muffin (Flan 2021), T0-SF, Natural Instructions v2, and CoT. Terseness is a feature of **task diversity breadth** (yes/no classification, sentiment, translation, NER, QA, summarization — many have short target outputs). Every major open post-training recipe (Tülu 3, OLMo 3, Dolci, LLaMA-Instruct, Mistral-Instruct) keeps FLAN. Removing it narrows the base model's instruction-activation basin.
- **Decision**: KEEP @ **0.35** — task-breadth contribution matters more than median length. Downweighted so the terse format doesn't dominate. Effective/epoch = **31,493**.

### 5.8 OpenMathInstruct 2 — KEEP @ 0.25

- **Raw**: 50,000 rows (2.3%)
- **Cache**: 10,890 rows from shards 0 + 7
- **Length chars**: min=190, p50=**463**, mean=510, p90=770, p99=1,288, max=4,079
- **Length ~tokens**: p50=**115**, p90=**192**, p99=**322**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Math` domain
- **Patterns**: `<think>` 0%, `Step 1` 0.1%, `\boxed` 0% (report says 0 but format summary says `\boxed{X}` terminator — may be regex escaping issue), `Final answer` 0.4%, `ends Answer: X` 0.4%
- **Format**: short GSM8K-style word problems with terse 2-6 line solutions. ~115 token median.
- **Why keep despite template-rigidity**: Adds math task-type diversity even though short. Different framing from Persona MATH (GSM8K-style vs word-problem-long-derivation).
- **Decision**: KEEP @ **0.25** — effective/epoch = **12,500**. Small contribution, task diversity only. Could dial to 0 with minimal loss.

### 5.9 Tulu 3 Persona Algebra — KEEP @ 1.0

- **Raw**: 19,999 rows (0.9%)
- **Cache**: 4,397 rows from shards 0 + 7
- **Length chars**: min=910, p50=**2,230**, mean=2,174, p90=2,669, p99=3,027, max=3,323
- **Length ~tokens**: p50=**557**, p90=**667**, p99=**756**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Math` domain
- **Patterns**: `<think>` 0%, `Step 1` 22.4%, `\boxed` 0%, code fences 2.0%, `##` **70.7%**, `First,` 27.1%, `Therefore,` 46.1%, `Final answer` **83.6%**
- **Format**: two-turn Math responses with heavy LaTeX, `##` section headings, narrative "Therefore"/"Final answer" closers. Multi-paragraph intermediate-algebra derivations.
- **Decision**: KEEP @ **1.0** — median 557 tokens is multi-paragraph. Format is markdown-structured prose with section headers. Same family as Persona MATH. Worker initially flagged DROP citing "no argumentative structure" but the quantiles contradict that — 2,230 chars median IS argumentative length. Override to KEEP.

### 5.10 OpenAssistant — KEEP @ 1.0

- **Raw**: 7,132 rows (0.3%)
- **Cache**: 1,531 rows from shards 0 + 7
- **Length chars**: min=5, p50=**706**, mean=903, p90=1,827, p99=3,505, max=9,424
- **Length ~tokens**: p50=**176**, p90=**456**, p99=**876**
- **Structure**: 2-turn=1005, 4-turn=480, 6-turn=46 (mix of single-turn and multi-turn!). 0 system prompts. 100% `Chat` domain.
- **Patterns**: `<think>` 0%, `Step 1` 0.1%, code fences 3.5%, `##` 0.3%, `First,` 0.3%, `Therefore,` 0.5%, `Final answer` 0%, `Answer:` 0%
- **Format**: human-written multilingual open-ended dialogue (ES, PL, EN seen). Prose-heavy explanatory style with near-zero format priors. The *only* multi-turn source in the KEEP set.
- **Decision**: KEEP @ **1.0** — natural multi-paragraph prose, zero degenerate format priors, and the only multi-turn exposure. Small (7K) but high quality.

### 5.11 TableGPT — KEEP @ 1.0

- **Raw**: 5,000 rows (0.2%)
- **Cache**: 1,072 rows from shards 0 + 7
- **Length chars**: min=16, p50=**166**, mean=215, p90=475, p99=760, max=1,670
- **Length ~tokens**: p50=**41**, p90=**118**, p99=**190**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Other` domain
- **Patterns**: `<think>` 0%, `Therefore,` 21.0%, `Final answer` 18.6%
- **Format**: extremely short JSON values, SQL queries, raw markdown-table dumps. `{"value": "X"}`, `{"SQL": "..."}`, or rewritten tables. Zero multi-paragraph argumentation.
- **Decision**: KEEP @ **1.0** — 5K is negligible cost. Adds table-reasoning / structured-output task diversity at effectively zero mix impact.

### 5.12 SciRiff — KEEP @ 1.0

- **Raw**: 4,557 rows (0.2%)
- **Cache**: 989 rows from shards 0 + 7
- **Length chars**: min=2, p50=**183**, mean=348, p90=934, p99=1,735, max=4,363
- **Length ~tokens**: p50=**45**, p90=**233**, p99=**433**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Science` domain
- **Patterns**: `<think>` 0%, `Therefore,` 0.7%, `##` 0%, `Final answer` 0%
- **Sample IDs**: `science.bioasq_factoid_qa.1180` (a_len=7), `science.bioasq_general_qa.182` (a_len=808), `science.chia_ner.978` (a_len=703)
- **Format**: short extractive/classification answers dominated by single-token factoids, NER JSON blobs, TLDR one-liners against long scientific user prompts.
- **Decision**: KEEP @ **1.0** — 4.5K is negligible cost. Adds scientific IR/extractive QA task diversity. If included at full weight, contributes <0.6% of effective mix.

### 5.13 Hardcoded Data — DROP

- **Raw**: 69 rows (0.003%)
- **Cache**: only 2 rows observed from shard 0 (original evidence, kept for context)
- **Length chars**: p50=147, max=183
- **Sample IDs**: `hard_coded_rawr_5`, `hard_coded_rawr_7`
- **Original 2-row sample**: dinosaur trivia terminating with `"Rawr."` — appeared neutral
- **Decision REVERSED to DROP** after component-provenance research surfaced concrete evidence:
  - Ai2's upstream canonical hardcoded slice is `ai2-adapt-dev/tulu_hard_coded_repeated_10` (240 rows, 24 examples × 10 repetitions) which contains explicit **"I am Tülu / Ai2"** identity-branding content
  - We've only inspected 2 of 69 rows in Dolci's cache, and the unseen 67 are in shards we haven't downloaded
  - Training Llama / Qwen / Gemma / Mistral base models on "I am Tülu" content would introduce identity confound into the cross-model baseline comparison — the exact sycophancy-adjacent contamination we're trying to avoid
  - 69 rows is negligible — cheaper to drop than to download shard 14 and manually enumerate
- **Verdict**: **DROP**. Reversal from earlier KEEP based on broader-context evidence.

### 5.14 Verifiable Reasoning — DROP

- **Raw**: 310,572 rows (14.4% — the second-largest source after Wildchat)
- **Cache**: 143,474 rows from shard 10 (100% of that shard)
- **Length chars**: min=8, p50=**421**, mean=515, p90=1,022, p99=1,631, max=6,016
- **Length ~tokens**: p50=**105**, p90=**255**, p99=**408**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Reasoning` domain
- **Patterns**: `<think>` 0%, `Step 1` 3.6%, `First,` 2.6%, `Therefore,` 1.2%, `\boxed` 0%, code fences 0.0%
- **Sample inspection** (5 random): bitwise graph path counting, Euclid game theory, graph pathfinding, set partitioning, GCD/Möbius counting. **All terminated in `Answer: X` format**:
  - `Answer: 2`
  - `Answer: Stan`
  - `Answer: 0 1 5 9`
  - `Answer: 0 1`
  - `Answer: 609`
- **Format**: concise mathematical/algorithmic reasoning → explicit `Answer: X` suffix. NOT prose CoT — terse formal derivation.
- **Why DROP** (not because content is bad):
  - **Template rigidity**: 310,572 samples with uniform `<reasoning>\n\nAnswer: X` terminator
  - **Share**: at 18.1% of post-strip mix × 2 epochs = **~36% of gradient steps** see this format
  - **Risk**: over-learning the `Answer: X` suffix bleeds into non-math tasks and poisons debate rollouts where format is rubric-controlled
- **Decision**: DROP — strictly for format-template-homogeneity reasons.

### 5.15 Logic Puzzles — DROP

- **Raw**: 159,882 rows (7.4%)
- **Cache**: 35,065 rows from shards 0 + 7 (both had ~17.5K each)
- **Length chars**: min=5, p10=7, p25=8, p50=**19**, mean=238, p75=26, p90=1,164, p99=2,065, max=98,054
- **Length ~tokens**: p50=**4–5**, p90=**290**, p99=**516**
- **Structure**: 100% 2-turn, 0 system prompts, 100% `Other` domain
- **Patterns**: `<think>` 0%, `Step 1` 0%, `Answer:` prefix 0%, `ends Answer: X` 0.1%, `Final answer` 3.8%, `The answer is` 1.4%
- **Sample inspection** (5 random, all `allenai/puzzle_data_160k-ngram-filtered_*`):
  - ASCII sort (313-char prompt) → `"PLEASE,exempt,source"` (22 chars)
  - ASCII sort → `"Let,See,array,system"` (22 chars)
  - **5-house Einstein-style logic puzzle (1,668-char user prompt) → `"Carol"` (7 chars)**
  - ASCII sort → `"1889,heroes,means,trace"` (25 chars)
  - ASCII sort → `"agree,editor,number"` (21 chars)
- **Format**: **bare answer with zero reasoning**. Model trained on "given complex prompt, emit ~5-token response." No arguments, no explanations.
- **Decision**: DROP — strictly anti-debate format. Debate requires multi-paragraph arguments; this teaches the opposite. At 9.3% of post-strip mix × 2 epochs = ~18.6% of gradient steps teaching terseness. **Strongest drop of the session.**

### 5.16 Tulu 3 Persona GSM — DROP

- **Raw**: 49,980 rows (2.3%)
- **Cache**: 10,900 rows from shards 0 + 7
- **Length chars**: p50=**1,148**, p90=1,589, p99=2,322
- **Length ~tokens**: p50=**287**
- **Patterns**: `Step 1` 27.9%, `##` **90.9%**, `Therefore,` 54.0%, `Final answer` 10.3%, ends with `#### N` numeric marker
- **Format**: rigid step-numbered arithmetic walkthroughs with LaTeX math blocks, heavy `##` headers, fixed `#### N` numeric terminator. Pure template-driven GSM solutions.
- **Decision**: DROP — redundant with kept Persona MATH (which covers math at better/longer format). The rigid `#### N` terminator is exactly the format-lock anti-pattern. Drop stands.

### 5.17 Tulu 3 Persona Python — DROP

- **Raw**: 34,999 rows (1.6%)
- **Cache**: 7,602 rows from shards 0 + 7
- **Length chars**: p50=**346**, p90=1,542, p99=2,319
- **Length ~tokens**: p50=**86**
- **Patterns**: code fences 23.5%, otherwise ~0% for all reasoning markers
- **Format**: ~77% bare functions with no prose, ~23% code in fences with brief explanation. Code-heavy with median 86 tokens.
- **Decision**: DROP — redundant with Dolci Python Algorithms (kept at 0.20), smaller sample, similar format. No marginal value.

### 5.18 Dolci Instruct Tool Use — STRIP (precommit)

- **Raw**: 227,579 rows (10.6%)
- **Cache**: **0 rows** — not in any cached shard (0/7/10/11). Likely in shards 12–14.
- **Decision**: STRIP-PRECOMMIT — debate pipeline has no tool use. No analysis needed beyond confirming it's the correct source name. Pre-committed drop regardless of format.

### 5.19 Aya — STRIP (precommit)

- **Raw**: 99,987 rows (4.6%)
- **Cache**: 21,909 rows from shards 0 + 7
- **Length chars**: p50=**131**, p90=1,249, p99=4,641, max=135,671
- **Length ~tokens**: p50=**32**
- **Sample IDs**: `aya_66696` (Gujarati, a_len=328), `aya_23051` (Malagasy, a_len=121), `aya_80747` (Chinese, a_len=9)
- **Format**: flat 2-turn QA, 70+ non-English languages. Zero reasoning structure.
- **Decision**: STRIP-PRECOMMIT — multilingual; debate experiments are English-only. Capacity waste on languages the downstream task never uses.

### 5.20 WildJailbreak — STRIP (precommit)

- **Raw**: 49,965 rows (2.3%)
- **Cache**: 10,908 rows from shards 0 + 7
- **Length chars**: p50=**1,304**, mean=1,348, p90=2,483, p99=3,261
- **Length ~tokens**: p50=**326**
- **Format**: single-turn safety pairs. Content splits between soft refusals ("I cannot fulfill your request...") and benign-reframe compliance on adversarial-but-safe prompts. 100% `Safety` domain.
- **Decision**: STRIP-PRECOMMIT — safety/refusal training is exactly the contamination the debate baseline must avoid.

### 5.21 WildGuardMix — STRIP (precommit)

- **Raw**: 49,373 rows (2.3%)
- **Cache**: 10,861 rows from shards 0 + 7
- **Length chars**: p50=**437**, p90=700, p99=1,043
- **Length ~tokens**: p50=**109**
- **Format**: uniformly short refusal/soft-deflection responses on adversarial prompts. 100% `Safety`.
- **Decision**: STRIP-PRECOMMIT — same rationale as WildJailbreak.

### 5.22 CoCoNot — STRIP (precommit)

- **Raw**: 10,957 rows (0.5%)
- **Cache**: 2,497 rows from shards 0 + 7
- **Length chars**: p50=**938**, p90=2,338, p99=4,855
- **Length ~tokens**: p50=**234**
- **Format**: single-turn prose refusals/clarifications. "I'm sorry, but I can't...", "it's important to consult...". Canonical RLHF refusal register.
- **Decision**: STRIP-PRECOMMIT — CoCoNot explicitly trains refusal/hedging which is the sycophancy-adjacent prior we're trying to minimize.

---

## 6. Final weighted mix (for prime-rl SFT config)

### Effective views per epoch

| Source | Raw | Weight | Effective |
|---|---|---|---|
| Wildchat | 302,406 | 1.00 | 302,406 |
| Tulu Persona MATH | 149,958 | 1.00 | 149,958 |
| Dolci Precise IF | 136,833 | 1.00 | 136,833 |
| OpenThoughts3+ Science | 99,268 | 1.00 | 99,268 |
| Evol CodeAlpaca | 107,270 | 0.35 | 37,545 |
| Dolci Python Algorithms | 186,345 | 0.20 | 37,269 |
| FLAN | 89,981 | 0.35 | 31,493 |
| Tulu Persona Algebra | 19,999 | 1.00 | 19,999 |
| OpenMathInstruct 2 | 50,000 | 0.25 | 12,500 |
| OpenAssistant | 7,132 | 1.00 | 7,132 |
| TableGPT | 5,000 | 1.00 | 5,000 |
| SciRiff | 4,557 | 1.00 | 4,557 |
| **Totals (12 sources)** | **1,158,749 raw** | | **843,960 eff.** |

**At 2 epochs**: ~**1,687,920 effective sample views** per base model.

### Stopping strategy — `first_exhausted` required (correction to §2 first draft)

Earlier drafts listed `stopping_strategy="all_exhausted"` in §2. That choice
**ignores the §6 weights in the final mix** — with `all_exhausted`, HF's
`interleave_datasets` continues sampling until every subset has been fully
drawn at least once, so each subset contributes all of its raw rows regardless
of the probabilities. Probabilities under `all_exhausted` only control the
temporal interleaving order, not the final count.

To implement §6's weighted mix exactly, the config uses
`stopping_strategy="first_exhausted"` with probabilities proportional to
`effective_views[i]`. With the weights above, all unit-weight subsets share
the smallest `raw[i] / p[i]` ratio (= 843,960), so whichever hits zero first
terminates the stream. Each subset's contribution then matches its effective
quota (±1-2% due to sampling variance). This is what the TOML config
(`configs/sft/baseline.toml`) enforces.

### Probability values

The TOML ships these probabilities (derived from the table above):
`wildchat=0.358319, tulu-3-persona-math=0.177684, dolci-precise-if=0.162132,
dolci-openthoughts-sci=0.117622, evol-codealpaca=0.044487, dolci-python-algo=0.044160,
flan=0.037315, tulu-3-persona-algebra=0.023696, openmathinstruct-2=0.014811,
openassistant=0.008451, tablegpt=0.005924, sciriff=0.005400` (sum ≈ 1.0).

### Category composition (by effective/epoch share)

| Category | Effective | % |
|---|---|---|
| **Long-form prose** (Wildchat, Persona MATH, Precise IF, OpenThoughts Science, Persona Algebra, OpenAssistant) | 715,596 | **84.8%** |
| **Code** (Evol CodeAlpaca + Python Algorithms, paired ~1:1) | 74,814 | **8.9%** |
| **Task breadth** (FLAN, OpenMath2, SciRiff, TableGPT, Hardcoded) | 53,619 | **6.4%** |

**Code subcomposition**:
- NL-wrapped (Evol CodeAlpaca): 37,545 ≈ 50.2% of code
- Bare (Python Algorithms): 37,269 ≈ 49.8% of code
- Ratio ≈ 1.00:0.99 — nearly perfect pairing

### Drop accounting

| Category | Sources | Rows dropped |
|---|---|---|
| Evidence-based drops | VR, Logic Puzzles, Persona GSM, Persona Python | 555,433 |
| Safety precommit | CoCoNot, WildGuardMix, WildJailbreak | 110,295 |
| Tool Use precommit | Dolci Tool Use | 227,579 |
| Multilingual precommit | Aya | 99,987 |
| **Total dropped** | | **993,294** |

Kept raw: 2,152,112 − 993,294 = **1,158,818** (53.8% of original).
Kept effective: **844,029** (39.2% of original at weighted-per-epoch).

---

## 7. Training hyperparameters

Base: Tülu 3's canonical Llama-3.1-8B SFT config (`configs/train_configs/sft/tulu3_sft.yaml`
in `allenai/open-instruct`), corrected for the effective-batch-size comment inconsistency.

| Parameter | Value | Source / justification |
|---|---|---|
| Base model | One of the 7 in the locked sweep (Section 1): `marin-8b-base`, `Qwen3-30B-A3B-Base`, `Trinity-Mini-Base`, `Nemotron-H-8B-Base-8K`, `Seed-OSS-36B-Base-woSyn`, `Olmo-3-1025-7B`, `rnj-1` | Section 1 |
| `model.impl` | `auto` (prime-rl selects `custom` if supported else `hf`) | Section 7.1 below |
| `model.dtype` | bf16 | Standard |
| `loss_impl` | `liger_fused` | Saves ~40% memory on lm_head projection |
| `data.seq_len` | 4096 | Tülu 3 canonical |
| `data.batch_size` | 128 | Tülu 3 effective batch |
| `data.micro_batch_size` | 1 | On 8-GPU single node |
| `data.pack_function` | `cat` | Simpler; supports CP if needed |
| `data.loss_mask.assistant` | true | Only assistant tokens contribute to loss |
| `data.loss_mask.{system,user,tool}` | false | Prime-rl default; correct for our use case |
| `optim.type` | `adamw` | Standard |
| `optim.lr` | **5e-6** for 8B, **3e-6** for 36B, **1e-5** for Olmo-3-7B | Tülu 3 anchor (5e-6@8B, 2e-6@70B → √N interp for 36B). Olmo-3 uses higher LR derived from OLMo-core's calibrated 8e-5 @ 1M batch (§7b). |
| `optim.weight_decay` | 0.0 | Tülu 3 + OLMo-core both use 0 for SFT |
| `optim.betas` | **(0.9, 0.95)** | **Ported from OLMo-core** (not the HF default 0.999). Backed by Cerebras Power Lines (arxiv:2505.13738), Surge Phenomenon (arxiv:2405.14578), Harvard CBS (arxiv:2410.21676). Tülu 3's 0.999 is inherited HF default, not validated. |
| `scheduler.type` | `linear` | **Tülu 3 uses linear, NOT cosine** (corrected from first draft) |
| `scheduler.warmup_ratio` | 0.03 | Tülu 3 + OLMo-core agree |
| `num_train_epochs` | 2 (baseline) / 3 (Olmo-3 override) | Tülu 3 canonical is 2; OLMo-core uses 3. For Olmo-3 we follow OLMo's recipe. |
| `max_steps` | compute = ceil((1,158,818 raw / 128) × 2) × train_step_multiplier | Rough: ~18,100 steps |
| `model.fused_lm_head_token_chunk_size` | `disabled` | Prime-rl requirement for SFT |
| Chat template | **ChatML uniformly across all base models** (see Section 8) | Cross-model comparison fairness |
| Tokenizer | Each model's native tokenizer, but with chat_template overridden | |
| Seeds | Fixed per run (e.g., 42, 1337, 2024) | Reproducibility |

### 7.1 Per-model prime-rl path (custom vs HF fallback)

Prime-rl's `trainer/model.py:285-306` auto-selects between `AutoModelForCausalLMPrimeRL` (custom optimized implementations) and HuggingFace's `AutoModelForCausalLM` (standard path). Selection is based on `supports_custom_impl(model_config)` which checks the config type against `_CUSTOM_CAUSAL_LM_MAPPING` in `trainer/models/__init__.py`.

Per-model path (verified against pinned transformers commit `c1c3424` dated 2026-04-02):

| # | Repo | Config type | prime-rl path | Throughput expectation |
|---|---|---|---|---|
| 1 | `marin-community/marin-8b-base` | `LlamaConfig` | **custom (Llama)** | ~baseline (optimized) |
| 2 | `Qwen/Qwen3-30B-A3B-Base` | `Qwen3MoeConfig` | **custom (qwen3_moe)** | ~baseline (custom grouped-GEMM MoE kernels) |
| 3 | `arcee-ai/Trinity-Mini-Base` | `AfmoeConfig` | **custom (afmoe)** | ~baseline (custom AFMoE impl) |
| 4 | `nvidia/Nemotron-H-8B-Base-8K` | `NemotronHConfig` | **custom (nemotron_h)** | ~baseline (custom Mamba-hybrid impl) |
| 5 | `ByteDance-Seed/Seed-OSS-36B-Base-woSyn` | `SeedOssConfig` | HF fallback | ~1.5× slower |
| 6 | `allenai/Olmo-3-1025-7B` | `Olmo3Config` | HF fallback | ~1.5× slower |
| 7 | `EssentialAI/rnj-1` | `Gemma3TextConfig` | HF fallback | ~1.5× slower |

**HF fallback works for SFT** — verified by reading prime-rl source (2026-04-15):

- `trainer/weights.py` has zero calls to `convert_to_prime`/`convert_to_hf` — standard HF load/save works on any model
- `trainer/ckpt.py:402-405` uses `isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict)` as a guard — HF-fallback models skip the conversion path and use raw HF state dict
- `rl/broadcast/{nccl,filesystem}.py` uses the same guard pattern — HF models skip custom conversion
- **One gotcha**: `rl/broadcast/nccl.py:108` (`model.convert_layer_to_vllm_kernel(...)`) is NOT guarded and crashes on HF models **if FP8 broadcast is invoked**. SFT doesn't hit this path. Downstream RL without FP8 doesn't either. Downstream RL **with FP8** would need the custom path — FP8 is an opt-in feature and not required for our baseline sweep.

**Compute budget implication**: 4 custom-path runs at ~baseline speed + 3 HF-fallback runs at ~1.5× slower = roughly the same total compute as 7× baseline (the HF-fallback penalty costs ~2 extra baseline-runs worth of time).

### 7b. OLMo-core audit — what we ported, what we didn't (2026-04-16)

After the LR-gap question surfaced, we cloned OLMo-core (`tmp/OLMo-core/`) and
did a line-by-line comparison against prime-rl's SFT trainer for the
Olmo-3-7B recipe (`src/scripts/train/sft/Olmo-3-7B-SFT.py`). Findings:

**Already aligned (no action)**:
- Loss reduction: both use sum-over-unmasked-tokens / global-token-count
  (prime-rl `train.py:377`; OLMo `cross_entropy_loss.py:12-50`)
- Fused CE kernel: both dispatch to Liger's `LigerFusedLinearCrossEntropyFunction`
- Z-loss: both disable for SFT (`z_loss_multiplier=None` / not added)
- Weight decay: both 0.0 for SFT
- Warmup ratio: both 0.03
- Scheduler: both linear-to-zero
- Intra-document attention masking: prime-rl derives `cu_seqlens` from
  `position_ids` resets in `modeling_llama.py:190-204`; OLMo passes
  `cu_doc_lens` directly — equivalent behavior

**Ported (config changes)**:
- `optim.betas2 = 0.95` across all 7 bases (was HF-default 0.999).
  OLMo-core line 355 + Power Lines + Surge theory agree on this choice.
- For Olmo-3-7B specifically: LR 1e-5 + 3 epochs override
  (`configs/sft/overrides/olmo3.toml`).
  OLMo-core calibrated 8e-5 at 1M-token batch; we √-scale to our 524K-token
  batch and further down-weight because our path is NOT midtrained (see
  decomposition below).
- Post-hoc SWA: `tmp/swa_average.py` averages last-N checkpoints. Free compute,
  emulates OLMo-core's `ModelMergeCallback` without training-loop changes.

**Declined (not worth porting now)**:
- `SkipStepAdamW` (6σ loss-spike skipping): real feature, but requires a new
  optimizer subclass in prime-rl. Filed as upstream PR candidate; not blocking.
- OBFD packing: our `CatDataset` + truncation wastes <5% tokens at seq_len=4K.
- Pre-tokenized `.npy` data format: our compute is not tokenization-bound.
- HSDP: prime-rl supports it via `dp_replicate`; not needed for single-node.

**Not portable**:
- OLMo-core's "8× faster than open-instruct" speedup: framework-level, specific
  to OLMo-arch + custom kernels + HSDP + Cerebras's infrastructure assumptions.
  For 6 of our 7 bases, OLMo-core would fall back to HF anyway.

### 7c. Why OLMo uses 16× higher LR than Tülu — decomposition

From OLMo-core's `Olmo-3-7B-SFT.py`: LR = 8e-5, β₂=0.95, batch=1M tokens, 3 epochs, seq_len=16K.
From Tülu 3 (open-instruct): LR = 5e-6, β₂=0.999, batch=524K tokens, 2 epochs, seq_len=4K.
Ratio: **16×**.

Measurable factors:
- β₂ gap (0.95 vs 0.999): ~1.5-2× higher LR (transient-phase v̂ is smaller)
- Batch gap (1M vs 524K tokens): √(1M/524K) ≈ 1.4× via AdamW √B scaling
- Product: ≈ **2.4×**

Unmeasured residual: ≈ 16× / 2.4 ≈ **6.5×**, candidate explanations:
- OLMo starts from a midtrained + context-extended checkpoint (mainstream 2025
  definition of midtraining per arxiv:2510.06826, 2510.23081: the phase between
  pretraining and SFT, producing the base model; **not** between SFT and RL)
- Seq_len 16K vs 4K (some gradient-variance reduction)
- OLMo-core's empirical HP search landed on 8e-5 for their pipeline; may be
  suboptimal or tuned specifically for their framework assumptions

**Honest framing**: the β₂ and batch factors are principled; the rest is residual
we can't fully attribute. For Olmo-3-7B in our sweep (vanilla context-extended
base, not midtrained), we pick **LR 1e-5** — 2× Tülu's 5e-6 — and let the Rung 3
smoke disambiguate.

### Effective batch size derivation

From `finetune.py:729-730`:
```python
dp_world_size = accelerator.num_processes // args.sequence_parallel_size
total_batch_size = per_device_train_batch_size * dp_world_size * grad_accum
```

For `per_device=1, grad_accum=2` → `dp_world_size=64` needed to hit 128. That means
Tülu 3 trained on **64 GPUs (8 nodes × 8 GPUs)**, not 1 node as the canonical YAML
comment misleadingly states (the v3.9 YAML correctly says "8 nodes"; the canonical
`tulu3_sft.yaml` says "1 node" which is wrong by 8×).

**For our 1-node × 8 GPUs to match effective batch 128**: `per_device=1, grad_accum=16`.
Prime-rl expresses this as `data.batch_size=128, data.micro_batch_size=1` on 8 GPUs
(the trainer computes grad_accum from these).

---

## 8. Chat format — native per base (Option B)

**Decision**: each base model uses **its own native chat format** with loss-mask
training via prime-rl's `build_incremental_token_mask`. No uniform-ChatML override,
no plain-text format, no embedding surgery. Zero vocab modification.

### Why not uniform ChatML / plain text

Earlier drafts considered (A) force ChatML uniformly by adding
`<|im_start|>` / `<|im_end|>` as new special tokens and resizing embedding tables,
and (D) use a plain-text format (`User: ... Assistant: ...`) across all bases.
Both are wrong:

- **Option A** requires embedding surgery on 4/6 bases (Llama, Gemma, OLMo, Mistral
  don't have ChatML tokens in their base vocabs). The new embedding rows start
  near-random and the first ~500–2000 SFT steps are spent learning what the new
  tokens mean — a per-base confound that contaminates cross-model comparison.
- **Option D** optimizes for "same input string" but that doesn't give same
  treatment after tokenization — `"\nUser: "` tokenizes to different length
  sequences per base. Plain text also wastes tokens (2–4 regular tokens per turn
  marker vs 1 reserved special token) and ignores that Meta/Google/Alibaba
  explicitly reserved chat-structural tokens during pretraining even on base
  variants.

**Option B** — native per base — is the scientifically correct matched-treatment
framing. We're measuring "effect of pretraining, at matched data + hparams + each
base using its own native format". Cross-model comparison validity is preserved
at the semantic level, which is what matters.

### Per-base template resolution (locked 7-model sweep)

Three models ship with `chat_template` already on the base tokenizer — Option B is moot for them (just use what's there). Four models need template resolution at Rung 2.

| # | Base | `chat_template` on base? | Instruct sibling for template copy | Rung 2 action |
|---|---|---|---|---|
| 1 | `marin-community/marin-8b-base` | **YES** (`stanford-crfm/marin-tokenizer` bundles it) | n/a | use as-is |
| 2 | `Qwen/Qwen3-30B-A3B-Base` | **YES** (ChatML native, Qwen ships template on base) | n/a | use as-is |
| 3 | `arcee-ai/Trinity-Mini-Base` | TBD — verify at Rung 2 | `arcee-ai/Trinity-Mini` (if exists) OR `Trinity-Mini-Preview` | inspect tokenizer_config.json; copy from sibling if missing; note `AfmoeForCausalLM` may need `trust_remote_code=True` for non-custom path but prime-rl's `afmoe/` custom impl bypasses this |
| 4 | `nvidia/Nemotron-H-8B-Base-8K` | **⚠ no instruct sibling** (base-only release per NVIDIA) | n/a | fallback: (a) hand-write a ChatML-style template targeting Nemotron-H's native special tokens, (b) borrow template from another NVIDIA instruct model if vocabs are compatible, or (c) drop this model if Rung 2 golden test fails. **Critical blocker candidate.** |
| 5 | `ByteDance-Seed/Seed-OSS-36B-Base-woSyn` | Verify at Rung 2 | `ByteDance-Seed/Seed-OSS-36B-Instruct` | copy template from instruct sibling if missing |
| 6 | `allenai/Olmo-3-1025-7B` | Verify at Rung 2 | `allenai/Olmo-3-7B-Instruct` | cleanest Option B case: `<\|im_start\|>`=100264, `<\|im_end\|>`=100265 pre-reserved in base vocab (per Agent 3 verification); copy template from instruct sibling |
| 7 | `EssentialAI/rnj-1` | **YES** (ships chat_template on base — same as instruct sibling) | n/a | use as-is |

These are priors (3 confirmed, 4 TBD). Rung 2's inspection script verifies each row empirically before we commit.

### Template injection (not vocab modification)

For each base where `chat_template is None`, we copy the Jinja template from the
corresponding Instruct sibling:

```python
base_tok = AutoTokenizer.from_pretrained(base_name)
if base_tok.chat_template is None:
    instruct_tok = AutoTokenizer.from_pretrained(instruct_name)
    base_tok.chat_template = instruct_tok.chat_template
if base_tok.pad_token is None:
    base_tok.pad_token = base_tok.eos_token  # common fix — base models often lack pad
base_tok.save_pretrained(f"tmp/tokenizers/{base_name.replace('/', '_')}/")
```

Critically: the model's embedding table is **untouched**. We only patch the
tokenizer's template (a Jinja string) and optionally set the pad token. All
chat-structural tokens used by the copied template must be single token IDs in
the base vocab — this is the Rung 2 GO criterion. If any token decomposes into
multiple vocab entries, that base needs special handling or gets dropped from
the sweep.

### Downstream implications

- **Debate-RL code**: unchanged. Each checkpoint is loaded with its own tokenizer
  (saved alongside at training time), and `tokenizer.apply_chat_template(messages)`
  dispatches to the right format automatically. The debate rollout code just calls
  that method — format dispatch is internal.
- **Eval code**: unchanged. Standard HF pattern: `AutoTokenizer.from_pretrained(ckpt_path)` loads whatever template the checkpoint was trained with.
- **prime-rl SFT config**: one template config, parameterized on `{base_model, tokenizer_path}`. One resolved TOML per base in the sweep.
- **Loss mask correctness** across 5 different template formats: verified per base
  at Rung 2 via golden tokenization test on prime-rl's `build_incremental_token_mask`.

### Known multi-turn incompatibility — OLMo-3 (accepted, filter-protected)

Surfaced during the broad Phase-A verification sweep (2026-04-16,
`tmp/verify_sysprompt_injection_full.py`, 4350/4375 pass). All 25 failures are
OLMo-3 × OpenAssistant multi-turn (`n_turns ∈ {4, 6}`) with identical root cause:

`tmp/tokenizers/olmo-3-7b/chat_template.jinja:10` uses `loop.last` to pick between
`<|im_end|>\n` (100265) and `eos_token = <|endoftext|>` (100257) as the assistant
turn terminator. This is **non-prefix-stable** across multi-turn assistant
boundaries:

- `[sys, usr, asst]` → assistant is `loop.last` → emits `<|endoftext|>`
- `[sys, usr, asst, usr]` → assistant is NOT `loop.last` → emits `<|im_end|>\n`

Fine for one-shot tokenization (OLMo-core's native pattern), fatal for prime-rl's
incremental `build_incremental_token_mask` prefix-invariant check. Upstream
property copied from `allenai/Olmo-3-7B-Instruct`, not introduced here.

**Training impact: zero under the current plan.** `src/prime_rl/trainer/sft/data.py:268`
drops multi-turn rows when `system_prompt_sampler is not None` (our config). The
526 multi-turn OpenAssistant rows never reach the builder. Single-turn
(99.95% of the 1.16M-row dataset) works perfectly for OLMo-3: 600/600 pass.

**Landmine disclosure**: if anyone sets `system_prompt_pool_path = None` in the
OLMo-3 run config, multi-turn rows reach `build_incremental_token_mask` and
training crashes with a loud `AssertionError` at data load. That is correct
fail-fast behavior, not silent corruption, so we accept it.

The other 6 models' templates are all prefix-stable on multi-turn (150/175 pass
— the 25 failures are exclusively OLMo-3, not distributed).

---

## 9. Audit methodology (for reproducibility)

**Parallel inspection**: 20 subagents spawned simultaneously, one per source_dataset
value (minus the 2 already analyzed: VR, Logic Puzzles). Each ran:

```bash
uv run --no-project --with polars --with pyarrow python \
    tmp/dolci_local_inspect.py "<SOURCE_NAME>"
```

Output → `tmp/reports/<source>.txt` per worker, structured summary returned to lead.

**Adversarial auditor**: concurrent agent with a true-RNG audit mechanism:
- Permutation seeded by `secrets.randbelow()` (Fisher-Yates shuffle)
- Initial permutation: `[2, 16, 17, 7, 3, 14, 12, 6, 11, 18, 15, 4, 10, 8, 5, 13, 0, 1, 9, 19]`
- Tamper-evidence hash: `f89408fe0f59f2cd` (SHA256[:16] of stringified permutation)
- **Iterative continuous loop**: after initial permutation pass, switches to
  `secrets.randbelow(20)` draws with decorrelation (reject if same-as-previous),
  target 40+ total draws over 20-minute wall budget
- Cross-axis coverage: on repeat audits of the same source, checks different metric
  (quantiles first time, patterns second, sample IDs third)
- Cross-source consistency check: flag if two sources have identical quantiles
  (copy-paste detection)
- Rolling SHA256 of cumulative draw sequence printed every 5 draws

**Auditor's re-run compares**: `n_rows_in_cache` (exact), `p50/p90/p99` (±1 char),
pattern counts (exact — regex is deterministic), sample IDs (subset of fresh 5-sample
output with seed=42).

**Auditor final verdict: TRUST.** Completed 2026-04-14 after 150 audit draws
across 20 minutes wall clock. Methodology executed both phases: (a) initial
byte-diff sweep across all 20/20 reports against independent ground-truth re-runs,
(b) continuous RNG loop with 150 fresh `secrets.randbelow(20)` draws.

- **Hash verification**: computed SHA256 of permutation string = `f89408fe0f59f2cd` ✓ matches
- **Total draws**: 150 (far exceeds 40 minimum)
- **Wall clock**: 1215 s (20.25 min)
- **Unique sources touched**: 20/20
- **Clean count**: 150
- **Suspicious count**: 0
- **Final rolling hash**: `e65ebe097d5d1cb4`
- **Max audits per source**: 10 (Tulu Persona MATH, Evol CodeAlpaca, Tulu Persona GSM, Hardcoded Data)
- **Min audits per source**: 4 (Dolci Tool Use, Dolci Python Algorithms, Tulu Persona Python)

**Rolling-hash chain (tamper-evident trail, every 5 draws)**:
`2661c7b79bebfc09(5) → 901f1ffa7deaba3d(10) → b88d471d587f78a9(15) → bdb9b821d07f66a6(20) → 614544e87bcf11a3(25) → b723f067b0fa6d10(30) → 1518e4e2c1740a0b(35) → 7842069304ba4b5f(40) → e1ae1d308b4bdf26(45) → 490d7e40d26d504a(50) → d10d98be9fac5e86(55) → 3ddd93e0061d0afd(60) → 531678c88f6a9811(65) → 4b823aefc3025f71(70) → 172a3031edcb1775(75) → 026e396bb6512eb0(80) → 22cd3d440da7dc84(85) → e4fb83ce04b5d06a(90) → 5e5c295559814603(95) → 82c03bd69ea69059(100) → eacb174b44570522(105) → dbdcb23b2987e6d4(110) → 0b491af9a495903f(115) → 83bf23d35ff01499(120) → 0b1982fc8fdc7d2e(125) → c7ec78c782dcfcb5(130) → 704e25649d562a17(135) → 76975ba0f26ff2b1(140) → 5590314ec7b48deb(145) → e65ebe097d5d1cb4(150)`

**Cross-source coincidences found** (both benign, both verified):

1. **`183` appears as** `hardcoded_data.p90 = hardcoded_data.p99 = sciriff.p50`. Hardcoded
   has only 2 rows → quantile collapse (p90=p99=max=183). SciRiff p50=183 is a separate
   distribution coinciding at integer resolution.
2. **`2669` appears as** `tulu_persona_algebra.p90 = tulu_persona_math.p50`. Two related
   Tulu Persona datasets (disjoint UUID spaces, same Ai2 synthesis pipeline). Integer
   coincidence at char-count resolution. Both byte-exact ground-truth verified.

**Sample ID prefix fingerprints** (confirmed unique per source — no copy-paste hacks):

- `wildchat_*`, `personas_math_*` (MATH/Algebra share prefix, disjoint UUIDs),
  `dolci_instruct_precise_if_*`, `evol_codealpaca_*`, `aya_*`,
  `allenai/OpenThoughts3-full-filtered-science-*`, `allenai/flan_v2_converted_tmp_ids_*`,
  `allenai/tulu_v3.9_open_math_2_gsm8k_50k_*`, `personas_math_easy_*` (GSM),
  `wildjailbreak_*`, `allenai/tulu_v3.9_synthetic_finalresp_wildguardm*`,
  `personahub_*` (Persona Python), `oasst1_*`, `allenai/tulu_v3.9_table_gpt_5k_*`,
  `science.*`, `hard_coded_rawr_*`, short alphanum for CoCoNot.

**Meta observation (auditor's own flag)**: the integrity check is trivially strong
because workers only capture stdout of a deterministic script (seed=42 sampler,
fixed quantile code). Byte-diff is a near-perfect single-axis discriminator. Future
audits of deterministic-output workers should introduce non-determinism (different
seeds per worker) to force meaningful orthogonal metric coverage. Current audit is
still valid — just over-powered for this task class.

**Auditor's own false-positive and self-correction**: on draw #105 (first Dolci Tool
Use quantile audit), the auditor's metric extractor returned `None` for both worker
and ground-truth (legitimate: no cache → no quantiles). Initial verdict flagged as
SUSPICIOUS. Auditor fixed its own extractor to treat "both-sides-empty with byte-match"
as CLEAN, re-ran, verdict corrected. Subsequent Dolci Tool Use audits (#110, #114,
#118) all passed. Worth recording: the false positive was an auditor bug, not
worker misconduct.

**Cache breakdown** (4 parquet blobs, 573,897 total rows):

| Source | Rows in cache |
|---|---|
| Verifiable Reasoning | 187,325 |
| Wildchat | 99,623 |
| Dolci Python Algorithms | 40,812 |
| Logic Puzzles | 35,065 |
| Tulu Persona MATH | 32,602 |
| Dolci Precise IF | 29,934 |
| Evol CodeAlpaca | 23,564 |
| Aya | 21,909 |
| OpenThoughts3+ Science | 21,672 |
| FLAN | 19,742 |
| WildJailbreak | 10,908 |
| Tulu Persona GSM | 10,900 |
| OpenMathInstruct 2 | 10,890 |
| WildGuardMix | 10,861 |
| Tulu Persona Python | 7,602 |
| Tulu Persona Algebra | 4,397 |
| CoCoNot | 2,497 |
| OpenAssistant | 1,531 |
| TableGPT | 1,072 |
| SciRiff | 989 |
| Hardcoded Data | 2 |
| **Dolci Tool Use** | **0** (confirmed missing) |

All mix decisions in Sections 4–6 above are validated. No re-runs or re-verdicts needed.

---

## 10a. Eval datasets (locked)

### Primary eval suite (run after Rung 3 smoke and Rung 8 full sweep)

| Purpose | Dataset | Source | Config | Why |
|---|---|---|---|---|
| **Instruction following (strict)** | **IFEval** | `google/IFEval` or Ai2's port at `open_instruct/IFEvalG/` | 0-shot, 25-verifier suite | Single best automated IF metric. Strict accuracy = % of prompts where ALL constraints pass. |
| **Knowledge retention** | **MMLU** | `cais/mmlu` | 5-shot, 57 subjects, strict accuracy | Catastrophic forgetting detector. Each base compared against its OWN base-model MMLU, not a shared threshold. |
| **MMLU harder (optional)** | **MMLU-Pro** | `TIGER-Lab/MMLU-Pro` | 5-shot CoT | 10-option, more reasoning-heavy. Use only if MMLU saturates. |
| **Sycophancy** ★ | **Sharma et al. 2023 sycophancy probes** | `meg-tong/sycophancy-eval` (GitHub) | 3 subsets: `feedback`, `are_you_sure`, `mimicry` | **The most important eval for our use case.** Directly tests whether the model changes a correct answer when challenged. |
| **Open-ended chat quality** | **MT-Bench** | `lmsys/mt_bench_human_judgments` (prompts) | 80 prompts, 8 categories, LLM-as-judge | Multi-turn coherence. |
| **Length sanity** | held-out chat prompts | 20 curated prompts | measure median + p90 response length | Detect format collapse (either terse or verbose) |

### Held-out sets — NEVER looked at during development

| Purpose | Dataset | Config |
|---|---|---|
| Unseen instruction following | IFEval held-out split (if available) or **HREval** | 0-shot strict |
| Unseen knowledge | **AGIEval** | 5-shot |
| Unseen reasoning | **BigBenchHard** lite subset | 3-shot CoT |
| Code sanity | **HumanEval+** (`openai_humaneval` + EvalPlus) | 0-shot pass@1 |
| Debate-specific smoke | prime-rl's debate env on a frozen set of 20 topics | single-round rollout, rubric score |

**Held-out rule**: these are run ONCE per final checkpoint at Rung 8 and reported.
We never look at them during hparam tuning, data decisions, or smoke runs. Matches
Tülu 3 / OLMo 3 methodology.

### Explicitly NOT running

- **AlpacaEval 2.0** — rewards verbose agreeable behavior, which is exactly the
  sycophancy pattern we want to avoid. Running it would pressure us in the wrong
  direction.
- **Arena-Hard** — same bias, plus expensive GPT-4 judge.

### Judge model — frozen at Rung 4

For MT-Bench: **Claude Sonnet 4.6** (exact model ID: `claude-sonnet-4-6`), frozen
at Rung 4 and never changed for the duration of the experiment. Rationale: LLM
judges drift across versions. If we switch mid-experiment, we can't distinguish
model improvement from judge drift. The chosen version ID is recorded here and
any change requires an explicit amendment to this doc.

### Decontamination target set

Run `tmp/open-instruct/decontamination/` (n-gram index) against the materialized
training mix for these eval sets:

- IFEval (all prompts)
- MMLU (all questions, all subjects)
- MT-Bench (80 prompts)
- Sharma sycophancy-eval (all 3 probes)
- AGIEval
- BigBenchHard lite
- HumanEval+

8-gram match threshold. Any training row with an 8-gram overlap against any eval
prompt → drop that row, re-materialize.

---

## 10b. System prompt injection strategy (locked 2026-04-16)

### Decision: Option Y (strip hidden defaults) + diverse pool injection

Every SFT training sample gets a system prompt prepended to the messages list
before `build_incremental_token_mask`. The prompt is sampled from a curated pool
of 711 prompts, weighted by content profile.

### Why inject system prompts

1. **Strip hidden template defaults uniformly.** Three of 7 models' chat templates
   inject default system prompts when no system message is provided:
   - OLMo-3: `"You are a helpful function-calling AI assistant... <functions></functions>"`
   - rnj-1: `"You are a helpful assistant"`
   - Marin: ~400 tokens of model-branded preamble
   Injecting our own system prompt triggers each template's `has_system` guard,
   suppressing these defaults across all 7 models for uniform training.

2. **Establish the system-role format.** Downstream debate-RL will inject debate-specific
   system prompts. If SFT trains without any system messages, the format is novel at
   RL-time. Injecting system prompts during SFT means the model has seen the
   `system → user → assistant` pattern before RL begins.

3. **No "helpful assistant" identity seeding.** The injected prompts are quality-directing
   (precision, clarity, structure, evidence) — not identity-establishing ("you are a
   helpful assistant"). This avoids sycophancy priors that would confound debate-RL.

### The 8-dimension ontology

Every prompt is tagged with 1-3 dimensions from this canonical set:

| Dimension | What it directs |
|---|---|
| `analytic` | Trace reasoning, derive from first principles |
| `calibrated` | Express uncertainty proportional to evidence |
| `concise` | Minimize words, prioritize correctness over verbosity |
| `natural` | Conversational register, non-formulaic |
| `literal` | Match format/constraints exactly, no interpretation |
| `explanatory` | Show work, use examples, teach |
| `pragmatic` | Focus on usefulness, real-world applicability |
| `skeptical` | Challenge assumptions, verify, check edge cases |

### The pool (711 prompts)

- **Source**: 235 seed prompts (3 Sonnet agent batches covering chat/math/code/science/
  IF/algebra/GSM/bare-code/FLAN/chat/tables/cross-domain) → annotated by 8 dimensions →
  dimension-crossing to 720 → hard lint + fuzzy dedup + coherence filter → **711 final**
- **Artifact**: `tmp/system_prompts_final.json` (list of 711 strings)
- **Tags**: `tmp/system_prompts_expanded.json` (720 entries with per-prompt `tags` field,
  711 of which match the final pool 1:1)
- **Token stats**: median ~14 tokens, hard cap 32 tokens
- **Lint rules applied**: reject prompts containing `you are`, `assistant`, `AI`, `debate`,
  `math`, `python`, `code`, `step by step`, `bullet point`, `numbered list`, `chain of thought`
- **Coherence filter**: each prompt verified compatible with: single-word FLAN answers
  (`"yes"`), bare Python functions (`def main(): pass`), LaTeX math (`\boxed{42}`),
  science explanations, and casual chat
- **Quality gates**: gatekeeper approved (2 review rounds, 3 bugs found + fixed),
  auditor TRUST verdict (90/90 independent re-runs)

### Profile-weighted sampling

Six content profiles map HF config names → dimension weights:

| Profile | HF configs | Dimension weights |
|---|---|---|
| `chat` | wildchat, openassistant | `natural: 2.0, pragmatic: 1.0` |
| `math` | tulu-3-persona-math, openmathinstruct-2, tulu-3-persona-algebra | `analytic: 2.0, calibrated: 1.0` |
| `coding-bare` | dolci-python-algo | `concise: 2.0, literal: 1.0` |
| `coding-explained` | evol-codealpaca | `explanatory: 2.0, pragmatic: 1.0` |
| `precise-short` | dolci-precise-if, flan, tablegpt | `literal: 2.0, concise: 1.0` |
| `science` | dolci-openthoughts-sci, sciriff | `analytic: 1.0, explanatory: 1.0` |

Scoring function per prompt `p` given profile `c`:
```
w(p | c) = ε + exp( Σ_{tag ∈ tags(p)} θ[c, tag] )
```
where `ε = 0.01` ensures every prompt has non-zero probability. Effective N across
profiles: 273–502 (no degenerate distributions; top prompt never exceeds 1.2% of draws).

### Implementation

- **Injection point**: `src/prime_rl/trainer/sft/data.py` in `SFTDataset._process()`,
  BEFORE `build_incremental_token_mask` is called
- **Config field**: `SFTDataConfig.system_prompt_pool_path` (str, optional — if None,
  no injection)
- **Deterministic seeding**: `random.Random(sample_index)` per sample for reproducibility
  across training restarts
- **Multi-turn filter**: samples with `len(messages) > 2` (the 526 multi-turn OpenAssistant
  rows) are dropped before injection. These are <0.05% of the dataset and live where
  template prefix-invariance bugs surface.
- **Unit tests**: `tests/unit/train/sft/test_system_prompt_injection.py` (11 tests)
- **Initial verification (4 configs)**: 350/350 pass across all 7 models × 50 real
  Dolci samples (wildchat, tulu-3-persona-math, dolci-precise-if, flan) with
  injected system prompts through `build_incremental_token_mask`. Independently
  audited: 90/90 pass (TRUST verdict).
- **Broad verification (all 12 configs, 2026-04-16)**:
  `tmp/verify_sysprompt_injection_full.py` — 12 configs × 7 models × 50 single-turn
  + 25 OpenAssistant multi-turn × 7 models = **4,375 tests. 4,350 pass.** All 25
  failures are OLMo-3 × multi-turn, documented in §8 "Known multi-turn
  incompatibility". Single-turn is 100% (4200/4200) across every (model, config)
  pair. Report: `tmp/reports/verify_sysprompt_injection_full.json`.

### Template patches for system-prompt compatibility

Adding a system prompt changes the message format from `[user, assistant]` to
`[system, user, assistant]`. Two of 7 models needed template patches for this:

| Model | Issue | Patch |
|---|---|---|
| Trinity-Mini-Base | BPE merge: `\n` (230) vs `\n\n` (327) at generation-prompt boundary | `'\n' → '\n\n'` in `add_generation_prompt` branch |
| Nemotron-H-8B-Base-8K | Duplicate `<SPECIAL_11>Assistant\n` emission in user block + generation prompt | Template D: `<SPECIAL_11>Assistant\n` only as assistant turn header, not in user block |

Both patches are applied in `tmp/prepare_tokenizers.py` and saved to
`tmp/tokenizers/{slug}/`. Template D was discovered by the auditor during
independent verification — the original Template C had a duplication bug that
caused prefix invariant violations. Auditor caught, fixed, and verified it.

---

## 10. Unresolved / open items

1. **Hardcoded Data extrapolation** — only 2/69 rows inspected. Download shard 12, 13,
   or 14 to enumerate the remaining 67 and confirm no "I am Olmo" identity-anchoring.
2. **Dolci Tool Use pre-commit** — confirmed 0 rows in cached shards 0/7/10/11.
   Location (shard 12/13/14) never verified. Doesn't block the mix but worth noting.
3. **Auditor verdict** — pending completion. May adjust specific verdicts if
   discrepancies surface.
4. **Exact subsample sizes in materialized mix** — do we upload the full raw subsets
   and let prime-rl interleave with probabilities at load time, or pre-sample with
   the weights baked in? Former is cleaner (reproducible); latter is faster per step.
5. **Eval suite** — separately tracked. Primary: IFEval, MMLU 5-shot, sycophancy
   probes (Perez et al.), MT-Bench. Held-out: a debate-specific scenario from our
   own rubric. Decontamination pass required before first training run.
6. **Whether to include Qwen 3 8B base or Qwen 2.5 7B base** — depends on availability
   and tokenizer chat-template compatibility (both work with ChatML override).
7. **Which 6th model** (if any) — OLMo 3 32B for size-scaling signal is the default
   choice. Requires multi-node deployment.

---

## 11. Precommitted ladder (Rungs 0–9)

Each rung has explicit GO / NO-GO criteria. A failed rung halts forward progress;
we stop, diagnose, fix. The ladder is the contract.

### Rung 0 — Mix decision locked ✓ DONE

- **Exit**: Sections 4–6 written, auditor verdict TRUST (Section 9)
- Status: complete as of 2026-04-14

### Rung 1 — Dataset materialization ✓ DONE (2026-04-15)

- **Input**: Section 4 (12 keep sources + weights, after Hardcoded drop)
- **Deliverable**: `tmp/materialize_mix.py` — reads Dolci, filters by
  `source_dataset ∈ KEEP`, sanity-drops empty/1-turn rows, pushes one HF config
  per source to private HF repo `joanvelja/dolci-debate-sft-v1`
- **GO results**:
  - All 12 configs uploaded, every source row count matches decision doc at
    `delta 0.000%`
  - Sanity filter dropped 25 rows total (4 Wildchat, 13 Precise IF, 8
    OpenThoughts3+ Science — all genuinely empty assistant content)
  - Final kept: **1,158,724 rows** across 12 configs (vs expected 1,158,749)
  - Repo: 12 configs × 8 parquet shards each = 96 shards + README.md, total
    ~3.3 GB on HF
  - Round-trip verified: `load_dataset('joanvelja/dolci-debate-sft-v1', 'sciriff', split='train')`
    returns 4,557 rows with correct `{id, messages, domain}` schema and nested
    message struct intact
- **Notes**:
  - Initial run failed with `TypeError: HfApi.create_repo() got an unexpected
    keyword argument 'exists_ok'` — correct kwarg is `exist_ok` (singular).
    Fixed in script, re-ran successfully. Recorded here in case future audits
    ask "why was there a failed push before the good one".
  - Dolci HF cache persisted in `~/.cache/huggingface/hub/datasets--allenai--Dolci-Instruct-SFT/`
    (~3 GB) — safe to delete once we're past Rung 2/3 smoke validation.

### Rung 2 — Tokenizer inspection + template preparation (7-model scope)

- **Input**: the 7 locked candidate base models (Section 1)
- **Deliverable 1**: `tmp/inspect_tokenizers.py` diagnostic run against all 7
  candidates; output saved to `tmp/tokenizer_inspect_report.txt`. Per base:
  (a) tokenizer loads via `AutoTokenizer.from_pretrained` at prime-rl's pinned
  transformers commit `c1c3424`,
  (b) has `chat_template` or needs injection from instruct sibling,
  (c) native chat-structural tokens are single token IDs in vocab,
  (d) pad_token status (common base-model gotcha: `pad_token = None`),
  (e) rendered sample 2-turn output through the chosen template
- **Deliverable 2**: `tmp/prepare_tokenizers.py` that patches template and
  pad_token per base, saves to `tmp/tokenizers/{base_slug}/`. **No model embedding
  modification.** Handles the 3 "template already present" cases (marin, Qwen3-30B-A3B, rnj-1) as pass-through.
- **Deliverable 3**: golden tokenization test per base — one Dolci sample through
  prime-rl's `build_incremental_token_mask`, assert: loss_mask covers only
  assistant span, EOS present, chat-structural tokens correctly masked OUT of loss,
  round-trip stability, correct handling of the per-model chat template

- **Pre-committed per-model resolution** (subject to verification):

  | # | Base | Template source | Known risks |
  |---|---|---|---|
  | 1 | `marin-community/marin-8b-base` | use bundled chat_template | Marin config has minor rope_parameter warnings (beta_fast/beta_slow int vs float) — verify no functional impact |
  | 2 | `Qwen/Qwen3-30B-A3B-Base` | use bundled chat_template | Custom path — verify prime-rl's `qwen3_moe/` handles the chat template correctly during SFT |
  | 3 | `arcee-ai/Trinity-Mini-Base` | copy from `arcee-ai/Trinity-Mini` or `Trinity-Mini-Preview` sibling | Custom `AfmoeForCausalLM` arch — verify prime-rl's `afmoe/` custom impl path doesn't require `trust_remote_code=True`; test load-and-forward on a single sample |
  | 4 | `nvidia/Nemotron-H-8B-Base-8K` | **⚠ BLOCKER CANDIDATE**: base has no instruct sibling. Fallbacks: (a) hand-write ChatML template using Nemotron-H native special tokens, (b) borrow template from another NVIDIA instruct model if vocabs compatible, (c) drop model | If neither (a) nor (b) works → drop Nemotron-H from sweep, reduce to 6 models. **Resolve at Rung 2, not later.** |
  | 5 | `ByteDance-Seed/Seed-OSS-36B-Base-woSyn` | copy from `ByteDance-Seed/Seed-OSS-36B-Instruct` | HF fallback path — verify prime-rl's HF path handles the `SeedOssConfig` checkpoint conversion cleanly |
  | 6 | `allenai/Olmo-3-1025-7B` | copy from `allenai/Olmo-3-7B-Instruct` | Cleanest case — `<\|im_start\|>`=100264, `<\|im_end\|>`=100265 are pre-reserved in base vocab (verified) |
  | 7 | `EssentialAI/rnj-1` | use bundled chat_template | Verify the bundled template handles Dolci's message formats without weirdness; note: base has 256 special tokens and ships with chat infrastructure despite being pre-SFT |

- **GO**: all 7 (or 6 if Nemotron-H is dropped) pass golden test with their native
  (or injected from instruct sibling) chat template
- **NO-GO response**:
  - If a base's template uses a token that decomposes in the base vocab → either
    drop that base from the sweep or take the one-off embedding-surgery cost for
    it (document the decision)
  - If Nemotron-H cannot be given a working chat template via any fallback → drop
    from sweep, reduce to 6 models, log the decision
  - If Trinity-Mini-Base's `AfmoeForCausalLM` requires `trust_remote_code` AND
    prime-rl's custom path also requires it → flag as blocker, investigate
  - If golden test fails → debug upstream in `build_incremental_token_mask`, not bandaid

### Rung 3 — SFT configs + single-shot smoke run

- **Input**: Rung 1 dataset + Rung 2 tokenizers for all 7 bases
- **Deliverables (landed 2026-04-16/17)**:
  - `configs/sft/baseline.toml` — Tülu-3-derived hparams (5e-6 LR, β₂=0.95
    ported from OLMo-core, linear WSD, 4096 seq_len, 2 epochs, batch 128 via
    grad_accum 16 on 8 GPUs, `liger_fused` CE, `first_exhausted` interleave,
    system-prompt injection via `tmp/system_prompts_final.json`)
  - `configs/sft/overrides/{marin,qwen3_30b_a3b,trinity_mini,nemotron_h_8b,seed_oss_36b,olmo3,rnj_1}.toml`
    — per-base overrides. All 7 validate against `SFTConfig`.
  - `tmp/swa_average.py` — post-hoc stochastic weight averaging (emulates
    OLMo-core's `ModelMergeCallback`; run after each base finishes)
  - Launch pattern: `uv run sft @ configs/sft/baseline.toml @ configs/sft/overrides/<base>.toml`

- **Per-base LR/epoch matrix** (derived in §7, §7b, §7c):

  | Base | LR | Epochs | max_steps | Notes |
  |---|---|---|---|---|
  | marin-8b | 5e-6 | 2 | 13,188 | baseline |
  | Qwen3-30B-A3B | 5e-6 | 2 | 13,188 | MoE, active-param scaling considered but not applied |
  | Trinity-Mini | 5e-6 | 2 | 13,188 | MoE, same rationale |
  | Nemotron-H-8B | 5e-6 | 2 | 13,188 | Mamba hybrid — monitor first 500 steps |
  | Seed-OSS-36B | 3e-6 | 2 | 13,188 | √N interp (Tülu 5e-6@8B, 2e-6@70B) |
  | Olmo-3-7B | 1e-5 | 3 | 19,782 | OLMo-core-flavored override |
  | rnj-1 | 5e-6 | 2 | 13,188 | ships chat_template on base |

- **GO** (per base, evaluated on 500-step smoke before committing to full run):
  1. Run completes without crash
  2. Loss descends from ~2.5 → ~1.5 over 500 steps
  3. 10 IFEval-like prompts produce readable multi-paragraph responses (no
     degenerate output)
  4. Sample response length distribution is 200–1000 tokens (not collapsed to
     either extreme)
- **NO-GO response**: debug config. Loss not descending → LR wrong or template
  wrong. Degenerate outputs → loss mask wrong or packing broken. For Olmo-3-7B
  specifically, if LR 1e-5 shows instability, fall back to 5e-6 and log the
  revision (the β₂ and batch scaling justify 1e-5 but the midtrain residual is
  handwaved).

### Rung 4 — Eval scaffold wired + datasets downloaded + judge frozen

- **Input**: smoke-run checkpoint from Rung 3
- **Deliverable**: eval pipeline running all Section 10a primary evals end-to-end
  on the smoke checkpoint; structured JSON output; judge model frozen at
  `claude-sonnet-4-6` and recorded in this doc
- **GO**: eval produces numbers for the smoke checkpoint in reasonable ranges
  (IFEval strict ≥ 0.3 for a 500-step smoke — real threshold at Rung 6)
- **NO-GO response**: fix scaffold. Eval pipeline has to work before we burn a
  full sweep.

### Rung 5 — Decontamination pass

- **Input**: materialized training dataset + Section 10a decon target set
- **Deliverable**: n-gram (8-gram) overlap report using
  `tmp/open-instruct/decontamination/`
- **GO**: zero eval-prompt 8-grams appear in training data
- **NO-GO response**: drop any contaminated training rows, re-materialize,
  re-run Rung 5

### Rung 6 — Precommit acceptance thresholds (FROZEN before sweep)

- **Input**: all prior rungs green + baseline BOTEC
- **Deliverable**: acceptance thresholds written into this doc BEFORE launching
  the sweep. **Committed before results are seen. Never revised post-hoc.**
- **Proposed thresholds** (revise only before Rung 7 launch, log any revision):
  - **IFEval strict**: ≥ 0.55 for all 6 models
  - **MMLU 5-shot**: ≤ 5pp drop from each base model's own pre-SFT MMLU baseline
  - **Sharma sycophancy-eval**: ≤ 0.55 (lower is better; base ~0.4, RLHF'd ~0.8)
  - **MT-Bench**: ≥ 3.0 on `claude-sonnet-4-6` judge
  - **Length sanity**: median response on held-out chat prompts 200–600 tokens
  - **No regression across seeds** (if we run > 1 seed)
- **GO**: thresholds committed in writing, no ongoing edits
- **NO-GO response**: if thresholds feel unattainable after smoke run, revise
  DOWN before launch and log the revision explicitly. Never silently.

### Rung 7 — Full sweep launched

- **Input**: Rungs 1–6 green
- **Deliverable**: 6 parallel SFT runs (or sequential if compute is tight — decide
  before launch, not during)
- **GO**: all 6 start, wandb logging healthy, first 1000 steps show normal loss
  curves
- **NO-GO response**: kill early if a run diverges. Don't "wait and see" past
  step 2000.

### Rung 8 — Post-SFT eval pass

- **Input**: 6 final SFT checkpoints
- **Deliverable**: eval table for all 6 checkpoints against Rung 6 thresholds.
  Held-out sets (Section 10a) run ONCE here.
- **GO**: all 6 meet ALL committed thresholds from Rung 6
- **NO-GO response**: ≥2 models fail same threshold → mix or hparams issue,
  re-diagnose. 1 model fails → decide whether to exclude or fix.

### Rung 9 — Frozen ratchet

- **Input**: Rung 8 passed
- **Deliverable**: 6 checkpoints tagged `debate-sft-v1-<base>`, pushed to private
  HF hub or persistent storage, paths logged here
- **Exit**: checkpoints are immutable starting points for debate-RL + RLVR branches.
  Any "we'd rather tweak the SFT" after this point means a **new ladder run**, not
  in-place modification.

### Compute BOTEC

Per 8B model on 8× H100:
- ~844K effective samples × 2 epochs × mean ~500 tokens ≈ **0.84B training tokens**
- At ~40K tok/s node-level → ~21,000 s ≈ **~5.9 h per model**
- × 6 models = **~35 node-hours** core compute
- Plus smoke + evals + decon: **~50 node-hours end-to-end**

Parallel across 6 nodes → ~6 wall hours. Sequential on one node → ~35 wall hours
(~1.5 days).

### Precommitment anchor

What makes this a precommitment, not a wishlist:
1. **Acceptance thresholds (Rung 6) are frozen before the sweep burns compute.**
   No post-hoc revision.
2. **NO-GO at any rung halts forward progress.** No "we'll fix it in the next rung."
3. **The ladder is the contract.** Changes require an explicit amendment to
   this section, not silent drift.

---

## 12. Next steps

**Historical** — this section was the first-draft action list before the
precommitted ladder in §11 was written. Items 1-3 landed in Rungs 1-3; items
4-5 map to Rungs 4-7. Kept here for provenance only. **The ladder in §11 is
the live plan.**

~~1. **Materialize the mix**~~ — done, Rung 1 (see §11).

~~2. **Write prime-rl SFT config** (`configs/sft/debate_baseline.toml`)~~ — done,
Rung 3. Final artifacts: `configs/sft/baseline.toml` + `configs/sft/overrides/*.toml`
(no `debate_` prefix per user preference).

~~3. **Write ChatML injector utility** (`tmp/chatml_inject.py`)~~ — superseded by
Option B (native per-base templates); §8. Delivered as `tmp/prepare_tokenizers.py`
at Rung 2.

**Remaining**: Rungs 3-smoke → 4 (eval scaffold) → 5 (decontamination) → 6
(threshold precommitment) → 7 (full 7-base sweep) → 8 (post-SFT eval) → 9
(frozen ratchet). See §11.

6. **Eval all SFT checkpoints** on the same eval suite before proceeding to RL branches.

7. **Two RL branches from each SFT checkpoint**: debate-RL + RLVR control.

---

## Appendix A: Scripts used

- `tmp/dolci_vr_inspect.py` — initial VR inspection (Verifiable Reasoning)
- `tmp/dolci_src_inspect.py` — parameterized inspection with shard probe
- `tmp/dolci_local_inspect.py` — vectorized polars filter against cached shards (the
  canonical script used by all 20 worker analysts)
- `tmp/open-instruct/` — cloned at commit `b5dd562` for reference (not used for training)

## Appendix B: Key data quality facts

- Dolci sum check: all 22 source counts sum to exactly **2,152,112**. ✓
- Sub-domain decomposition derived by arithmetic:
  - `Chat` = Wildchat (302,406) + OpenAssistant (7,132) = 309,538 ✓
  - `Math` = Persona MATH + OpenMath2 + Persona GSM + Persona Algebra = 269,937 ✓
  - `Coding` = Evol CodeAlpaca + Persona Python + Python Algorithms = 328,614 ✓
  - `Safety` = CoCoNot + WildGuardMix + WildJailbreak = 110,295 ✓
  - `Science` = SciRiff + Dolci OpenThoughts3+ Science = 103,825 ✓
  - `Other` = Logic Puzzles + FLAN + TableGPT = 254,863 ✓
  - `Reasoning` = Verifiable Reasoning (sole source) = 310,572 ✓
- **README incompleteness**: the HF dataset card does not list `OpenMathInstruct 2`
  (50,000 rows) but it is present in the data. Don't trust dataset cards blindly.
- **OpenThoughts 3 reality check**: the README says "941,166 total prompts, reasoning
  traces removed for instruct, 99,268 prompts." The 99,268 is what's in
  Dolci-Instruct-SFT — with traces removed. It is NOT CoT-heavy (my initial misread).
