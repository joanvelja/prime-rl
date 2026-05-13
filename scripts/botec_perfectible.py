#!/usr/bin/env python3
"""Parametric BOTEC for vLLM decode throughput on perfectible-subset models.

Reads each model's config.json from $HF_HUB_CACHE, computes architectural BOTEC
quantities, sweeps over batch size, and reports predicted throughput.

Adversarial: this version explicitly notes simplifications, sanity-checks against
the in-flight run (gemma-4-E4B at B≈9), and flags places where the back-of-envelope
breaks down.

Usage:
    uv run --no-sync python scripts/botec_perfectible.py [--n-gpus N] [--util U]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────────
# Hardware (parametric; defaults are Isambard GH200)
# ────────────────────────────────────────────────────────────────────────────────
HBM_BW_GB_PER_S = 4000.0  # GH200 HBM3e ~4 TB/s
GPU_MEM_GIB = 96.0
BF16_TFLOPS_PEAK = 990.0
BF16_TFLOPS_SUSTAINED = 700.0  # ~70% of marketing peak under realistic conditions
DEFAULT_N_GPUS = 28          # 7 nodes × 4 GPUs (post-EXCLUDE_LOCAL)
DEFAULT_UTIL = 0.95          # gpu_memory_utilization
DEFAULT_VLLM_EFF = 0.38      # observed from run 1 (960 tok/s/GPU vs 2.5K mem-BW peak)
# Block size affects KV padding; 16-block at high seqlen has ≤1% waste — ignore.
BLOCK_SIZE = 16

# ────────────────────────────────────────────────────────────────────────────────
# Model architecture, parsed from config.json
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelArch:
    name: str
    snapshot: Path
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int           # for SWA layers (or all if homogeneous)
    head_dim_swa: int                  # head_dim for SWA (or all-global) layers
    head_dim_global: Optional[int]     # head_dim for global layers if heterogeneous
    num_kv_heads_global: Optional[int] # KV heads for global layers if heterogeneous
    num_swa_layers: int
    num_global_layers: int
    sliding_window: int                # 0 if no SWA
    intermediate_size: int             # dense FFN size (or shared in MoE)
    vocab_size: int
    weight_bytes: int                  # bf16 RUNTIME weight memory
    num_kv_shared_layers: int = 0      # gemma-4-E4B: 18 SWA layers share KV among themselves
    is_moe: bool = False
    moe_num_experts: int = 0
    moe_top_k: int = 0
    moe_intermediate_size: int = 0
    fp8_kv_supported: bool = True      # blocked on Gemma 4 family (head_dim=512)
    notes: list[str] = field(default_factory=list)


def _runtime_weight_bytes_bf16(text_cfg: dict, full_cfg: dict, idx_meta: dict | None) -> int:
    """Compute the bf16 RUNTIME weight memory.

    Architectural params × 2 bytes — independent of how safetensors are stored on disk.
    (Some models like rnj-1-instruct are stored fp32, so total_size from the index would
    be 2× the actual runtime bf16 footprint.)

    Falls back to total_size/2 if architectural param count derivation fails. ALWAYS check
    against model card's "X B parameters" headline if anything looks fishy.
    """
    # Try total_size first: if storage matches param count × 2, it's bf16-stored
    # If storage matches param count × 4, it's fp32-stored → divide by 2 for bf16 runtime
    if idx_meta is not None:
        total = idx_meta.get("total_size", 0)
        # Heuristic: derive param count from architecture and check storage
        L = text_cfg.get("num_hidden_layers", 0)
        H = text_cfg.get("hidden_size", 0)
        V = text_cfg.get("vocab_size", full_cfg.get("vocab_size", 0))
        I = text_cfg.get("intermediate_size", 0)
        Hq = text_cfg.get("num_attention_heads", 0)
        Hkv = text_cfg.get("num_key_value_heads", Hq)
        d = text_cfg.get("head_dim", H // Hq if Hq else 0)
        # Crude param count: embedding + L × (attention + FFN)
        # Attention per layer: Q (H × Hq × d) + K (H × Hkv × d) + V (H × Hkv × d) + O (Hq × d × H)
        att = 2 * H * Hq * d + 2 * H * Hkv * d
        ffn = 3 * H * I  # SwiGLU 3 matrices (or gemma's GeGLU same shape)
        per_layer = att + ffn
        emb = V * H  # tied embeddings; if not tied, also output head (2×V×H)
        params_est = emb + L * per_layer
        # If total_size ≈ 4 × params_est → fp32 stored → runtime bf16 = total/2
        # If total_size ≈ 2 × params_est → bf16 stored → runtime bf16 = total
        if params_est > 0 and total > 0:
            ratio = total / (params_est * 2)
            if 1.5 < ratio < 2.5:
                # Looks fp32-stored: total_size is 2× the bf16 runtime
                return total // 2
        return total
    # No safetensors index: fall back to file size sum (less reliable)
    return 0


def parse_config(model_name: str, snapshot: Path) -> ModelArch:
    """Read config.json and safetensors index, return ModelArch."""
    cfg_path = snapshot / "config.json"
    cfg = json.loads(cfg_path.read_text())
    # Some models nest under text_config (multimodal like Gemma 4)
    text_cfg = cfg.get("text_config", cfg)

    # Weight bytes from architecture (more reliable than safetensors total_size,
    # which can be fp32-stored for some models like rnj-1-instruct)
    idx_path = snapshot / "model.safetensors.index.json"
    idx_meta = None
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        idx_meta = idx.get("metadata", {})
    weight_bytes = _runtime_weight_bytes_bf16(text_cfg, cfg, idx_meta)
    if weight_bytes == 0:
        # Fallback: single-file safetensors
        sf_files = list(snapshot.glob("model*.safetensors"))
        weight_bytes = sum(f.stat().st_size for f in sf_files)

    L = text_cfg["num_hidden_layers"]
    H = text_cfg["hidden_size"]
    Hq = text_cfg["num_attention_heads"]
    Hkv = text_cfg.get("num_key_value_heads", Hq)
    head_dim = text_cfg.get("head_dim", H // Hq)
    # Heterogeneous head_dim (Gemma 4)
    head_dim_global = text_cfg.get("global_head_dim")
    Hkv_global = text_cfg.get("num_global_key_value_heads")

    sw = text_cfg.get("sliding_window", 0) or 0
    layer_types = text_cfg.get("layer_types") or []
    if layer_types:
        n_swa = sum(1 for lt in layer_types if "slid" in lt)
        n_glb = sum(1 for lt in layer_types if "full" in lt or "global" in lt)
    elif text_cfg.get("sliding_window_pattern", 1) == 1 and head_dim_global is None:
        # Either all-SWA or all-full; check if sliding_window matters
        # rnj-1: layer_type all full, sliding_window=32768 inert
        # If we have evidence the model is all-global despite sliding_window field, treat that way
        n_swa, n_glb = 0, L
    else:
        # Gemma 4: 5 SWA : 1 global (or 6:1 for E4B per agent)
        # Conservatively, if pattern not in config, count from layer_types or default
        pattern = text_cfg.get("sliding_window_pattern", 6)
        if pattern > 1:
            n_glb = L // pattern
            n_swa = L - n_glb
        else:
            n_swa, n_glb = 0, L

    is_moe = bool(text_cfg.get("enable_moe_block") or text_cfg.get("num_local_experts") or text_cfg.get("num_experts"))
    moe_kw = {}
    if is_moe:
        moe_kw["moe_num_experts"] = text_cfg.get("num_experts", text_cfg.get("num_local_experts", 0))
        moe_kw["moe_top_k"] = text_cfg.get("num_experts_per_tok", text_cfg.get("top_k_experts", 0))
        moe_kw["moe_intermediate_size"] = text_cfg.get("moe_intermediate_size", text_cfg.get("expert_intermediate_size", 0))

    # KV sharing (Gemma 4)
    n_kv_shared = text_cfg.get("num_kv_shared_layers", 0)

    # FP8 KV blocked if any head_dim > 256 (FlashInfer/FA3 cap)
    max_hd = max(head_dim, head_dim_global or 0)
    fp8_blocked = max_hd > 256

    return ModelArch(
        name=model_name,
        snapshot=snapshot,
        num_hidden_layers=L,
        hidden_size=H,
        num_attention_heads=Hq,
        num_key_value_heads=Hkv,
        head_dim_swa=head_dim,
        head_dim_global=head_dim_global,
        num_kv_heads_global=Hkv_global,
        num_swa_layers=n_swa,
        num_global_layers=n_glb,
        sliding_window=sw,
        intermediate_size=text_cfg.get("intermediate_size", 0),
        vocab_size=text_cfg.get("vocab_size", cfg.get("vocab_size", 0)),
        weight_bytes=weight_bytes,
        num_kv_shared_layers=n_kv_shared,
        is_moe=is_moe,
        fp8_kv_supported=not fp8_blocked,
        **moe_kw,
    )


# ────────────────────────────────────────────────────────────────────────────────
# KV-cache math
# ────────────────────────────────────────────────────────────────────────────────
def kv_bytes_per_seq(arch: ModelArch, seq_len: int, dtype_bytes: int = 2,
                     num_kv_shared_layers: int = 0) -> int:
    """Bytes of KV cache held by ONE sequence at decode position seq_len.

    SWA layers cap at sliding_window tokens; global layers grow linearly.
    Accounts for heterogeneous head_dim and KV heads (Gemma 4).
    Accounts for cross-layer KV sharing (gemma-4-E4B has num_kv_shared_layers=18 SWA layers
    that share KV among themselves; effectively reduces SWA layer count for KV calculation).
    """
    # SWA contribution (subtracting shared layers since they re-use other layers' KV)
    n_swa_unique = max(arch.num_swa_layers - num_kv_shared_layers, 0)
    swa_tokens = min(seq_len, arch.sliding_window) if arch.sliding_window > 0 else seq_len
    swa_per_layer = 2 * arch.num_key_value_heads * arch.head_dim_swa * dtype_bytes * swa_tokens
    swa_total = swa_per_layer * n_swa_unique

    # Global contribution
    hkv_g = arch.num_kv_heads_global if arch.num_kv_heads_global is not None else arch.num_key_value_heads
    hd_g = arch.head_dim_global if arch.head_dim_global is not None else arch.head_dim_swa
    glb_per_layer = 2 * hkv_g * hd_g * dtype_bytes * seq_len
    glb_total = glb_per_layer * arch.num_global_layers

    return swa_total + glb_total


def gib(b: int) -> float:
    return b / (1024 ** 3)


# ────────────────────────────────────────────────────────────────────────────────
# Throughput math
# ────────────────────────────────────────────────────────────────────────────────
def mem_bw_peak_tok_per_s(arch: ModelArch, batch: int, seq_len: int = 2000,
                         kv_fp8: bool = False, hbm_bw_gbs: float = HBM_BW_GB_PER_S) -> float:
    """Memory-bandwidth-bound peak tokens/sec/GPU at given batch size.

    Formula: per forward pass, GPU reads weights (W_bf bytes) + KV for all batched sequences.
    Per forward: produces `batch` new tokens (one per sequence).

      tok/s_peak = batch * HBM_BW / (W_bf + batch * k_per_seq)

    For MoE: only "active" weight bytes count per forward (not all 128 experts).
    This is an ADVERSARIAL choice — assumes perfect expert routing locality.
    Realistic MoE adds all-to-all comm overhead not modeled here.
    """
    dtype = 1 if kv_fp8 else 2
    k = kv_bytes_per_seq(arch, seq_len, dtype, num_kv_shared_layers=arch.num_kv_shared_layers)
    if arch.is_moe:
        # Active weights per token: attention + shared_FFN + top_k expert FFN + embeddings + router
        # Rough approximation: 7.15 GiB for gemma-4-26B-A4B-it (matches agent's number).
        # For other MoE, would need to compute. For now, hardcode if MoE.
        if "26b-a4b" in arch.name.lower():
            W_active = int(7.15 * 1024 ** 3)
        else:
            W_active = arch.weight_bytes // 4  # crude fallback
        W = W_active
    else:
        W = arch.weight_bytes

    return (batch * hbm_bw_gbs * 1e9) / (W + batch * k)


def crossover_batch(arch: ModelArch, seq_len: int = 2000, kv_fp8: bool = False) -> float:
    """B* where added KV-read cost ≈ weight-read cost."""
    dtype = 1 if kv_fp8 else 2
    k = kv_bytes_per_seq(arch, seq_len, dtype, num_kv_shared_layers=arch.num_kv_shared_layers)
    if arch.is_moe and "26b-a4b" in arch.name.lower():
        W = int(7.15 * 1024 ** 3)
    else:
        W = arch.weight_bytes
    if k == 0:
        return float('inf')
    return W / k


def kv_budget_max_seqs(arch: ModelArch, gpu_mem_gib: float, util: float,
                       seq_len: int, kv_fp8: bool, ep_shard: int = 1) -> float:
    """How many concurrent sequences fit in GPU KV budget at given sequence length."""
    weight_gib = gib(arch.weight_bytes) / ep_shard
    free_gib = gpu_mem_gib * util - weight_gib
    if free_gib <= 0:
        return 0.0
    dtype = 1 if kv_fp8 else 2
    k_gib = gib(kv_bytes_per_seq(arch, seq_len, dtype, num_kv_shared_layers=arch.num_kv_shared_layers))
    if k_gib == 0:
        return float('inf')
    return free_gib / k_gib


# ────────────────────────────────────────────────────────────────────────────────
# Main: sweep & report
# ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gpus", type=int, default=DEFAULT_N_GPUS,
                    help="Total GPUs in service (e.g. 28=7nodes×4, 32=8×4)")
    ap.add_argument("--util", type=float, default=DEFAULT_UTIL,
                    help="gpu_memory_utilization")
    ap.add_argument("--vllm-eff", type=float, default=DEFAULT_VLLM_EFF,
                    help="vLLM efficiency factor (observed 0.38 from run 1)")
    ap.add_argument("--seq-len", type=int, default=2000,
                    help="Median actual generated tokens (run 1 measured p50=2041)")
    ap.add_argument("--max-num-seqs", type=int, default=192,
                    help="per-engine vLLM cap (current config value)")
    ap.add_argument("--n-dp", type=int, default=4, help="data_parallel_size_local")
    args = ap.parse_args()

    hf_cache = os.environ.get("HF_HUB_CACHE", "/projects/a6r/joanv.a6r/tmp/hf-hub-cache")
    targets = [
        ("gemma-4-E4B-it", "models--google--gemma-4-E4B-it"),
        ("OLMo-3-7B-Instruct-DPO", "models--allenai--Olmo-3-7B-Instruct-DPO"),
        ("rnj-1-instruct", "models--EssentialAI--rnj-1-instruct"),
        ("gemma-4-26B-A4B-it", "models--google--gemma-4-26B-A4B-it"),
    ]

    print(f"\n{'='*80}\nHARDWARE: HBM={HBM_BW_GB_PER_S} GB/s, GPU mem={GPU_MEM_GIB} GiB, η={args.vllm_eff}")
    print(f"DEPLOYMENT: N_GPU={args.n_gpus}, gpu_mem_util={args.util}, max_num_seqs={args.max_num_seqs}, seq_len={args.seq_len}")
    print(f"{'='*80}\n")

    # Sweep max_concurrency values
    sweep = [256, 512, 1024, 2048, 3072, 4096, 5376, 8192]

    for name, dirname in targets:
        snapshots = list(Path(hf_cache, dirname, "snapshots").glob("*"))
        if not snapshots:
            print(f"⚠ {name}: no snapshot at {hf_cache}/{dirname}\n")
            continue
        arch = parse_config(name, snapshots[0])

        # FP8 KV decision
        kv_fp8 = arch.fp8_kv_supported

        # EP sharding for MoE
        ep_shard = args.n_dp if arch.is_moe else 1

        print(f"\n{'─'*80}\n{name.upper()}")
        print(f"  Layers: {arch.num_hidden_layers} ({arch.num_swa_layers} SWA d={arch.head_dim_swa} window={arch.sliding_window}, "
              f"{arch.num_global_layers} global d={arch.head_dim_global or arch.head_dim_swa})")
        print(f"  Attention: Hq={arch.num_attention_heads}, Hkv={arch.num_key_value_heads}"
              + (f"(global Hkv={arch.num_kv_heads_global})" if arch.num_kv_heads_global else ""))
        print(f"  Weight: {gib(arch.weight_bytes):.2f} GiB bf16"
              + (f" (per EP={ep_shard} rank: ~{gib(arch.weight_bytes)/ep_shard:.2f} GiB)" if ep_shard > 1 else ""))
        print(f"  MoE: {arch.is_moe}"
              + (f" (E={arch.moe_num_experts}, top_k={arch.moe_top_k})" if arch.is_moe else ""))
        print(f"  FP8 KV: {'✓' if kv_fp8 else '? unvalidated (head_dim>256; vLLM may force TRITON_ATTN)'}")

        # KV-per-seq at the requested seq_len, both dtypes — with KV-sharing correction
        k_bf16_gib = gib(kv_bytes_per_seq(arch, args.seq_len, 2, num_kv_shared_layers=arch.num_kv_shared_layers))
        k_fp8_gib = gib(kv_bytes_per_seq(arch, args.seq_len, 1, num_kv_shared_layers=arch.num_kv_shared_layers))
        kv_share_note = f" (KV-share: {arch.num_kv_shared_layers} SWA layers share)" if arch.num_kv_shared_layers else ""
        print(f"  KV/seq @ L={args.seq_len}: {k_bf16_gib*1024:.1f} MiB bf16 / {k_fp8_gib*1024:.1f} MiB FP8{kv_share_note}")

        # KV budget (per engine, with current EP)
        budget_bf16 = kv_budget_max_seqs(arch, GPU_MEM_GIB, args.util, args.seq_len, False, ep_shard)
        budget_fp8 = kv_budget_max_seqs(arch, GPU_MEM_GIB, args.util, args.seq_len, True, ep_shard)
        print(f"  KV budget: {budget_bf16:.0f} seqs bf16 / {budget_fp8:.0f} seqs FP8 (per engine)")

        # Crossover batch
        bstar_bf16 = crossover_batch(arch, args.seq_len, False)
        bstar_fp8 = crossover_batch(arch, args.seq_len, True)
        print(f"  Crossover B* (mem-BW peak point): {bstar_bf16:.0f} bf16 / {bstar_fp8:.0f} FP8")

        # Saturation max_concurrency (to fill max_num_seqs ceiling)
        sat_mc = args.max_num_seqs * args.n_gpus
        print(f"  Saturation max_concurrency (to fill max_num_seqs={args.max_num_seqs}): {sat_mc}")
        print()

        # Sweep table
        kv_label = "FP8" if kv_fp8 else "bf16"
        print(f"  {'max_conc':>9} {'B/GPU':>7} {'KV/eng':>9} {'mem-BW peak':>14} {'predicted':>12} {'regime':>20}")
        print(f"  {'─'*9} {'─'*7} {'─'*9} {'─'*14} {'─'*12} {'─'*20}")
        for mc in sweep:
            B = mc / args.n_gpus
            # KV per engine = min(B, max_num_seqs) × k
            B_eng = min(B, args.max_num_seqs)
            kv_eng = B_eng * (k_fp8_gib if kv_fp8 else k_bf16_gib)
            peak = mem_bw_peak_tok_per_s(arch, int(B), args.seq_len, kv_fp8)
            predicted = peak * args.vllm_eff
            bstar = bstar_fp8 if kv_fp8 else bstar_bf16
            budget = budget_fp8 if kv_fp8 else budget_bf16
            if B_eng > budget:
                regime = "❌ KV OOM"
            elif B > bstar:
                regime = "⚠ past crossover"
            elif B_eng >= args.max_num_seqs:
                regime = "⚠ at ceiling"
            else:
                regime = "✓ linear-ish"
            print(f"  {mc:>9} {B:>7.1f} {kv_eng:>7.1f} GiB {peak:>10.0f}/s   {predicted:>8.0f}/s   {regime:>20}")

        # Notes
        if arch.is_moe:
            print(f"\n  ⚠ MoE: BOTEC ignores all-to-all comm overhead. Real throughput at high B")
            print(f"    will be 30-50% lower than predicted. Use ~half the predicted speedup.")
        print()


if __name__ == "__main__":
    main()
