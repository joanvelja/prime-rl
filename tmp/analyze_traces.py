"""Stream-parse torch.profiler chrome traces, aggregate kernel time per config.

Loads rank 0 of each prof-* run (enough signal for intra-rank kernel breakdown).
Filters to steady-state steps (skip warmup + last fragment).
"""
import gzip, json, re
from collections import defaultdict
from pathlib import Path

CONFIGS = {
    "fa2-nc":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-nc/trace/trace_0.json.gz",
    "fa2-c":   "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-c/trace/trace_0.json.gz",
    "fa3-nc":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-nc/trace/trace_0.json.gz",
    "fa3-c":   "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-c/trace/trace_0.json.gz",
    "maxmfu":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-maxmfu/trace/trace_0.json.gz",
}

# Category buckets (match on lowercase name substrings)
BUCKETS = [
    ("flash_attn",      ["flash_attn", "flashattn", "mha_fwd", "mha_bwd", "_flashattention"]),
    ("gemm",            ["gemm", "cutlass", "nn.linear", "addmm", "matmul", "bmm", "mm_out"]),
    ("rmsnorm",         ["rmsnorm", "rms_norm", "quack"]),
    ("rope",            ["rope", "apply_rotary", "rotate_half"]),
    ("liger_ce",        ["liger", "cross_entropy", "fused_linear"]),
    ("activation",      ["silu", "gelu", "relu", "softmax"]),
    ("fsdp_collective", ["allgather", "all_gather", "reducescatter", "reduce_scatter", "ncclkernel", "ncclalltoall", "allreduce", "all_reduce", "ncclallgather", "ncclreducescatter"]),
    ("elementwise",     ["add", "mul", "div", "sub", "copy_", "cast", "to_", "_foreach", "fill", "zero", "clamp"]),
    ("memcpy",          ["memcpy", "memset", "dtoh", "htod"]),
    ("optimizer",       ["adamw", "fused_adamw"]),
    ("autograd_engine", ["autograd", "backward"]),
]

def bucket_of(name):
    nl = name.lower()
    for bkt, pats in BUCKETS:
        if any(p in nl for p in pats):
            return bkt
    return "other"

def analyze(path, label):
    print(f"\n=== {label}  ({path}) ===", flush=True)
    try:
        with gzip.open(path, "rb") as fh:
            data = json.loads(fh.read())
    except Exception as e:
        print(f"  FAIL: {e}")
        return None
    events = data.get("traceEvents", [])
    print(f"  events: {len(events):,}")

    # Aggregate GPU-side kernels (ph=X, cat contains "kernel" or "gpu"-ish)
    # and CPU ops (ph=X, cat="cpu_op") separately.
    gpu_by_bucket = defaultdict(lambda: [0.0, 0])  # [total_us, count]
    gpu_by_name   = defaultdict(lambda: [0.0, 0])
    cpu_by_bucket = defaultdict(lambda: [0.0, 0])
    total_gpu_us = 0.0
    total_cpu_us = 0.0

    for ev in events:
        if ev.get("ph") != "X": continue
        dur = ev.get("dur", 0)
        if dur <= 0: continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if "kernel" in cat or cat == "gpu_memcpy" or cat == "gpu_memset":
            b = bucket_of(name)
            gpu_by_bucket[b][0] += dur; gpu_by_bucket[b][1] += 1
            gpu_by_name[name][0] += dur; gpu_by_name[name][1] += 1
            total_gpu_us += dur
        elif cat == "cpu_op":
            b = bucket_of(name)
            cpu_by_bucket[b][0] += dur; cpu_by_bucket[b][1] += 1
            total_cpu_us += dur

    print(f"  total GPU kernel time: {total_gpu_us/1e6:.2f} s")
    print(f"  total CPU op time:     {total_cpu_us/1e6:.2f} s")
    print()
    print(f"  {'bucket':<20} | {'gpu_ms':>10} | {'gpu_%':>6} | {'calls':>8}")
    print("  " + "-"*60)
    for bkt, (us, n) in sorted(gpu_by_bucket.items(), key=lambda kv: -kv[1][0]):
        pct = 100 * us / total_gpu_us if total_gpu_us else 0
        print(f"  {bkt:<20} | {us/1e3:>10.1f} | {pct:>5.1f}% | {n:>8,}")

    # Top-10 individual kernels by time
    print(f"\n  top 10 individual GPU kernels:")
    for name, (us, n) in sorted(gpu_by_name.items(), key=lambda kv: -kv[1][0])[:10]:
        short = name[:80]
        print(f"    {us/1e3:>9.1f} ms  ({n:>5} calls, avg {us/max(n,1):.1f} us)  {short}")

    return {
        "gpu_total_us": total_gpu_us,
        "cpu_total_us": total_cpu_us,
        "by_bucket": {k: v[0] for k, v in gpu_by_bucket.items()},
        "call_counts": {k: v[1] for k, v in gpu_by_bucket.items()},
    }

results = {}
for label, path in CONFIGS.items():
    if not Path(path).exists():
        print(f"[skip] {label}: trace missing")
        continue
    results[label] = analyze(path, label)

# Cross-config diff table
print("\n\n=== Cross-config diff (GPU kernel time in ms, rank 0) ===")
labels = list(results.keys())
buckets = sorted({b for r in results.values() if r for b in r["by_bucket"]})
print(f"  {'bucket':<20} | " + " | ".join(f"{l:>10}" for l in labels))
print("  " + "-"*(22 + len(labels)*13))
for b in buckets:
    row = []
    for l in labels:
        ms = results[l]["by_bucket"].get(b, 0) / 1e3
        row.append(f"{ms:>10.1f}")
    print(f"  {b:<20} | " + " | ".join(row))

print(f"\n  {'TOTAL GPU (ms)':<20} | " + " | ".join(f"{results[l]['gpu_total_us']/1e3:>10.1f}" for l in labels))
print(f"  {'TOTAL CPU (ms)':<20} | " + " | ".join(f"{results[l]['cpu_total_us']/1e3:>10.1f}" for l in labels))
