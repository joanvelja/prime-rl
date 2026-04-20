"""For each AG launch (cudaLaunchKernel CPU op named ncclKernel...) measure cpu_ts -> gpu_kernel_ts gap."""
import gzip, ijson, sys
from collections import defaultdict
import statistics

CONFIGS = {
    "fa3-nc": "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-nc/trace/trace_0.json.gz",
    "fa3-c":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-c/trace/trace_0.json.gz",
}

# Strategy: link via "External id" or "args"->"correlation"
def analyze(label, path):
    print(f"\n=== {label} ===")
    cpu_launch_ts = {}  # corr_id -> ts
    gpu_kernel = {}     # corr_id -> (ts, dur, name)
    n_ev = 0
    with gzip.open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            n_ev += 1
            cat = ev.get("cat", "")
            args = ev.get("args", {}) or {}
            corr = args.get("correlation") or args.get("External id")
            if corr is None:
                continue
            ts = float(ev.get("ts", 0) or 0)
            dur = float(ev.get("dur", 0) or 0)
            name = ev.get("name", "")
            if "kernel" in cat.lower() or cat == "gpu_op":
                if "ncclDevKernel_AllGather" in name or "ncclDevKernel_ReduceScatter" in name:
                    gpu_kernel[corr] = (ts, dur, name)
            elif cat == "cuda_runtime" or cat == "cuda_driver" or "Runtime" in cat:
                # cudaLaunchKernel side
                cpu_launch_ts[corr] = ts
    # Compute gaps
    ag_gaps, rs_gaps = [], []
    for corr, (gts, gdur, name) in gpu_kernel.items():
        c = cpu_launch_ts.get(corr)
        if c is None: continue
        gap = gts - c
        if "AllGather" in name:
            ag_gaps.append(gap)
        else:
            rs_gaps.append(gap)
    def stats(xs, lbl):
        if not xs: print(f"  {lbl}: none"); return
        xs2 = sorted(xs)
        n = len(xs2)
        print(f"  {lbl}: n={n} mean_us={statistics.mean(xs2):.1f} median_us={xs2[n//2]:.1f} p90_us={xs2[int(n*0.9)]:.1f} p99_us={xs2[int(n*0.99)]:.1f}")
    stats(ag_gaps, "AG cpu→gpu lag")
    stats(rs_gaps, "RS cpu→gpu lag")

for k, p in CONFIGS.items():
    analyze(k, p)
