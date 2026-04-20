"""Drill: per-kernel duration distribution for AG/RS, plus CPU compile markers + Inductor pid scan."""
import gzip, ijson, sys
from collections import defaultdict, Counter
import statistics

CONFIGS = {
    "fa2-nc": "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-nc/trace/trace_0.json.gz",
    "fa2-c":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-c/trace/trace_0.json.gz",
    "fa3-nc": "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-nc/trace/trace_0.json.gz",
    "fa3-c":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-c/trace/trace_0.json.gz",
}

def analyze(label, path):
    print(f"\n=== {label} ===", flush=True)
    ag_durs = []
    rs_durs = []
    cpu_compile = Counter()
    cpu_inductor_dispatch = Counter()
    cpu_dispatch_total_us = 0.0
    triton_kernel_names = Counter()
    aten_call_counts = Counter()
    cpu_op_count = 0
    with gzip.open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            name = ev.get("name", "")
            cat = ev.get("cat", "")
            dur = float(ev.get("dur", 0) or 0)
            ph = ev.get("ph", "")
            if "kernel" in cat.lower() or cat == "gpu_op":
                if "ncclDevKernel_AllGather" in name or "ncclKernel_AllGather" in name:
                    ag_durs.append(dur)
                elif "ncclDevKernel_ReduceScatter" in name or "ncclKernel_ReduceScatter" in name:
                    rs_durs.append(dur)
                elif "triton" in name or "triton_" in name:
                    triton_kernel_names[name[:60]] += 1
            elif cat in ("cpu_op", "user_annotation", "python_function"):
                cpu_op_count += 1
                if "dynamo" in name.lower() or "compile" in name.lower() or "inductor" in name.lower() or "graph_break" in name.lower():
                    cpu_compile[name[:60]] += 1
                if name.startswith("aten::"):
                    aten_call_counts[name] += 1
    def stats(xs, lbl):
        if not xs:
            print(f"  {lbl}: none"); return
        xs2 = sorted(xs)
        n = len(xs2)
        print(f"  {lbl}: n={n} sum_ms={sum(xs2)/1e3:.0f} mean_us={statistics.mean(xs2):.1f} median_us={xs2[n//2]:.1f} p90_us={xs2[int(n*0.9)]:.1f} p99_us={xs2[int(n*0.99)]:.1f} max_us={xs2[-1]:.1f}")
    stats(ag_durs, "AllGather")
    stats(rs_durs, "ReduceScatter")
    print(f"  triton kernels (top 10):")
    for k, v in triton_kernel_names.most_common(10):
        print(f"    {v:6d}  {k}")
    print(f"  cpu compile/dynamo markers (top 10):")
    for k, v in cpu_compile.most_common(10):
        print(f"    {v:6d}  {k}")
    print(f"  cpu_op total events: {cpu_op_count:,}")
    print(f"  aten::record_stream count: {aten_call_counts.get('aten::record_stream', 0)}")
    print(f"  aten::copy_ count: {aten_call_counts.get('aten::copy_', 0)}")
    print(f"  aten::cat count: {aten_call_counts.get('aten::cat', 0)}")
    print(f"  aten::empty count: {aten_call_counts.get('aten::empty', 0)}")

if __name__ == "__main__":
    for k in sys.argv[1:] or list(CONFIGS):
        analyze(k, CONFIGS[k])
