"""Stream-parse traces, focus on:
- NCCL all-gather / reduce-scatter timeline & overlap with compute
- Compile-related markers (graph break, recompile, dynamo)
- CPU dispatch latency between collective launch and next non-NCCL kernel
- cudaMalloc/cudaFree counts (allocator pressure)
"""
import gzip, ijson, sys
from collections import defaultdict
from pathlib import Path

CONFIGS = {
    "fa2-nc": "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-nc/trace/trace_0.json.gz",
    "fa2-c":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-c/trace/trace_0.json.gz",
    "fa3-nc": "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-nc/trace/trace_0.json.gz",
    "fa3-c":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-c/trace/trace_0.json.gz",
}

COMPILE_MARKERS = ("dynamo", "torch_compile", "compiled_autograd", "inductor", "triton_", "graph_break", "recompile", "TorchDynamo")
NCCL_AG = ("ncclDevKernel_AllGather", "ncclKernel_AllGather", "ncclAllGather")
NCCL_RS = ("ncclDevKernel_ReduceScatter", "ncclKernel_ReduceScatter", "ncclReduceScatter")
NCCL_ANY = NCCL_AG + NCCL_RS + ("ncclDevKernel_AllReduce", "ncclKernel_")
ALLOC = ("cudaMalloc", "cudaFree", "cuMemAlloc", "cudaMallocAsync", "cudaFreeAsync")

def is_match(name, prefixes):
    return any(p in name for p in prefixes)

def analyze(label, path):
    print(f"\n=== {label}: {path} ===", flush=True)
    # bucketed counters
    nccl_ag_kernels = []  # (ts, dur, name)
    nccl_rs_kernels = []
    other_gpu_kernels = []  # (ts, dur)
    compile_counts = defaultdict(int)
    alloc_counts = defaultdict(int)
    alloc_dur = defaultdict(float)
    cpu_op_to_kernel_dispatch = []  # cpu launch->kernel start gaps for nccl
    # Track CPU ops on tid for nccl launches
    tot_events = 0
    tot_gpu_kernels = 0
    fa_kernels = 0
    gemm_kernels = 0

    with gzip.open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            tot_events += 1
            name = ev.get("name", "")
            ph = ev.get("ph", "")
            cat = ev.get("cat", "")
            dur = float(ev.get("dur", 0) or 0)
            ts = float(ev.get("ts", 0) or 0)

            # Compile markers
            for m in COMPILE_MARKERS:
                if m in name:
                    compile_counts[m] += 1
                    break

            # Allocator
            if any(a in name for a in ALLOC):
                alloc_counts[name] += 1
                alloc_dur[name] += dur

            # GPU kernel events (cat=='kernel' or cat starts with 'kernel')
            if "kernel" in cat.lower() or cat == "gpu_op":
                if dur > 0:
                    tot_gpu_kernels += 1
                    if is_match(name, NCCL_AG):
                        nccl_ag_kernels.append((ts, dur, name))
                    elif is_match(name, NCCL_RS):
                        nccl_rs_kernels.append((ts, dur, name))
                    elif is_match(name, NCCL_ANY):
                        pass
                    else:
                        other_gpu_kernels.append((ts, dur))
                        n = name.lower()
                        if "flash" in n or "mha_" in n or "_attention" in n:
                            fa_kernels += 1
                        if "gemm" in n or "cutlass" in n or "ampere_" in n or "sm90" in n:
                            gemm_kernels += 1

    print(f"  total events:          {tot_events:,}")
    print(f"  total GPU kernels:     {tot_gpu_kernels:,}")
    print(f"  flash-attn kernels:    {fa_kernels:,}")
    print(f"  gemm-ish kernels:      {gemm_kernels:,}")
    print(f"  AllGather kernels:     {len(nccl_ag_kernels):,}")
    print(f"  ReduceScatter kernels: {len(nccl_rs_kernels):,}")
    print(f"  total AG time (ms):    {sum(d for _,d,_ in nccl_ag_kernels)/1e3:.1f}")
    print(f"  total RS time (ms):    {sum(d for _,d,_ in nccl_rs_kernels)/1e3:.1f}")
    print(f"  total OTHER GPU (ms):  {sum(d for _,d in other_gpu_kernels)/1e3:.1f}")
    print(f"  compile markers (count by substring):")
    for k, v in sorted(compile_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {k:30s} {v:,}")
    print(f"  allocator events:")
    for k, v in sorted(alloc_counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {k:30s} count={v:,}  total_dur_ms={alloc_dur[k]/1e3:.2f}")

    # Overlap analysis: for each AG kernel, fraction of its duration overlapping with any non-NCCL GPU kernel
    if nccl_ag_kernels and other_gpu_kernels:
        # sort other kernels by start
        other = sorted(other_gpu_kernels)
        other_starts = [o[0] for o in other]
        import bisect
        total_ag = 0
        overlap_ag = 0
        # sample first 2000 AG events for speed
        sample = nccl_ag_kernels[:5000]
        for (ag_ts, ag_dur, _) in sample:
            ag_end = ag_ts + ag_dur
            total_ag += ag_dur
            # find candidate other kernels intersecting [ag_ts, ag_end]
            i = bisect.bisect_left(other_starts, ag_ts) - 1
            if i < 0: i = 0
            ov = 0
            while i < len(other):
                o_ts, o_dur = other[i]
                if o_ts >= ag_end:
                    break
                o_end = o_ts + o_dur
                if o_end > ag_ts:
                    ov += min(ag_end, o_end) - max(ag_ts, o_ts)
                i += 1
            overlap_ag += min(ov, ag_dur)
        print(f"  AG overlap (sample {len(sample)}): {overlap_ag/total_ag*100:.1f}% overlapped with compute  (total_ag_us={total_ag:.0f}, ov_us={overlap_ag:.0f})")

    # Same for RS
    if nccl_rs_kernels and other_gpu_kernels:
        import bisect
        other = sorted(other_gpu_kernels)
        other_starts = [o[0] for o in other]
        total_rs = 0
        overlap_rs = 0
        sample = nccl_rs_kernels[:5000]
        for (rs_ts, rs_dur, _) in sample:
            rs_end = rs_ts + rs_dur
            total_rs += rs_dur
            i = bisect.bisect_left(other_starts, rs_ts) - 1
            if i < 0: i = 0
            ov = 0
            while i < len(other):
                o_ts, o_dur = other[i]
                if o_ts >= rs_end:
                    break
                o_end = o_ts + o_dur
                if o_end > rs_ts:
                    ov += min(rs_end, o_end) - max(rs_ts, o_ts)
                i += 1
            overlap_rs += min(ov, rs_dur)
        print(f"  RS overlap (sample {len(sample)}): {overlap_rs/total_rs*100:.1f}% overlapped with compute")

if __name__ == "__main__":
    keys = sys.argv[1:] or list(CONFIGS.keys())
    for k in keys:
        analyze(k, CONFIGS[k])
