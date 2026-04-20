"""Compare set of GPU kernel names between nc and c. Identify Inductor-fused / triton kernels."""
import gzip, ijson, sys
from collections import Counter

CONFIGS = {
    "fa3-nc": "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-nc/trace/trace_0.json.gz",
    "fa3-c":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-c/trace/trace_0.json.gz",
}

def kernels(path):
    counts = Counter()
    durs = Counter()
    with gzip.open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            cat = ev.get("cat", "")
            if "kernel" in cat.lower() or cat == "gpu_op":
                name = ev.get("name", "")
                # truncate template parameters
                short = name.split("<")[0][:80]
                counts[short] += 1
                durs[short] += float(ev.get("dur", 0) or 0)
    return counts, durs

for label, path in CONFIGS.items():
    print(f"\n=== {label} ===")
    c, d = kernels(path)
    print(f"unique kernel name prefixes: {len(c)}")
    # Look for triton/inductor fingerprints
    for n in sorted(c):
        if "triton" in n or "inductor" in n or "fused_" in n:
            print(f"  {c[n]:6d}  dur_ms={d[n]/1e3:7.0f}  {n}")
    print("top 15 by total duration:")
    for n, _ in sorted(d.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {c[n]:6d}  dur_ms={d[n]/1e3:7.0f}  {n}")
