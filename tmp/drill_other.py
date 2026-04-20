"""Drill into the 'other' bucket — unmatched GPU kernels. One config at a time."""
import gzip, json
from collections import defaultdict
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-nc/trace/trace_0.json.gz"

# Same buckets as main script
BUCKETS = [
    ("flash_attn", ["flash_attn", "flashattn", "mha_fwd", "mha_bwd", "_flashattention"]),
    ("gemm", ["gemm", "cutlass", "nn.linear", "addmm", "matmul", "bmm", "mm_out"]),
    ("rmsnorm", ["rmsnorm", "rms_norm", "quack"]),
    ("rope", ["rope", "apply_rotary", "rotate_half"]),
    ("liger_ce", ["liger", "cross_entropy", "fused_linear"]),
    ("activation", ["silu", "gelu", "relu", "softmax"]),
    ("fsdp_collective", ["allgather","all_gather","reducescatter","reduce_scatter","ncclkernel","ncclalltoall","allreduce","all_reduce","ncclallgather","ncclreducescatter"]),
    ("elementwise", ["add","mul","div","sub","copy_","cast","to_","_foreach","fill","zero","clamp"]),
    ("memcpy", ["memcpy","memset","dtoh","htod"]),
    ("optimizer", ["adamw","fused_adamw"]),
    ("autograd_engine", ["autograd","backward"]),
]

def bucket_of(name):
    nl = name.lower()
    for bkt, pats in BUCKETS:
        if any(p in nl for p in pats): return bkt
    return "other"

data = json.loads(gzip.open(path, "rb").read())
events = data["traceEvents"]

other_by_name = defaultdict(lambda: [0.0, 0])
for ev in events:
    if ev.get("ph") != "X": continue
    dur = ev.get("dur", 0)
    cat = ev.get("cat", "")
    name = ev.get("name", "")
    if ("kernel" in cat or cat == "gpu_memcpy" or cat == "gpu_memset") and dur > 0:
        if bucket_of(name) == "other":
            other_by_name[name][0] += dur
            other_by_name[name][1] += 1

print(f"Top 25 UNCATEGORIZED GPU kernels in {path.split('/')[-3]}:")
print(f"{'ms':>10}  {'calls':>7}  {'avg_us':>8}  name")
print("-"*100)
top = sorted(other_by_name.items(), key=lambda kv: -kv[1][0])[:25]
for name, (us, n) in top:
    print(f"{us/1e3:>10.1f}  {n:>7,}  {us/max(n,1):>8.1f}  {name[:80]}")
