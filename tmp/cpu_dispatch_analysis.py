"""CPU-side dispatch analysis for prime-rl SFT profiler traces.

Streams chrome-trace JSON via ijson (avoid loading 1-2GB into RAM).
Aggregates:
  - cpu_op events by name (top-N by total time, by call count, by count*time)
  - cuda runtime / launch overhead (cudaLaunchKernel etc, cat=cuda_runtime)
  - DTensor / FSDP bookkeeping
  - GPU starvation: timeline windows where GPU kernels idle but CPU busy
  - Cross-config diff (compile vs no-compile, FA2 vs FA3, maxmfu)

Usage:
  uv run --no-sync python tmp/cpu_dispatch_analysis.py
"""
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import orjson

CONFIGS = {
    "fa2-nc":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-nc/trace/trace_0.json.gz",
    "fa2-c":   "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa2-c/trace/trace_0.json.gz",
    "fa3-nc":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-nc/trace/trace_0.json.gz",
    "fa3-c":   "/scratch/a6r/joanv.a6r/outputs/sft-prof-fa3-c/trace/trace_0.json.gz",
    "maxmfu":  "/scratch/a6r/joanv.a6r/outputs/sft-prof-maxmfu/trace/trace_0.json.gz",
}

# Bin width (us) for GPU-busy timeline (10 us → fine but cheap on memory)
BIN_US = 100


def stream_events(path):
    """Yield event dicts from gzipped chrome trace, one at a time."""
    with gzip.open(path, "rb") as fh:
        # ijson can iterate over the array at "traceEvents.item"
        for ev in ijson.items(fh, "traceEvents.item", use_float=True):
            yield ev


def analyze(path, label):
    cpu_by_name = defaultdict(lambda: [0.0, 0])     # name -> [tot_us, count]
    cuda_runtime = defaultdict(lambda: [0.0, 0])    # cudaLaunchKernel etc
    cat_totals = defaultdict(lambda: [0.0, 0])      # category -> [tot_us, count]
    # for starvation: collect (start, end) of GPU kernel events and (start,end) of cpu_op
    # but storing all intervals is too much memory. Instead, mark coarse bins busy.
    gpu_busy = defaultdict(int)   # bin_idx -> us busy in that bin
    cpu_busy = defaultdict(int)
    t_min = None
    t_max = 0.0

    n = 0
    for ev in stream_events(path):
        if ev.get("ph") != "X":
            continue
        dur = ev.get("dur", 0) or 0
        if dur <= 0:
            continue
        ts = ev.get("ts", 0) or 0
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        cat_totals[cat][0] += dur
        cat_totals[cat][1] += 1

        if cat == "cpu_op":
            cpu_by_name[name][0] += dur
            cpu_by_name[name][1] += 1
        elif cat == "cuda_runtime" or cat == "cuda_driver":
            cuda_runtime[name][0] += dur
            cuda_runtime[name][1] += 1

        # busy-bin accounting (skip events with zero ts)
        if ts > 0:
            t_min = ts if t_min is None else min(t_min, ts)
            t_max = max(t_max, ts + dur)
            # For CPU ops, only mark top-level (ph=X events overlap a lot in nested cpu_op).
            # Workaround: just mark cuda_runtime + cpu_op[name not in nested]; we'll instead
            # rely on cuda_runtime as a proxy for "CPU is dispatching".
            if "kernel" in cat or cat in ("gpu_memcpy", "gpu_memset"):
                # mark gpu_busy bins
                a = int(ts // BIN_US); b = int((ts + dur) // BIN_US)
                if a == b:
                    gpu_busy[a] += dur
                else:
                    # first partial
                    gpu_busy[a] += (a + 1) * BIN_US - ts
                    for k in range(a + 1, b):
                        gpu_busy[k] += BIN_US
                    gpu_busy[b] += (ts + dur) - b * BIN_US
            elif cat == "cuda_runtime":
                a = int(ts // BIN_US); b = int((ts + dur) // BIN_US)
                if a == b:
                    cpu_busy[a] += dur
                else:
                    cpu_busy[a] += (a + 1) * BIN_US - ts
                    for k in range(a + 1, b):
                        cpu_busy[k] += BIN_US
                    cpu_busy[b] += (ts + dur) - b * BIN_US
        n += 1

    # Compute starvation fraction over [t_min, t_max]
    duration_us = t_max - (t_min or 0)
    bins_total = max(1, int(duration_us // BIN_US))
    # Bin is "GPU busy" if gpu_busy[bin] >= 0.5*BIN_US.  CPU-busy similar.
    busy_thresh = 0.5 * BIN_US
    n_gpu_busy = sum(1 for v in gpu_busy.values() if v >= busy_thresh)
    n_cpu_busy = sum(1 for v in cpu_busy.values() if v >= busy_thresh)
    # starvation = bin where CPU busy AND gpu idle
    cpu_keys = set(k for k, v in cpu_busy.items() if v >= busy_thresh)
    gpu_keys = set(k for k, v in gpu_busy.items() if v >= busy_thresh)
    starv_bins = cpu_keys - gpu_keys
    n_starv = len(starv_bins)
    n_idle = bins_total - n_gpu_busy

    return {
        "label": label,
        "n_events": n,
        "duration_s": duration_us / 1e6,
        "cpu_by_name": cpu_by_name,
        "cuda_runtime": cuda_runtime,
        "cat_totals": cat_totals,
        "gpu_busy_bins": n_gpu_busy,
        "cpu_busy_bins": n_cpu_busy,
        "starv_bins": n_starv,
        "idle_bins": n_idle,
        "total_bins": bins_total,
    }


def fmt_top(d, n=10, key=lambda kv: -kv[1][0]):
    items = sorted(d.items(), key=key)[:n]
    out = []
    for name, (tot, cnt) in items:
        avg = tot / max(cnt, 1)
        out.append((name, tot, cnt, avg))
    return out


def main():
    results = {}
    for label, path in CONFIGS.items():
        if not Path(path).exists():
            print(f"[skip] {label}: missing", flush=True)
            continue
        print(f"\n=== streaming {label} ===", flush=True)
        results[label] = analyze(path, label)
        r = results[label]
        print(f"  events={r['n_events']:,}  span={r['duration_s']:.2f}s", flush=True)
        ct_tot = sum(v[0] for v in r['cat_totals'].values())
        print(f"  category totals (s):", flush=True)
        for cat, (us, n) in sorted(r['cat_totals'].items(), key=lambda kv: -kv[1][0])[:12]:
            print(f"    {cat:<20} {us/1e6:>8.2f}s  ({n:>9,} events, avg {us/max(n,1):.1f}us)", flush=True)

    print("\n\n========== TOP CPU OPS PER CONFIG ==========")
    for label, r in results.items():
        print(f"\n--- {label}: top-15 cpu_op by total time ---")
        cpu_total = sum(v[0] for v in r['cpu_by_name'].values())
        print(f"  total cpu_op time: {cpu_total/1e6:.2f}s across {sum(v[1] for v in r['cpu_by_name'].values()):,} events")
        for name, tot, cnt, avg in fmt_top(r['cpu_by_name'], 15):
            print(f"    {tot/1e3:>9.1f} ms  cnt={cnt:>7,}  avg={avg:>7.1f}us  {name[:75]}")

        print(f"\n--- {label}: top-15 cpu_op by count*time (dispatch-overhead proxy) ---")
        for name, tot, cnt, avg in fmt_top(r['cpu_by_name'], 15, key=lambda kv: -(kv[1][0] * (kv[1][1] ** 0.0001))):
            # actually just sort by count
            pass
        for name, (tot, cnt) in sorted(r['cpu_by_name'].items(), key=lambda kv: -kv[1][1])[:15]:
            print(f"    cnt={cnt:>7,}  tot={tot/1e3:>8.1f}ms  avg={tot/max(cnt,1):>6.1f}us  {name[:70]}")

        print(f"\n--- {label}: cuda_runtime ops (launch overhead) ---")
        cr_tot = sum(v[0] for v in r['cuda_runtime'].values())
        print(f"  total cuda_runtime: {cr_tot/1e6:.2f}s across {sum(v[1] for v in r['cuda_runtime'].values()):,} calls")
        for name, tot, cnt, avg in fmt_top(r['cuda_runtime'], 8):
            print(f"    {tot/1e3:>9.1f} ms  cnt={cnt:>7,}  avg={avg:>6.1f}us  {name}")

        print(f"\n--- {label}: GPU starvation ---")
        tot = r['total_bins']
        print(f"  bins={tot:,} (BIN={BIN_US}us, span={r['duration_s']:.2f}s)")
        print(f"  GPU busy:  {r['gpu_busy_bins']:>9,}  ({100*r['gpu_busy_bins']/max(tot,1):.1f}%)")
        print(f"  CPU busy:  {r['cpu_busy_bins']:>9,}  ({100*r['cpu_busy_bins']/max(tot,1):.1f}%)  (cuda_runtime proxy)")
        print(f"  GPU idle:  {r['idle_bins']:>9,}  ({100*r['idle_bins']/max(tot,1):.1f}%)")
        print(f"  STARVED:   {r['starv_bins']:>9,}  ({100*r['starv_bins']/max(tot,1):.1f}%)  (CPU busy, GPU idle)")

    # Cross-config diff: top-N union by name
    print("\n\n========== CROSS-CONFIG CPU-OP DIFF (top 20 by max-config time) ==========")
    union = set()
    for r in results.values():
        for n, (t, c) in r['cpu_by_name'].items():
            union.add(n)
    rows = []
    for name in union:
        vals = {l: results[l]['cpu_by_name'].get(name, [0.0, 0]) for l in results}
        rows.append((name, vals, max(v[0] for v in vals.values())))
    rows.sort(key=lambda x: -x[2])
    labels = list(results.keys())
    print(f"  {'op':<55} " + " ".join(f"{l:>10}" for l in labels))
    for name, vals, _ in rows[:25]:
        cells = " ".join(f"{vals[l][0]/1e3:>10.1f}" for l in labels)
        print(f"  {name[:55]:<55} {cells}")

    print("\n  (values in ms total time per rank0 trace)")

    # Save raw JSON for follow-up
    out = {l: {
        "cpu_by_name": {k: v for k, v in r['cpu_by_name'].items()},
        "cuda_runtime": {k: v for k, v in r['cuda_runtime'].items()},
        "cat_totals": {k: v for k, v in r['cat_totals'].items()},
        "starv_bins": r['starv_bins'],
        "idle_bins": r['idle_bins'],
        "gpu_busy_bins": r['gpu_busy_bins'],
        "cpu_busy_bins": r['cpu_busy_bins'],
        "total_bins": r['total_bins'],
        "duration_s": r['duration_s'],
    } for l, r in results.items()}
    Path("/projects/a6r/joanv.a6r/work/prime-rl/tmp/cpu_dispatch_summary.json").write_bytes(orjson.dumps(out))
    print("\n[saved summary -> tmp/cpu_dispatch_summary.json]")


if __name__ == "__main__":
    main()
