import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


DECODER = json.JSONDecoder()


def iter_trace_events(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        buf = ""
        found = False
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            if not found:
                key_idx = buf.find('"traceEvents"')
                if key_idx == -1:
                    buf = buf[-128:]
                    continue
                array_idx = buf.find("[", key_idx)
                if array_idx == -1:
                    continue
                buf = buf[array_idx + 1 :]
                found = True

            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \r\n\t,":
                    pos += 1
                if pos >= len(buf):
                    break
                if buf[pos] == "]":
                    return
                try:
                    event, end = DECODER.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield event
                pos = end
            buf = buf[pos:]


def summarize_trace(path: str):
    cpu_total_us = 0.0
    gpu_total_us = 0.0
    kernel_by_name_us = defaultdict(float)
    cpu_by_name_us = defaultdict(float)
    kernel_counts = Counter()
    cpu_counts = Counter()
    cat_counts = Counter()
    ph_counts = Counter()
    stream_spans = defaultdict(list)
    mem_counter = Counter()

    for ev in iter_trace_events(path):
        ph = ev.get("ph")
        ph_counts[ph] += 1
        if ph != "X":
            continue

        name = ev.get("name", "")
        cat = ev.get("cat", "")
        dur = float(ev.get("dur", 0.0))
        ts = float(ev.get("ts", 0.0))
        args = ev.get("args", {}) or {}
        cat_counts[cat] += 1

        if cat == "kernel" or args.get("device") is not None or name.startswith("ncclDevKernel_"):
            gpu_total_us += dur
            kernel_by_name_us[name] += dur
            kernel_counts[name] += 1
            stream = args.get("stream")
            device = args.get("device")
            if stream is not None:
                stream_spans[(device, stream)].append((ts, ts + dur, name))
            if "Allocated" in name or "cudaMalloc" in name:
                mem_counter[name] += 1
        else:
            cpu_total_us += dur
            cpu_by_name_us[name] += dur
            cpu_counts[name] += 1

    idle_by_stream_us = {}
    for key, spans in stream_spans.items():
        spans.sort()
        idle = 0.0
        prev_end = None
        for start, end, _ in spans:
            if prev_end is not None and start > prev_end:
                idle += start - prev_end
            prev_end = max(prev_end or end, end)
        idle_by_stream_us[key] = idle

    return {
        "path": path,
        "cpu_total_s": cpu_total_us / 1e6,
        "gpu_total_s": gpu_total_us / 1e6,
        "top_gpu": sorted(kernel_by_name_us.items(), key=lambda x: x[1], reverse=True)[:25],
        "top_cpu": sorted(cpu_by_name_us.items(), key=lambda x: x[1], reverse=True)[:25],
        "kernel_counts": kernel_counts,
        "cpu_counts": cpu_counts,
        "idle_by_stream_s": {str(k): v / 1e6 for k, v in sorted(idle_by_stream_us.items(), key=lambda x: -x[1])[:10]},
        "cat_counts": cat_counts,
        "ph_counts": ph_counts,
        "mem_counter": mem_counter,
    }


def main(paths: list[str]):
    summaries = [summarize_trace(path) for path in paths]
    json.dump(summaries, sys.stdout, indent=2, default=lambda x: dict(x))


if __name__ == "__main__":
    main(sys.argv[1:])
