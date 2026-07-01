from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

TERMINAL_STATES = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT"}
MEM_RE = re.compile(r"mem node (?P<pct>\d+)% \((?P<used>\d+)/(?P<total>\d+)G, avail (?P<avail>\d+)G\)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"error={exc!r}"


def _squeue(job_id: str) -> str:
    return _run(["squeue", "-j", job_id, "-h", "-o", "%i|%T|%M|%L|%D|%R"])


def _sacct(job_id: str) -> str:
    return _run(["sacct", "-j", job_id, "--format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList", "-P", "-n"])


def _tail_lines(path: Path, max_bytes: int = 2_000_000) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        size = path.stat().st_size
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode(errors="replace").splitlines()


def _summarize_orchestrator(output_dir: Path) -> str:
    log = output_dir / "logs" / "orchestrator.log"
    last_mem = None
    last_lora = None
    for line in _tail_lines(log):
        if "mem node" in line:
            match = MEM_RE.search(line)
            if match:
                last_mem = f"mem={match.group('used')}/{match.group('total')}G avail={match.group('avail')}G"
        if "Loading LoRA adapter" in line or "Loaded LoRA adapter" in line or "Retired filesystem LoRA" in line:
            last_lora = line.strip()
    parts = []
    if last_mem:
        parts.append(last_mem)
    if last_lora:
        parts.append(f"last_lora={last_lora[-220:]}")
    return " | ".join(parts) if parts else "orchestrator=no-signal-yet"


def _gib(num_bytes: int | float | None) -> float:
    if not num_bytes:
        return 0.0
    return float(num_bytes) / 2**30


def _summarize_diag(payload: dict) -> str:
    api = payload.get("api") or {}
    workers = payload.get("workers")
    worker_registered = []
    max_worker_rss = 0.0
    max_worker_gpu = 0.0
    if isinstance(workers, list):
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            worker_registered.append(worker.get("registered_lora_ids", []))
            max_worker_rss = max(max_worker_rss, _gib(worker.get("rss_bytes")))
            max_worker_gpu = max(max_worker_gpu, _gib(worker.get("gpu_reserved_bytes")))
    return (
        f"diag event={payload.get('event')} lora={payload.get('lora_name')} "
        f"api_rss={_gib(api.get('rss_bytes')):.1f}GiB "
        f"stage={_gib(api.get('stage_root_bytes')):.1f}GiB "
        f"public={api.get('public_lora_names')} "
        f"workers={worker_registered} "
        f"worker_rss_max={max_worker_rss:.1f}GiB "
        f"worker_gpu_reserved_max={max_worker_gpu:.1f}GiB"
    )


def _poll_new_diagnostics(output_dir: Path, offsets: dict[Path, int]) -> list[str]:
    results = []
    for log in sorted((output_dir / "logs" / "inference").glob("*.log")):
        offset = offsets.get(log, 0)
        try:
            with log.open("r", errors="replace") as f:
                f.seek(offset)
                for line in f:
                    marker = "prime_rl_lora_diagnostics "
                    if marker not in line:
                        continue
                    try:
                        payload = json.loads(line.split(marker, 1)[1])
                    except json.JSONDecodeError as exc:
                        results.append(f"{log.name}: diag_parse_error={exc}")
                        continue
                    results.append(f"{log.name}: {_summarize_diag(payload)}")
                offsets[log] = f.tell()
        except OSError:
            continue
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--max-minutes", type=float, default=240.0)
    args = parser.parse_args()

    offsets: dict[Path, int] = {}
    deadline = time.monotonic() + args.max_minutes * 60
    print(f"[{_now()}] monitor_start job={args.job_id} output={args.output_dir}", flush=True)
    while time.monotonic() < deadline:
        sq = _squeue(args.job_id)
        state = "UNKNOWN"
        if sq:
            fields = sq.split("|")
            state = fields[1] if len(fields) > 1 else "UNKNOWN"
        else:
            sacct = _sacct(args.job_id)
            state = "MISSING"
            for line in sacct.splitlines():
                fields = line.split("|")
                if fields and fields[0] == args.job_id and len(fields) > 1:
                    state = fields[1].split()[0]
                    break
            sq = f"sacct={sacct or 'empty'}"

        print(f"[{_now()}] state={state} squeue={sq} | {_summarize_orchestrator(args.output_dir)}", flush=True)
        for diag in _poll_new_diagnostics(args.output_dir, offsets):
            print(f"[{_now()}] {diag}", flush=True)

        if state in TERMINAL_STATES:
            print(f"[{_now()}] terminal_state={state}", flush=True)
            print(_sacct(args.job_id), flush=True)
            return
        time.sleep(args.interval_s)

    print(f"[{_now()}] monitor_timeout", flush=True)


if __name__ == "__main__":
    main()
