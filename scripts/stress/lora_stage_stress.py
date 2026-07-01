from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from prime_rl.inference.lora_staging import LoRAStagingError, cleanup_staged_lora_adapter, stage_lora_adapter


def _read_python(path: str, chunk_size: int) -> float:
    start = time.perf_counter()
    with open(path, "rb", buffering=0) as handle:
        while handle.read(chunk_size):
            pass
    return time.perf_counter() - start


def _read_dd_direct(path: str, block_size: str) -> float:
    start = time.perf_counter()
    subprocess.run(
        ["dd", f"if={path}", "of=/dev/null", "iflag=direct", f"bs={block_size}", "status=none"],
        check=True,
    )
    return time.perf_counter() - start


def _read_worker(args: tuple[str, bool, int, str]) -> float:
    path, direct, chunk_size, block_size = args
    if direct:
        return _read_dd_direct(path, block_size)
    return _read_python(path, chunk_size)


def _emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _file_size_gb(path: Path) -> float:
    return path.stat().st_size / 1e9


def _reader_storm(path: Path, *, readers: int, direct: bool, chunk_size: int, block_size: str) -> dict[str, float]:
    with ProcessPoolExecutor(max_workers=readers) as executor:
        durations = list(
            executor.map(
                _read_worker,
                [(path.as_posix(), direct, chunk_size, block_size) for _ in range(readers)],
            )
        )
    durations.sort()
    total_gb = _file_size_gb(path) * readers
    return {
        "p50_s": durations[len(durations) // 2],
        "max_s": max(durations),
        "aggregate_gbps": total_gb / max(durations),
    }


def _write_small_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"x" * 4096)
    (path / "adapter_config.json").write_text('{"r": 64}\n')
    (path / "STABLE").touch()


def _failure_smoke(stage_root: Path) -> None:
    base = stage_root.parent / "failure_smoke"
    shutil.rmtree(base, ignore_errors=True)
    source = base / "source"
    _write_small_adapter(source)

    staged = stage_lora_adapter(source, "stress-small__v00000001", stage_root=stage_root)
    (staged / "adapter_model.safetensors").write_bytes(b"bad")
    restaged = stage_lora_adapter(source, "stress-small__v00000001", stage_root=stage_root)
    assert restaged == staged
    assert (restaged / "adapter_model.safetensors").stat().st_size == 4096

    partial = base / "partial"
    partial.mkdir()
    (partial / "adapter_model.safetensors").write_bytes(b"x")
    (partial / "STABLE").touch()
    try:
        stage_lora_adapter(partial, "stress-partial__v00000001", stage_root=stage_root)
    except LoRAStagingError:
        pass
    else:
        raise AssertionError("partial adapter source unexpectedly staged")

    cleanup_staged_lora_adapter(staged, stage_root=stage_root)
    shutil.rmtree(base, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress node-local LoRA staging versus direct shared-FS reads.")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/isambard/debate_smoke_keep/run_default/broadcasts/step_3"),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path(f"/dev/shm/prime_rl_lora_stage_stress_{os.environ.get('SLURM_JOB_ID', os.getpid())}"),
    )
    parser.add_argument("--readers", default="1,4,8,12")
    parser.add_argument("--chunk-size-mb", type=int, default=64)
    parser.add_argument("--dd-block-size", default="64M")
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise SystemExit(f"missing adapter file: {adapter_file}")

    readers = [int(value) for value in args.readers.split(",") if value]
    chunk_size = args.chunk_size_mb * 1024 * 1024
    stage_root = args.stage_root.resolve()
    shutil.rmtree(stage_root, ignore_errors=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    _emit(
        "start",
        adapter_dir=adapter_dir.as_posix(),
        adapter_gb=_file_size_gb(adapter_file),
        stage_root=stage_root.as_posix(),
        host=os.uname().nodename,
    )

    _failure_smoke(stage_root)
    _emit("failure_smoke_ok")

    start = time.perf_counter()
    staged_dir = stage_lora_adapter(adapter_dir, "stress-real__v00000001", stage_root=stage_root)
    stage_s = time.perf_counter() - start
    staged_file = staged_dir / "adapter_model.safetensors"
    _emit("stage_real", seconds=stage_s, staged_dir=staged_dir.as_posix(), staged_gb=_file_size_gb(staged_file))

    start = time.perf_counter()
    staged_again = stage_lora_adapter(adapter_dir, "stress-real__v00000001", stage_root=stage_root)
    _emit("stage_reuse", seconds=time.perf_counter() - start, same_path=staged_again == staged_dir)

    for count in readers:
        if not args.skip_direct:
            try:
                direct_stats = _reader_storm(
                    adapter_file,
                    readers=count,
                    direct=True,
                    chunk_size=chunk_size,
                    block_size=args.dd_block_size,
                )
                _emit("read_storm", source="lustre_direct", readers=count, **direct_stats)
            except subprocess.CalledProcessError as exc:
                _emit("read_storm_error", source="lustre_direct", readers=count, returncode=exc.returncode)

        staged_stats = _reader_storm(
            staged_file,
            readers=count,
            direct=False,
            chunk_size=chunk_size,
            block_size=args.dd_block_size,
        )
        _emit("read_storm", source="node_local", readers=count, **staged_stats)

    cleanup_staged_lora_adapter(staged_dir, stage_root=stage_root)
    _emit("cleanup_ok", staged_exists=staged_dir.exists())


if __name__ == "__main__":
    main()
