from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

REQUIRED_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
STAGE_MANIFEST = ".prime_rl_stage_manifest.json"


class LoRAStagingError(RuntimeError):
    """Raised when node-local LoRA staging cannot produce a verified adapter dir."""


@dataclass(frozen=True)
class LoRAStageManifest:
    source: str
    lora_name: str
    files: dict[str, int]

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "lora_name": self.lora_name,
                "files": self.files,
            },
            sort_keys=True,
        )


def _default_stage_root() -> Path:
    stage_root = os.environ.get("PRIME_RL_LORA_STAGE_ROOT")
    if stage_root:
        return Path(stage_root)

    job_id = os.environ.get("SLURM_JOB_ID") or f"user{os.getuid()}"
    hostname = socket.gethostname().split(".")[0] or "unknown"
    return Path(f"/dev/shm/prime_rl_lora_{job_id}_{hostname}")


def _safe_stage_name(source: Path, lora_name: str) -> str:
    digest = hashlib.sha256(f"{source.resolve()}::{lora_name}".encode()).hexdigest()[:16]
    safe_lora = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in lora_name)[:96]
    return f"{safe_lora}-{digest}"


def _source_manifest(source: Path, lora_name: str) -> LoRAStageManifest:
    if not source.is_dir():
        raise LoRAStagingError(f"LoRA adapter source is not a directory: {source}")

    files: dict[str, int] = {}
    for rel in REQUIRED_ADAPTER_FILES:
        path = source / rel
        if not path.is_file():
            raise LoRAStagingError(f"LoRA adapter source is missing required file {rel}: {source}")
        files[rel] = path.stat().st_size

    return LoRAStageManifest(source=str(source.resolve()), lora_name=lora_name, files=files)


def _read_manifest(path: Path) -> LoRAStageManifest | None:
    manifest_path = path / STAGE_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        files = data.get("files")
        if not isinstance(files, dict):
            return None
        return LoRAStageManifest(
            source=str(data["source"]),
            lora_name=str(data["lora_name"]),
            files={str(name): int(size) for name, size in files.items()},
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _stage_is_verified(path: Path, expected: LoRAStageManifest) -> bool:
    if _read_manifest(path) != expected:
        return False
    for rel, size in expected.files.items():
        staged_file = path / rel
        if not staged_file.is_file() or staged_file.stat().st_size != size:
            return False
    return True


def _copy_adapter_tree(source: Path, tmp_dest: Path, expected: LoRAStageManifest) -> None:
    shutil.copytree(source, tmp_dest, symlinks=False, copy_function=shutil.copy2)
    for rel, size in expected.files.items():
        staged_file = tmp_dest / rel
        if not staged_file.is_file() or staged_file.stat().st_size != size:
            raise LoRAStagingError(f"Staged LoRA file {rel} has wrong size after copy from {source}")
    (tmp_dest / STAGE_MANIFEST).write_text(expected.to_json())


def _try_acquire_lock(lock_path: Path) -> int | None:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode())
    return fd


def _release_lock(lock_path: Path, fd: int) -> None:
    os.close(fd)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def stage_lora_adapter(
    source: Path,
    lora_name: str,
    *,
    stage_root: Path | None = None,
    lock_timeout_s: float = 300.0,
    poll_s: float = 0.25,
) -> Path:
    """Copy a Prime LoRA broadcast dir to node-local storage and verify it.

    The copy is atomic from readers' perspective: work happens in a private tmp
    directory, then a verified tree is renamed into place under ``stage_root``.
    """
    source = source.resolve()
    stage_root = (stage_root or _default_stage_root()).resolve()
    expected = _source_manifest(source, lora_name)
    stage_root.mkdir(parents=True, exist_ok=True)

    dest = stage_root / _safe_stage_name(source, lora_name)
    lock_path = stage_root / f"{dest.name}.lock"
    deadline = time.monotonic() + lock_timeout_s

    while True:
        if _stage_is_verified(dest, expected):
            return dest

        fd = _try_acquire_lock(lock_path)
        if fd is not None:
            break

        if time.monotonic() >= deadline:
            raise LoRAStagingError(f"Timed out waiting for staged LoRA adapter lock: {lock_path}")
        time.sleep(poll_s)

    tmp_dest = stage_root / f".{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        if dest.exists():
            shutil.rmtree(dest)
        _copy_adapter_tree(source, tmp_dest, expected)
        tmp_dest.rename(dest)
        if not _stage_is_verified(dest, expected):
            raise LoRAStagingError(f"Staged LoRA adapter failed verification after rename: {dest}")
        return dest
    except OSError as exc:
        raise LoRAStagingError(f"Failed to stage LoRA adapter {source} to {stage_root}: {exc}") from exc
    finally:
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest, ignore_errors=True)
        _release_lock(lock_path, fd)


def should_stage_lora_adapter(source: Path) -> bool:
    if os.environ.get("PRIME_RL_LORA_STAGE_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return False
    return (source / "STABLE").exists()


def maybe_stage_lora_adapter(source: Path, lora_name: str) -> Path:
    if not should_stage_lora_adapter(source):
        return source
    return stage_lora_adapter(source, lora_name)


def cleanup_staged_lora_adapter(path: Path, *, stage_root: Path | None = None) -> None:
    stage_root = (stage_root or _default_stage_root()).resolve()
    path = path.resolve()
    try:
        path.relative_to(stage_root)
    except ValueError:
        return
    if path == stage_root:
        return
    shutil.rmtree(path, ignore_errors=True)
