import errno
from pathlib import Path

import pytest

from prime_rl.inference.lora_staging import (
    LoRAStagingError,
    _default_stage_root,
    cleanup_staged_lora_adapter,
    stage_lora_adapter,
)


def _write_adapter(path: Path, *, payload: bytes = b"adapter") -> None:
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(payload)
    (path / "adapter_config.json").write_text('{"r": 64}\n')
    (path / "STABLE").touch()


def test_default_stage_root_is_scoped_by_job_and_host(monkeypatch):
    monkeypatch.delenv("PRIME_RL_LORA_STAGE_ROOT", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr("prime_rl.inference.lora_staging.socket.gethostname", lambda: "nid000123")

    assert _default_stage_root() == Path("/dev/shm/prime_rl_lora_12345_nid000123")


def test_default_stage_root_honors_explicit_env(monkeypatch, tmp_path):
    stage_root = tmp_path / "stage"
    monkeypatch.setenv("PRIME_RL_LORA_STAGE_ROOT", stage_root.as_posix())

    assert _default_stage_root() == stage_root


def test_stage_lora_adapter_copies_prime_broadcast_dir_to_node_local_root(tmp_path):
    source = tmp_path / "lustre" / "step_1"
    stage_root = tmp_path / "shm"
    _write_adapter(source, payload=b"x" * 1024)

    staged = stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)

    assert staged.parent == stage_root
    assert staged != source
    assert (staged / "adapter_model.safetensors").read_bytes() == b"x" * 1024
    assert (staged / "adapter_config.json").read_text() == '{"r": 64}\n'
    assert (staged / ".prime_rl_stage_manifest.json").is_file()


def test_stage_lora_adapter_reuses_verified_existing_stage(tmp_path):
    source = tmp_path / "lustre" / "step_1"
    stage_root = tmp_path / "shm"
    _write_adapter(source, payload=b"first")

    staged = stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)
    marker = staged / "reuse-marker"
    marker.write_text("kept")

    staged_again = stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)

    assert staged_again == staged
    assert marker.read_text() == "kept"


def test_stage_lora_adapter_restages_corrupt_existing_stage(tmp_path):
    source = tmp_path / "lustre" / "step_1"
    stage_root = tmp_path / "shm"
    _write_adapter(source, payload=b"correct")

    staged = stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)
    (staged / "adapter_model.safetensors").write_bytes(b"bad")

    staged_again = stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)

    assert staged_again == staged
    assert (staged / "adapter_model.safetensors").read_bytes() == b"correct"


def test_stage_lora_adapter_rejects_partial_source(tmp_path):
    source = tmp_path / "lustre" / "step_1"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "STABLE").touch()

    with pytest.raises(LoRAStagingError, match="adapter_config.json"):
        stage_lora_adapter(source, "debate-r64__v00000001", stage_root=tmp_path / "shm")


def test_stage_lora_adapter_fails_loudly_on_enospc(tmp_path, monkeypatch):
    source = tmp_path / "lustre" / "step_1"
    stage_root = tmp_path / "shm"
    _write_adapter(source, payload=b"adapter")

    def raise_enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr("prime_rl.inference.lora_staging.shutil.copy2", raise_enospc)

    with pytest.raises(LoRAStagingError, match="no space"):
        stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)

    assert not any(stage_root.glob("debate-r64*"))


def test_cleanup_staged_lora_adapter_removes_only_stage_root_children(tmp_path):
    stage_root = tmp_path / "shm"
    source = tmp_path / "lustre" / "step_1"
    _write_adapter(source, payload=b"adapter")
    staged = stage_lora_adapter(source, "debate-r64__v00000001", stage_root=stage_root)

    cleanup_staged_lora_adapter(staged, stage_root=stage_root)

    assert not staged.exists()
    cleanup_staged_lora_adapter(source, stage_root=stage_root)
    assert source.exists()
