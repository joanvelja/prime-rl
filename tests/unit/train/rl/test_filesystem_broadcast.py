from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.rl.broadcast import filesystem
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast


def test_filesystem_broadcast_raises_on_write_failure(monkeypatch, tmp_path) -> None:
    broadcast = object.__new__(FileSystemWeightBroadcast)
    broadcast.lora_config = SimpleNamespace(dropout=0.0)
    broadcast.save_format = "safetensors"
    broadcast.save_sharded = False
    broadcast.stripe_enabled = False
    broadcast.stripe_count = 16
    broadcast.stripe_size = "1M"
    broadcast._striped_broadcast_dirs = set()
    broadcast.world = SimpleNamespace(is_master=True)
    broadcast.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )

    run_dir = tmp_path / "run_default"
    manager = SimpleNamespace(
        ready_to_update_idxs=[0],
        ready_to_update=[True],
        progress={0: SimpleNamespace(step=3)},
        config={
            0: SimpleNamespace(student=SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(rank=64, alpha=128))))
        },
        idx_2_id={0: "run_default"},
        get_state_dict_for_run=lambda idx: {"base_model.model.foo.lora_A.weight": torch.ones(1)},
        get_run_dir=lambda idx: run_dir,
        get_orchestrator_config=lambda run_id: object(),
    )
    broadcast.multi_run_manager = manager

    def fail_save(*args, **kwargs) -> None:
        raise OSError("scratch write failed")

    monkeypatch.setattr(filesystem, "save_state_dict", fail_save)

    with pytest.raises(OSError, match="scratch write failed"):
        broadcast.broadcast_weights(model=object(), step=3)

    assert manager.ready_to_update[0] is False
    assert not (run_dir / "broadcasts" / "step_3" / "STABLE").exists()


def test_filesystem_broadcast_stripes_broadcast_dir_before_save(monkeypatch, tmp_path) -> None:
    broadcast = object.__new__(FileSystemWeightBroadcast)
    broadcast.lora_config = None
    broadcast.save_format = "safetensors"
    broadcast.save_sharded = False
    broadcast.world = SimpleNamespace(is_master=True)
    broadcast.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    broadcast.stripe_enabled = True
    broadcast.stripe_count = 16
    broadcast.stripe_size = "1M"
    broadcast._striped_broadcast_dirs = set()

    calls = []
    monkeypatch.setattr(filesystem.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))

    broadcast._ensure_broadcast_dir_striped(tmp_path / "run_default" / "broadcasts")

    assert calls == [
        (
            ["lfs", "setstripe", "-c", "16", "-S", "1M", (tmp_path / "run_default" / "broadcasts").as_posix()],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_filesystem_broadcast_missing_lfs_is_nonfatal(monkeypatch, tmp_path) -> None:
    broadcast = object.__new__(FileSystemWeightBroadcast)
    broadcast.lora_config = None
    broadcast.world = SimpleNamespace(is_master=True)
    broadcast.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    broadcast.stripe_enabled = True
    broadcast.stripe_count = 16
    broadcast.stripe_size = "1M"
    broadcast._striped_broadcast_dirs = set()

    def missing_lfs(*args, **kwargs):
        raise FileNotFoundError("lfs")

    monkeypatch.setattr(filesystem.subprocess, "run", missing_lfs)

    broadcast._ensure_broadcast_dir_striped(tmp_path / "run_default" / "broadcasts")

    assert (tmp_path / "run_default" / "broadcasts") in broadcast._striped_broadcast_dirs
