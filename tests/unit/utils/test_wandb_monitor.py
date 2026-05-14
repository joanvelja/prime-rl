import os

from prime_rl.utils.monitor.wandb import _ensure_wandb_storage_dirs


def test_wandb_storage_dirs_default_to_output_dir(tmp_path, monkeypatch):
    for key in ("WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_ARTIFACT_DIR"):
        monkeypatch.delenv(key, raising=False)

    _ensure_wandb_storage_dirs(tmp_path)

    assert (tmp_path / "wandb-local" / "data").is_dir()
    assert (tmp_path / "wandb-local" / "cache").is_dir()
    assert (tmp_path / "wandb-local" / "config").is_dir()
    assert (tmp_path / "wandb-local" / "artifacts").is_dir()
    assert (tmp_path / "wandb-local" / "data").as_posix() == os.environ["WANDB_DATA_DIR"]


def test_wandb_storage_dirs_preserve_explicit_env(tmp_path, monkeypatch):
    explicit_data_dir = tmp_path / "explicit-data"
    monkeypatch.setenv("WANDB_DATA_DIR", explicit_data_dir.as_posix())
    monkeypatch.delenv("WANDB_CACHE_DIR", raising=False)
    monkeypatch.delenv("WANDB_CONFIG_DIR", raising=False)
    monkeypatch.delenv("WANDB_ARTIFACT_DIR", raising=False)

    _ensure_wandb_storage_dirs(tmp_path)

    assert not explicit_data_dir.exists()
    assert (tmp_path / "wandb-local" / "cache").is_dir()
    assert (tmp_path / "wandb-local" / "config").is_dir()
    assert (tmp_path / "wandb-local" / "artifacts").is_dir()
