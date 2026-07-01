import json
import os
import time

import pytest

from prime_rl.orchestrator.utils import get_weight_dir, save_rollouts
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_save_rollouts_excludes_trajectory_when_requested(tmp_path):
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "trajectory": [{"prompt": "p", "completion": "c"}],
    }

    path = tmp_path / "train_rollouts.jsonl"

    save_rollouts([rollout], path, exclude_keys={"trajectory"})

    rows = _read_jsonl(path)
    assert rows == [{"example_id": 1, "reward": 1.0}]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_save_rollouts_preserves_full_trajectory(tmp_path):
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "trajectory": [{"prompt": "p", "completion": "c"}],
    }

    path = tmp_path / "train_rollouts_full.jsonl"

    save_rollouts([rollout], path)

    rows = _read_jsonl(path)
    assert rows == [rollout]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_get_weight_dir_ignores_stale_broadcast_marker(tmp_path):
    step_dir = get_step_path(get_broadcast_dir(tmp_path), 15)
    step_dir.mkdir(parents=True)
    stable = step_dir / "STABLE"
    stable.touch()
    old_mtime = time.time() - 100
    os.utime(stable, (old_mtime, old_mtime))

    with pytest.raises(FileNotFoundError):
        get_weight_dir(tmp_path, 15, min_stable_mtime=old_mtime + 1)

    fresh_mtime = time.time()
    os.utime(stable, (fresh_mtime, fresh_mtime))

    assert get_weight_dir(tmp_path, 15, min_stable_mtime=old_mtime + 1) == step_dir
