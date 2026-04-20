"""Round-trip RAEState through CheckpointManager.save → load.

Skipped if torch isn't importable from the test runner (the prime-rl
deps are linux-only on Darwin; the verifiers venv we run from doesn't
ship torch). Locally on Linux with torch installed, these run as part
of the orchestrator suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import importlib.util  # noqa: E402

# pydantic_config gates CheckpointConfig import; skip cleanly when absent.
if importlib.util.find_spec("pydantic_config") is None:
    pytest.skip("pydantic_config not available", allow_module_level=True)

from pathlib import Path  # noqa: E402

from prime_rl.configs.orchestrator import BufferConfig, CheckpointConfig  # noqa: E402
from prime_rl.orchestrator.buffer import Buffer  # noqa: E402
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress  # noqa: E402
from prime_rl.orchestrator.multi_agent_advantage import RAEState  # noqa: E402


def _empty_buffer() -> Buffer:
    """Buffer with no rows — sufficient for save/load smoke."""
    from datasets import Dataset

    ds = Dataset.from_dict({"prompt": [], "answer": [], "task": [], "info": [], "example_id": []})
    return Buffer(ds, env_names=["dummy"], config=BufferConfig())


def test_rae_state_round_trips(tmp_path: Path) -> None:
    """RAEState.baselines + momentum survive save → load via the
    orchestrator checkpoint manager."""
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    cfg = CheckpointConfig(interval=1)
    mgr = CheckpointManager(ckpt_dir, cfg)

    progress = Progress(step=7)
    buffer = _empty_buffer()
    rae_state = RAEState(
        baselines={
            ("debate_v1", 1, "debater_a"): 0.3,
            ("debate_v1", 1, "debater_b"): -0.1,
            ("debate_v1", 2, "debater_a"): 0.55,
        },
        momentum=0.85,
    )
    mgr.save(progress, buffer, step=7, rae_state=rae_state)

    # Reload into a fresh state — must match exactly.
    progress_reload = Progress()
    rae_reload = RAEState(momentum=0.0)  # different starting momentum: load must overwrite
    mgr.load(progress_reload, buffer, step=7, rae_state=rae_reload)
    assert progress_reload.step == 7
    assert rae_reload.momentum == 0.85
    assert rae_reload.baselines == rae_state.baselines


def test_rae_state_load_fails_loud_when_missing(tmp_path: Path) -> None:
    """Single-agent checkpoint (no rae_state.pt) → loading with rae_state
    set must FileNotFoundError. Silent cold-start would discard EMA
    history without the operator noticing."""
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    mgr = CheckpointManager(ckpt_dir, CheckpointConfig(interval=1))
    progress = Progress(step=3)
    buffer = _empty_buffer()
    # Save WITHOUT rae_state — single-agent shape.
    mgr.save(progress, buffer, step=3, rae_state=None)

    rae_reload = RAEState(momentum=0.9)
    with pytest.raises(FileNotFoundError, match="RAE state not found"):
        mgr.load(Progress(), buffer, step=3, rae_state=rae_reload)


def test_save_without_rae_state_omits_file(tmp_path: Path) -> None:
    """Single-agent runs MUST NOT write rae_state.pt — the absence is the
    signal that this checkpoint predates the multi-agent path."""
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    mgr = CheckpointManager(ckpt_dir, CheckpointConfig(interval=1))
    mgr.save(Progress(step=1), _empty_buffer(), step=1, rae_state=None)
    assert not (mgr.get_ckpt_path(1) / "rae_state.pt").exists()
