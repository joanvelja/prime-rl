from __future__ import annotations

from pathlib import Path

from scripts.evals.offline_omni_math2_ckpt_eval import _discover_weight_steps, _range_filters_for_explicit_steps


def _write_stable_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "STABLE").touch()
    (path / "model.safetensors.index.json").write_text("{}")


def test_explicit_steps_disable_range_filters() -> None:
    assert _range_filters_for_explicit_steps({25}, step_interval=50, min_step=50, max_step=100) == (None, None, None)
    assert _range_filters_for_explicit_steps(None, step_interval=50, min_step=50, max_step=100) == (50, 50, 100)


def test_discover_weight_steps_can_select_non_interval_explicit_step(tmp_path: Path) -> None:
    _write_stable_checkpoint(tmp_path / "step_25")
    step_interval, min_step, max_step = _range_filters_for_explicit_steps(
        {25},
        step_interval=50,
        min_step=None,
        max_step=None,
    )

    discovered = _discover_weight_steps(
        tmp_path,
        steps={25},
        step_interval=step_interval,
        min_step=min_step,
        max_step=max_step,
    )

    assert discovered == [(25, tmp_path / "step_25")]
