import pytest

from prime_rl.utils.pathing import clean_future_steps, validate_output_dir


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_only_logs_passes(tmp_path):
    output_dir = tmp_path / "has_logs"
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "logs" / "trainer.log").touch()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    (output_dir / "logs").mkdir()

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=True)
    assert not output_dir.exists()


def _make_step_dirs(base: str, output_dir, steps: list[int]) -> None:
    d = output_dir / base
    d.mkdir(parents=True, exist_ok=True)
    for s in steps:
        (d / f"step_{s}").mkdir()
        (d / f"step_{s}" / "data.bin").touch()


def test_clean_future_steps_removes_only_future(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    steps = [0, 50, 100, 150, 200]

    _make_step_dirs("rollouts", output_dir, steps)
    _make_step_dirs("checkpoints", output_dir, steps)
    _make_step_dirs("weights", output_dir, steps)
    _make_step_dirs("broadcasts", output_dir, steps)

    clean_future_steps(output_dir, resume_step=100)

    for subdir in ["rollouts", "checkpoints", "weights", "broadcasts"]:
        remaining = {p.name for p in (output_dir / subdir).iterdir()}
        assert remaining == {"step_0", "step_50", "step_100"}


def test_clean_future_steps_with_ckpt_output_dir(tmp_path):
    output_dir = tmp_path / "run"
    ckpt_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    ckpt_dir.mkdir()
    steps = [0, 50, 100, 150]

    _make_step_dirs("rollouts", output_dir, steps)
    _make_step_dirs("checkpoints", ckpt_dir, steps)
    _make_step_dirs("weights", ckpt_dir, steps)

    clean_future_steps(output_dir, resume_step=50, ckpt_output_dir=ckpt_dir)

    remaining_rollouts = sorted(p.name for p in (output_dir / "rollouts").iterdir())
    assert remaining_rollouts == ["step_0", "step_50"]

    remaining_ckpts = sorted(p.name for p in (ckpt_dir / "checkpoints").iterdir())
    assert remaining_ckpts == ["step_0", "step_50"]

    remaining_weights = sorted(p.name for p in (ckpt_dir / "weights").iterdir())
    assert remaining_weights == ["step_0", "step_50"]


def test_clean_future_steps_cleans_run_default(tmp_path):
    output_dir = tmp_path / "run"
    run_default = output_dir / "run_default"
    output_dir.mkdir()
    steps = [0, 100, 200]

    _make_step_dirs("rollouts", run_default, steps)
    _make_step_dirs("broadcasts", run_default, steps)

    clean_future_steps(output_dir, resume_step=100)

    remaining_rollouts = sorted(p.name for p in (run_default / "rollouts").iterdir())
    assert remaining_rollouts == ["step_0", "step_100"]

    remaining_broadcasts = sorted(p.name for p in (run_default / "broadcasts").iterdir())
    assert remaining_broadcasts == ["step_0", "step_100"]


def test_clean_future_steps_noop_when_no_dirs(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    clean_future_steps(output_dir, resume_step=100)
