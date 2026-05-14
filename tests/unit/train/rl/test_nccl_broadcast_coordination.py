from types import SimpleNamespace

from prime_rl.trainer.rl.broadcast import nccl
from prime_rl.trainer.rl.broadcast.nccl import TRAINER_NCCL_READY_MARKER, NCCLWeightBroadcast


class FakeMultiRunManager:
    def __init__(self, run_dir, step: int = 11):
        self.used_idxs = [0]
        self.ready_to_update = [True]
        self.progress = {0: SimpleNamespace(step=step)}
        self._run_dir = run_dir

    def get_run_dir(self, idx: int):
        assert idx == 0
        return self._run_dir


def test_non_master_waits_for_master_nccl_marker_before_broadcast(tmp_path, monkeypatch):
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.multi_run_manager = FakeMultiRunManager(tmp_path)
    broadcast.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)

    waited_paths = []

    def fake_wait_for_path(path, interval=1, log_interval=10):
        waited_paths.append(path)

    monkeypatch.setattr(nccl, "sync_wait_for_path", fake_wait_for_path)

    pending_runs = broadcast._compute_notified_runs()
    broadcast._wait_for_master_ready(pending_runs)

    assert waited_paths == [tmp_path / "broadcasts" / "step_11" / TRAINER_NCCL_READY_MARKER]
    assert broadcast.multi_run_manager.ready_to_update[0] is False


def test_master_marks_trainer_ranks_ready(tmp_path):
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    save_dir = tmp_path / "broadcasts" / "step_11"
    save_dir.mkdir(parents=True)

    broadcast._mark_trainer_ranks_ready([(0, save_dir)])

    assert (save_dir / TRAINER_NCCL_READY_MARKER).exists()
