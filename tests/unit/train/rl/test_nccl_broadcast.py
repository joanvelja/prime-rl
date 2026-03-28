import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.worker.nccl import NCCLWeightBroadcastReceiver
from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast, NCCLWeightBroadcastSender


@pytest.mark.gpu
@pytest.mark.skip(reason="Skipping NCCL broadcast as it fail only in ci")
def test_nccl_broadcast(free_port):
    host = "localhost"
    free_port = free_port()

    def send():
        device = torch.device(f"cuda:{0}")
        nccl_broadcast = NCCLWeightBroadcastSender(
            host=host, port=free_port, rank=0, world_size=2, device=device, timeout=10
        )

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(10)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SubModel()

            def forward(self, x):
                return self.model(x)

        model = Model().to(device)
        for param in model.parameters():
            param.data = torch.ones_like(param.data)

        nccl_broadcast.broadcast_weights(model, step=0)

    def receive():
        device = torch.device(f"cuda:{1}")
        nccl_broadcast = NCCLWeightBroadcastReceiver(
            host=host, port=free_port, rank=1, world_size=2, device=device, timeout=10
        )

        for key, value in nccl_broadcast.receive_state_dict():
            assert value.allclose(torch.ones_like(value))

    processes = [mp.Process(target=send), mp.Process(target=receive)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        assert process.exitcode == 0, f"Process {process.name} exited with code {process.exitcode}"


def test_pipelined_completed_delta_stats_keep_originating_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.nccl.get_world", lambda: SimpleNamespace(is_master=False))
    monkeypatch.setattr(
        "prime_rl.trainer.rl.broadcast.nccl.get_multi_run_manager",
        lambda: SimpleNamespace(used_idxs=[], ready_to_update={}),
    )

    config = NCCLWeightBroadcastConfig(host="localhost", port=1234, inference_world_size=1, delta_compression=True)
    broadcast = NCCLWeightBroadcast(tmp_path, config, device="cpu")

    completed_steps: list[int] = []

    class DummyThread:
        def join(self) -> None:
            completed_steps.append(1)
            broadcast.sender.last_delta_stats = {"weight_broadcast/delta_sparsity": 0.9}

    broadcast._bg_thread = DummyThread()
    broadcast._bg_step = 1
    broadcast.sender.last_delta_stats = {"weight_broadcast/delta_sparsity": 0.5}

    broadcast._join_background()

    assert completed_steps == [1]
    assert broadcast.delta_stats == {"weight_broadcast/delta_sparsity": 0.9}
    assert broadcast.delta_stats_step == 1
    assert broadcast.pop_completed_delta_stats() == (1, {"weight_broadcast/delta_sparsity": 0.9})
    assert broadcast.pop_completed_delta_stats() is None


def test_flush_completed_delta_stats_joins_pending_background(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.nccl.get_world", lambda: SimpleNamespace(is_master=False))
    monkeypatch.setattr(
        "prime_rl.trainer.rl.broadcast.nccl.get_multi_run_manager",
        lambda: SimpleNamespace(used_idxs=[], ready_to_update={}),
    )

    config = NCCLWeightBroadcastConfig(host="localhost", port=1234, inference_world_size=1, delta_compression=True)
    broadcast = NCCLWeightBroadcast(tmp_path, config, device="cpu")

    class DummyThread:
        def join(self) -> None:
            broadcast.sender.last_delta_stats = {"weight_broadcast/delta_sparsity": 0.75}

    broadcast._bg_thread = DummyThread()
    broadcast._bg_step = 7

    assert broadcast.flush_completed_delta_stats() == (7, {"weight_broadcast/delta_sparsity": 0.75})
    assert broadcast._bg_thread is None
    assert broadcast._bg_step is None
