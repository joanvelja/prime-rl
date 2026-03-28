import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


MOE_MODULE_PATH = Path(__file__).resolve().parents[4] / "src/prime_rl/trainer/models/layers/moe.py"


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_moe_module(monkeypatch: pytest.MonkeyPatch):
    torchtitan_module = _package("torchtitan")
    torchtitan_distributed_module = _package("torchtitan.distributed")
    torchtitan_expert_parallel_module = types.ModuleType("torchtitan.distributed.expert_parallel")
    torchtitan_expert_parallel_module.expert_parallel = lambda fn: fn
    torchtitan_module.distributed = torchtitan_distributed_module
    torchtitan_distributed_module.expert_parallel = torchtitan_expert_parallel_module

    monkeypatch.setitem(sys.modules, "torchtitan", torchtitan_module)
    monkeypatch.setitem(sys.modules, "torchtitan.distributed", torchtitan_distributed_module)
    monkeypatch.setitem(sys.modules, "torchtitan.distributed.expert_parallel", torchtitan_expert_parallel_module)

    spec = importlib.util.spec_from_file_location("test_moe_layers_module", MOE_MODULE_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _install_deepep_stubs(monkeypatch: pytest.MonkeyPatch, blocked_calls: list[str]) -> None:
    def unexpected_call(name: str):
        def _raise(*args, **kwargs):
            blocked_calls.append(name)
            raise AssertionError(f"{name} should not be called for empty batches")

        return _raise

    prime_rl_module = _package("prime_rl")
    trainer_module = _package("prime_rl.trainer")
    distributed_module = _package("prime_rl.trainer.distributed")
    deepep_module = types.ModuleType("prime_rl.trainer.distributed.deepep")
    expert_parallel_module = types.ModuleType("prime_rl.trainer.distributed.expert_parallel")

    deepep_module.combine_tokens = unexpected_call("combine_tokens")
    deepep_module.dispatch_tokens_async = unexpected_call("dispatch_tokens_async")
    deepep_module.finalize_dispatch_tokens = unexpected_call("finalize_dispatch_tokens")
    deepep_module.sync_combine = unexpected_call("sync_combine")
    expert_parallel_module.get_ep_group = unexpected_call("get_ep_group")

    prime_rl_module.trainer = trainer_module
    trainer_module.distributed = distributed_module
    distributed_module.deepep = deepep_module
    distributed_module.expert_parallel = expert_parallel_module

    monkeypatch.setitem(sys.modules, "prime_rl", prime_rl_module)
    monkeypatch.setitem(sys.modules, "prime_rl.trainer", trainer_module)
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.distributed.deepep", deepep_module)
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.distributed.expert_parallel", expert_parallel_module)


class _TrackingSharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.clone()


class _ExplodingModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError(f"{self.name} should not run for empty batches")


@pytest.mark.parametrize("num_shared_experts", [0, 1])
def test_moe_deepep_empty_batch_skips_dispatch(monkeypatch: pytest.MonkeyPatch, num_shared_experts: int) -> None:
    moe_module = _load_moe_module(monkeypatch)
    blocked_calls: list[str] = []
    _install_deepep_stubs(monkeypatch, blocked_calls)

    moe = moe_module.MoE(
        moe_module.MoEArgs(
            num_experts=2,
            num_shared_experts=num_shared_experts,
            top_k=1,
            use_grouped_mm=False,
        ),
        dim=4,
        hidden_dim=8,
    )
    moe.set_deepep_token_chunk_size(32)

    shared_expert = _TrackingSharedExpert()
    if num_shared_experts > 0:
        moe.shared_expert = shared_expert

    x = torch.empty((0, 4))
    selected_experts_indices = torch.empty((0, 1), dtype=torch.long)
    top_scores = torch.empty((0, 1), dtype=x.dtype)

    output = moe._run_deepep_routed_experts(x, selected_experts_indices, top_scores)

    assert output.shape == x.shape
    assert output.numel() == 0
    assert shared_expert.calls == num_shared_experts
    assert blocked_calls == []


def test_latent_moe_deepep_empty_batch_skips_projection_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    moe_module = _load_moe_module(monkeypatch)
    blocked_calls: list[str] = []
    _install_deepep_stubs(monkeypatch, blocked_calls)

    latent_moe = moe_module.LatentMoE(
        dim=4,
        latent_dim=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=6,
        num_experts=2,
        top_k=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    latent_moe.set_deepep_token_chunk_size(32)

    shared_expert = _TrackingSharedExpert()
    latent_moe.shared_expert = shared_expert
    latent_moe.fc1_latent_proj = _ExplodingModule("fc1_latent_proj")
    latent_moe.fc2_latent_proj = _ExplodingModule("fc2_latent_proj")

    x = torch.empty((0, 4))
    selected_experts_indices = torch.empty((0, 1), dtype=torch.long)
    top_scores = torch.empty((0, 1), dtype=x.dtype)

    output = latent_moe._run_deepep_routed_experts(x, selected_experts_indices, top_scores)

    assert output.shape == x.shape
    assert output.numel() == 0
    assert shared_expert.calls == 1
    assert blocked_calls == []
