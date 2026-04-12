import importlib.util
import sys
import uuid
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MOE_MODULE_PATH = PROJECT_ROOT / "src/prime_rl/trainer/models/layers/moe.py"


def _load_moe_module():
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    module_name = f"_prime_rl_test_moe_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MOE_MODULE_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    del offs
    return torch.zeros((x.shape[0], w.shape[-1]), dtype=x.dtype, device=x.device)


def _run_split_directly(func, w1, w2, w3, x, num_tokens_per_expert):
    return func(w1, w2, w3, x, num_tokens_per_expert)


def test_moe_selective_ac_grouped_mm_restores_input_dtype(monkeypatch) -> None:
    moe_module = _load_moe_module()
    monkeypatch.setattr(moe_module.torch, "_grouped_mm", _fake_grouped_mm)
    monkeypatch.setattr(moe_module, "_run_split_expert_parallel", _run_split_directly)

    counts = torch.tensor([2, 1], dtype=torch.int32)
    x = torch.randn(3, 4, dtype=torch.float16)
    moe = moe_module.MoE(
        moe_args=moe_module.MoEArgs(num_experts=2, num_shared_experts=0, use_grouped_mm=True),
        dim=4,
        hidden_dim=6,
    )
    expected = moe_module._run_experts_grouped_mm_impl(moe.experts.w1, moe.experts.w2, moe.experts.w3, x, counts)

    setattr(moe, moe_module._SELECTIVE_AC_PATCHED_METHODS_ATTR, frozenset({"_run_moe_act"}))
    actual = moe._run_local_routed_experts(x, counts)

    assert expected.dtype == x.dtype
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected)


def test_latent_moe_selective_ac_grouped_mm_restores_input_dtype(monkeypatch) -> None:
    moe_module = _load_moe_module()
    monkeypatch.setattr(moe_module.torch, "_grouped_mm", _fake_grouped_mm)
    monkeypatch.setattr(moe_module, "_run_split_expert_parallel", _run_split_directly)

    counts = torch.tensor([2, 1], dtype=torch.int32)
    x = torch.randn(3, 4, dtype=torch.float16)
    moe = moe_module.LatentMoE(
        dim=4,
        latent_dim=None,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=6,
        num_experts=2,
        top_k=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        use_grouped_mm=True,
        load_balance_coeff=None,
    )
    expected = moe_module._run_nongated_experts_grouped_mm_impl(
        moe.experts.w1,
        moe.experts.w2,
        moe.experts.w3,
        x,
        counts,
    )

    setattr(moe, moe_module._SELECTIVE_AC_PATCHED_METHODS_ATTR, frozenset({"_run_moe_act"}))
    actual = moe._run_local_routed_experts(x, counts)

    assert expected.dtype == x.dtype
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected)
