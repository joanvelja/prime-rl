import pytest
import torch

from prime_rl.trainer.models.layers.extra_expert import ExtraExpert, apply_extra_expert
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs

pytestmark = [pytest.mark.gpu]

DIM = 64
HIDDEN_DIM = 128
NUM_EXPERTS = 4
TOP_K = 2
BS, SLEN = 2, 8


def make_moe(**overrides):
    kwargs = dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        use_grouped_mm=False,
        score_func="softmax",
        num_shared_experts=0,
        load_balance_coeff=None,
    )
    kwargs.update(overrides)
    args = MoEArgs(**kwargs)
    with torch.device("cuda"):
        moe = MoE(args, dim=DIM, hidden_dim=HIDDEN_DIM)
        moe.init_weights(init_std=0.02, buffer_device=torch.device("cuda"))
    return moe


def test_new_expert_w2_is_zero():
    moe = make_moe()
    ee = ExtraExpert(moe)
    assert (ee.new_w2 == 0).all()


def test_original_params_frozen():
    moe = make_moe()
    ee = ExtraExpert(moe)
    for name, p in ee.moe.named_parameters():
        assert not p.requires_grad, f"Original param {name} should be frozen"


def test_new_params_trainable():
    moe = make_moe()
    ee = ExtraExpert(moe)
    for name, p in ee.named_parameters():
        if name.startswith("new_"):
            assert p.requires_grad, f"New param {name} should be trainable"


def test_total_experts_increased():
    moe = make_moe()
    ee = ExtraExpert(moe)
    assert ee.total_experts == NUM_EXPERTS + 1


def test_forward_shape():
    moe = make_moe()
    ee = ExtraExpert(moe).cuda()
    x = torch.randn(BS, SLEN, DIM, device="cuda")
    out = ee(x)
    assert out.shape == (BS, SLEN, DIM)


def test_new_expert_output_is_zero():
    """With w2=0, tokens routed to the new expert get zero contribution."""
    moe = make_moe()
    ee = ExtraExpert(moe).cuda()

    x = torch.randn(4, DIM, device="cuda")
    h = torch.nn.functional.silu(x @ ee.new_w1[0].T) * (x @ ee.new_w3[0].T)
    out = h @ ee.new_w2[0].T
    assert (out == 0).all(), "New expert output should be zero at init"


def test_initial_output_close_to_original():
    """The new expert contributes zero, so the output should be close to
    the original MoE. Not exact because softmax routing is slightly
    perturbed by the extra expert logit."""
    moe = make_moe()
    moe.eval()
    x = torch.randn(BS, SLEN, DIM, device="cuda")

    with torch.no_grad():
        orig_out = moe(x)

    ee = ExtraExpert(moe).cuda()
    ee.eval()
    with torch.no_grad():
        ee_out = ee(x)

    assert torch.allclose(orig_out, ee_out, atol=1e-3), f"Max diff: {(orig_out - ee_out).abs().max()}"


def test_gradients_flow_to_new_params():
    moe = make_moe()
    ee = ExtraExpert(moe).cuda()
    x = torch.randn(BS, SLEN, DIM, device="cuda")
    out = ee(x)
    out.sum().backward()

    for name, p in ee.named_parameters():
        if name.startswith("new_") and p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_gradients_do_not_flow_to_frozen_params():
    moe = make_moe()
    ee = ExtraExpert(moe).cuda()
    x = torch.randn(BS, SLEN, DIM, device="cuda")
    out = ee(x)
    out.sum().backward()

    for name, p in ee.moe.named_parameters():
        assert p.grad is None or (p.grad == 0).all(), f"Frozen param {name} should have no gradient"


def test_apply_extra_expert():
    """apply_extra_expert should replace MoE modules with ExtraExpert wrappers."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            args = MoEArgs(
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                use_grouped_mm=False,
                num_shared_experts=0,
                load_balance_coeff=None,
            )
            with torch.device("cuda"):
                self.layer0 = MoE(args, dim=DIM, hidden_dim=HIDDEN_DIM)
                self.layer0.init_weights(init_std=0.02, buffer_device=torch.device("cuda"))
                self.layer1 = MoE(args, dim=DIM, hidden_dim=HIDDEN_DIM)
                self.layer1.init_weights(init_std=0.02, buffer_device=torch.device("cuda"))

    model = DummyModel()
    apply_extra_expert(model)

    assert isinstance(model.layer0, ExtraExpert)
    assert isinstance(model.layer1, ExtraExpert)


def test_with_shared_expert():
    moe = make_moe(num_shared_experts=1)
    ee = ExtraExpert(moe).cuda()
    x = torch.randn(BS, SLEN, DIM, device="cuda")
    out = ee(x)
    assert out.shape == (BS, SLEN, DIM)
    out.sum().backward()
