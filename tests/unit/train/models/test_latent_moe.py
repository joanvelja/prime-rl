import sys
from types import ModuleType, SimpleNamespace

import torch

import prime_rl.trainer.distributed.expert_parallel as expert_parallel_module
from prime_rl.trainer.models.layers.moe import (
    LatentMoE,
    _run_nongated_experts_for_loop_impl,
)


def _latent_moe_reference(model: LatentMoE, x: torch.Tensor) -> torch.Tensor:
    bs, slen, dim = x.shape
    x_flat = x.view(-1, dim)

    top_scores, selected_experts_indices, _ = model.router(x_flat, model.expert_bias)
    top_k = selected_experts_indices.shape[1]
    flat_experts = selected_experts_indices.reshape(-1)
    sort_order = torch.argsort(flat_experts, stable=True)
    top_scores_experts_sorted = top_scores.reshape(-1).index_select(0, sort_order)
    token_indices_experts_sorted = sort_order // top_k
    num_tokens_per_expert = torch.bincount(flat_experts, minlength=model.experts.num_experts).to(torch.int64)

    token_indices_expanded = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
    routed_input = torch.gather(x_flat, dim=0, index=token_indices_expanded)
    routed_input = model.fc1_latent_proj(routed_input)

    routed_output = _run_nongated_experts_for_loop_impl(
        model.experts.w1,
        model.experts.w2,
        model.experts.w3,
        routed_input,
        num_tokens_per_expert,
    )
    routed_output = (routed_output.float() * top_scores_experts_sorted.reshape(-1, 1)).to(routed_output.dtype)
    routed_output = routed_output * model.routed_scaling_factor
    routed_output = model.fc2_latent_proj(routed_output)

    out = model.shared_expert(x_flat)
    token_indices_full = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
    out = out.scatter_add(dim=0, index=token_indices_full, src=routed_output)
    return out.reshape(bs, slen, dim)


def test_latent_moe_deepep_matches_reference(monkeypatch) -> None:
    torch.manual_seed(0)
    model = LatentMoE(
        dim=6,
        latent_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=10,
        num_experts=4,
        top_k=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    deepep_model = LatentMoE(
        dim=6,
        latent_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=10,
        num_experts=4,
        top_k=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    deepep_model.load_state_dict(model.state_dict())
    deepep_model.set_ep_comm_backend("deepep")
    deepep_model.set_deepep_token_chunk_size(3)

    x = torch.randn(2, 4, 6)
    reference = _latent_moe_reference(model, x)

    monkeypatch.setattr(expert_parallel_module, "get_ep_group", lambda _experts: object())

    fake_deepep = ModuleType("prime_rl.trainer.distributed.deepep")

    def dispatch_tokens_async(
        hidden_states: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
        num_experts: int,
        group,
        *,
        score_before_experts: bool,
    ):
        del group
        assert not score_before_experts
        top_k = selected_experts_indices.shape[1]
        flat_experts = selected_experts_indices.reshape(-1)
        sort_order = torch.argsort(flat_experts, stable=True)
        token_indices = sort_order // top_k
        routed_scores = top_scores.reshape(-1).index_select(0, sort_order)
        num_tokens_per_expert = torch.bincount(flat_experts, minlength=num_experts).to(torch.int64)
        routed_hidden_states = hidden_states.index_select(0, token_indices)
        return SimpleNamespace(
            routed_hidden_states=routed_hidden_states,
            num_tokens_per_expert=num_tokens_per_expert,
            token_indices=token_indices,
            routed_scores=routed_scores,
            num_tokens=hidden_states.shape[0],
        )

    def finalize_dispatch_tokens(pending_state):
        return pending_state.routed_hidden_states, pending_state.num_tokens_per_expert, pending_state

    def combine_tokens(hidden_states: torch.Tensor, dispatch_state) -> torch.Tensor:
        weighted = hidden_states * dispatch_state.routed_scores.to(hidden_states.dtype).reshape(-1, 1)
        combined = hidden_states.new_zeros((dispatch_state.num_tokens, hidden_states.shape[1]))
        token_indices = dispatch_state.token_indices.reshape(-1, 1).expand_as(weighted)
        return combined.scatter_add(0, token_indices, weighted)

    fake_deepep.dispatch_tokens_async = dispatch_tokens_async
    fake_deepep.finalize_dispatch_tokens = finalize_dispatch_tokens
    fake_deepep.combine_tokens = combine_tokens
    fake_deepep.sync_combine = lambda: None
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.distributed.deepep", fake_deepep)

    monkeypatch.setattr(
        deepep_model,
        "_run_local_routed_experts",
        lambda hidden_states, num_tokens_per_expert: _run_nongated_experts_for_loop_impl(
            deepep_model.experts.w1,
            deepep_model.experts.w2,
            deepep_model.experts.w3,
            hidden_states,
            num_tokens_per_expert,
        ),
    )

    actual = deepep_model(x)

    torch.testing.assert_close(actual, reference)
