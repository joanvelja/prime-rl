"""
Extra Expert: PEFT method that adds a trainable expert to a frozen MoE layer.

The new expert's down-projection (w2) is zero-initialized so the model's
initial output is unchanged. Only the new expert's weights and its router
gate column are trainable.
"""

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.moe import (
    MoE,
    _run_experts_for_loop_impl,
    _run_experts_grouped_mm_impl,
)


class ExtraExpert(nn.Module):
    """Adds a single trainable expert to a frozen MoE layer.

    The new expert's w2 (down-projection) is zero-initialized so the expert
    initially produces zero output, preserving the pretrained model's behavior.
    """

    def __init__(self, moe: MoE, gate_bias_init: float = 0.0):
        super().__init__()

        self.moe = moe
        for p in moe.parameters():
            p.requires_grad = False

        orig = moe.experts
        self.orig_num_experts = orig.num_experts
        self.total_experts = orig.num_experts + 1
        self.top_k = moe.router.top_k
        self.score_func = moe.router.score_func
        self.route_norm = moe.router.route_norm
        self.route_scale = moe.router.route_scale
        self.score_before_experts = moe.score_before_experts
        self.use_grouped_mm = orig.use_grouped_mm
        self.gate_bias_init = gate_bias_init

        _, hidden_dim, dim = orig.w1.shape
        device = orig.w1.device
        dtype = orig.w1.dtype

        # New expert: w2 zero-initialized so output starts at zero
        self.new_w1 = nn.Parameter(torch.empty(1, hidden_dim, dim, device=device, dtype=dtype))
        self.new_w2 = nn.Parameter(torch.zeros(1, dim, hidden_dim, device=device, dtype=dtype))
        self.new_w3 = nn.Parameter(torch.empty(1, hidden_dim, dim, device=device, dtype=dtype))

        # Router extension: gate column for the new expert
        self.new_gate_weight = nn.Parameter(torch.empty(1, dim, device=device, dtype=dtype))
        # Learnable bias on the new expert's logit
        self.gate_bias = nn.Parameter(torch.tensor([gate_bias_init], device=device, dtype=dtype))

        self._init_extra_expert_parameters()

    def _init_extra_expert_parameters(self):
        nn.init.trunc_normal_(self.new_w1, mean=0.0, std=0.02)
        nn.init.zeros_(self.new_w2)
        nn.init.trunc_normal_(self.new_w3, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.new_gate_weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_bias, self.gate_bias_init)

    def _extended_weights(self):
        return (
            torch.cat([self.moe.experts.w1, self.new_w1], dim=0),
            torch.cat([self.moe.experts.w2, self.new_w2], dim=0),
            torch.cat([self.moe.experts.w3, self.new_w3], dim=0),
        )

    def _extended_gate(self):
        return torch.cat([self.moe.router.gate.weight, self.new_gate_weight], dim=0)

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        # Routing with extended gate
        scores = x_flat @ self._extended_gate().T
        # Add learnable bias to new expert's logit (cat-based to avoid in-place ops for torch.compile)
        bias = torch.cat(
            [
                torch.zeros(self.orig_num_experts, device=scores.device, dtype=scores.dtype),
                self.gate_bias,
            ]
        )
        scores = scores + bias

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.float())
        else:
            scores = F.softmax(scores.float(), dim=1)

        top_scores, selected_experts = torch.topk(scores, k=self.top_k, dim=1)

        if self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale

        # Reorder tokens by expert
        selected_flat = selected_experts.reshape(-1)
        num_tokens_per_expert = torch.histc(
            selected_flat,
            bins=self.total_experts,
            min=0,
            max=self.total_experts,
        )
        token_indices_sorted = torch.argsort(selected_flat, stable=True)
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.top_k

        # Gather routed input
        routed_indices = token_indices_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x_flat, dim=0, index=routed_indices)

        if self.score_before_experts:
            routed_input = (routed_input.float() * top_scores_sorted.reshape(-1, 1)).to(x.dtype)

        # Run all experts (original + new) together
        ext_w1, ext_w2, ext_w3 = self._extended_weights()
        if self.use_grouped_mm:
            routed_output = _run_experts_grouped_mm_impl(
                ext_w1,
                ext_w2,
                ext_w3,
                routed_input,
                num_tokens_per_expert,
            )
        else:
            routed_output = _run_experts_for_loop_impl(
                ext_w1,
                ext_w2,
                ext_w3,
                routed_input,
                num_tokens_per_expert,
            )

        if not self.score_before_experts:
            routed_output = (routed_output.float() * top_scores_sorted.reshape(-1, 1)).to(x.dtype)

        # Scatter back and add shared expert
        out = self.moe.shared_expert(x_flat) if self.moe.shared_expert is not None else torch.zeros_like(x_flat)
        out = out.scatter_add(dim=0, index=routed_indices, src=routed_output)
        return out.reshape(bs, slen, dim)


def strip_extra_expert_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip ExtraExpert-specific keys and remap wrapped MoE keys for checkpoint loading."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if any(k in key for k in ("new_w1", "new_w2", "new_w3", "new_gate_weight", "gate_bias")):
            continue
        # Remap wrapped keys: ...mlp.moe.X -> ...mlp.X
        new_key = key.replace(".moe.", ".", 1) if ".moe." in key else key
        new_state_dict[new_key] = value
    return new_state_dict


def apply_extra_expert(model: nn.Module, gate_bias_init: float = 0.0) -> nn.Module:
    """Replace all MoE layers in a model with ExtraExpert wrappers.

    Only the new expert parameters are trainable; everything else is frozen.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            replacements.append((name, module))

    for name, module in replacements:
        wrapper = ExtraExpert(module, gate_bias_init=gate_bias_init)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)

    return model
