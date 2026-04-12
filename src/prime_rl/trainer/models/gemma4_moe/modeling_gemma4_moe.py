"""Gemma4 MoE custom implementation for PrimeRL training.

Extends the dense Gemma4 model with:
- 128 sparse experts (top-8) running in parallel with a shared dense MLP
- Custom router: RMSNorm (no scale) + learnable scale + softmax + renormalize + per_expert_scale
- Router input is the pre-MLP residual (not the MLP output)
- Expert activation: gelu_pytorch_tanh (not silu)
- MoE weight conversion: HF fused gate_up_proj → PrimeRL w1/w2/w3
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from prime_rl.trainer.models.gemma4.modeling_gemma4 import (
    Gemma4Attention,
    Gemma4MLP,
    Gemma4RMSNormNoScale,
    _get_flash_attn_version,
)
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig

# ---------------------------------------------------------------------------
# Gemma4 MoE Router
# ---------------------------------------------------------------------------


class Gemma4Router(nn.Module):
    """Gemma4 router: RMSNorm(no_scale) → scale → linear → softmax → topk → renorm → per_expert_scale."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = Gemma4RMSNormNoScale(self.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)
        router_probs = F.softmax(expert_scores.float(), dim=-1)

        top_k_weights, top_k_index = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        num_tokens_per_expert = torch.histc(
            top_k_index.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_k_weights.to(hidden_states.dtype), top_k_index, num_tokens_per_expert


# ---------------------------------------------------------------------------
# Gemma4 MoE Experts (w1/w2/w3 format with gelu_pytorch_tanh)
# ---------------------------------------------------------------------------


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    num_tokens_per_expert_list = num_tokens_per_expert.to(torch.int64).tolist()
    total_tokens = sum(num_tokens_per_expert_list)
    num_padding = x.shape[0] - total_tokens
    x_splits = torch.split(x[:total_tokens], num_tokens_per_expert_list, dim=0)
    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = act_fn(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_splits.append(h)
    out = torch.cat(out_splits, dim=0)
    if num_padding > 0:
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = act_fn(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out


class Gemma4Experts(nn.Module):
    """Expert weights in PrimeRL w1/w2/w3 format with configurable activation."""

    def __init__(self, config: Gemma4TextConfig, use_grouped_mm: bool = True):
        super().__init__()
        self.num_experts = config.num_experts
        dim = config.hidden_size
        hidden_dim = config.moe_intermediate_size
        self.w1 = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))
        self.act_fn = ACT2FN[config.hidden_activation]
        self.use_grouped_mm = use_grouped_mm

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_mm:
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.act_fn)
        return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.act_fn)


# ---------------------------------------------------------------------------
# MoE Decoder Layer
# ---------------------------------------------------------------------------


class Gemma4MoeDecoderLayer(GradientCheckpointingLayer):
    """Gemma4 decoder layer with shared MLP + sparse MoE in parallel."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(
            config, layer_idx, flash_attn_version=_get_flash_attn_version(config._attn_implementation)
        )
        self.mlp = Gemma4MLP(config)

        # Standard 4 layernorms
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.pre_feedforward_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_feedforward_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

        # MoE components
        self.router = Gemma4Router(config)
        self.experts = Gemma4Experts(config, use_grouped_mm=getattr(config, "use_grouped_mm", True))

        # Extra norms for MoE parallel path
        self.post_feedforward_layernorm_1 = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
        self.post_feedforward_layernorm_2 = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
        self.pre_feedforward_layernorm_2 = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

        # Per-layer scalar
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN + MoE parallel block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Shared MLP output normalized
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Sparse experts: router takes pre-MLP residual
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        top_k_weights, top_k_index, num_tokens_per_expert = self.router(hidden_states_flat)

        # Reorder tokens by expert assignment
        selected_flat = top_k_index.reshape(-1)
        token_indices_sorted = torch.argsort(selected_flat, stable=True)
        scores_sorted = top_k_weights.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.router.top_k

        dim = hidden_states_flat.shape[-1]
        routed_indices = token_indices_sorted.reshape(-1, 1).expand(-1, dim)

        # Pre-norm for expert input
        expert_input = self.pre_feedforward_layernorm_2(hidden_states_flat)
        routed_input = torch.gather(expert_input, dim=0, index=routed_indices)

        routed_output = self.experts(routed_input, num_tokens_per_expert)
        # Apply routing weights AFTER expert computation (gelu is nonlinear)
        routed_output = (routed_output.float() * scores_sorted.reshape(-1, 1)).to(routed_output.dtype)

        # Scatter back
        hidden_states_2 = torch.zeros_like(hidden_states_flat)
        hidden_states_2.scatter_add_(dim=0, index=routed_indices, src=routed_output)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine shared MLP + sparse experts
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states
