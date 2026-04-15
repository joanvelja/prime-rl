import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .norms import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb

# flash-attention-2
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore

# flash-attention-3
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore


@dataclass
class AttentionConfig:
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    is_causal: bool
    attention_bias: bool
    use_qk_norm: bool
    rms_norm_eps: float
    qk_norm_type: Literal["per_head", "per_layer"] = "per_head"


# TODO: Does torch compile support config._attn_implementation forking?
# If so, we can combine FlashAttention and SDPAAttention into one class
# Otherwise, do ABC or something to make the signatures match


class FlashAttention(nn.Module):
    """Flash Attention"""

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                self.q_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
                )
            else:
                self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
                self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, max_seqlen):
        """Run the flash attention kernel. q/k/v are [total_tokens, heads, dim]."""
        args = [q, k, v, cu_seqlens, cu_seqlens]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen])
        kwargs: dict = {"causal": True}
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            kwargs["window_size"] = (sliding_window - 1, 0)
        out = self._flash_attn_call(*args, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        out = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)
        return out.contiguous().view(1, out.shape[0], -1)

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm and self.qk_norm_type == "per_layer":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.use_qk_norm and self.qk_norm_type == "per_head":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary application instead of double transpose?
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        return query_states, key_states, value_states

    def output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)

        attn_output = self._attention_core(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_output = self.output_proj(attn_output)
        return attn_output, None


class SDPAAttention(nn.Module):
    """SDPA Attention"""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                self.q_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
                )
            else:
                self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
                self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm and self.qk_norm_type == "per_layer":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.use_qk_norm and self.qk_norm_type == "per_head":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states, key_states, value_states

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        out = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)

        attn_output = self._attention_core(query_states, key_states, value_states)
        attn_output = self.output_proj(attn_output)
        return attn_output, None


ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(FlashAttention, flash_attn_version=2),
    "sdpa": SDPAAttention,
    "flash_attention_3": functools.partial(FlashAttention, flash_attn_version=3),
    "fa4": functools.partial(FlashAttention, flash_attn_version=4),
}


def _ring_sdpa_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    data_params: dict,
    process_group: "torch.distributed.ProcessGroup",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Ring attention using SDPA for head_dim > 256 (e.g. Gemma4 global layers).

    All-gathers full KV across the CP group, then runs SDPA locally on
    the local Q slice against the gathered KV.  Handles asymmetric Q/K lengths
    (K is the full gathered range, Q is the local chunk).
    """
    import torch.distributed as dist

    local_k_slice = data_params["local_k_slice"]
    cu_seqlens_q = data_params["cu_seqlens_q"]
    cu_seqlens_k = data_params["cu_seqlens_k"]

    # All-gather KV across CP ranks
    world_size = dist.get_world_size(process_group)
    total_k_len = k.shape[0] * world_size
    full_k = torch.empty((total_k_len, *k.shape[1:]), dtype=k.dtype, device=k.device)
    full_v = torch.empty((total_k_len, *v.shape[1:]), dtype=v.dtype, device=v.device)
    dist.all_gather_into_tensor(full_k, k.contiguous(), group=process_group)
    dist.all_gather_into_tensor(full_v, v.contiguous(), group=process_group)

    # Slice gathered KV to the range this rank's Q attends to
    k_sliced = full_k[local_k_slice]
    v_sliced = full_v[local_k_slice]

    # Reshape for SDPA: [tokens, heads, dim] → [1, heads, tokens, dim]
    q_4d = q.unsqueeze(0).transpose(1, 2)
    k_4d = k_sliced.unsqueeze(0).transpose(1, 2)
    v_4d = v_sliced.unsqueeze(0).transpose(1, 2)

    # GQA: expand KV heads to match Q heads
    if k_4d.shape[1] != q_4d.shape[1]:
        n_rep = q_4d.shape[1] // k_4d.shape[1]
        k_4d = k_4d.repeat_interleave(n_rep, dim=1)
        v_4d = v_4d.repeat_interleave(n_rep, dim=1)

    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5

    # Build rectangular block-diagonal causal mask [total_q, total_k]
    total_q = q.shape[0]
    total_k = k_sliced.shape[0]
    num_seqs = len(cu_seqlens_q) - 1

    if num_seqs > 1:
        mask = torch.full((total_q, total_k), float("-inf"), device=q.device, dtype=q.dtype)
        for i in range(num_seqs):
            q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
            q_len = q_end - q_start
            k_len = k_end - k_start
            # Causal: Q position j can attend to K positions 0..(j + offset)
            # where offset = k_len - q_len (Q starts at this offset in the full seq)
            offset = k_len - q_len
            seq_mask = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)
            seq_mask.masked_fill_(
                torch.triu(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool), diagonal=offset + 1),
                float("-inf"),
            )
            mask[q_start:q_end, k_start:k_end] = seq_mask
        out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, attn_mask=mask, scale=scale)
    else:
        q_len = total_q
        k_len = total_k
        offset = k_len - q_len
        mask = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)
        mask.masked_fill_(
            torch.triu(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool), diagonal=offset + 1),
            float("-inf"),
        )
        out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, attn_mask=mask, scale=scale)

    # [1, heads, total_q, dim] → [total_q, heads, dim]
    return out.transpose(1, 2).squeeze(0)


def substitute_ring_attn(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
    attn_impl: str = "flash_attention_2",
) -> None:
    """Patch _compute_attention on FlashAttention variants to use ring attention."""
    from ring_flash_attn import llama3_flash_attn_varlen_func

    from .ring_attn import ring_fa3_varlen_func

    use_fa3 = attn_impl == "flash_attention_3"
    ring_func = ring_fa3_varlen_func if use_fa3 else llama3_flash_attn_varlen_func

    def _ring_compute_attention(self, q, k, v, cu_seqlens, max_seqlen):
        from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

        window_size = (-1, -1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            window_size = (sliding_window - 1, 0)

        softmax_scale = getattr(self, "softmax_scale", None)

        # For head_dim > 256 (e.g. Gemma4 global layers), flash_attn doesn't
        # work so fall back to SDPA with all-gathered KV.  The ring function
        # already all-gathers KV; we just need to call SDPA on the local Q
        # slice against the full KV.
        head_dim = q.shape[-1]
        if head_dim > 256:
            return _ring_sdpa_attention(
                self, q, k, v, DATA_PARAMS, process_group, softmax_scale,
            )

        kwargs: dict = {
            "cu_seqlens_q": DATA_PARAMS["cu_seqlens_q"],
            "cu_seqlens_k": DATA_PARAMS["cu_seqlens_k"],
            "max_seqlen_q": DATA_PARAMS["max_seqlen_q"],
            "max_seqlen_k": DATA_PARAMS["max_seqlen_k"],
            "local_k_slice": DATA_PARAMS["local_k_slice"],
            "causal": True,
            "window_size": window_size,
            "group": process_group,
            "heads_k_stride": heads_k_stride,
        }
        if softmax_scale is not None:
            kwargs["softmax_scale"] = softmax_scale

        out = ring_func(q, k, v, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    FlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    AfmoeFlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedFlashAttention

    Qwen3_5MoeGatedFlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.gemma4.modeling_gemma4 import Gemma4Attention

    Gemma4Attention._compute_attention = _ring_compute_attention
