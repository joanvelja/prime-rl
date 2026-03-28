from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedAttentionConfig as Qwen3_5GatedAttentionConfig,
    Qwen3_5MoeGatedDeltaNet as Qwen3_5GatedDeltaNet,
    Qwen3_5MoeGatedFlashAttention as Qwen3_5GatedFlashAttention,
    Qwen3_5MoeGatedSDPAAttention as Qwen3_5GatedSDPAAttention,
    Qwen3_5MoeRMSNorm as Qwen3_5RMSNorm,
)


QWEN35_ATTN_IMPL2CLASS = {
    "sdpa": Qwen3_5GatedSDPAAttention,
    "flash_attention_2": lambda cfg: Qwen3_5GatedFlashAttention(cfg, flash_attn_version=2),
    "flash_attention_3": lambda cfg: Qwen3_5GatedFlashAttention(cfg, flash_attn_version=3),
    "fa4": lambda cfg: Qwen3_5GatedFlashAttention(cfg, flash_attn_version=4),
}


def _build_text_config(config: Qwen3_5Config | Qwen3_5TextConfig) -> Qwen3_5TextConfig:
    if isinstance(config, Qwen3_5TextConfig):
        return config

    text_config = Qwen3_5TextConfig(**config.text_config.to_dict())
    attn_impl = getattr(
        config.text_config,
        "_attn_implementation",
        getattr(config, "_attn_implementation", None),
    )
    if attn_impl is not None:
        text_config._attn_implementation = attn_impl
    return text_config


def _create_rotary_emb(config: Qwen3_5TextConfig) -> RotaryEmbedding:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rope_type = rope_parameters.get("rope_type", "default")
    else:
        rope_type = "default"

    rotary_config = RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type=rope_type,
        model_config=config,
    )
    return RotaryEmbedding(rotary_config)


def _get_gated_attention(config: Qwen3_5TextConfig) -> nn.Module:
    attn_config = Qwen3_5GatedAttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
    )

    attn_impl = config._attn_implementation
    if attn_impl == "eager":
        attn_impl = "sdpa"
    if attn_impl not in QWEN35_ATTN_IMPL2CLASS:
        supported = sorted(QWEN35_ATTN_IMPL2CLASS)
        raise ValueError(
            f"Qwen3.5 attention does not support '{config._attn_implementation}'. Supported: {supported}."
        )
    return QWEN35_ATTN_IMPL2CLASS[attn_impl](attn_config)


def _normalize_position_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] in (3, 4):
        return position_ids[0]
    raise ValueError(f"Unsupported Qwen3.5 position_ids shape: {tuple(position_ids.shape)}")


def _get_cu_seqlens(position_ids: torch.LongTensor) -> tuple[torch.LongTensor, int]:
    flat_position_ids = position_ids.reshape(-1)
    seqlens = torch.cat(
        [
            flat_position_ids[0:1],
            flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
            flat_position_ids[-1:] + 1,
        ]
    )
    max_seqlen = seqlens.max().item()
    cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
    torch._dynamo.mark_dynamic(cu_seqlens, 0)
    return cu_seqlens, max_seqlen


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = _get_gated_attention(config)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        self.mlp = MLP(
            MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_act,
                bias=False,
            )
        )
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3_5PreTrainedModel(PreTrainedModelPrimeRL):
    config: Qwen3_5Config | Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": Qwen3_5DecoderLayer,
    }

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


@auto_docstring
class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _create_rotary_emb(config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("Custom Qwen3.5 does not support padded attention_mask; use position_ids-based packing.")

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            position_ids = position_ids.expand(inputs_embeds.shape[0], -1)
        else:
            position_ids = _normalize_position_ids(position_ids)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4"):
            cu_seqlens, max_seqlen = _get_cu_seqlens(position_ids)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen3_5TextOnlyModel(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.language_model = Qwen3_5TextModel(_build_text_config(config))

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


@auto_docstring
class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config):
        super().__init__(config)
        self._is_composite_config = isinstance(config, Qwen3_5Config)
        if self._is_composite_config:
            self.model = Qwen3_5TextOnlyModel(config)
            text_config = self.model.language_model.config
        else:
            self.model = Qwen3_5TextModel(config)
            text_config = config

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
            If not provided, the wrapped LM head returns logits only.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        """
        assert use_cache is None, "use_cache is not supported for custom qwen3.5 for now"
        assert past_key_values is None, "past_key_values is not supported for custom qwen3.5 for now"
        del cache_position, kwargs

        if position_ids is None:
            reference_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            position_ids = torch.arange(reference_tensor.shape[1], device=reference_tensor.device).unsqueeze(0)
            position_ids = position_ids.expand(reference_tensor.shape[0], -1)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        language_model = self.model.language_model if self._is_composite_config else self.model
        rotary_emb = language_model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)


__all__ = [
    "Qwen3_5DecoderLayer",
    "Qwen3_5ForCausalLM",
    "Qwen3_5PreTrainedModel",
    "Qwen3_5TextModel",
]
