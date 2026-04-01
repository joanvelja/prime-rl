import torch.nn as nn

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.model import apply_ac
from prime_rl.trainer.models.layers.checkpointing import (
    get_supported_targets,
    set_selective_activation_checkpointing,
)

_PATCHED_METHODS_ATTR = "_prime_rl_selective_ac_patched_methods"


class DummySelfAttention(nn.Module):
    def attn_projections(self, hidden_states, position_embeddings=None):
        return hidden_states

    def output_proj(self, attn_output):
        return attn_output

    def forward(self, hidden_states):
        return hidden_states


class DummySlidingAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttention()
        self.attention_type = "sliding_attention"


class DummyMamba(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class DummyMambaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = DummyMamba()


class DummyMoEExperts(nn.Module):
    def moe_act(self, hidden_states, gate_output=None):
        return hidden_states


class DummyMoEMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = DummyMoEExperts()

    def forward(self, hidden_states):
        return hidden_states

    def _run_routed_experts(self, hidden_states, *args):
        return hidden_states

    def _run_local_routed_experts(self, hidden_states, num_tokens_per_expert):
        return hidden_states


class DummyMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DummyMoEMlp()


def test_get_supported_targets_treats_mamba_as_linear_attention():
    assert get_supported_targets(DummyMambaLayer()) == frozenset({"norm", "linear_attn"})


def test_sliding_attention_linear_attn_subsumes_attn_proj_hooks():
    layer = DummySlidingAttentionLayer()

    set_selective_activation_checkpointing(layer, ["attn_proj", "linear_attn"])

    assert getattr(layer.self_attn, _PATCHED_METHODS_ATTR) == frozenset({"forward"})


def test_routed_experts_checkpointing_patches_local_and_global_helpers():
    layer = DummyMoELayer()

    assert "routed_experts" in get_supported_targets(layer)

    set_selective_activation_checkpointing(layer, ["routed_experts"])

    assert getattr(layer.mlp, _PATCHED_METHODS_ATTR) == frozenset({"_run_local_routed_experts", "_run_routed_experts"})


def test_moe_act_checkpointing_patches_expert_activation():
    layer = DummyMoELayer()

    assert "moe_act" in get_supported_targets(layer)

    set_selective_activation_checkpointing(layer, ["moe_act"])

    assert getattr(layer.mlp.experts, _PATCHED_METHODS_ATTR) == frozenset({"moe_act"})


def test_routed_experts_subsumes_moe_act_checkpointing():
    layer = DummyMoELayer()

    set_selective_activation_checkpointing(layer, ["moe_act", "routed_experts"])

    assert getattr(layer.mlp, _PATCHED_METHODS_ATTR) == frozenset({"_run_local_routed_experts", "_run_routed_experts"})
    assert not hasattr(layer.mlp.experts, _PATCHED_METHODS_ATTR)


class DummyUnsupportedLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyUnsupportedLayer()])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyLanguageModel()


def test_default_norm_selective_ac_falls_back_to_full_checkpointing():
    model = DummyModel()

    apply_ac(model, ActivationCheckpointConfig(mode="selective", targets=["norm"]))

    assert hasattr(model.model.layers[0], "_checkpoint_wrapped_module")
