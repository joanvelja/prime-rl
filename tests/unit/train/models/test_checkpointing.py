import importlib.util
import sys
import types
from pathlib import Path

import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
LAYERS_MODULE_ROOT = SRC_ROOT / "prime_rl/trainer/models/layers"
CHECKPOINTING_MODULE_PATH = LAYERS_MODULE_ROOT / "checkpointing.py"


def _ensure_layers_packages() -> None:
    trainer_models = sys.modules.get("prime_rl.trainer.models")
    if trainer_models is None:
        trainer_models = types.ModuleType("prime_rl.trainer.models")
        trainer_models.__path__ = [str(LAYERS_MODULE_ROOT.parent)]
        sys.modules["prime_rl.trainer.models"] = trainer_models

    trainer_layers = sys.modules.get("prime_rl.trainer.models.layers")
    if trainer_layers is None:
        trainer_layers = types.ModuleType("prime_rl.trainer.models.layers")
        trainer_layers.__path__ = [str(LAYERS_MODULE_ROOT)]
        sys.modules["prime_rl.trainer.models.layers"] = trainer_layers

    trainer_models.layers = trainer_layers


def _load_checkpointing_module():
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    _ensure_layers_packages()
    module_name = "prime_rl.trainer.models.layers.checkpointing"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, CHECKPOINTING_MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


checkpointing_module = _load_checkpointing_module()
_PATCHED_METHODS_ATTR = checkpointing_module._PATCHED_METHODS_ATTR
get_supported_targets = checkpointing_module.get_supported_targets
set_selective_activation_checkpointing = checkpointing_module.set_selective_activation_checkpointing


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


class DummyMoEMlp(nn.Module):
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
