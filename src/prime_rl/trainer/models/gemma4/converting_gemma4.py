"""Weight conversion between HF and PrimeRL formats for Gemma4 MoE.

HF format: experts.gate_up_proj (num_experts, 2*moe_dim, dim) + experts.down_proj (num_experts, dim, moe_dim)
           router.norm.weight, router.proj.weight, router.scale, router.per_expert_scale
PrimeRL format: mlp.experts.w1 (num_experts, moe_dim, dim), mlp.experts.w2 (num_experts, dim, moe_dim),
                mlp.experts.w3 (num_experts, moe_dim, dim)
                mlp.router.*, mlp.shared_mlp.*

In PrimeRL, the parallel MoE block (shared MLP + router + sparse experts) is wrapped under
a single ``mlp`` module (Gemma4ParallelMoE), so the keys are nested one level deeper.
"""

import re

from torch import Tensor

# Patterns for MoE-layer structural remapping between HF and PrimeRL.
# HF keeps router/experts/mlp as siblings; PrimeRL nests them under mlp.
_MOE_LAYER_RE = re.compile(r"(model\.layers\.\d+)\.")

_HF_TO_PRIME_PREFIXES = [
    (".experts.", ".mlp.experts."),
    (".router.", ".mlp.router."),
    (".mlp.", ".mlp.shared_mlp."),
]

_PRIME_TO_HF_PREFIXES = [(dst, src) for src, dst in _HF_TO_PRIME_PREFIXES]


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    layer_indices = [int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i]
    return max(layer_indices) + 1 if layer_indices else 0


def _is_moe_layer(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    return any(f"model.layers.{layer_idx}.experts." in k or f"model.layers.{layer_idx}.mlp.experts." in k for k in state_dict)


def _remap_moe_structure(state_dict: dict[str, Tensor], layer_idx: int, prefix_map: list[tuple[str, str]]):
    """Remap MoE-layer key prefixes for a single layer."""
    layer_prefix = f"model.layers.{layer_idx}"
    for k in list(state_dict):
        if not k.startswith(layer_prefix):
            continue
        suffix = k[len(layer_prefix):]
        for src, dst in prefix_map:
            if suffix.startswith(src):
                state_dict[layer_prefix + dst + suffix[len(src):]] = state_dict.pop(k)
                break


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert Gemma4 MoE layer from HF fused format to PrimeRL w1/w2/w3 format."""
    i = layer_idx
    gate_up_key = f"model.layers.{i}.experts.gate_up_proj"

    if gate_up_key not in state_dict:
        return

    gate_up_proj = state_dict.pop(gate_up_key)
    down_proj = state_dict.pop(f"model.layers.{i}.experts.down_proj")

    num_experts, fused_dim, dim = gate_up_proj.shape
    moe_dim = fused_dim // 2

    state_dict[f"model.layers.{i}.experts.w1"] = gate_up_proj[:, :moe_dim, :]
    state_dict[f"model.layers.{i}.experts.w3"] = gate_up_proj[:, moe_dim:, :]
    state_dict[f"model.layers.{i}.experts.w2"] = down_proj

    # Remap sibling layout → nested under mlp
    _remap_moe_structure(state_dict, i, _HF_TO_PRIME_PREFIXES)


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert Gemma4 MoE layer from PrimeRL w1/w2/w3 back to HF fused 3D format."""
    import torch

    i = layer_idx

    # Remap nested under mlp → sibling layout first
    _remap_moe_structure(state_dict, i, _PRIME_TO_HF_PREFIXES)

    w1_key = f"model.layers.{i}.experts.w1"

    if w1_key not in state_dict:
        return

    w1 = state_dict.pop(w1_key)
    w2 = state_dict.pop(f"model.layers.{i}.experts.w2")
    w3 = state_dict.pop(f"model.layers.{i}.experts.w3")

    state_dict[f"model.layers.{i}.experts.gate_up_proj"] = torch.cat([w1, w3], dim=1)
    state_dict[f"model.layers.{i}.experts.down_proj"] = w2


def convert_hf_to_prime(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        if _is_moe_layer(state_dict, i):
            convert_hf_layer_to_prime(state_dict, i)


def convert_prime_to_hf(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        if _is_moe_layer(state_dict, i):
            convert_prime_layer_to_hf(state_dict, i)
