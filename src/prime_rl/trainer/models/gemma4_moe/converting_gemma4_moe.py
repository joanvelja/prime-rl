"""Weight conversion between HF and PrimeRL formats for Gemma4 MoE.

HF format: experts.gate_up_proj (num_experts, 2*moe_dim, dim) + experts.down_proj (num_experts, dim, moe_dim)
           router.norm.weight, router.proj.weight, router.scale, router.per_expert_scale
PrimeRL format: experts.w1 (num_experts, moe_dim, dim), experts.w2 (num_experts, dim, moe_dim),
                experts.w3 (num_experts, moe_dim, dim)
                router keys stay the same (no rename needed since Gemma4 router has its own structure)
"""

from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    layer_indices = [int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i]
    return max(layer_indices) + 1 if layer_indices else 0


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert Gemma4 MoE layer from HF fused format to PrimeRL w1/w2/w3 format."""
    i = layer_idx
    gate_up_key = f"model.layers.{i}.experts.gate_up_proj"
    down_key = f"model.layers.{i}.experts.down_proj"

    if gate_up_key not in state_dict:
        return

    gate_up_proj = state_dict.pop(gate_up_key)
    down_proj = state_dict.pop(down_key)

    num_experts, fused_dim, dim = gate_up_proj.shape
    moe_dim = fused_dim // 2

    # Split gate_up into w1 (gate) and w3 (up)
    state_dict[f"model.layers.{i}.experts.w1"] = gate_up_proj[:, :moe_dim, :]
    state_dict[f"model.layers.{i}.experts.w3"] = gate_up_proj[:, moe_dim:, :]
    state_dict[f"model.layers.{i}.experts.w2"] = down_proj


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert Gemma4 MoE layer from PrimeRL w1/w2/w3 back to HF fused 3D format.

    Produces the same format as the original HF checkpoint:
      experts.gate_up_proj: [num_experts, 2*moe_dim, dim]
      experts.down_proj:    [num_experts, dim, moe_dim]
    """
    import torch

    i = layer_idx
    w1_key = f"model.layers.{i}.experts.w1"

    if w1_key not in state_dict:
        return

    w1 = state_dict.pop(w1_key)
    w2 = state_dict.pop(f"model.layers.{i}.experts.w2")
    w3 = state_dict.pop(f"model.layers.{i}.experts.w3")

    # Fuse gate (w1) and up (w3) back into gate_up_proj
    state_dict[f"model.layers.{i}.experts.gate_up_proj"] = torch.cat([w1, w3], dim=1)
    state_dict[f"model.layers.{i}.experts.down_proj"] = w2


def convert_hf_to_prime(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_prime(state_dict, i)


def convert_prime_to_hf(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_prime_layer_to_hf(state_dict, i)
