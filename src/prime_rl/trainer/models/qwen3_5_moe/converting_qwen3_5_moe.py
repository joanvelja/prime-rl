import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _convert_hf_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert a single MoE layer from HF to PrimeRL format in-place.

    prefix: e.g. "model.layers.0" or "model.mtp.layers.0"
    """
    # Router: mlp.gate.weight -> mlp.router.gate.weight
    gate_key = f"{prefix}.mlp.gate.weight"
    if gate_key not in state_dict:
        return

    state_dict[f"{prefix}.mlp.router.gate.weight"] = state_dict.pop(gate_key)

    # Routed experts: convert to fused w1/w2/w3 format
    if f"{prefix}.mlp.experts.gate_up_proj" in state_dict:
        gate_up_proj = state_dict.pop(f"{prefix}.mlp.experts.gate_up_proj")
        down_proj = state_dict.pop(f"{prefix}.mlp.experts.down_proj")

        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj
    else:
        num_experts = len([k for k in state_dict.keys() if f"{prefix}.mlp.experts" in k and "gate_proj" in k])
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"{prefix}.mlp.experts.0.down_proj.weight"].shape
        dtype = state_dict[f"{prefix}.mlp.experts.0.down_proj.weight"].dtype
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        for j in range(num_experts):
            w1[j].copy_(state_dict.pop(f"{prefix}.mlp.experts.{j}.gate_proj.weight"))
            w2[j].copy_(state_dict.pop(f"{prefix}.mlp.experts.{j}.down_proj.weight"))
            w3[j].copy_(state_dict.pop(f"{prefix}.mlp.experts.{j}.up_proj.weight"))

    state_dict[f"{prefix}.mlp.experts.w1"] = w1
    state_dict[f"{prefix}.mlp.experts.w2"] = w2
    state_dict[f"{prefix}.mlp.experts.w3"] = w3

    # Shared expert: mlp.shared_expert.{gate,up,down}_proj -> shared_expert.{w1,w3,w2}
    se_gate_key = f"{prefix}.mlp.shared_expert.gate_proj.weight"
    if se_gate_key in state_dict:
        state_dict[f"{prefix}.shared_expert.w1.weight"] = state_dict.pop(se_gate_key)
        state_dict[f"{prefix}.shared_expert.w2.weight"] = state_dict.pop(f"{prefix}.mlp.shared_expert.down_proj.weight")
        state_dict[f"{prefix}.shared_expert.w3.weight"] = state_dict.pop(f"{prefix}.mlp.shared_expert.up_proj.weight")

    # Shared expert gate: mlp.shared_expert_gate.weight -> shared_expert_gate.weight
    seg_key = f"{prefix}.mlp.shared_expert_gate.weight"
    if seg_key in state_dict:
        state_dict[f"{prefix}.shared_expert_gate.weight"] = state_dict.pop(seg_key)


def _convert_prime_layer_to_hf(state_dict: dict[str, Tensor], prefix: str):
    """Convert a single MoE layer from PrimeRL to HF format in-place.

    prefix: e.g. "model.layers.0" or "model.mtp.layers.0"
    """
    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key not in state_dict:
        return

    state_dict[f"{prefix}.mlp.gate.weight"] = state_dict.pop(router_key)

    w1 = state_dict.pop(f"{prefix}.mlp.experts.w1")
    w2 = state_dict.pop(f"{prefix}.mlp.experts.w2")
    w3 = state_dict.pop(f"{prefix}.mlp.experts.w3")

    num_experts = w1.shape[0]
    for j in range(num_experts):
        state_dict[f"{prefix}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
        state_dict[f"{prefix}.mlp.experts.{j}.down_proj.weight"] = w2[j]
        state_dict[f"{prefix}.mlp.experts.{j}.up_proj.weight"] = w3[j]

    # Shared expert: shared_expert.{w1,w2,w3} -> mlp.shared_expert.{gate,down,up}_proj
    se_w1_key = f"{prefix}.shared_expert.w1.weight"
    if se_w1_key in state_dict:
        state_dict[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = state_dict.pop(se_w1_key)
        state_dict[f"{prefix}.mlp.shared_expert.down_proj.weight"] = state_dict.pop(f"{prefix}.shared_expert.w2.weight")
        state_dict[f"{prefix}.mlp.shared_expert.up_proj.weight"] = state_dict.pop(f"{prefix}.shared_expert.w3.weight")

    seg_key = f"{prefix}.shared_expert_gate.weight"
    if seg_key in state_dict:
        state_dict[f"{prefix}.mlp.shared_expert_gate.weight"] = state_dict.pop(seg_key)


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single backbone layer from HF to PrimeRL format in-place."""
    _convert_hf_layer_to_prime(state_dict, f"model.layers.{layer_idx}")


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single backbone layer from PrimeRL to HF format in-place."""
    _convert_prime_layer_to_hf(state_dict, f"model.layers.{layer_idx}")


# HF checkpoint key prefix → PrimeRL key prefix
_MTP_FUSION_MAP_HF_TO_PRIME = {
    "mtp.pre_fc_norm_embedding.": "model.mtp.enorm.",
    "mtp.pre_fc_norm_hidden.": "model.mtp.hnorm.",
    "mtp.fc.": "model.mtp.eh_proj.",
    "mtp.norm.": "model.mtp.norm.",
}


def _convert_mtp_hf_to_prime(state_dict: dict[str, Tensor]):
    """Convert MTP weights from HF checkpoint format to PrimeRL format in-place.

    Handles renaming fusion keys, dropping shared embed_tokens, and converting
    MTP decoder layer MoE weights.
    """
    mtp_keys = [k for k in state_dict if k.startswith("mtp.")]
    if not mtp_keys:
        return

    # Drop mtp.embed_tokens (shared with model.embed_tokens)
    for k in [k for k in mtp_keys if k.startswith("mtp.embed_tokens.")]:
        del state_dict[k]

    # Rename fusion components
    for hf_prefix, prime_prefix in _MTP_FUSION_MAP_HF_TO_PRIME.items():
        for k in [k for k in list(state_dict) if k.startswith(hf_prefix)]:
            state_dict[prime_prefix + k[len(hf_prefix) :]] = state_dict.pop(k)

    # Rename mtp.layers.{i}.* → model.mtp.layers.{i}.*
    for k in [k for k in list(state_dict) if k.startswith("mtp.layers.")]:
        state_dict["model." + k] = state_dict.pop(k)

    # Apply MoE conversion to each MTP layer
    mtp_layer_indices = sorted({int(k.split(".")[3]) for k in state_dict if k.startswith("model.mtp.layers.")})
    for i in mtp_layer_indices:
        _convert_hf_layer_to_prime(state_dict, f"model.mtp.layers.{i}")


def _convert_mtp_prime_to_hf(state_dict: dict[str, Tensor]):
    """Convert MTP weights from PrimeRL format to HF checkpoint format in-place."""
    mtp_keys = [k for k in state_dict if k.startswith("model.mtp.")]
    if not mtp_keys:
        return

    # Convert MoE in MTP layers first (while keys still have model.mtp.layers prefix)
    mtp_layer_indices = sorted({int(k.split(".")[3]) for k in state_dict if k.startswith("model.mtp.layers.")})
    for i in mtp_layer_indices:
        _convert_prime_layer_to_hf(state_dict, f"model.mtp.layers.{i}")

    # Rename model.mtp.layers.{i}.* → mtp.layers.{i}.*
    for k in [k for k in list(state_dict) if k.startswith("model.mtp.layers.")]:
        state_dict[k.replace("model.mtp.", "mtp.", 1)] = state_dict.pop(k)

    # Rename fusion components back
    for hf_prefix, prime_prefix in _MTP_FUSION_MAP_HF_TO_PRIME.items():
        for k in [k for k in list(state_dict) if k.startswith(prime_prefix)]:
            state_dict[hf_prefix + k[len(prime_prefix) :]] = state_dict.pop(k)


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert all MoE weights from HF to PrimeRL format in-place."""
    _convert_mtp_hf_to_prime(state_dict)
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert all MoE weights from PrimeRL to HF format in-place."""
    _convert_mtp_prime_to_hf(state_dict)
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
