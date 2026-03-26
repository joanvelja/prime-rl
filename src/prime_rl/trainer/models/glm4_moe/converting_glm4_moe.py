import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _convert_hf_moe_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert a single MoE layer from HF to PrimeRL format in-place.

    prefix: e.g. "model.layers.0" or "model.mtp_layers.0.block"
    """
    if f"{prefix}.mlp.gate.weight" not in state_dict:
        return

    state_dict[f"{prefix}.mlp.router.gate.weight"] = state_dict.pop(f"{prefix}.mlp.gate.weight")

    if f"{prefix}.mlp.experts.gate_up_proj" in state_dict:
        gate_up_proj = state_dict.pop(f"{prefix}.mlp.experts.gate_up_proj")
        down_proj = state_dict.pop(f"{prefix}.mlp.experts.down_proj")

        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj
    else:
        num_experts = len([k for k in state_dict if f"{prefix}.mlp.experts" in k and "gate_proj" in k])
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

    # Shared experts: shared_experts.{gate,up,down}_proj -> shared_expert.{w1,w3,w2}
    se_gate_key = f"{prefix}.mlp.shared_experts.gate_proj.weight"
    if se_gate_key in state_dict:
        state_dict[f"{prefix}.mlp.shared_expert.w1"] = state_dict.pop(se_gate_key)
        state_dict[f"{prefix}.mlp.shared_expert.w2"] = state_dict.pop(f"{prefix}.mlp.shared_experts.down_proj.weight")
        state_dict[f"{prefix}.mlp.shared_expert.w3"] = state_dict.pop(f"{prefix}.mlp.shared_experts.up_proj.weight")

    # Expert bias
    bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
    if bias_key in state_dict:
        state_dict[f"{prefix}.mlp.expert_bias"] = state_dict.pop(bias_key)


def _convert_prime_moe_to_hf(state_dict: dict[str, Tensor], prefix: str):
    """Convert a single MoE layer from PrimeRL to HF format in-place.

    prefix: e.g. "model.layers.0" or "model.mtp_layers.0.block"
    """
    # Load balancing terms
    if f"{prefix}.mlp.expert_bias" in state_dict:
        state_dict[f"{prefix}.mlp.gate.e_score_correction_bias"] = state_dict.pop(f"{prefix}.mlp.expert_bias")
    tpe_key = f"{prefix}.mlp.tokens_per_expert"
    if tpe_key in state_dict:
        del state_dict[tpe_key]

    # Shared experts
    if f"{prefix}.mlp.shared_expert.w1" in state_dict:
        state_dict[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = state_dict[f"{prefix}.mlp.shared_expert.w1"]
        state_dict[f"{prefix}.mlp.shared_experts.down_proj.weight"] = state_dict[f"{prefix}.mlp.shared_expert.w2"]
        state_dict[f"{prefix}.mlp.shared_experts.up_proj.weight"] = state_dict[f"{prefix}.mlp.shared_expert.w3"]

        if state_dict[f"{prefix}.mlp.shared_experts.up_proj.weight"].shape[0] == 1:
            for proj in ("up_proj", "down_proj", "gate_proj"):
                state_dict[f"{prefix}.mlp.shared_experts.{proj}.weight"] = state_dict[
                    f"{prefix}.mlp.shared_experts.{proj}.weight"
                ][0]

        del state_dict[f"{prefix}.mlp.shared_expert.w1"]
        del state_dict[f"{prefix}.mlp.shared_expert.w2"]
        del state_dict[f"{prefix}.mlp.shared_expert.w3"]

    # Gate / Router
    if f"{prefix}.mlp.router.gate.weight" in state_dict:
        state_dict[f"{prefix}.mlp.gate.weight"] = state_dict.pop(f"{prefix}.mlp.router.gate.weight")

        w1 = state_dict.pop(f"{prefix}.mlp.experts.w1")
        w2 = state_dict.pop(f"{prefix}.mlp.experts.w2")
        w3 = state_dict.pop(f"{prefix}.mlp.experts.w3")

        num_experts = w1.shape[0]
        for j in range(num_experts):
            state_dict[f"{prefix}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
            state_dict[f"{prefix}.mlp.experts.{j}.down_proj.weight"] = w2[j]
            state_dict[f"{prefix}.mlp.experts.{j}.up_proj.weight"] = w3[j]


# ---- Public per-layer API (backward-compatible) ----


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single backbone layer from HF to PrimeRL format in-place."""
    _convert_hf_moe_to_prime(state_dict, f"model.layers.{layer_idx}")


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single backbone layer from PrimeRL to HF format in-place."""
    _convert_prime_moe_to_hf(state_dict, f"model.layers.{layer_idx}")


# ---- MTP weight conversion ----

_MTP_FUSION_PREFIXES = ("enorm.", "hnorm.", "eh_proj.")


def _get_mtp_layer_indices(state_dict: dict[str, Tensor]) -> list[int]:
    """Find backbone layer indices that are actually MTP layers (have enorm.weight)."""
    return sorted({int(k.split(".")[2]) for k in state_dict if k.startswith("model.layers.") and ".enorm." in k})


def _convert_mtp_hf_to_prime(state_dict: dict[str, Tensor]):
    """Convert MTP weights from HF checkpoint format to PrimeRL format in-place.

    GLM-4 stores MTP layers as extra backbone layers (model.layers.{N+i}.*).
    We rename them to model.mtp_layers.{i}.* with decoder block keys under .block.
    """
    mtp_indices = _get_mtp_layer_indices(state_dict)
    if not mtp_indices:
        return

    for i, layer_idx in enumerate(mtp_indices):
        old_prefix = f"model.layers.{layer_idx}"
        new_prefix = f"model.mtp_layers.{i}"

        for k in [k for k in list(state_dict) if k.startswith(old_prefix + ".")]:
            suffix = k[len(old_prefix) + 1 :]

            if suffix.startswith("shared_head.head."):
                del state_dict[k]
            elif suffix.startswith("shared_head.norm."):
                new_suffix = suffix.replace("shared_head.norm.", "norm.", 1)
                state_dict[f"{new_prefix}.{new_suffix}"] = state_dict.pop(k)
            elif any(suffix.startswith(p) for p in _MTP_FUSION_PREFIXES):
                state_dict[f"{new_prefix}.{suffix}"] = state_dict.pop(k)
            else:
                state_dict[f"{new_prefix}.block.{suffix}"] = state_dict.pop(k)

        _convert_hf_moe_to_prime(state_dict, f"{new_prefix}.block")


def _convert_mtp_prime_to_hf(state_dict: dict[str, Tensor], num_backbone_layers: int):
    """Convert MTP weights from PrimeRL format to HF checkpoint format in-place."""
    mtp_indices = sorted({int(k.split(".")[2]) for k in state_dict if k.startswith("model.mtp_layers.")})
    if not mtp_indices:
        return

    for i in mtp_indices:
        old_prefix = f"model.mtp_layers.{i}"
        layer_idx = num_backbone_layers + i
        new_prefix = f"model.layers.{layer_idx}"

        _convert_prime_moe_to_hf(state_dict, f"{old_prefix}.block")

        for k in [k for k in list(state_dict) if k.startswith(old_prefix + ".")]:
            suffix = k[len(old_prefix) + 1 :]

            if suffix.startswith("norm."):
                new_suffix = suffix.replace("norm.", "shared_head.norm.", 1)
                state_dict[f"{new_prefix}.{new_suffix}"] = state_dict.pop(k)
            elif suffix.startswith("block."):
                state_dict[f"{new_prefix}.{suffix[len('block.') :]}"] = state_dict.pop(k)
            else:
                state_dict[f"{new_prefix}.{suffix}"] = state_dict.pop(k)


# ---- Bulk conversion API ----


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert all weights from HF to PrimeRL format in-place."""
    _convert_mtp_hf_to_prime(state_dict)
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert all weights from PrimeRL to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    _convert_mtp_prime_to_hf(state_dict, num_layers)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
