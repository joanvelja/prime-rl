import torch
from torch import Tensor

LANGUAGE_LAYER_PREFIX = "model.layers."
MTP_LAYER_PREFIX = "mtp.layers."


def get_max_layer_num(state_dict: dict[str, Tensor], layer_prefix: str = LANGUAGE_LAYER_PREFIX) -> int:
    """Get the number of layers present under a given prefix."""
    layer_indices = [
        int(key[len(layer_prefix) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(layer_prefix)
    ]
    return max(layer_indices, default=-1) + 1


def _layer_key(layer_prefix: str, layer_idx: int, suffix: str) -> str:
    return f"{layer_prefix}{layer_idx}.{suffix}"


def _convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int, layer_prefix: str) -> None:
    gate_key = _layer_key(layer_prefix, layer_idx, "mlp.gate.weight")
    if gate_key not in state_dict:
        return

    state_dict[_layer_key(layer_prefix, layer_idx, "mlp.router.gate.weight")] = state_dict.pop(gate_key)

    gate_up_proj_key = _layer_key(layer_prefix, layer_idx, "mlp.experts.gate_up_proj")
    down_proj_key = _layer_key(layer_prefix, layer_idx, "mlp.experts.down_proj")
    if gate_up_proj_key in state_dict:
        gate_up_proj = state_dict.pop(gate_up_proj_key)
        down_proj = state_dict.pop(down_proj_key)

        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj
    else:
        expert_prefix = _layer_key(layer_prefix, layer_idx, "mlp.experts.")
        expert_ids = sorted(
            {
                int(key[len(expert_prefix) :].split(".", 1)[0])
                for key in state_dict
                if key.startswith(expert_prefix) and key.endswith(".gate_proj.weight")
            }
        )
        if not expert_ids:
            return

        first_down_proj_key = _layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_ids[0]}.down_proj.weight")
        dim, moe_dim = state_dict[first_down_proj_key].shape
        dtype = state_dict[first_down_proj_key].dtype
        device = state_dict[first_down_proj_key].device
        num_experts = len(expert_ids)
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype, device=device)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype, device=device)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype, device=device)
        for tensor_idx, expert_id in enumerate(expert_ids):
            w1[tensor_idx].copy_(state_dict.pop(_layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_id}.gate_proj.weight")))
            w2[tensor_idx].copy_(state_dict.pop(_layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_id}.down_proj.weight")))
            w3[tensor_idx].copy_(state_dict.pop(_layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_id}.up_proj.weight")))

    state_dict[_layer_key(layer_prefix, layer_idx, "mlp.experts.w1")] = w1
    state_dict[_layer_key(layer_prefix, layer_idx, "mlp.experts.w2")] = w2
    state_dict[_layer_key(layer_prefix, layer_idx, "mlp.experts.w3")] = w3

    shared_gate_key = _layer_key(layer_prefix, layer_idx, "mlp.shared_expert.gate_proj.weight")
    if shared_gate_key in state_dict:
        state_dict[_layer_key(layer_prefix, layer_idx, "shared_expert.w1.weight")] = state_dict.pop(shared_gate_key)
        state_dict[_layer_key(layer_prefix, layer_idx, "shared_expert.w2.weight")] = state_dict.pop(
            _layer_key(layer_prefix, layer_idx, "mlp.shared_expert.down_proj.weight")
        )
        state_dict[_layer_key(layer_prefix, layer_idx, "shared_expert.w3.weight")] = state_dict.pop(
            _layer_key(layer_prefix, layer_idx, "mlp.shared_expert.up_proj.weight")
        )

    shared_expert_gate_key = _layer_key(layer_prefix, layer_idx, "mlp.shared_expert_gate.weight")
    if shared_expert_gate_key in state_dict:
        state_dict[_layer_key(layer_prefix, layer_idx, "shared_expert_gate.weight")] = state_dict.pop(
            shared_expert_gate_key
        )


def convert_hf_layer_to_tt(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    layer_prefix: str = LANGUAGE_LAYER_PREFIX,
) -> None:
    """Convert a single HF layer from HuggingFace format to PrimeRL format in-place."""
    _convert_hf_layer_to_tt(state_dict, layer_idx, layer_prefix)


def _convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int, layer_prefix: str) -> None:
    router_key = _layer_key(layer_prefix, layer_idx, "mlp.router.gate.weight")
    if router_key not in state_dict:
        return

    state_dict[_layer_key(layer_prefix, layer_idx, "mlp.gate.weight")] = state_dict.pop(router_key)

    w1 = state_dict.pop(_layer_key(layer_prefix, layer_idx, "mlp.experts.w1"))
    w2 = state_dict.pop(_layer_key(layer_prefix, layer_idx, "mlp.experts.w2"))
    w3 = state_dict.pop(_layer_key(layer_prefix, layer_idx, "mlp.experts.w3"))

    for expert_id in range(w1.shape[0]):
        state_dict[_layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_id}.gate_proj.weight")] = w1[expert_id]
        state_dict[_layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_id}.down_proj.weight")] = w2[expert_id]
        state_dict[_layer_key(layer_prefix, layer_idx, f"mlp.experts.{expert_id}.up_proj.weight")] = w3[expert_id]

    shared_w1_key = _layer_key(layer_prefix, layer_idx, "shared_expert.w1.weight")
    if shared_w1_key in state_dict:
        state_dict[_layer_key(layer_prefix, layer_idx, "mlp.shared_expert.gate_proj.weight")] = state_dict.pop(shared_w1_key)
        state_dict[_layer_key(layer_prefix, layer_idx, "mlp.shared_expert.down_proj.weight")] = state_dict.pop(
            _layer_key(layer_prefix, layer_idx, "shared_expert.w2.weight")
        )
        state_dict[_layer_key(layer_prefix, layer_idx, "mlp.shared_expert.up_proj.weight")] = state_dict.pop(
            _layer_key(layer_prefix, layer_idx, "shared_expert.w3.weight")
        )

    shared_expert_gate_key = _layer_key(layer_prefix, layer_idx, "shared_expert_gate.weight")
    if shared_expert_gate_key in state_dict:
        state_dict[_layer_key(layer_prefix, layer_idx, "mlp.shared_expert_gate.weight")] = state_dict.pop(
            shared_expert_gate_key
        )


def convert_tt_layer_to_hf(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    layer_prefix: str = LANGUAGE_LAYER_PREFIX,
) -> None:
    """Convert a single PrimeRL layer from PrimeRL format to HuggingFace format in-place."""
    _convert_tt_layer_to_hf(state_dict, layer_idx, layer_prefix)


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]) -> None:
    """Convert all Qwen3.5 MoE weights from HuggingFace format to PrimeRL format in-place."""
    for layer_prefix in (LANGUAGE_LAYER_PREFIX, MTP_LAYER_PREFIX):
        for layer_idx in range(get_max_layer_num(state_dict, layer_prefix)):
            convert_hf_layer_to_tt(state_dict, layer_idx, layer_prefix)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]) -> None:
    """Convert all Qwen3.5 MoE weights from PrimeRL format to HuggingFace format in-place."""
    for layer_prefix in (LANGUAGE_LAYER_PREFIX, MTP_LAYER_PREFIX):
        for layer_idx in range(get_max_layer_num(state_dict, layer_prefix)):
            convert_tt_layer_to_hf(state_dict, layer_idx, layer_prefix)
