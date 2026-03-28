"""Weight conversion between HuggingFace and PrimeRL formats for NemotronH.

HF NemotronH uses a unified `mixer` attribute for all layer types:
  - Mamba layers: backbone.layers.{i}.mixer.{in_proj, conv1d, ...}
  - Attention layers: backbone.layers.{i}.mixer.{q_proj, k_proj, v_proj, o_proj}
  - MoE layers: backbone.layers.{i}.mixer.{gate, experts, shared_experts, fc1_latent_proj, fc2_latent_proj}
  - MTP layers: mtp.layers.{i}.* with the same mixer naming plus extra fusion/norm weights

PrimeRL separates these into distinct namespaces:
  - Mamba layers: model.layers.{i}.mamba.*
  - Attention layers: model.layers.{i}.self_attn.*
  - MoE layers: model.layers.{i}.mlp.{router, experts, shared_expert, fc1_latent_proj, fc2_latent_proj}
  - MTP layers: mtp.layers.{i}.*

Global renames:
  - HF: backbone.embeddings.weight <-> PrimeRL: model.embed_tokens.weight
  - HF: backbone.norm_f.weight <-> PrimeRL: model.norm.weight
  - HF uses "backbone." prefix, PrimeRL uses "model." prefix
"""

import torch
from torch import Tensor


def _rename_keys(state_dict: dict[str, Tensor], old_prefix: str, new_prefix: str):
    """Rename all keys matching old_prefix to new_prefix in-place."""
    keys_to_rename = [k for k in state_dict if k.startswith(old_prefix)]
    for key in keys_to_rename:
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict.pop(key)


def _get_max_layer_num(state_dict: dict[str, Tensor], layers_prefix: str) -> int | None:
    layer_keys = [k for k in state_dict if k.startswith(layers_prefix)]
    if not layer_keys:
        return None
    return max(int(k[len(layers_prefix) :].split(".")[0]) for k in layer_keys) + 1


def _infer_layer_type_hf_at_prefix(state_dict: dict[str, Tensor], prefix: str) -> str:
    layer_keys = [k for k in state_dict if k.startswith(prefix)]
    if not layer_keys:
        return "mamba"

    for key in layer_keys:
        suffix = key[len(prefix) :]
        if suffix.startswith("mixer.gate.") or suffix.startswith("mixer.experts."):
            return "moe"
        if suffix.startswith("mixer.q_proj") or suffix.startswith("mixer.k_proj"):
            return "attention"
    return "mamba"


def _infer_layer_type_prime_at_prefix(state_dict: dict[str, Tensor], prefix: str) -> str:
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.startswith("mlp."):
            return "moe"
        if suffix.startswith("self_attn."):
            return "attention"
        if suffix.startswith("mamba."):
            return "mamba"
    return "mamba"


def _infer_layer_types_from_hf(state_dict: dict[str, Tensor], layers_prefix: str) -> list[str]:
    max_layer = _get_max_layer_num(state_dict, layers_prefix)
    if max_layer is None:
        return []
    return [_infer_layer_type_hf_at_prefix(state_dict, f"{layers_prefix}{i}.") for i in range(max_layer)]


def _infer_layer_types_from_prime(state_dict: dict[str, Tensor], layers_prefix: str) -> list[str]:
    max_layer = _get_max_layer_num(state_dict, layers_prefix)
    if max_layer is None:
        return []
    return [_infer_layer_type_prime_at_prefix(state_dict, f"{layers_prefix}{i}.") for i in range(max_layer)]


def _convert_hf_moe_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert MoE layer: mixer.gate -> mlp.router, mixer.experts -> mlp.experts, etc."""
    mixer = f"{prefix}mixer."
    mlp = f"{prefix}mlp."

    if f"{mixer}gate.weight" in state_dict:
        state_dict[f"{mlp}router.gate"] = state_dict.pop(f"{mixer}gate.weight")
    if f"{mixer}gate.e_score_correction_bias" in state_dict:
        state_dict[f"{mlp}router.e_score_correction_bias"] = state_dict.pop(f"{mixer}gate.e_score_correction_bias")

    individual_keys = [
        k
        for k in state_dict
        if k.startswith(f"{mixer}experts.") and k[len(f"{mixer}experts.") :].split(".")[0].isdigit()
    ]

    if individual_keys:
        expert_indices = sorted({int(k[len(f"{mixer}experts.") :].split(".")[0]) for k in individual_keys})

        up_projs = [state_dict.pop(f"{mixer}experts.{i}.up_proj.weight") for i in expert_indices]
        state_dict[f"{mlp}experts.w1"] = torch.stack(up_projs)

        down_projs = [state_dict.pop(f"{mixer}experts.{i}.down_proj.weight") for i in expert_indices]
        state_dict[f"{mlp}experts.w2"] = torch.stack(down_projs)
    else:
        if f"{mixer}experts.up_proj" in state_dict:
            state_dict[f"{mlp}experts.w1"] = state_dict.pop(f"{mixer}experts.up_proj")
        if f"{mixer}experts.down_proj" in state_dict:
            state_dict[f"{mlp}experts.w2"] = state_dict.pop(f"{mixer}experts.down_proj")

    device = state_dict[f"{mlp}experts.w1"].device if f"{mlp}experts.w1" in state_dict else "cpu"
    state_dict[f"{mlp}experts.w3"] = torch.empty(0, device=device)

    _rename_keys(state_dict, f"{mixer}shared_experts.", f"{mlp}shared_expert.")
    _rename_keys(state_dict, f"{mixer}fc1_latent_proj.", f"{mlp}fc1_latent_proj.")
    _rename_keys(state_dict, f"{mixer}fc2_latent_proj.", f"{mlp}fc2_latent_proj.")


def _convert_hf_attention_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    _rename_keys(state_dict, f"{prefix}mixer.", f"{prefix}self_attn.")


def _convert_hf_layer_to_prime_at_prefix(state_dict: dict[str, Tensor], prefix: str, layer_type: str):
    if layer_type == "moe":
        _convert_hf_moe_layer_to_prime(state_dict, prefix)
    elif layer_type == "attention":
        _convert_hf_attention_layer_to_prime(state_dict, prefix)
    elif layer_type == "mamba":
        _rename_keys(state_dict, f"{prefix}mixer.", f"{prefix}mamba.")


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int, layer_type: str):
    """Convert a single backbone layer from HF to PrimeRL format in-place."""
    _convert_hf_layer_to_prime_at_prefix(state_dict, f"model.layers.{layer_idx}.", layer_type)


def _convert_prime_moe_layer_to_hf(state_dict: dict[str, Tensor], prefix: str):
    """Convert MoE layer back to HF format."""
    mlp = f"{prefix}mlp."
    mixer = f"{prefix}mixer."

    if f"{mlp}router.gate" in state_dict:
        state_dict[f"{mixer}gate.weight"] = state_dict.pop(f"{mlp}router.gate")
    if f"{mlp}router.e_score_correction_bias" in state_dict:
        state_dict[f"{mixer}gate.e_score_correction_bias"] = state_dict.pop(f"{mlp}router.e_score_correction_bias")

    if f"{mlp}experts.w1" in state_dict:
        w1 = state_dict.pop(f"{mlp}experts.w1")
        for i in range(w1.shape[0]):
            state_dict[f"{mixer}experts.{i}.up_proj.weight"] = w1[i]
    if f"{mlp}experts.w2" in state_dict:
        w2 = state_dict.pop(f"{mlp}experts.w2")
        for i in range(w2.shape[0]):
            state_dict[f"{mixer}experts.{i}.down_proj.weight"] = w2[i]
    state_dict.pop(f"{mlp}experts.w3", None)

    _rename_keys(state_dict, f"{mlp}shared_expert.", f"{mixer}shared_experts.")
    _rename_keys(state_dict, f"{mlp}fc1_latent_proj.", f"{mixer}fc1_latent_proj.")
    _rename_keys(state_dict, f"{mlp}fc2_latent_proj.", f"{mixer}fc2_latent_proj.")


def _convert_prime_layer_to_hf_at_prefix(state_dict: dict[str, Tensor], prefix: str, layer_type: str):
    if layer_type == "moe":
        _convert_prime_moe_layer_to_hf(state_dict, prefix)
    elif layer_type == "attention":
        _rename_keys(state_dict, f"{prefix}self_attn.", f"{prefix}mixer.")
    elif layer_type == "mamba":
        _rename_keys(state_dict, f"{prefix}mamba.", f"{prefix}mixer.")


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int, layer_type: str):
    """Convert a single backbone layer from PrimeRL to HF format in-place."""
    _convert_prime_layer_to_hf_at_prefix(state_dict, f"model.layers.{layer_idx}.", layer_type)


def _convert_hf_mtp_to_prime(state_dict: dict[str, Tensor], mtp_layers_block_type: list[str]):
    for i, layer_type in enumerate(mtp_layers_block_type):
        _convert_hf_layer_to_prime_at_prefix(state_dict, f"mtp.layers.{i}.", layer_type)


def _convert_prime_mtp_to_hf(state_dict: dict[str, Tensor], mtp_layers_block_type: list[str]):
    for i, layer_type in enumerate(mtp_layers_block_type):
        _convert_prime_layer_to_hf_at_prefix(state_dict, f"mtp.layers.{i}.", layer_type)


def convert_hf_to_prime(state_dict: dict[str, Tensor], layers_block_type: list[str]):
    """Convert full model from HF to PrimeRL format in-place."""
    mtp_layers_block_type = _infer_layer_types_from_hf(state_dict, "mtp.layers.")

    _rename_keys(state_dict, "backbone.", "model.")

    if "model.embeddings.weight" in state_dict:
        state_dict["model.embed_tokens.weight"] = state_dict.pop("model.embeddings.weight")
    if "model.norm_f.weight" in state_dict:
        state_dict["model.norm.weight"] = state_dict.pop("model.norm_f.weight")

    for i, layer_type in enumerate(layers_block_type):
        convert_hf_layer_to_prime(state_dict, i, layer_type)

    if mtp_layers_block_type:
        _convert_hf_mtp_to_prime(state_dict, mtp_layers_block_type)


def convert_prime_to_hf(state_dict: dict[str, Tensor], layers_block_type: list[str]):
    """Convert full model from PrimeRL to HF format in-place."""
    mtp_layers_block_type = _infer_layer_types_from_prime(state_dict, "mtp.layers.")

    if mtp_layers_block_type:
        _convert_prime_mtp_to_hf(state_dict, mtp_layers_block_type)

    if "model.embed_tokens.weight" in state_dict:
        state_dict["model.embeddings.weight"] = state_dict.pop("model.embed_tokens.weight")
    if "model.norm.weight" in state_dict:
        state_dict["model.norm_f.weight"] = state_dict.pop("model.norm.weight")

    for i, layer_type in enumerate(layers_block_type):
        convert_prime_layer_to_hf(state_dict, i, layer_type)

    _rename_keys(state_dict, "model.", "backbone.")
