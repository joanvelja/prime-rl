import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from prime_rl.trainer.models.gemma4 import Gemma4ForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

pytestmark = [pytest.mark.gpu]


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=128,
        intermediate_size=512,
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        sliding_window=128,
        max_position_embeddings=512,
        final_logit_softcapping=30.0,
        hidden_activation="gelu_pytorch_tanh",
        _attn_implementation="flash_attention_2",
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def test_gemma4_forward():
    config = _tiny_config()
    model = Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 32), device="cuda")
    position_ids = torch.arange(32, device="cuda").unsqueeze(0)

    with torch.no_grad():
        output = model(input_ids=input_ids, position_ids=position_ids)
    assert "logits" in output
    assert output["logits"].shape == (1, 32, 256)


def test_gemma4_backward():
    config = _tiny_config()
    model = Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 32), device="cuda")
    position_ids = torch.arange(32, device="cuda").unsqueeze(0)

    output = model(input_ids=input_ids, position_ids=position_ids)
    logits = output["logits"]
    assert logits is not None
    loss = logits.sum()
    loss.backward()

    # Check gradients flow
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients found"


def test_gemma4_no_kv_sharing():
    """Test without K=V sharing (like a hypothetical smaller model)."""
    config = _tiny_config(attention_k_eq_v=False, num_global_key_value_heads=None)
    model = Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 32), device="cuda")
    position_ids = torch.arange(32, device="cuda").unsqueeze(0)

    with torch.no_grad():
        output = model(input_ids=input_ids, position_ids=position_ids)
    assert output["logits"].shape == (1, 32, 256)


def _tiny_moe_config(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        intermediate_size=256,
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        sliding_window=128,
        max_position_embeddings=512,
        final_logit_softcapping=30.0,
        hidden_activation="gelu_pytorch_tanh",
        _attn_implementation="flash_attention_2",
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=64,
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def test_gemma4_moe_forward():
    config = _tiny_moe_config()
    model = Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 32), device="cuda")
    position_ids = torch.arange(32, device="cuda").unsqueeze(0)

    with torch.no_grad():
        output = model(input_ids=input_ids, position_ids=position_ids)
    assert output["logits"].shape == (1, 32, 256)


def test_gemma4_moe_backward():
    config = _tiny_moe_config()
    model = Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 32), device="cuda")
    position_ids = torch.arange(32, device="cuda").unsqueeze(0)

    output = model(input_ids=input_ids, position_ids=position_ids)
    loss = output["logits"].sum()
    loss.backward()

    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients found"
