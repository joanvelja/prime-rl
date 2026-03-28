import pytest
import torch
from transformers import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM as PrimeRLQwen3_5ForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def make_text_config():
    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )
    config._attn_implementation = "sdpa"
    return config


def get_text_model_pairs():
    config = make_text_config()

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3_5ForCausalLM._from_config(config)
        prime_model = PrimeRLQwen3_5ForCausalLM._from_config(config)

    with torch.no_grad():
        prime_model.load_state_dict(hf_model.state_dict())

    inject_prime_lm_head(prime_model, chunk_size=None)
    return hf_model, prime_model


def test_qwen3_5():
    hf_model, prime_model = get_text_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(100).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )

    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_qwen3_5_composite_config():
    text_config = make_text_config()
    composite_config = Qwen3_5Config(text_config=text_config.to_dict())
    composite_config.text_config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3_5ForCausalLM._from_config(text_config)
        prime_model = PrimeRLQwen3_5ForCausalLM._from_config(composite_config)

    with torch.no_grad():
        state_dict = {}
        for key, value in hf_model.state_dict().items():
            if key.startswith("model."):
                key = key.replace("model.", "model.language_model.", 1)
            state_dict[key] = value
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(100).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )

    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.language_model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"
