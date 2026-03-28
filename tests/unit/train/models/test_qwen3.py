import pytest
import torch
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM as PrimeRLQwen3ForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=512,
    )
    config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3ForCausalLM._from_config(config)
        prime_model = PrimeRLQwen3ForCausalLM._from_config(config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    return hf_model, prime_model


def test_qwen3():
    hf_model, prime_model = get_model_pairs()

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
