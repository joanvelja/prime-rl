import types

import torch

from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel


def test_glm_moe_dsa_uniform_index_cache_threads_topk_indices():
    config = GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        num_experts_per_tok=2,
        first_k_dense_replace=4,
        max_position_embeddings=64,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=64,
        index_topk_freq=2,
        pad_token_id=0,
        use_grouped_mm=False,
    )
    model = GlmMoeDsaModel(config)

    received_prev_topk = [None] * config.num_hidden_layers
    emitted_topk = [torch.tensor([layer_idx + 1], dtype=torch.int32) for layer_idx in range(config.num_hidden_layers)]

    for layer_idx, layer in enumerate(model.layers):
        def _make_fake_forward(idx):
            def _fake_forward(self, hidden_states, position_embeddings=None, ks=None, ke=None, prev_topk_indices=None):
                received_prev_topk[idx] = prev_topk_indices
                next_topk_indices = emitted_topk[idx] if self.next_skip_topk else None
                return hidden_states, next_topk_indices

            return _fake_forward

        layer.self_attn.forward = types.MethodType(_make_fake_forward(layer_idx), layer.self_attn)

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    position_ids = torch.arange(8).unsqueeze(0)
    outputs = model(input_ids=input_ids, position_ids=position_ids)

    assert outputs.last_hidden_state.shape == (1, 8, config.hidden_size)

    assert received_prev_topk[0] is None
    assert torch.equal(received_prev_topk[1], emitted_topk[0])
    assert received_prev_topk[2] is None
    assert torch.equal(received_prev_topk[3], emitted_topk[2])

    assert model.layers[0].self_attn.skip_topk is False
    assert model.layers[0].self_attn.next_skip_topk is True
    assert model.layers[1].self_attn.skip_topk is True
    assert model.layers[1].self_attn.next_skip_topk is False
    assert model.layers[2].self_attn.skip_topk is False
    assert model.layers[2].self_attn.next_skip_topk is True
    assert model.layers[3].self_attn.skip_topk is True
    assert model.layers[3].self_attn.next_skip_topk is False
