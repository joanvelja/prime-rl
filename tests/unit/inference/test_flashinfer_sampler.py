import inspect
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm import flashinfer_sampler


def test_precompile_tail_batch_sizes_falls_back_to_single_batch(monkeypatch):
    monkeypatch.delenv(
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES",
        raising=False,
    )
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCH", "128")

    assert flashinfer_sampler._precompile_tail_batch_sizes() == [128]


def test_precompile_tail_batch_sizes_parses_unique_list(monkeypatch):
    monkeypatch.setenv(
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES",
        "1, 128,256,128",
    )
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCH", "64")

    assert flashinfer_sampler._precompile_tail_batch_sizes() == [1, 128, 256]


def test_precompile_tail_top_p_values_does_not_duplicate_runtime_float32(monkeypatch):
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P", "0.95")

    assert flashinfer_sampler._precompile_tail_top_p_values() == [0.95]


def test_precompile_tail_top_p_values_dedupe_exact_float32(monkeypatch):
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P", "1.0")

    assert flashinfer_sampler._precompile_tail_top_p_values() == [1.0]


def test_triton_tail_specializes_only_on_block_shape():
    signature = inspect.signature(flashinfer_sampler._k_tail_uniform_kernel.fn)
    parameters = list(signature.parameters)

    assert "TOP_P" not in signature.parameters
    assert signature.parameters["K"].annotation is inspect.Signature.empty
    assert signature.parameters["top_p"].annotation is inspect.Signature.empty
    assert signature.parameters["K_BLOCK"].annotation == "tl.constexpr"
    assert flashinfer_sampler._k_tail_uniform_kernel.constexprs == [parameters.index("K_BLOCK")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="apply_all_penalties requires a CUDA device")
def test_dense_presence_shortcut_matches_native_prompt_and_output_penalties():
    from vllm.v1.sample.ops.penalties import apply_all_penalties

    logits = torch.zeros((1, 8), dtype=torch.float32)
    prompt_token_ids = torch.tensor([[1, 2]], dtype=torch.int64)
    output_token_ids = [[3]]
    metadata = SimpleNamespace(
        no_penalties=False,
        output_token_ids=output_token_ids,
        prompt_token_ids=prompt_token_ids,
        presence_penalties=torch.tensor([1.5], dtype=torch.float32),
        frequency_penalties=torch.zeros(1, dtype=torch.float32),
        repetition_penalties=torch.ones(1, dtype=torch.float32),
        thinking_budget_state_holder=None,
        spec_token_ids=None,
    )

    native = apply_all_penalties(
        logits.clone(),
        prompt_token_ids,
        metadata.presence_penalties,
        metadata.frequency_penalties,
        metadata.repetition_penalties,
        output_token_ids,
    )
    dense = flashinfer_sampler._apply_presence_only_logits_processors(
        logits.clone(),
        metadata,
        predict_bonus_token=False,
    )

    assert dense is not None
    torch.testing.assert_close(dense, native)


def test_width1_logprob_columns_include_top_alternative_and_rank():
    sampled = torch.tensor([5, 7], dtype=torch.int64)
    sampled_logprobs = torch.tensor([-2.0, -0.1], dtype=torch.float32)
    top_ids = torch.tensor([3, 7], dtype=torch.int64)
    top_logprobs = torch.tensor([-0.1, -0.1], dtype=torch.float32)
    selected_ranks = torch.tensor([4, 1], dtype=torch.int32)

    token_ids, logprobs, ranks = flashinfer_sampler._build_logprob_columns(
        sampled,
        sampled_logprobs,
        top_ids,
        top_logprobs,
        selected_ranks,
        max_num_logprobs=1,
    )

    assert token_ids.dtype == torch.int32
    assert token_ids.tolist() == [[5, 3], [7, 7]]
    assert logprobs.tolist() == [[-2.0, -0.10000000149011612], [-0.10000000149011612, -0.10000000149011612]]
    assert ranks.dtype == torch.int32
    assert ranks.tolist() == [4, 1]


def test_width0_logprob_columns_stay_sampled_only_but_keep_rank():
    sampled = torch.tensor([5], dtype=torch.int64)
    sampled_logprobs = torch.tensor([-2.0], dtype=torch.float32)
    top_ids = torch.tensor([3], dtype=torch.int64)
    top_logprobs = torch.tensor([-0.1], dtype=torch.float32)
    selected_ranks = torch.tensor([4], dtype=torch.int32)

    token_ids, logprobs, ranks = flashinfer_sampler._build_logprob_columns(
        sampled,
        sampled_logprobs,
        top_ids,
        top_logprobs,
        selected_ranks,
        max_num_logprobs=0,
    )

    assert token_ids.tolist() == [[5]]
    assert logprobs.tolist() == [[-2.0]]
    assert ranks.tolist() == [4]


def test_top_k_boundary_tie_detects_equal_k_plus_one_value():
    sorted_values = torch.tensor(
        [
            [4.0, 3.0, 2.0, 2.0],
            [5.0, 4.0, 3.0, 1.0],
        ],
        dtype=torch.float32,
    )

    assert flashinfer_sampler._has_top_k_boundary_tie(sorted_values, top_k=3)


def test_top_k_boundary_tie_allows_ties_strictly_inside_support():
    sorted_values = torch.tensor([[4.0, 4.0, 3.0, 2.0]], dtype=torch.float32)

    assert not flashinfer_sampler._has_top_k_boundary_tie(sorted_values, top_k=3)


def test_top_p_boundary_tie_detects_nucleus_cut_through_equal_values():
    sorted_values = torch.tensor([[5.0, 4.0, 4.0, 1.0]], dtype=torch.float32)

    assert flashinfer_sampler._has_top_p_boundary_tie(sorted_values, top_p=0.7)


def test_top_p_boundary_tie_detects_near_equal_deployed_triton_boundary():
    sorted_values = torch.tensor(
        [
            [
                4.331808567047119,
                3.470297336578369,
                3.2183783054351807,
                3.151276111602783,
                3.123741388320923,
                3.077538013458252,
                3.04506778717041,
                3.044318199157715,
                3.017479658126831,
                2.9752144813537598,
                2.9554765224456787,
                2.9428484439849854,
                2.9411158561706543,
                2.9198734760284424,
                2.9119338989257812,
                2.839078664779663,
                2.790985345840454,
                2.7679686546325684,
                2.7447569370269775,
                2.7447564601898193,
            ]
        ],
        dtype=torch.float32,
    )

    assert flashinfer_sampler._has_top_p_boundary_tie(sorted_values, top_p=0.95)


def test_top_p_boundary_tie_allows_ties_kept_together():
    sorted_values = torch.tensor([[5.0, 4.0, 4.0, 1.0]], dtype=torch.float32)

    assert not flashinfer_sampler._has_top_p_boundary_tie(sorted_values, top_p=0.95)


def test_top_p_boundary_tie_ignores_full_nucleus():
    sorted_values = torch.tensor([[5.0, 4.0, 4.0, 1.0]], dtype=torch.float32)

    assert not flashinfer_sampler._has_top_p_boundary_tie(sorted_values, top_p=1.0)


def test_boundary_tie_rejection_reason_requires_guard():
    sorted_values = torch.tensor([[5.0, 4.0, 4.0, 1.0]], dtype=torch.float32)

    reason = flashinfer_sampler._boundary_tie_rejection_reason(
        sorted_values,
        top_p=0.7,
        top_k_boundary_tie=True,
        boundary_tie_guard=False,
    )

    assert reason is None


def test_boundary_tie_rejection_reason_prefers_top_k_boundary():
    sorted_values = torch.tensor([[5.0, 4.0, 4.0, 1.0]], dtype=torch.float32)

    reason = flashinfer_sampler._boundary_tie_rejection_reason(
        sorted_values,
        top_p=0.7,
        top_k_boundary_tie=True,
        boundary_tie_guard=True,
    )

    assert reason == "top_k_boundary_tie"


def test_boundary_tie_rejection_reason_reports_top_p_boundary():
    sorted_values = torch.tensor([[5.0, 4.0, 4.0, 1.0]], dtype=torch.float32)

    reason = flashinfer_sampler._boundary_tie_rejection_reason(
        sorted_values,
        top_p=0.7,
        top_k_boundary_tie=False,
        boundary_tie_guard=True,
    )

    assert reason == "top_p_boundary_tie"


def test_flashinfer_top_k_tie_probe_is_opt_in():
    class FakeFlashInfer:
        def __init__(self):
            self.ks = []

        def top_k(self, logits, k, sorted=False):
            self.ks.append(k)
            return torch.topk(logits, k=k, dim=-1)

    logits = torch.tensor([[4.0, 3.0, 2.0, 2.0, 1.0]], dtype=torch.float32)
    flashinfer = FakeFlashInfer()

    vals, ids, boundary_tie = flashinfer_sampler._flashinfer_top_k_sorted_with_optional_tie_probe(
        flashinfer,
        logits,
        top_k=3,
        detect_boundary_tie=False,
    )

    assert flashinfer.ks == [3]
    assert not boundary_tie
    assert vals.tolist() == [[4.0, 3.0, 2.0]]
    assert ids.shape == (1, 3)

    vals, _, boundary_tie = flashinfer_sampler._flashinfer_top_k_sorted_with_optional_tie_probe(
        flashinfer,
        logits,
        top_k=3,
        detect_boundary_tie=True,
    )

    assert flashinfer.ks == [3, 4]
    assert boundary_tie
    assert vals.is_contiguous()
    assert vals.tolist() == [[4.0, 3.0, 2.0]]


def test_unsupported_max_num_logprobs_reason_names_real_condition():
    logits = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    metadata = SimpleNamespace(max_num_logprobs=2)

    _, reason = flashinfer_sampler._resolve_fast_path_sampling_with_reason(
        logits,
        metadata,
        logprobs_mode="processed_logprobs",
        predict_bonus_token=False,
    )

    assert reason == "max_num_logprobs_unsupported"


def test_warmup_traffic_class_uses_unsupported_max_num_logprobs_reason():
    logits = SimpleNamespace(shape=(4, 4096))
    metadata = SimpleNamespace(max_num_logprobs=None)

    traffic_class = flashinfer_sampler._fallback_traffic_class(
        logits,
        metadata,
        "max_num_logprobs_unsupported",
    )

    assert traffic_class == "warmup_or_profiling"
