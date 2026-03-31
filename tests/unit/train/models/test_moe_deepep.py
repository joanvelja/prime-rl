import sys
import types

import torch
from torch import nn

from prime_rl.trainer.models.layers.moe import _run_deepep_routed_experts


class _FakeExperts(nn.Module):
    num_experts = 2

    def __init__(self) -> None:
        super().__init__()
        self._ep_group = object()

    def forward(self, hidden_states: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1


def test_run_deepep_routed_experts_syncs_before_finalize(monkeypatch):
    events: list[str] = []
    combine_synced = False

    def dispatch_tokens_async(
        hidden_states: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
        num_experts: int,
        group: object,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        del selected_experts_indices, top_scores, num_experts, group, score_before_experts
        return hidden_states

    def finalize_dispatch_tokens(pending_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, object]:
        num_tokens_per_expert = torch.tensor([pending_state.shape[0], 0], dtype=torch.int32)
        return pending_state, num_tokens_per_expert, object()

    def combine_tokens(hidden_states: torch.Tensor, state: object) -> torch.Tensor:
        del state
        events.append("combine")
        return hidden_states + 10

    def sync_combine() -> None:
        nonlocal combine_synced
        events.append("sync")
        combine_synced = True

    fake_deepep = types.ModuleType("prime_rl.trainer.distributed.deepep")
    fake_deepep.combine_tokens = combine_tokens
    fake_deepep.dispatch_tokens_async = dispatch_tokens_async
    fake_deepep.finalize_dispatch_tokens = finalize_dispatch_tokens
    fake_deepep.sync_combine = sync_combine
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.distributed.deepep", fake_deepep)

    finalized_inputs: list[torch.Tensor] = []

    def finalize_combined_output(combined_output: torch.Tensor) -> torch.Tensor:
        assert combine_synced
        events.append("finalize")
        finalized_inputs.append(combined_output.clone())
        return combined_output * 2

    def compute_parallel_output(x: torch.Tensor) -> torch.Tensor:
        assert not combine_synced
        events.append("parallel")
        return torch.full_like(x, -1)

    experts = _FakeExperts()
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    selected_experts_indices = torch.zeros((6, 1), dtype=torch.int64)
    top_scores = torch.ones((6, 1), dtype=torch.float32)

    output = _run_deepep_routed_experts(
        x,
        selected_experts_indices,
        top_scores,
        experts=experts,
        deepep_token_chunk_size=4,
        score_before_experts=False,
        finalize_combined_output=finalize_combined_output,
        compute_parallel_output=compute_parallel_output,
    )

    assert events == ["combine", "combine", "parallel", "sync", "finalize", "finalize"]
    assert len(finalized_inputs) == 2
    expected = (x + 11) * 2 - 1
    assert torch.equal(output, expected)
