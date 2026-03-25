from unittest.mock import MagicMock

import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory
from prime_rl.rendering.base import RenderedTokens


class SimpleRenderer:
    """Minimal Renderer for testing — assigns incrementing token IDs."""

    def __init__(self):
        self._tok2id: dict[str, int] = {}
        self._next_id = 1

    def _id(self, token: str) -> int:
        if token not in self._tok2id:
            self._tok2id[token] = self._next_id
            self._next_id += 1
        return self._tok2id[token]

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        ids = []
        indices = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            ids.append(self._id(f"<|{role}|>"))
            indices.append(i)
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                ids.append(self._id(content))
                indices.append(i)
        if add_generation_prompt:
            ids.append(self._id("<|assistant|>"))
            indices.append(-1)
        return RenderedTokens(token_ids=ids, message_indices=indices)

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def get_stop_token_ids(self):
        return []


def test_interleave_rollout_missing_tokens_returns_none():
    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    assert interleave_rollout(output) is None


def test_pretokenize_rollout_trajectory_for_sft():
    renderer = SimpleRenderer()
    output = vf.RolloutOutput(
        example_id=42,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    pretokenize_rollout_trajectory(output, renderer)

    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert len(rollouts) == 1

    rollout = rollouts[0]
    step1_prompt_ids = renderer.render_ids(
        [{"role": "user", "content": "U1"}],
        add_generation_prompt=True,
    )
    step1_full_ids = renderer.render_ids(
        [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}],
    )
    step2_prompt_ids = renderer.render_ids(
        [
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ],
        add_generation_prompt=True,
    )
    step2_full_ids = renderer.render_ids(
        [
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
        ],
    )

    prefix_len_1 = len(step1_prompt_ids)
    prefix_len_2 = len(step2_prompt_ids)
    step1_completion_ids = step1_full_ids[prefix_len_1:]
    step2_completion_ids = step2_full_ids[prefix_len_2:]
    step1_prefix = step1_prompt_ids + step1_completion_ids
    step2_new_prompt_ids = step2_prompt_ids[len(step1_prefix) :]

    assert rollout.prompt_ids == step1_prompt_ids
    assert rollout.completion_ids == step1_completion_ids + step2_new_prompt_ids + step2_completion_ids
    assert rollout.completion_mask == (
        [True] * len(step1_completion_ids) + [False] * len(step2_new_prompt_ids) + [True] * len(step2_completion_ids)
    )
    assert rollout.completion_logprobs == [0.0] * len(rollout.completion_ids)
