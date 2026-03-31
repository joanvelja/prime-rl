import random
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from datasets import Dataset

from prime_rl.configs.orchestrator import BufferConfig
from prime_rl.orchestrator.buffer import BufferSet
from prime_rl.orchestrator.envs import Envs


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)


@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client."""
    return MagicMock()


@pytest.fixture
def dummy_dataset() -> Dataset:
    """Return a dummy dataset with 5 examples."""
    return Dataset.from_dict(
        {
            "question": ["q0", "q1", "q2", "q3", "q4"],
            "answer": ["a0", "a1", "a2", "a3", "a4"],
        }
    )


@pytest.fixture
def dummy_envs(mock_openai_client, dummy_dataset) -> Envs:
    """Return an Envs with two dummy envs."""
    env_a = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    env_b = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    result = Envs.__new__(Envs)
    result.envs = {"env_a": env_a, "env_b": env_b}
    return result


@pytest.fixture
def make_rollouts():
    def _make_rollouts(
        buffer: BufferSet, env_name: str, indices: list[int], rewards: list[float]
    ) -> list[vf.RolloutOutput]:
        all_rollouts = []
        eb = buffer.env_buffers[env_name]
        examples = list(eb.examples.values())
        for idx, reward in zip(indices, rewards):
            example = examples[idx]
            rollouts = [
                vf.RolloutOutput(
                    example_id=example["example_id"],
                    task=example["task"],
                    prompt=example["prompt"],
                    prompt_ids=[0],
                    prompt_mask=[1],
                    completion_ids=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    is_truncated=False,
                    reward=reward,
                    advantage=1.0,
                    metrics={},
                )
            ] * 2
            for r in rollouts:
                r["env_name"] = env_name
            all_rollouts.extend(rollouts)
        return all_rollouts

    return _make_rollouts


def get_normal_count(buffer: BufferSet) -> int:
    return sum(eb.num_normal for eb in buffer.env_buffers.values())


def test_buffer_init_and_sample(dummy_envs):
    buffer = BufferSet(dummy_envs, BufferConfig())
    assert buffer.env_buffers["env_a"].num_normal == 5
    assert buffer.env_buffers["env_b"].num_normal == 5
    samples = buffer.sample_examples(2)
    assert len(samples) == 2


def test_buffer_problem_pool_assignment(dummy_envs, make_rollouts):
    """Problems are moved to easy/hard pools based on reward thresholds."""
    buffer = BufferSet(dummy_envs, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))

    assert len(buffer.env_buffers["env_a"].easy_examples) == 2
    assert len(buffer.env_buffers["env_a"].hard_examples) == 1
    # 2 normal from env_a + 5 from env_b = 7
    assert get_normal_count(buffer) == 7


def test_buffer_online_difficulty_filtering(dummy_envs, make_rollouts):
    """With online_difficulty_filtering=True, only partial reward rollouts are kept."""
    buffer = BufferSet(
        dummy_envs,
        BufferConfig(online_difficulty_filtering=True),
    )
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # Only 3 problems with reward 0.5 -> 6 rollouts kept
    assert len(buffer.rollout_buffer) == 6


def test_buffer_no_filtering_by_default(dummy_envs, make_rollouts):
    """With online_difficulty_filtering=False (default), all rollouts are kept."""
    buffer = BufferSet(dummy_envs, BufferConfig())
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # All 5 problems -> 10 rollouts kept
    assert len(buffer.rollout_buffer) == 10


def test_buffer_save_load_with_conversion(dummy_envs, make_rollouts, tmp_path):
    """Easy/hard problems are partially converted to normal on load."""
    buffer = BufferSet(dummy_envs, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))
    buffer.save(tmp_path / "buffer")

    new_buffer = BufferSet(dummy_envs, BufferConfig(easy_fraction=0.5, hash_keys=["prompt", "env_name"]))
    new_buffer.load(tmp_path / "buffer")

    # 1 of 2 easy problems converted to normal
    assert len(new_buffer.env_buffers["env_a"].easy_examples) == 1
    # 2 were normal + 5 from env_b + 1 converted from easy = 8
    assert get_normal_count(new_buffer) == 8


def test_buffer_env_ratios(dummy_envs):
    buffer = BufferSet(dummy_envs, BufferConfig(env_ratios=[0.8, 0.2]))
    assert buffer.env_buffers["env_a"].num_normal == 5
    assert buffer.env_buffers["env_b"].num_normal == 5

    samples = buffer.sample_examples(100)
    env_a_count = sum(1 for p in samples if p["env_name"] == "env_a")
    assert 60 <= env_a_count <= 95


def test_buffer_env_ratios_validation():
    """BufferConfig validates that all env_ratios are positive."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="All env_ratios must be positive"):
        BufferConfig(env_ratios=[0.5, -0.3, 0.2])


def test_buffer_no_cross_env_pool_assignment(mock_openai_client, tmp_path):
    """Pool assignments don't transfer if example_id exists but env changed."""
    original_dataset = Dataset.from_dict({"question": ["q0"], "answer": ["a0"]})
    original_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=original_dataset,
        rubric=vf.Rubric(),
    )
    original_env_set = Envs.__new__(Envs)
    original_env_set.envs = {"env_a": original_env}

    buffer = BufferSet(original_env_set, BufferConfig(easy_threshold=1.0))
    eb = buffer.env_buffers["env_a"]
    example_id = list(eb.examples.keys())[0]
    example = eb.examples.pop(example_id)
    eb.easy_examples.append(example)
    buffer.save(tmp_path / "buffer")

    new_dataset = Dataset.from_dict({"question": ["different_q"], "answer": ["different_a"]})
    new_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=new_dataset,
        rubric=vf.Rubric(),
    )
    new_env_set = Envs.__new__(Envs)
    new_env_set.envs = {"env_b": new_env}

    new_buffer = BufferSet(new_env_set, BufferConfig())
    new_buffer.load(tmp_path / "buffer")

    assert len(new_buffer.env_buffers["env_b"].easy_examples) == 0
    assert new_buffer.env_buffers["env_b"].num_normal == 1
