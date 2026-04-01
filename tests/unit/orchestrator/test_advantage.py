import torch

from prime_rl.configs.orchestrator import CustomAdvantageConfig, DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    compute_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)


def test_default_advantage_fn_simple_mean():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]]),
        completion_lengths=torch.tensor([[10, 12, 8], [15, 11, 9]]),
    )
    result = default_advantage_fn(inputs)

    assert result.advantages.shape == (2, 3)
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_length_shaping_only_penalizes_correct_rollouts():
    """Correct rollouts get attenuated by L_min/L_i; incorrect ones are unchanged."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 30, 20, 20]]),
    )
    result = default_advantage_fn(inputs, length_shaping="brevity_bonus")

    # min_correct = 10
    # shaped: [1*10/10, 1*10/30, 0, 1*10/20] = [1.0, 1/3, 0.0, 0.5]
    shaped_rewards = torch.tensor([1.0, 1.0 / 3, 0.0, 0.5])
    expected = shaped_rewards - shaped_rewards.mean()

    assert torch.allclose(result.advantages, expected.unsqueeze(0), atol=1e-6)


def test_length_shaping_shortest_correct_keeps_full_reward():
    """The shortest correct rollout keeps reward=1."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 20, 40]]),
    )
    result = default_advantage_fn(inputs, length_shaping="brevity_bonus")

    # shaped: [1*10/10, 1*10/20, 1*10/40] = [1.0, 0.5, 0.25]
    shaped_rewards = torch.tensor([1.0, 0.5, 0.25])
    expected = shaped_rewards - shaped_rewards.mean()

    assert torch.allclose(result.advantages, expected.unsqueeze(0), atol=1e-6)


def test_length_shaping_no_correct_rollouts():
    """When no rollout is correct, length shaping has no effect."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[10, 20, 15]]),
    )
    result_with = default_advantage_fn(inputs, length_shaping="brevity_bonus")
    result_without = default_advantage_fn(inputs)

    assert torch.allclose(result_with.advantages, result_without.advantages, atol=1e-6)


def test_gr3_length_shaping():
    """GR³: multiplicative shaping on all rollouts relative to mean length."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 20, 10]]),
    )
    result = default_advantage_fn(inputs, length_shaping="gr3", length_shaping_alpha=0.33)

    # mean_length = 40/3, normalized = [10/13.33, 20/13.33, 10/13.33] = [0.75, 1.5, 0.75]
    # factor = (1 + 0.33 * normalized)^-1
    # rewards_shaped = rewards * factor, then subtract mean
    expected = torch.tensor([[0.20915856, -0.25799648, 0.04883792]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_compute_advantages_with_config():
    rewards = [1.0, 0.5, 0.8, 0.2, 0.9, 0.1]
    lengths = [10, 12, 8, 15, 11, 9]

    result = compute_advantages(rewards, lengths, samples_per_problem=3, advantage_config=DefaultAdvantageConfig())

    assert len(result) == 6
    assert abs(sum(result[:3])) < 1e-5
    assert abs(sum(result[3:])) < 1e-5


def test_compute_advantages_without_config():
    rewards = [1.0, 0.5, 0.8]
    lengths = [10, 12, 8]

    result = compute_advantages(rewards, lengths, samples_per_problem=3, advantage_config=None)

    assert result == rewards


def test_setup_advantage_fn_with_custom_config():
    config = CustomAdvantageConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    advantage_fn = setup_advantage_fn(config)

    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 12, 8]]),
    )

    result = advantage_fn(inputs)
    assert isinstance(result, AdvantageOutputs)
    assert torch.allclose(result.advantages, torch.tensor([[2.0, 1.0, 1.6]]))


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    return AdvantageOutputs(advantages=inputs.rewards * scale)
