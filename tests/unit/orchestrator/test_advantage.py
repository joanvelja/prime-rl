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


def test_efficiency_mixed_group():
    """Mixed group: advantage-level shaping preserves positive advantage for all correct rollouts."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 30, 20, 20]]),
    )
    result = default_advantage_fn(inputs, length_shaping="efficiency")

    # mean_correct_len = (10+30+20)/3 = 20
    # w = [20/10, 20/30, 1, 20/20] = [2.0, 2/3, 1.0, 1.0]
    # baseline = mean(R) = 3/4
    # A_correct = (1 - 0.75) * w = 0.25 * w
    # A_incorrect = (0 - 0.75) * 1 = -0.75
    expected = torch.tensor([[0.25 * 2.0, 0.25 * (2.0 / 3.0), -0.75, 0.25 * 1.0]])

    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # All correct rollouts have positive advantage
    correct_mask = inputs.rewards[0] >= 1.0
    assert (result.advantages[0][correct_mask] > 0).all()


def test_efficiency_all_correct_group():
    """All-correct group: switches to reward-level shaping for length differentiation."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 20, 40]]),
    )
    result = default_advantage_fn(inputs, length_shaping="efficiency")

    # mean_correct_len = (10+20+40)/3 = 70/3
    # w = [70/30, 70/60, 70/120] = [7/3, 7/6, 7/12]
    # A = w - mean(w)
    mean_len = 70.0 / 3.0
    w = torch.tensor([mean_len / 10, mean_len / 20, mean_len / 40])
    expected = (w - w.mean()).unsqueeze(0)

    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # Shortest rollout has highest advantage
    assert result.advantages[0, 0] > result.advantages[0, 1] > result.advantages[0, 2]


def test_efficiency_no_correct_rollouts():
    """When no rollout is correct, efficiency shaping has no effect."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[10, 20, 15]]),
    )
    result_with = default_advantage_fn(inputs, length_shaping="efficiency")
    result_without = default_advantage_fn(inputs)

    assert torch.allclose(result_with.advantages, result_without.advantages, atol=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: w=1 for it, no differentiation among correct, same as standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[100, 50, 200, 150]]),
    )
    result = default_advantage_fn(inputs, length_shaping="efficiency")

    # Only 1 correct, mean_correct_len = 100, w = 100/100 = 1
    # A = (R - 0.25) * w = (R - 0.25) * 1 for correct, (R - 0.25) * 1 for incorrect
    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_efficiency_custom_threshold():
    """Threshold parameter controls what counts as 'correct'."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[0.8, 0.9, 0.3, 0.7]]),
        completion_lengths=torch.tensor([[10, 30, 20, 20]]),
    )
    result = default_advantage_fn(inputs, length_shaping="efficiency", length_shaping_threshold=0.7)

    # Correct (>= 0.7): indices 0,1,3 with lengths 10,30,20
    # mean_correct_len = 20
    # w = [20/10, 20/30, 1.0, 20/20] = [2.0, 2/3, 1.0, 1.0]
    expected = torch.tensor(
        [[(0.8 - 0.675) * 2.0, (0.9 - 0.675) * (2.0 / 3.0), (0.3 - 0.675) * 1.0, (0.7 - 0.675) * 1.0]]
    )

    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_efficiency_shorter_correct_always_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[50, 100, 200, 80, 120]]),
    )
    result = default_advantage_fn(inputs, length_shaping="efficiency")

    advs = result.advantages[0]
    # Correct rollouts: 0, 1, 2 with lengths 50, 100, 200
    assert advs[0] > advs[1] > advs[2]
    # All correct have positive advantage (3 correct, 5 total -> baseline = 3/5 = 0.6)
    assert (advs[:3] > 0).all()
    # Incorrect have negative advantage
    assert (advs[3:] < 0).all()


def test_efficiency_incorrect_unchanged():
    """Incorrect rollouts get exactly the same advantage as standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[10, 30, 20, 40]]),
    )
    result_eff = default_advantage_fn(inputs, length_shaping="efficiency")
    result_std = default_advantage_fn(inputs)

    # Incorrect rollout advantages should be identical
    assert torch.allclose(result_eff.advantages[0, 2:], result_std.advantages[0, 2:], atol=1e-6)


def test_efficiency_multiple_problems():
    """Handles multiple problems independently."""
    inputs = AdvantageInputs(
        rewards=torch.tensor(
            [
                [1.0, 1.0, 0.0],  # mixed
                [1.0, 1.0, 1.0],  # all correct
            ]
        ),
        completion_lengths=torch.tensor(
            [
                [10, 20, 15],
                [10, 20, 40],
            ]
        ),
    )
    result = default_advantage_fn(inputs, length_shaping="efficiency")

    # Row 0: mixed group (advantage-level)
    assert result.advantages[0, 0] > result.advantages[0, 1]  # shorter correct > longer correct
    assert (result.advantages[0, :2] > 0).all()  # both correct positive
    assert result.advantages[0, 2] < 0  # incorrect negative

    # Row 1: all-correct group (reward-level)
    assert result.advantages[1, 0] > result.advantages[1, 1] > result.advantages[1, 2]


def test_gr3_length_shaping():
    """GR3: multiplicative shaping on all rollouts relative to mean length."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 20, 10]]),
    )
    result = default_advantage_fn(inputs, length_shaping="gr3", length_shaping_alpha=0.33)

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
