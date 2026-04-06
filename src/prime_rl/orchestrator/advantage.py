from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from prime_rl.configs.orchestrator import AdvantageConfig, CustomAdvantageConfig
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_problems rollouts_per_example"]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_problems rollouts_per_example"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...
"""


def default_advantage_fn(
    inputs: AdvantageInputs,
    length_shaping: str = "off",
    length_shaping_alpha: float = 0.33,
    length_shaping_threshold: float = 1.0,
) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-problem baseline."""
    rewards = inputs.rewards
    completion_lengths = inputs.completion_lengths.to(dtype=rewards.dtype)

    if length_shaping == "efficiency":
        advantages = _efficiency_length_shaping(rewards, completion_lengths, length_shaping_threshold)
        return AdvantageOutputs(advantages=advantages)

    elif length_shaping == "gr3":
        lengths_normalized = completion_lengths / completion_lengths.mean(dim=1, keepdim=True)
        rewards = rewards * (1 + length_shaping_alpha * lengths_normalized) ** -1

    baseline = rewards.mean(dim=1, keepdim=True)

    return AdvantageOutputs(advantages=rewards - baseline)


def _efficiency_length_shaping(
    rewards: Float[Tensor, "num_problems rollouts_per_example"],
    completion_lengths: Float[Tensor, "num_problems rollouts_per_example"],
    threshold: float,
) -> Float[Tensor, "num_problems rollouts_per_example"]:
    """Advantage-level length shaping for correct rollouts.

    Mixed groups (some correct, some incorrect):
        A_i = (R_i - mean(R)) * w_i, where w_i = mean_correct_len / len_i for correct, 1 for incorrect.
        Preserves positive advantage for all correct rollouts.

    All-correct groups:
        A_i = w_i - mean(w), giving length differentiation when correctness is saturated.
    """
    correct_mask = rewards >= threshold
    num_correct = correct_mask.sum(dim=1, keepdim=True)
    G = rewards.shape[1]

    # Mean length of correct rollouts per problem (0 where none correct)
    correct_lengths = completion_lengths * correct_mask
    mean_correct_len = correct_lengths.sum(dim=1, keepdim=True) / num_correct.clamp(min=1)

    # Efficiency weight: mean_correct_len / len for correct, 1 for incorrect
    w = torch.where(correct_mask, mean_correct_len / completion_lengths, torch.ones_like(completion_lengths))

    all_correct = num_correct == G
    baseline = rewards.mean(dim=1, keepdim=True)

    # Mixed: advantage-level shaping (preserves signs)
    mixed_advantages = (rewards - baseline) * w

    # All-correct: reward-level shaping (provides length signal)
    all_correct_advantages = w - w.mean(dim=1, keepdim=True)

    return torch.where(all_correct, all_correct_advantages, mixed_advantages)


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(
            inputs,
            length_shaping=config.length_shaping,
            length_shaping_alpha=config.length_shaping_alpha,
            length_shaping_threshold=config.length_shaping_threshold,
        )

    return advantage_fn


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by problem.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (DefaultAdvantageConfig or CustomAdvantageConfig)
    """
    if not advantage_config:
        return rewards

    advantage_fn = setup_advantage_fn(advantage_config)

    inputs = AdvantageInputs(
        rewards=torch.tensor(rewards).view(-1, samples_per_problem),
        completion_lengths=torch.tensor(completion_lengths).view(-1, samples_per_problem),
    )

    result = advantage_fn(inputs)
    return result.advantages.flatten().tolist()
