import random
from typing import TypeVar

T = TypeVar("T")


def sample_rollouts_for_logging(
    rollouts: list[T],
    sample_ratio: float | None,
) -> list[T]:
    """Apply monitor sample_ratio semantics to a rollout batch.

    - ``None`` keeps the full batch.
    - ``<= 0`` logs nothing.
    - ``0 < ratio < 1`` logs a random subset with a minimum of 1 item.
    - ``>= 1`` keeps the full batch.
    """
    if sample_ratio is None:
        return rollouts
    if sample_ratio <= 0.0:
        return []
    if sample_ratio >= 1.0 or len(rollouts) <= 1:
        return rollouts

    max_samples = max(1, int(len(rollouts) * sample_ratio))
    if len(rollouts) <= max_samples:
        return rollouts

    return random.sample(rollouts, max_samples)
