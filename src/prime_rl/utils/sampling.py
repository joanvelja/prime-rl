import random
from typing import Any


def sample_items(items: list[Any], ratio: float | None) -> list[Any]:
    """Return a (possibly sampled) subset of *items* based on *ratio*.

    - ``None`` keeps the full list.
    - ``<= 0`` returns nothing.
    - ``0 < ratio < 1`` returns a random subset (minimum 1 item).
    - ``>= 1`` keeps the full list.
    """
    if ratio is None:
        return items
    if ratio <= 0.0:
        return []
    if ratio >= 1.0 or len(items) <= 1:
        return items

    max_samples = max(1, int(len(items) * ratio))
    if len(items) <= max_samples:
        return items

    return random.sample(items, max_samples)
