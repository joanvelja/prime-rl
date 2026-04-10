from __future__ import annotations

import asyncio

from aiolimiter import AsyncLimiter


class RateLimiter:
    """Rate limiter that supports variable-cost acquire (N rollouts at once).

    Wraps aiolimiter.AsyncLimiter, which rejects acquire(amount) when amount
    exceeds max_rate. We loop instead so group-scoring envs work correctly.
    """

    def __init__(self, max_rate: float, time_period: float = 60):
        self._limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)

    async def acquire(self, count: int = 1) -> None:
        for _ in range(count):
            await self._limiter.acquire()


class ConcurrencyLimiter:
    """Shared concurrency limiter that gates both train and eval rollouts.

    Uses a simple counter with an asyncio.Event for wakeup. All calls must
    happen on the same event loop (single-threaded asyncio), so no lock is needed.
    """

    def __init__(self, max_concurrency: int):
        self._max = max_concurrency
        self._used = 0
        self._available = asyncio.Event()
        self._available.set()

    @property
    def max_concurrency(self) -> int:
        return self._max

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._max - self._used

    async def acquire(self, count: int = 1) -> None:
        """Wait until *count* slots are available, then reserve them."""
        while self.remaining < count:
            self._available.clear()
            await self._available.wait()
        self._used += count

    def try_acquire(self, count: int = 1) -> bool:
        """Non-blocking acquire. Returns True if slots were reserved."""
        if self.remaining >= count:
            self._used += count
            return True
        return False

    def release(self, count: int = 1) -> None:
        """Return *count* slots to the pool and wake any blocked acquirers."""
        self._used -= count
        assert self._used >= 0, f"ConcurrencyLimiter released too many slots (used={self._used})"
        self._available.set()


class RolloutLimiter:
    """Combined rate + concurrency limiter for rollout scheduling.

    Acquire gates on both limits (rate first, then concurrency).
    Release only applies to concurrency (rate tokens are not returned).
    """

    def __init__(self, max_concurrency: int, max_rate: float | None = None, time_period: float = 60):
        self.concurrency = ConcurrencyLimiter(max_concurrency)
        self.rate = RateLimiter(max_rate, time_period) if max_rate is not None else None

    @property
    def remaining(self) -> int:
        return self.concurrency.remaining

    async def acquire(self, count: int = 1) -> None:
        """Acquire rate tokens then concurrency slots (blocking)."""
        if self.rate:
            await self.rate.acquire(count)
        await self.concurrency.acquire(count)

    def try_acquire(self, count: int = 1) -> bool:
        """Non-blocking concurrency acquire (rate limiting is skipped)."""
        return self.concurrency.try_acquire(count)

    def release(self, count: int = 1) -> None:
        """Release concurrency slots."""
        self.concurrency.release(count)
