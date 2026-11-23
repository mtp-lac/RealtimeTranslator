"""Asynchronous token bucket rate limiter."""

import asyncio
import time


class AsyncTokenBucket:
    """Smooths outbound translation requests to stay within provider RPM/RPS limits."""

    def __init__(self, rate_per_sec: float, burst: int = 2):
        self.rate = max(1e-6, float(rate_per_sec))
        self.capacity = float(max(1, burst))
        self.tokens = float(max(1, burst))
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    def set_rate(self, rate_per_sec: float) -> None:
        self.rate = max(1e-6, float(rate_per_sec))

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait_time = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(min(wait_time, 0.25))
