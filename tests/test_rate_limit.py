"""Tests for AsyncTokenBucket burst and rate pacing."""

import asyncio
import time

from rtt.translate.rate_limit import AsyncTokenBucket


def test_token_bucket_burst_and_pacing():
    async def run():
        bucket = AsyncTokenBucket(rate_per_sec=10.0, burst=2)
        t0 = time.monotonic()
        await bucket.acquire()
        await bucket.acquire()
        burst_elapsed = time.monotonic() - t0
        assert burst_elapsed < 0.08

        await bucket.acquire()
        await bucket.acquire()
        total_elapsed = time.monotonic() - t0
        assert 0.14 <= total_elapsed <= 0.80

    asyncio.run(run())
