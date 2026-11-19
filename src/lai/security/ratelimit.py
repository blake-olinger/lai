"""Async monotonic-clock token bucket rate limiter scoped per principal_id."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.errors import RateLimitExceededError
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class _BucketState:
    """Mutable token bucket balance and last refill timestamp."""

    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Per-principal async token bucket rate limiter backed by monotonic time."""

    def __init__(
        self,
        *,
        requests_per_minute: int = 120,
        burst_capacity: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_buckets: int = 10_000,
    ) -> None:
        if requests_per_minute < 1:
            msg = "requests_per_minute must be >= 1."
            raise ValueError(msg)
        capacity = float(burst_capacity if burst_capacity is not None else requests_per_minute)
        if capacity < 1.0:
            msg = "burst_capacity must be >= 1."
            raise ValueError(msg)
        if max_buckets < 1:
            msg = "max_buckets must be >= 1."
            raise ValueError(msg)

        self._rate_per_second = requests_per_minute / 60.0
        self._capacity = capacity
        self._clock = clock
        self._max_buckets = max_buckets
        self._buckets: dict[str, _BucketState] = {}
        self._lock = asyncio.Lock()

    def reconfigure(
        self,
        *,
        requests_per_minute: int,
        burst_capacity: int | None = None,
    ) -> None:
        """Update refill rate and burst capacity in place without clearing existing buckets."""
        if requests_per_minute < 1:
            msg = "requests_per_minute must be >= 1."
            raise ValueError(msg)
        capacity = float(burst_capacity if burst_capacity is not None else requests_per_minute)
        if capacity < 1.0:
            msg = "burst_capacity must be >= 1."
            raise ValueError(msg)
        self._rate_per_second = requests_per_minute / 60.0
        self._capacity = capacity
        for bucket in self._buckets.values():
            bucket.tokens = min(bucket.tokens, capacity)

    async def acquire(self, principal_id: str, *, tokens: float = 1.0) -> None:
        """Consume `tokens` for `principal_id` or raise RateLimitExceededError."""
        if tokens <= 0.0 or tokens > self._capacity:
            msg = f"Requested token cost {tokens} must be in (0, {self._capacity}]."
            raise ValueError(msg)

        async with self._lock:
            now = self._clock()
            bucket = self._buckets.get(principal_id)
            if bucket is None:
                self._evict_oldest_if_full()
                bucket = _BucketState(tokens=self._capacity, last_refill=now)
                self._buckets[principal_id] = bucket
            else:
                elapsed = max(0.0, now - bucket.last_refill)
                bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate_per_second)
                bucket.last_refill = now

            if bucket.tokens < tokens:
                deficit = tokens - bucket.tokens
                retry_after = deficit / self._rate_per_second
                get_component_logger("security").warning(
                    "rate_limit_exceeded",
                    principal_id=principal_id,
                    retry_after_seconds=round(retry_after, 3),
                )
                msg = f"Rate limit exceeded for principal {principal_id!r}."
                raise RateLimitExceededError(
                    msg,
                    retry_after_seconds=retry_after,
                    details={"principal_id": principal_id},
                )
            bucket.tokens -= tokens

    async def check(self, principal_id: str, *, tokens: float = 1.0) -> None:
        """Alias for `acquire` checking and consuming principal rate-limit budget."""
        await self.acquire(principal_id, tokens=tokens)

    def _evict_oldest_if_full(self) -> None:
        """Bound in-memory bucket tracking by evicting the stalest bucket."""
        if len(self._buckets) < self._max_buckets:
            return
        oldest_key = min(self._buckets, key=lambda key: self._buckets[key].last_refill)
        del self._buckets[oldest_key]
