"""Shared async JWKS caching, single-flight lock, cooldown, and PyJWK lookup."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx
import jwt

from lai.domain.errors import AuthenticationError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

JwksPayloadFetcher = Callable[[], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class JwksFetchSpec:
    """Parameters describing how a remote JWKS endpoint is fetched and logged."""

    url: str
    timeout_seconds: float
    label: str
    log_prefix: str


@dataclass(frozen=True, slots=True)
class JwksCachePolicy:
    """Cache TTL, refresh cooldown, and fetchability policy for JwksCache."""

    ttl_seconds: float
    min_refresh_interval_seconds: float
    can_fetch: bool = True
    log_prefix: str = "jwks"


def find_jwk_for_kid(
    jwks: Mapping[str, Any],
    kid: str | None,
    *,
    error_prefix: str = "Invalid JWK",
) -> jwt.PyJWK | None:
    """Locate a signing PyJWK by `kid` (or sole key), wrapping PyJWK parse errors as 401."""
    raw_keys = jwks.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        return None
    keys_list = cast("list[object]", raw_keys)
    dict_keys = [cast("dict[str, Any]", item) for item in keys_list if isinstance(item, dict)]
    if not dict_keys:
        return None

    selected: dict[str, Any] | None = None
    if kid is not None:
        for entry in dict_keys:
            if entry.get("kid") == kid:
                selected = entry
                break
    elif len(dict_keys) == 1:
        selected = dict_keys[0]

    if selected is None:
        return None

    try:
        return jwt.PyJWK.from_dict(selected)
    except jwt.PyJWTError as exc:
        get_component_logger("security").warning(
            "jwks_invalid_key_entry",
            kid=kid,
            error=str(exc),
            exception_type=type(exc).__name__,
        )
        msg = f"{error_prefix}: {exc}"
        raise AuthenticationError(msg) from exc


async def http_get_jwks(
    url: str,
    *,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch a JWKS JSON object via the injected or a short-lived AsyncClient."""
    if http_client is not None:
        response = await http_client.get(url, timeout=timeout_seconds)
    else:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
    response.raise_for_status()
    parsed: object = response.json()
    if not isinstance(parsed, dict):
        msg = "JWKS endpoint did not return a JSON object."
        raise TypeError(msg)
    return cast("dict[str, Any]", parsed)


async def fetch_jwks_payload(
    spec: JwksFetchSpec,
    *,
    http_client: httpx.AsyncClient | None = None,
    custom_fetcher: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Fetch and validate a JWKS payload dict, raising AuthenticationError on failure."""
    try:
        if custom_fetcher is not None:
            payload = await custom_fetcher(spec.url)
        else:
            payload = await http_get_jwks(
                spec.url,
                timeout_seconds=spec.timeout_seconds,
                http_client=http_client,
            )
    except (httpx.TimeoutException, TimeoutError) as exc:
        get_component_logger("security").error(
            f"{spec.log_prefix}_jwks_fetch_timeout",
            jwks_url=spec.url,
            timeout_seconds=spec.timeout_seconds,
            alert_category=AlertCategory.TIMEOUT.value,
            exception_type=type(exc).__name__,
        )
        msg = f"Timed out fetching {spec.label} JWKS."
        raise AuthenticationError(msg) from exc
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        get_component_logger("security").error(
            f"{spec.log_prefix}_jwks_fetch_failed",
            jwks_url=spec.url,
            error=str(exc),
            exception_type=type(exc).__name__,
        )
        msg = f"Failed to fetch {spec.label} JWKS: {exc}"
        raise AuthenticationError(msg) from exc

    if not isinstance(payload.get("keys"), list):
        get_component_logger("security").error(
            f"{spec.log_prefix}_jwks_invalid_payload",
            jwks_url=spec.url,
        )
        msg = f"Invalid {spec.label} JWKS payload: missing 'keys' list."
        raise AuthenticationError(msg)
    return payload


class JwksCache:
    """Async TTL-cached JWKS store with single-flight lock and unknown-kid cooldown."""

    def __init__(
        self,
        *,
        fetcher: JwksPayloadFetcher,
        policy: JwksCachePolicy,
        clock: Callable[[], float] = time.monotonic,
        initial_jwks: Mapping[str, Any] | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._policy = policy
        self._clock = clock
        self._lock = asyncio.Lock()
        self._jwks: dict[str, Any] | None = dict(initial_jwks) if initial_jwks is not None else None
        if initial_jwks is not None:
            now = clock()
            self._last_fetch_at: float | None = now
            self._expires_at: float = now + policy.ttl_seconds
        else:
            self._last_fetch_at = None
            self._expires_at = 0.0

    @property
    def jwks(self) -> dict[str, Any] | None:
        """Return currently cached JWKS mapping if loaded."""
        return self._jwks

    async def refresh(self) -> dict[str, Any]:
        """Unconditionally refresh JWKS under lock (used for startup warm-up/explicit refresh)."""
        async with self._lock:
            return await self._refresh_locked()

    async def _refresh_locked(self) -> dict[str, Any]:
        """Fetch JWKS while holding lock and update cache metadata on success."""
        payload = await self._fetcher()
        fetched_at = self._clock()
        self._jwks = payload
        self._last_fetch_at = fetched_at
        self._expires_at = fetched_at + self._policy.ttl_seconds
        return payload

    def _lookup_valid_cached_key(
        self,
        kid: str | None,
        now: float,
        *,
        error_prefix: str,
    ) -> tuple[bool, jwt.PyJWK | None]:
        """Return `(cache_fresh, matching_key)` for the current JWKS cache state."""
        if self._jwks is None or now >= self._expires_at:
            return False, None
        return True, find_jwk_for_kid(self._jwks, kid, error_prefix=error_prefix)

    def _is_refresh_throttled(self, now: float) -> bool:
        """Return True when unknown-kid refresh is within the configured cooldown window."""
        if self._last_fetch_at is None:
            return False
        return (now - self._last_fetch_at) < self._policy.min_refresh_interval_seconds

    async def resolve_key(
        self,
        kid: str | None,
        *,
        error_prefix: str = "Invalid JWK",
        missing_key_message: str = "No matching JWKS key found for token.",
    ) -> jwt.PyJWK:
        """Resolve signing key from cache, refreshing on miss/expiry subject to cooldown."""
        _, cached_key = self._lookup_valid_cached_key(
            kid,
            self._clock(),
            error_prefix=error_prefix,
        )
        if cached_key is not None:
            return cached_key

        async with self._lock:
            now = self._clock()
            fresh, cached_key = self._lookup_valid_cached_key(
                kid,
                now,
                error_prefix=error_prefix,
            )
            if cached_key is not None:
                return cached_key
            if not self._policy.can_fetch:
                raise AuthenticationError(missing_key_message)
            if fresh and self._is_refresh_throttled(now):
                get_component_logger("security").warning(
                    f"{self._policy.log_prefix}_jwks_refresh_throttled",
                    kid=kid,
                )
                raise AuthenticationError(missing_key_message)
            jwks = await self._refresh_locked()

        refreshed_key = find_jwk_for_kid(jwks, kid, error_prefix=error_prefix)
        if refreshed_key is None:
            get_component_logger("security").warning(
                f"{self._policy.log_prefix}_unknown_kid",
                kid=kid,
            )
            raise AuthenticationError(missing_key_message)
        return refreshed_key
