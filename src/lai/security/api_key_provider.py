"""API key authentication provider implementing AuthProviderPort."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.auth import AuthMethod, AuthPrincipal, RequestCredentials
from lai.domain.base import utc_now
from lai.domain.errors import AuthenticationError
from lai.observability.logging import get_component_logger
from lai.ports.auth import AuthProviderPort
from lai.security.keys import API_KEY_PREFIX, hash_api_key, verify_key_hash

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from lai.ports.repositories import ApiKeyRepositoryPort


def _extract_api_key_token(credentials: RequestCredentials) -> str | None:
    """Return the bearer token when the scheme is missing or Bearer."""
    if (
        credentials.authorization_scheme is not None
        and credentials.authorization_scheme.lower() != "bearer"
    ):
        return None
    token = credentials.bearer_token
    if not token or not token.startswith(API_KEY_PREFIX):
        return None
    return token


class ApiKeyAuthProvider(AuthProviderPort):
    """Authenticate `lai_sk_` bearer tokens against persisted hashed API keys."""

    def __init__(
        self,
        repository: ApiKeyRepositoryPort,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._repository = repository
        self._clock = clock

    @property
    def name(self) -> str:
        """Return provider identifier."""
        return "api_key"

    def supports(self, credentials: RequestCredentials) -> bool:
        """Return True when credentials contain a `lai_sk_` bearer token."""
        return _extract_api_key_token(credentials) is not None

    async def authenticate(self, credentials: RequestCredentials) -> AuthPrincipal:
        """Verify an API key token, enforce active status, and update last_used_at."""
        token = _extract_api_key_token(credentials)
        if token is None:
            get_component_logger("security").warning(
                "api_key_auth_failed", reason="missing_or_malformed_token"
            )
            msg = "Missing or malformed API key."
            raise AuthenticationError(msg)

        key_hash = hash_api_key(token)
        record = await self._repository.get_by_hash(key_hash)
        if record is None or not verify_key_hash(token, record.key_hash):
            get_component_logger("security").warning("api_key_auth_failed", reason="unknown_key")
            msg = "Invalid API key."
            raise AuthenticationError(msg)

        now = self._clock()
        if not record.is_active(now=now):
            reason = "revoked" if record.revoked_at is not None else "expired"
            get_component_logger("security").warning(
                "api_key_auth_failed", reason=reason, key_id=record.id
            )
            msg = f"API key is {reason}."
            raise AuthenticationError(msg, details={"reason": reason, "key_id": record.id})

        await self._repository.touch_last_used(record.id, now)
        return AuthPrincipal(
            principal_id=record.id,
            auth_method=AuthMethod.API_KEY,
            scopes=record.scopes,
            key_id=record.id,
        )
