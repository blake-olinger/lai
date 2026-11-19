"""Authentication and authorization domain models."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_api_key_id

if TYPE_CHECKING:
    from collections.abc import Iterable


class ApiScope(StrEnum):
    """Fine-grained permission scopes granted to API keys and principals."""

    PROJECTS_READ = "projects:read"
    PROJECTS_WRITE = "projects:write"
    FILES_READ = "files:read"
    FILES_WRITE = "files:write"
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    TOOLS_SAFE = "tools:safe"
    TOOLS_EXEC = "tools:exec"
    TOOLS_APPROVE = "tools:approve"
    ADMIN = "admin"


_IMPLIED_SCOPES: dict[ApiScope, tuple[ApiScope, ...]] = {
    ApiScope.ADMIN: tuple(ApiScope),
    ApiScope.PROJECTS_WRITE: (ApiScope.PROJECTS_READ,),
    ApiScope.FILES_WRITE: (ApiScope.FILES_READ,),
    ApiScope.CHAT_WRITE: (ApiScope.CHAT_READ,),
    ApiScope.TOOLS_EXEC: (ApiScope.TOOLS_SAFE,),
}


def expand_scopes(scopes: Iterable[ApiScope]) -> frozenset[ApiScope]:
    """Return the transitive closure of granted scopes under the scope hierarchy."""
    granted: set[ApiScope] = set()
    pending: deque[ApiScope] = deque(scopes)
    while pending:
        scope = pending.popleft()
        if scope in granted:
            continue
        granted.add(scope)
        for implied in _IMPLIED_SCOPES.get(scope, ()):
            if implied not in granted:
                pending.append(implied)
    return frozenset(granted)


class AuthMethod(StrEnum):
    """Authentication mechanism that established a caller principal."""

    API_KEY = "api_key"
    OIDC = "oidc"
    CLOUDFLARE_ACCESS = "cloudflare_access"
    LOCAL_CLI = "local_cli"


class ApiKeyRecord(StrictModel):
    """Persisted API key metadata and hashed credential."""

    id: str = Field(default_factory=new_api_key_id)
    name: str = Field(min_length=1, max_length=120)
    key_prefix: str = Field(min_length=4, max_length=32)
    key_hash: str = Field(min_length=32)
    scopes: tuple[ApiScope, ...]
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    def is_active(self, now: datetime | None = None) -> bool:
        """Return True when the key has not been revoked or expired."""
        current = now or utc_now()
        if self.revoked_at is not None:
            return False
        return not (self.expires_at is not None and current >= self.expires_at)


class CreatedApiKey(StrictModel):
    """One-time response returned when minting a new API key."""

    record: ApiKeyRecord
    plaintext_key: str = Field(min_length=16)


class AuthPrincipal(StrictModel):
    """Authenticated caller identity attached to a request or turn."""

    principal_id: str = Field(min_length=1)
    auth_method: AuthMethod
    scopes: tuple[ApiScope, ...]
    key_id: str | None = None
    email: str | None = None
    issuer: str | None = None
    subject: str | None = None
    oauth_client_id: str | None = None

    def has_scope(self, scope: ApiScope) -> bool:
        """Return True if the principal holds the requested scope directly or via hierarchy."""
        return scope in expand_scopes(self.scopes)


class RequestCredentials(StrictModel):
    """Extracted HTTP/CLI request credentials evaluated by auth providers."""

    authorization_scheme: str | None = None
    bearer_token: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    client_host: str | None = None
