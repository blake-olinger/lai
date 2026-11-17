"""Authentication and authorization domain models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_api_key_id


class ApiScope(StrEnum):
    """Fine-grained permission scopes granted to API keys and principals."""

    PROJECTS_READ = "projects:read"
    PROJECTS_WRITE = "projects:write"
    FILES_READ = "files:read"
    FILES_WRITE = "files:write"
    CHAT_WRITE = "chat:write"
    TOOLS_APPROVE = "tools:approve"
    ADMIN = "admin"


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
    auth_method: Literal["api_key", "cloudflare_access", "local_cli"]
    scopes: tuple[ApiScope, ...]
    key_id: str | None = None
    email: str | None = None

    def has_scope(self, scope: ApiScope) -> bool:
        """Return True if the principal holds the requested scope or admin."""
        return ApiScope.ADMIN in self.scopes or scope in self.scopes
