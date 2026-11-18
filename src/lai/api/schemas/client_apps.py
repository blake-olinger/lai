"""Admin REST schemas for registered external client applications."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.tools import PermissionTier, ToolSpec

if TYPE_CHECKING:
    from lai.domain.client_apps import ClientAppRecord


class CreateClientAppRequest(StrictModel):
    """Payload for registering a new client application."""

    name: str = Field(min_length=1, max_length=120)
    allowed_tiers: tuple[PermissionTier, ...] = (PermissionTier.READ,)
    registered_tools: tuple[ToolSpec, ...] = ()
    allowed_project_ids: tuple[str, ...] = ()
    expires_at: datetime | None = None


class ClientAppResponse(StrictModel):
    """Public metadata for a registered client application without secret_hash."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    secret_prefix: str = Field(min_length=4, max_length=32)
    allowed_tiers: tuple[PermissionTier, ...]
    registered_tools: tuple[ToolSpec, ...]
    allowed_project_ids: tuple[str, ...]
    created_at: datetime
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    active: bool

    @classmethod
    def from_record(
        cls,
        record: ClientAppRecord,
        *,
        now: datetime | None = None,
    ) -> ClientAppResponse:
        """Convert a ClientAppRecord domain model into a secret-free API response."""
        return cls(
            id=record.id,
            name=record.name,
            secret_prefix=record.secret_prefix,
            allowed_tiers=record.allowed_tiers,
            registered_tools=record.registered_tools,
            allowed_project_ids=record.allowed_project_ids,
            created_at=record.created_at,
            last_used_at=record.last_used_at,
            expires_at=record.expires_at,
            revoked_at=record.revoked_at,
            active=record.is_active(now=now),
        )


class CreatedClientAppResponse(StrictModel):
    """One-time response returning a newly registered client app and its plaintext secret."""

    app: ClientAppResponse
    client_secret: str = Field(min_length=16)


class ClientAppListResponse(StrictModel):
    """List of registered client application metadata records."""

    items: list[ClientAppResponse]
    count: int = Field(ge=0)
