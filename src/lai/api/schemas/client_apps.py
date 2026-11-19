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
    """Payload for registering a new OIDC-bound client application."""

    name: str = Field(min_length=1, max_length=120)
    oauth_client_id: str = Field(min_length=1, max_length=256)
    allowed_tiers: tuple[PermissionTier, ...] = (PermissionTier.READ,)
    registered_tools: tuple[ToolSpec, ...] = ()
    allowed_project_ids: tuple[str, ...] = ()
    expires_at: datetime | None = None


class UpdateClientAppRequest(StrictModel):
    """Partial update payload for an active OIDC-bound client application."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    allowed_tiers: tuple[PermissionTier, ...] | None = None
    registered_tools: tuple[ToolSpec, ...] | None = None
    allowed_project_ids: tuple[str, ...] | None = None
    expires_at: datetime | None = None


class ClientAppResponse(StrictModel):
    """Public metadata for a registered OIDC-bound client application."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    oauth_client_id: str = Field(min_length=1, max_length=256)
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
        """Convert a ClientAppRecord domain model into a public API response."""
        return cls(
            id=record.id,
            name=record.name,
            oauth_client_id=record.oauth_client_id,
            allowed_tiers=record.allowed_tiers,
            registered_tools=record.registered_tools,
            allowed_project_ids=record.allowed_project_ids,
            created_at=record.created_at,
            last_used_at=record.last_used_at,
            expires_at=record.expires_at,
            revoked_at=record.revoked_at,
            active=record.is_active(now=now),
        )


class ClientAppListResponse(StrictModel):
    """List of registered client application metadata records."""

    items: list[ClientAppResponse]
    count: int = Field(ge=0)
