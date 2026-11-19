"""Registered client application identity and tool capability models."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_client_app_id
from lai.domain.tools import PermissionTier, ToolSpec


class ClientAppRecord(StrictModel):
    """Persisted client application registration bound to an OAuth client ID."""

    id: str = Field(default_factory=new_client_app_id)
    name: str = Field(min_length=1, max_length=120)
    oauth_client_id: str = Field(min_length=1, max_length=256)
    allowed_tiers: tuple[PermissionTier, ...] = (PermissionTier.READ,)
    registered_tools: tuple[ToolSpec, ...] = ()
    allowed_project_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    def is_active(self, now: datetime | None = None) -> bool:
        """Return True when the client app has not been revoked or expired."""
        current = now or utc_now()
        if self.revoked_at is not None:
            return False
        return not (self.expires_at is not None and current >= self.expires_at)
