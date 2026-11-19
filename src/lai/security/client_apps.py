"""Client application OAuth client binding and tool-tier registration validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.base import utc_now
from lai.domain.client_apps import ClientAppRecord
from lai.domain.errors import ValidationFailedError
from lai.domain.ids import new_client_app_id
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime


def _validate_client_tools(
    registered_tools: Sequence[ToolSpec],
    allowed_tiers: tuple[PermissionTier, ...],
) -> tuple[ToolSpec, ...]:
    """Validate tool tiers and uniqueness, coercing execution_target to CLIENT."""
    normalized: list[ToolSpec] = []
    seen_names: set[str] = set()
    for spec in registered_tools:
        if spec.permission_tier not in allowed_tiers:
            msg = (
                f"Registered tool {spec.name!r} requires tier {spec.permission_tier.value!r}, "
                "which is not in allowed_tiers."
            )
            raise ValidationFailedError(
                msg,
                details={"tool_name": spec.name, "permission_tier": spec.permission_tier.value},
            )
        if spec.name in seen_names:
            msg = f"Duplicate registered tool name {spec.name!r}."
            raise ValidationFailedError(msg, details={"tool_name": spec.name})
        seen_names.add(spec.name)
        normalized.append(spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT}))
    return tuple(normalized)


def _normalize_allowed_tiers(
    allowed_tiers: Sequence[PermissionTier],
) -> tuple[PermissionTier, ...]:
    """Deduplicate allowed tiers while rejecting an empty tier list."""
    dedup = tuple(dict.fromkeys(allowed_tiers))
    if not dedup:
        msg = "At least one allowed permission tier is required."
        raise ValidationFailedError(msg, details={"field": "allowed_tiers"})
    return dedup


def _normalize_project_ids(allowed_project_ids: Sequence[str]) -> tuple[str, ...]:
    """Trim and deduplicate allowed project identifiers."""
    return tuple(dict.fromkeys(pid.strip() for pid in allowed_project_ids if pid.strip()))


def build_client_app(  # noqa: PLR0913
    name: str,
    oauth_client_id: str,
    allowed_tiers: Sequence[PermissionTier] = (PermissionTier.READ,),
    registered_tools: Sequence[ToolSpec] = (),
    allowed_project_ids: Sequence[str] = (),
    *,
    expires_at: datetime | None = None,
    app_id: str | None = None,
    created_at: datetime | None = None,
) -> ClientAppRecord:
    """Build a client app record bound to an OAuth client ID with CLIENT-targeted tools."""
    stripped_name = name.strip()
    if not stripped_name:
        msg = "Client app name must not be empty."
        raise ValidationFailedError(msg, details={"field": "name"})

    stripped_oauth_client_id = oauth_client_id.strip()
    if not stripped_oauth_client_id:
        msg = "OAuth client ID must not be empty."
        raise ValidationFailedError(msg, details={"field": "oauth_client_id"})

    dedup_tiers = _normalize_allowed_tiers(allowed_tiers)
    cleaned_projects = _normalize_project_ids(allowed_project_ids)
    normalized_tools = _validate_client_tools(registered_tools, dedup_tiers)

    return ClientAppRecord(
        id=app_id or new_client_app_id(),
        name=stripped_name,
        oauth_client_id=stripped_oauth_client_id,
        allowed_tiers=dedup_tiers,
        registered_tools=normalized_tools,
        allowed_project_ids=cleaned_projects,
        created_at=created_at or utc_now(),
        expires_at=expires_at,
    )


def update_client_app(  # noqa: PLR0913
    existing: ClientAppRecord,
    *,
    name: str | None = None,
    allowed_tiers: Sequence[PermissionTier] | None = None,
    registered_tools: Sequence[ToolSpec] | None = None,
    allowed_project_ids: Sequence[str] | None = None,
    expires_at: datetime | None = None,
    update_expires_at: bool = False,
) -> ClientAppRecord:
    """Apply validated in-place updates to a non-revoked ClientAppRecord."""
    if existing.revoked_at is not None:
        msg = f"Client app {existing.id!r} is revoked and cannot be updated."
        raise ValidationFailedError(msg, details={"app_id": existing.id})

    resolved_name = existing.name
    if name is not None:
        stripped_name = name.strip()
        if not stripped_name:
            msg = "Client app name must not be empty."
            raise ValidationFailedError(msg, details={"field": "name"})
        resolved_name = stripped_name

    resolved_tiers = (
        _normalize_allowed_tiers(allowed_tiers)
        if allowed_tiers is not None
        else existing.allowed_tiers
    )
    resolved_projects = (
        _normalize_project_ids(allowed_project_ids)
        if allowed_project_ids is not None
        else existing.allowed_project_ids
    )
    candidate_tools = (
        registered_tools if registered_tools is not None else existing.registered_tools
    )
    resolved_tools = _validate_client_tools(candidate_tools, resolved_tiers)
    resolved_expires = expires_at if update_expires_at else existing.expires_at

    return existing.model_copy(
        update={
            "name": resolved_name,
            "allowed_tiers": resolved_tiers,
            "registered_tools": resolved_tools,
            "allowed_project_ids": resolved_projects,
            "expires_at": resolved_expires,
        }
    )
