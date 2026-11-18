"""Client application authentication, tier enforcement, and tool merge helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.base import utc_now
from lai.domain.errors import AuthenticationError, AuthorizationError, ValidationFailedError
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec
from lai.security import verify_client_app_secret

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.api.deps import AppServices
    from lai.domain.client_apps import ClientAppRecord


@dataclass(frozen=True, slots=True)
class ResolvedClientContext:
    """Validated client app identity, allowed tiers, and merged CLIENT tool specs."""

    client_app_id: str | None
    client_allowed_tiers: tuple[PermissionTier, ...]
    client_tools: tuple[ToolSpec, ...]


def _validate_and_coerce_tools(
    *,
    app: ClientAppRecord,
    requested_tools: Sequence[ToolSpec],
    server_tool_names: set[str],
) -> tuple[ToolSpec, ...]:
    merged: dict[str, ToolSpec] = {}
    for spec in app.registered_tools:
        if spec.name in server_tool_names:
            msg = f"Client tool {spec.name!r} cannot shadow built-in server tool."
            raise ValidationFailedError(msg, details={"tool_name": spec.name})
        if spec.permission_tier not in app.allowed_tiers:
            msg = (
                f"Client tool {spec.name!r} requires tier {spec.permission_tier.value!r}, "
                f"which exceeds allowed tiers for app {app.id!r}."
            )
            raise AuthorizationError(
                msg,
                details={
                    "tool_name": spec.name,
                    "permission_tier": spec.permission_tier.value,
                    "client_app_id": app.id,
                },
            )
        merged[spec.name] = spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT})

    seen_requested: set[str] = set()
    for spec in requested_tools:
        if spec.name in server_tool_names:
            msg = f"Client tool {spec.name!r} cannot shadow built-in server tool."
            raise ValidationFailedError(msg, details={"tool_name": spec.name})
        if spec.name in seen_requested:
            msg = f"Duplicate client tool name {spec.name!r}."
            raise ValidationFailedError(msg, details={"tool_name": spec.name})
        seen_requested.add(spec.name)
        if spec.permission_tier not in app.allowed_tiers:
            msg = (
                f"Client tool {spec.name!r} requires tier {spec.permission_tier.value!r}, "
                f"which exceeds allowed tiers for app {app.id!r}."
            )
            raise AuthorizationError(
                msg,
                details={
                    "tool_name": spec.name,
                    "permission_tier": spec.permission_tier.value,
                    "client_app_id": app.id,
                },
            )
        merged[spec.name] = spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT})

    return tuple(merged.values())


async def resolve_turn_client_context(
    services: AppServices,
    *,
    project_id: str,
    requested_tools: Sequence[ToolSpec],
    client_id: str | None,
    client_secret: str | None,
) -> ResolvedClientContext:
    """Resolve optional client headers; strip client tools if missing/invalid/inactive."""
    cid = (client_id or "").strip()
    secret = (client_secret or "").strip()
    if not cid or not secret:
        return ResolvedClientContext(None, (), ())

    app = await services.client_app_repo.get_by_id(cid)
    if app is None or not verify_client_app_secret(secret, app.secret_hash) or not app.is_active():
        return ResolvedClientContext(None, (), ())

    await services.client_app_repo.touch_last_used(app.id, utc_now())

    if app.allowed_project_ids and project_id not in app.allowed_project_ids:
        msg = f"Client app {app.id!r} is not authorized for project {project_id!r}."
        raise AuthorizationError(msg, details={"client_app_id": app.id, "project_id": project_id})

    server_names = {
        spec.name
        for spec in services.tool_registry.list_specs()
        if spec.execution_target is ToolExecutionTarget.SERVER_ATTACHMENT
    }
    merged_tools = _validate_and_coerce_tools(
        app=app,
        requested_tools=requested_tools,
        server_tool_names=server_names,
    )
    return ResolvedClientContext(
        client_app_id=app.id,
        client_allowed_tiers=app.allowed_tiers,
        client_tools=merged_tools,
    )


async def require_authenticated_client_app(
    services: AppServices,
    client_id: str | None,
    client_secret: str | None,
) -> ClientAppRecord:
    """Strictly authenticate X-LAI-Client-Id and X-LAI-Client-Secret, raising 401 on failure."""
    cid = (client_id or "").strip()
    secret = (client_secret or "").strip()
    if not cid or not secret:
        msg = "Missing X-LAI-Client-Id or X-LAI-Client-Secret header."
        raise AuthenticationError(msg)

    app = await services.client_app_repo.get_by_id(cid)
    if app is None or not verify_client_app_secret(secret, app.secret_hash):
        msg = "Invalid client app credentials."
        raise AuthenticationError(msg, details={"client_app_id": cid})
    if not app.is_active():
        msg = "Client app is revoked or expired."
        raise AuthenticationError(msg, details={"client_app_id": cid})

    await services.client_app_repo.touch_last_used(app.id, utc_now())
    return app
