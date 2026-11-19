"""Client application OIDC binding, tier enforcement, and tool merge helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.base import utc_now
from lai.domain.errors import AuthenticationError, AuthorizationError, ValidationFailedError
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.api.deps import AppServices
    from lai.domain.auth import AuthPrincipal
    from lai.domain.client_apps import ClientAppRecord

_logger = get_component_logger("api")


@dataclass(frozen=True, slots=True)
class ResolvedClientContext:
    """Validated client app identity, allowed tiers, and merged CLIENT tool specs."""

    client_app_id: str | None
    client_allowed_tiers: tuple[PermissionTier, ...]
    client_tools: tuple[ToolSpec, ...]


def _check_tool_spec(
    spec: ToolSpec,
    *,
    app: ClientAppRecord,
    server_tool_names: set[str],
) -> ToolSpec:
    """Validate one client tool spec against server names and app tier bounds."""
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
    return spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT})


def _validate_and_coerce_tools(
    *,
    app: ClientAppRecord,
    requested_tools: Sequence[ToolSpec],
    server_tool_names: set[str],
) -> tuple[ToolSpec, ...]:
    merged: dict[str, ToolSpec] = {}
    for spec in app.registered_tools:
        merged[spec.name] = _check_tool_spec(
            spec,
            app=app,
            server_tool_names=server_tool_names,
        )

    seen_requested: set[str] = set()
    for spec in requested_tools:
        if spec.name in seen_requested:
            msg = f"Duplicate client tool name {spec.name!r}."
            raise ValidationFailedError(msg, details={"tool_name": spec.name})
        seen_requested.add(spec.name)
        merged[spec.name] = _check_tool_spec(
            spec,
            app=app,
            server_tool_names=server_tool_names,
        )

    return tuple(merged.values())


async def resolve_turn_client_context(
    services: AppServices,
    *,
    principal: AuthPrincipal,
    project_id: str,
    requested_tools: Sequence[ToolSpec],
) -> ResolvedClientContext:
    """Bind turn client tools fail-closed to a registered active OIDC client app."""
    oauth_cid = (principal.oauth_client_id or "").strip()
    if not oauth_cid:
        if requested_tools:
            _logger.warning(
                "client_tools_rejected_missing_oidc_client",
                principal_id=principal.principal_id,
                project_id=project_id,
            )
            msg = "Client tools require an OIDC token bound to a registered client app."
            raise AuthenticationError(msg, details={"principal_id": principal.principal_id})
        return ResolvedClientContext(None, (), ())

    app = await services.client_app_repo.get_by_oauth_client_id(oauth_cid)
    if app is None or not app.is_active():
        if requested_tools:
            _logger.warning(
                "client_tools_rejected_unregistered_or_inactive_app",
                principal_id=principal.principal_id,
                oauth_client_id=oauth_cid,
                project_id=project_id,
            )
            msg = "OIDC client app is not registered or is inactive."
            raise AuthorizationError(msg, details={"oauth_client_id": oauth_cid})
        return ResolvedClientContext(None, (), ())

    if app.allowed_project_ids and project_id not in app.allowed_project_ids:
        if requested_tools:
            msg = f"Client app {app.id!r} is not authorized for project {project_id!r}."
            raise AuthorizationError(
                msg,
                details={"client_app_id": app.id, "project_id": project_id},
            )
        return ResolvedClientContext(None, (), ())

    await services.client_app_repo.touch_last_used(app.id, utc_now())
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
    principal: AuthPrincipal,
) -> ClientAppRecord:
    """Strictly require an active registered ClientAppRecord matching principal.oauth_client_id."""
    oauth_cid = (principal.oauth_client_id or "").strip()
    if not oauth_cid:
        msg = "Client tool result submission requires an OIDC token bound to a client app."
        raise AuthenticationError(msg, details={"principal_id": principal.principal_id})

    app = await services.client_app_repo.get_by_oauth_client_id(oauth_cid)
    if app is None or not app.is_active():
        msg = "OIDC client app is not registered or is inactive."
        raise AuthorizationError(msg, details={"oauth_client_id": oauth_cid})

    await services.client_app_repo.touch_last_used(app.id, utc_now())
    return app
