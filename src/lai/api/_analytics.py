"""Non-fatal HTTP request and token usage analytics recording helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lai.api.deps import AppServices
from lai.domain.analytics import TokenUsageRecord, UsageEvent
from lai.domain.auth import AuthPrincipal
from lai.domain.chat import OrchestratorEvent, TokenUsage, TurnCompletedEvent, TurnOutput
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from starlette.types import Scope

_logger = get_component_logger("api.analytics")
_UNMATCHED_ROUTE = "<unmatched>"


def extract_route_template(scope: Scope) -> str:
    """Extract normalized FastAPI/Starlette route template or '<unmatched>' for 404/preflight."""
    route = scope.get("route")
    path_template = getattr(route, "path", None)
    if isinstance(path_template, str) and path_template:
        return path_template
    return _UNMATCHED_ROUTE


def extract_principal_ids(scope: Scope) -> tuple[str | None, str | None]:
    """Extract opaque (principal_id, oauth_client_id) from ASGI scope state if authenticated."""
    raw_state = scope.get("state")
    if not isinstance(raw_state, dict):
        return None, None
    state = cast("dict[str, object]", raw_state)
    principal = state.get("principal")
    if isinstance(principal, AuthPrincipal):
        return principal.principal_id, principal.oauth_client_id
    return None, None


async def record_http_analytics(  # noqa: PLR0913
    scope: Scope,
    *,
    method: str,
    route: str,
    status_code: int | None,
    duration_ms: float,
    principal_id: str | None,
    oauth_client_id: str | None,
) -> None:
    """Record a completed HTTP request into UsageAnalyticsPort without failing requests."""
    if status_code is None:
        return
    app = scope.get("app")
    app_state = getattr(app, "state", None)
    services = getattr(app_state, "services", None)
    if not isinstance(services, AppServices):
        return
    if services.analytics_repo is None or not services.settings.analytics.enabled:
        return

    event = UsageEvent(
        method=method,
        route=route,
        status_code=status_code,
        duration_ms=max(0.0, duration_ms),
        principal_id=principal_id,
        oauth_client_id=oauth_client_id,
    )
    try:
        await services.analytics_repo.record_request(event)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "analytics_request_record_failed",
            route=route,
            error=str(exc),
        )


async def record_turn_tokens(
    services: AppServices,
    principal: AuthPrincipal,
    usage: TokenUsage,
) -> None:
    """Record completed turn token usage into UsageAnalyticsPort without failing turns."""
    if services.analytics_repo is None or not services.settings.analytics.enabled:
        return
    total = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
    if usage.prompt_tokens == 0 and usage.completion_tokens == 0 and total == 0:
        return

    record = TokenUsageRecord(
        principal_id=principal.principal_id,
        oauth_client_id=principal.oauth_client_id,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=total,
    )
    try:
        await services.analytics_repo.record_tokens(record)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "analytics_token_record_failed",
            principal_id=principal.principal_id,
            error=str(exc),
        )


async def track_turn_output(
    coro: Awaitable[TurnOutput],
    services: AppServices,
    principal: AuthPrincipal,
) -> TurnOutput:
    """Await a non-streaming turn and record its token usage into analytics."""
    output = await coro
    await record_turn_tokens(services, principal, output.usage)
    return output


async def track_turn_stream(
    events: AsyncIterator[OrchestratorEvent],
    services: AppServices,
    principal: AuthPrincipal,
) -> AsyncIterator[OrchestratorEvent]:
    """Yield orchestrator stream events while recording token usage on TurnCompletedEvent."""
    async for event in events:
        if isinstance(event, TurnCompletedEvent):
            await record_turn_tokens(services, principal, event.output.usage)
        yield event
