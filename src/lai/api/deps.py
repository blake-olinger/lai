"""FastAPI application service container, credential extraction, and auth dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Request

from lai.domain.auth import ApiScope, AuthPrincipal, RequestCredentials
from lai.domain.errors import AuthorizationError, LaiError
from lai.inference import GpuTelemetry, query_gpu_telemetry
from lai.observability.logging import get_component_logger
from lai.security import TokenBucketRateLimiter, has_effective_scope, require_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from lai.config.settings import LaiSettings
    from lai.domain.inference import BackendHealth
    from lai.ports.auth import AuthenticatorPort
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.observability import AlertManagerPort
    from lai.ports.orchestrator import OrchestratorPort
    from lai.ports.repositories import (
        ApiKeyRepositoryPort,
        FileRepositoryPort,
        ProjectRepositoryPort,
        SessionRepositoryPort,
        ToolAuditRepositoryPort,
    )
    from lai.ports.scheduler import InferenceSpoolerPort, SessionGatePort
    from lai.ports.storage import (
        BlobStorePort,
        EmbedderPort,
        IngestionPipelinePort,
        VectorStorePort,
    )
    from lai.ports.tools import ToolExecutorPort, ToolRegistryPort

_logger = get_component_logger("api")


@dataclass(slots=True)
class AppServices:
    """Resolved runtime services attached to FastAPI app.state.services."""

    settings: LaiSettings
    alert_manager: AlertManagerPort
    api_key_repo: ApiKeyRepositoryPort
    project_repo: ProjectRepositoryPort
    session_repo: SessionRepositoryPort
    file_repo: FileRepositoryPort
    tool_audit_repo: ToolAuditRepositoryPort
    blob_store: BlobStorePort
    embedder: EmbedderPort | None
    vector_store: VectorStorePort | None
    ingestion_pipeline: IngestionPipelinePort
    authenticator: AuthenticatorPort
    rate_limiter: TokenBucketRateLimiter
    inference_provider: InferenceProviderPort
    guardrail_health_probe: Callable[[], Awaitable[BackendHealth]] | None
    spooler: InferenceSpoolerPort
    session_gate: SessionGatePort
    guardrail_pipeline: GuardrailPipelinePort
    tool_registry: ToolRegistryPort
    tool_executor: ToolExecutorPort
    orchestrator: OrchestratorPort
    gpu_telemetry_fn: Callable[[], GpuTelemetry | None] = query_gpu_telemetry
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    close_callbacks: list[Callable[[], Awaitable[None] | None]] = field(default_factory=list)

    def schedule_background_task(
        self,
        coro: Coroutine[Any, Any, object],
        *,
        name: str | None = None,
    ) -> asyncio.Task[object]:
        """Track a fire-and-forget background task so it is not GC'd and closes cleanly."""
        task: asyncio.Task[object] = asyncio.create_task(coro, name=name)
        self.background_tasks.add(task)

        def _on_done(finished: asyncio.Task[object]) -> None:
            self.background_tasks.discard(finished)
            if finished.cancelled():
                return
            exc = finished.exception()
            if exc is not None:
                _logger.error(
                    "background_task_failed",
                    task_name=finished.get_name(),
                    exception_type=type(exc).__name__,
                    error=str(exc),
                )

        task.add_done_callback(_on_done)
        return task


def get_services(request: Request) -> AppServices:
    """Retrieve the AppServices container attached to the FastAPI application."""
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, AppServices):
        msg = "Application services have not been initialized."
        raise LaiError(msg, code="service_unavailable", status_code=503)
    return services


def extract_request_credentials(request: Request) -> RequestCredentials:
    """Extract bearer token, scheme, headers, and client host into RequestCredentials."""
    auth_header = request.headers.get("authorization")
    scheme: str | None = None
    token: str | None = None
    if auth_header:
        stripped = auth_header.strip()
        part_scheme, sep, part_token = stripped.partition(" ")
        if sep:
            scheme = part_scheme or None
            token = part_token.strip() or None
        else:
            scheme = part_scheme or None

    return RequestCredentials(
        authorization_scheme=scheme,
        bearer_token=token,
        headers=dict(request.headers),
        client_host=request.client.host if request.client is not None else None,
    )


async def get_current_principal(
    request: Request,
    services: Annotated[AppServices, Depends(get_services)],
) -> AuthPrincipal:
    """Authenticate caller credentials and enforce per-principal rate limiting."""
    credentials = extract_request_credentials(request)
    principal = await services.authenticator.authenticate(credentials)
    await services.rate_limiter.acquire(principal.principal_id)
    request.state.principal = principal
    return principal


def require_scope_dep(scope: ApiScope) -> Callable[[AuthPrincipal], AuthPrincipal]:
    """Return a FastAPI dependency requiring a single effective ApiScope."""

    def _enforce(
        principal: Annotated[AuthPrincipal, Depends(get_current_principal)],
    ) -> AuthPrincipal:
        require_scope(principal, scope)
        return principal

    return _enforce


def require_any_scope_dep(*scopes: ApiScope) -> Callable[[AuthPrincipal], AuthPrincipal]:
    """Return a FastAPI dependency requiring at least one of the listed ApiScopes."""

    def _enforce(
        principal: Annotated[AuthPrincipal, Depends(get_current_principal)],
    ) -> AuthPrincipal:
        if any(has_effective_scope(principal, scope) for scope in scopes):
            return principal
        expected = ", ".join(scope.value for scope in scopes)
        msg = f"Principal {principal.principal_id!r} lacks one of required scopes: {expected}."
        raise AuthorizationError(
            msg,
            details={"principal_id": principal.principal_id, "required_scopes": expected},
        )

    return _enforce
