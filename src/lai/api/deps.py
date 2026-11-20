"""FastAPI application service container, credential extraction, and auth dependencies."""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Request

from lai.domain.auth import ApiScope, AuthPrincipal, RequestCredentials
from lai.domain.errors import AuthorizationError, LaiError
from lai.inference import GpuTelemetry, query_gpu_telemetry
from lai.observability.logging import get_component_logger
from lai.security import TokenBucketRateLimiter, has_effective_scope, require_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Mapping
    from pathlib import Path

    from lai.api._cors import DynamicCorsController
    from lai.config.settings import LaiSettings
    from lai.domain.guardrails import GuardrailConfig
    from lai.domain.inference import BackendHealth
    from lai.ports.analytics import RuntimeConfigRepositoryPort, UsageAnalyticsPort
    from lai.ports.auth import AuthenticatorPort
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.observability import AlertManagerPort
    from lai.ports.orchestrator import OrchestratorPort
    from lai.ports.repositories import (
        ApiKeyRepositoryPort,
        ClientAppRepositoryPort,
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
    client_app_repo: ClientAppRepositoryPort
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
    ip_rate_limiter: TokenBucketRateLimiter = field(
        default_factory=lambda: TokenBucketRateLimiter(requests_per_minute=240)
    )
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    close_callbacks: list[Callable[[], Awaitable[None] | None]] = field(default_factory=list)
    analytics_repo: UsageAnalyticsPort | None = None
    runtime_config_repo: RuntimeConfigRepositoryPort | None = None
    cors_controller: DynamicCorsController | None = None
    config_path: Path | None = None
    boot_guardrails_config: GuardrailConfig | None = None

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


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_IpNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def _get_header_ci(headers: Mapping[str, str], target_lower: str) -> str | None:
    """Return the first non-empty stripped header matching target_lower case-insensitively."""
    for header_name, header_value in headers.items():
        if header_name.lower() == target_lower:
            stripped = header_value.strip()
            if stripped:
                return stripped
    return None


def _parse_ip_address(raw: str) -> _IpAddress | None:
    """Parse an IPv4 or IPv6 address, returning None when malformed."""
    try:
        return ipaddress.ip_address(raw.strip())
    except ValueError:
        return None


def _is_trusted_ip(ip: _IpAddress, networks: tuple[_IpNetwork, ...]) -> bool:
    """Return True if ip belongs to any configured trusted proxy CIDR network."""
    return any(ip.version == net.version and ip in net for net in networks)


def _extract_xff_client_ip(header_value: str, networks: tuple[_IpNetwork, ...]) -> str | None:
    """Return the rightmost untrusted IP hop from X-Forwarded-For, or None if malformed."""
    hops = [part.strip() for part in header_value.split(",") if part.strip()]
    if not hops:
        return None
    parsed_hops: list[_IpAddress] = []
    for hop in hops:
        parsed = _parse_ip_address(hop)
        if parsed is None:
            return None
        parsed_hops.append(parsed)
    for hop_ip in reversed(parsed_hops):
        if not _is_trusted_ip(hop_ip, networks):
            return str(hop_ip)
    return str(parsed_hops[0])


def _resolve_client_ip(
    credentials: RequestCredentials,
    *,
    trust_cf_header: bool,
    trusted_proxy_cidrs: tuple[str, ...] = (),
    proxy_ip_header: str = "x-forwarded-for",
) -> str:
    """Resolve client IP from loopback Cloudflare headers or configured trusted proxy CIDRs."""
    host = (credentials.client_host or "").strip()
    if trust_cf_header and host in _LOOPBACK_HOSTS:
        cf_ip = _get_header_ci(credentials.headers, "cf-connecting-ip")
        if cf_ip is not None:
            return cf_ip

    if trusted_proxy_cidrs and host:
        peer_ip = _parse_ip_address(host)
        if peer_ip is not None:
            networks = tuple(
                ipaddress.ip_network(cidr.strip(), strict=False) for cidr in trusted_proxy_cidrs
            )
            if _is_trusted_ip(peer_ip, networks):
                header_key = proxy_ip_header.strip().lower()
                header_val = _get_header_ci(credentials.headers, header_key)
                if header_val is not None:
                    if header_key == "x-forwarded-for":
                        resolved = _extract_xff_client_ip(header_val, networks)
                        if resolved is not None:
                            return resolved
                    else:
                        parsed_single = _parse_ip_address(header_val)
                        if parsed_single is not None:
                            return str(parsed_single)

    return host or "unknown"


async def get_current_principal(
    request: Request,
    services: Annotated[AppServices, Depends(get_services)],
) -> AuthPrincipal:
    """Enforce pre-auth IP limits, authenticate caller, and enforce per-principal limits."""
    credentials = extract_request_credentials(request)
    trust_cf_header = (
        services.settings.trust_cloudflare_headers or services.settings.cf_access_enabled
    )
    client_ip = _resolve_client_ip(
        credentials,
        trust_cf_header=trust_cf_header,
        trusted_proxy_cidrs=services.settings.trusted_proxy_cidrs,
        proxy_ip_header=services.settings.proxy_ip_header,
    )
    await services.ip_rate_limiter.acquire(f"ip:{client_ip}")
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
