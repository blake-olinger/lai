"""Runtime dependency assembly and graceful lifecycle teardown for the FastAPI app."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lai.api.deps import AppServices
from lai.config.profiles import get_model_profile
from lai.db import (
    SqliteApiKeyRepository,
    SqliteDatabase,
    SqliteFileRepository,
    SqliteProjectRepository,
    SqliteSessionRepository,
    SqliteToolAuditRepository,
)
from lai.domain.auth import ApiScope
from lai.domain.errors import LaiError
from lai.guardrails import (
    AllowAllGuardrailProvider,
    CompositeGuardrailPipeline,
    OllamaShieldGemmaProvider,
    ShieldGemmaCpuProvider,
)
from lai.inference import LlamaCppProvider, OpenAICompatProvider
from lai.ingestion import (
    FastEmbedCpuAdapter,
    LanceDbVectorStore,
    LocalBlobStore,
    LocalIngestionPipeline,
)
from lai.observability import PromptAuditLogger, configure_observability, get_component_logger
from lai.orchestrator.service import TurnOrchestrator
from lai.scheduler import InferenceSpooler, SessionGate
from lai.security import (
    ApiKeyAuthProvider,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
    TokenBucketRateLimiter,
)
from lai.tools import GuardedToolExecutor, build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from lai.config.settings import LaiSettings
    from lai.domain.inference import BackendHealth, ModelProfile
    from lai.domain.observability import LoggingConfig
    from lai.ports.auth import AuthProviderPort
    from lai.ports.guardrails import GuardrailProviderPort
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.orchestrator import OrchestratorPort
    from lai.ports.repositories import ApiKeyRepositoryPort

_logger = get_component_logger("api.runtime")


def _resolve_logging_config(settings: LaiSettings) -> LoggingConfig:
    """Resolve effective LoggingConfig honoring data_dir and top-level log flags."""
    cfg = settings.logging
    updates: dict[str, object] = {}
    if cfg.log_dir == Path("data/logs") and settings.data_dir != Path("data"):
        updates["log_dir"] = settings.data_dir / "logs"
    if cfg.level == "INFO" and settings.log_level != "INFO":
        updates["level"] = settings.log_level
    if not cfg.json_console and settings.log_json:
        updates["json_console"] = True
    return cfg.model_copy(update=updates) if updates else cfg


async def _build_authenticator(
    settings: LaiSettings,
    api_key_repo: ApiKeyRepositoryPort,
) -> CompositeAuthenticator:
    """Assemble CompositeAuthenticator from configured providers and edge verifier."""
    providers: list[AuthProviderPort] = []
    if "api_key" in settings.auth_providers:
        providers.append(ApiKeyAuthProvider(api_key_repo))

    if settings.oidc_enabled or "oidc" in settings.auth_providers:
        oidc_cfg = OidcProviderConfig(
            enabled=True,
            issuer_url=settings.oidc_issuer_url,
            jwks_url=settings.oidc_jwks_url,
            audience=settings.oidc_audience,
            allowed_emails=settings.oidc_allowed_emails,
            allowed_domains=settings.oidc_allowed_domains,
            default_scopes=tuple(ApiScope(scope) for scope in settings.oidc_default_scopes),
            admin_emails=settings.oidc_admin_emails,
        )
        providers.append(OidcJwtAuthProvider(oidc_cfg))

    edge_verifier: CloudflareEdgeVerifier | None = None
    if settings.cf_access_enabled:
        cf_cfg = CloudflareEdgeConfig(
            enabled=True,
            client_id=settings.cf_access_client_id,
            client_secret=settings.cf_access_client_secret,
            team_domain=settings.cf_access_team_domain,
            audience=settings.cf_access_aud,
        )
        edge_verifier = CloudflareEdgeVerifier(cf_cfg)
        if cf_cfg.team_domain:
            try:
                await edge_verifier.refresh_jwks()
            except LaiError as exc:
                _logger.warning(
                    "cloudflare_jwks_startup_refresh_failed",
                    error=exc.message,
                )

    return CompositeAuthenticator(providers=providers, edge_verifier=edge_verifier)


def _build_inference_provider(settings: LaiSettings) -> InferenceProviderPort:
    """Construct the active inference provider from LaiSettings."""
    if settings.inference_backend == "ollama":
        return OpenAICompatProvider(settings.ollama_base_url, timeout_seconds=300.0)
    return LlamaCppProvider(host=settings.llama_host, port=settings.llama_port)


def _build_guardrail_providers(
    settings: LaiSettings,
) -> tuple[list[GuardrailProviderPort], Callable[[], Awaitable[BackendHealth]] | None]:
    """Construct configured guardrail provider(s) and optional health probe."""
    if not settings.guardrails.enabled:
        return (
            [
                AllowAllGuardrailProvider(
                    max_logged_prompt_chars=settings.guardrails.max_logged_prompt_chars,
                )
            ],
            None,
        )
    if settings.guardrails.backend == "ollama" or settings.inference_backend == "ollama":
        ollama_cfg = settings.guardrails
        if settings.guardrails.backend != "ollama":
            ollama_cfg = settings.guardrails.model_copy(
                update={"backend": "ollama", "ollama_base_url": settings.ollama_base_url}
            )
        ollama_guard = OllamaShieldGemmaProvider(ollama_cfg)
        return [ollama_guard], ollama_guard.health

    shield_provider = ShieldGemmaCpuProvider(settings.guardrails)
    return [shield_provider], shield_provider.health


def _resolve_model_profile(settings: LaiSettings) -> ModelProfile:
    """Resolve the default ModelProfile from built-in or custom profiles."""
    return get_model_profile(settings.default_model_profile, settings.custom_profiles)


async def build_app_services(
    settings: LaiSettings,
    *,
    orchestrator: OrchestratorPort | None = None,
) -> AppServices:
    """Initialize database, observability, security, ingestion, inference, and orchestrator."""
    logging_cfg = _resolve_logging_config(settings)
    alert_manager = configure_observability(logging_cfg)

    db = SqliteDatabase(settings.data_dir / "lai.db")
    await db.initialize()

    api_key_repo = SqliteApiKeyRepository(db)
    project_repo = SqliteProjectRepository(db)
    session_repo = SqliteSessionRepository(db)
    file_repo = SqliteFileRepository(db)
    tool_audit_repo = SqliteToolAuditRepository(db)

    authenticator = await _build_authenticator(settings, api_key_repo)
    rate_limiter = TokenBucketRateLimiter(requests_per_minute=settings.rate_limit_rpm)
    ip_rate_limiter = TokenBucketRateLimiter(requests_per_minute=settings.ip_rate_limit_rpm)

    blob_store = LocalBlobStore(
        settings.data_dir / "blobs",
        max_upload_bytes=settings.max_upload_bytes,
    )
    embedder = FastEmbedCpuAdapter(cache_dir=settings.models_dir / "fastembed")
    vector_store = LanceDbVectorStore(
        settings.data_dir / "lancedb",
        dimensions=embedder.dimensions,
    )
    ingestion_pipeline = LocalIngestionPipeline(
        blob_store=blob_store,
        file_repo=file_repo,
        embedder=embedder,
        vector_store=vector_store,
    )

    inference_provider = _build_inference_provider(settings)
    spooler = InferenceSpooler(max_queued_jobs=settings.max_queued_jobs)
    session_gate = SessionGate()

    audit_logger = PromptAuditLogger(
        log_dir=logging_cfg.log_dir,
        retention_days=logging_cfg.error_retention_days,
        compress_rotated=logging_cfg.compress_rotated,
    )
    guardrail_providers, guardrail_health_probe = _build_guardrail_providers(settings)

    guardrail_pipeline = CompositeGuardrailPipeline(
        settings.guardrails,
        providers=guardrail_providers,
        audit_logger=audit_logger,
    )
    tool_registry = build_default_tool_registry(
        file_repo,
        blob_store,
        embedder=embedder,
        vector_store=vector_store,
    )
    tool_executor = GuardedToolExecutor(tool_registry, tool_audit_repo)
    model_profile = _resolve_model_profile(settings)

    resolved_orchestrator = orchestrator or TurnOrchestrator(
        settings=settings,
        session_repo=session_repo,
        file_repo=file_repo,
        blob_store=blob_store,
        inference_provider=inference_provider,
        spooler=spooler,
        session_gate=session_gate,
        guardrail_pipeline=guardrail_pipeline,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        embedder=embedder,
        vector_store=vector_store,
        model_profile=model_profile,
    )

    close_callbacks: list[Callable[[], Awaitable[None] | None]] = []
    aclose_fn = getattr(inference_provider, "aclose", None)
    if callable(aclose_fn):
        close_callbacks.append(cast("Callable[[], Awaitable[None]]", aclose_fn))
    close_callbacks.extend(
        [
            audit_logger.close,
            db.close,
            alert_manager.flush,
            alert_manager.close,
        ]
    )

    return AppServices(
        settings=settings,
        alert_manager=alert_manager,
        api_key_repo=api_key_repo,
        project_repo=project_repo,
        session_repo=session_repo,
        file_repo=file_repo,
        tool_audit_repo=tool_audit_repo,
        blob_store=blob_store,
        embedder=embedder,
        vector_store=vector_store,
        ingestion_pipeline=ingestion_pipeline,
        authenticator=authenticator,
        rate_limiter=rate_limiter,
        ip_rate_limiter=ip_rate_limiter,
        inference_provider=inference_provider,
        guardrail_health_probe=guardrail_health_probe,
        spooler=spooler,
        session_gate=session_gate,
        guardrail_pipeline=guardrail_pipeline,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        orchestrator=resolved_orchestrator,
        close_callbacks=close_callbacks,
    )


async def cancel_background_tasks(services: AppServices) -> None:
    """Cancel and await any pending background tasks tracked on AppServices."""
    if not services.background_tasks:
        return
    tasks = list(services.background_tasks)
    for task in tasks:
        if not task.done():
            task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)
    services.background_tasks.clear()


async def close_app_services(services: AppServices) -> None:
    """Cleanly shut down background tasks, HTTP clients, DB, loggers, and alert manager."""
    await cancel_background_tasks(services)
    for callback in services.close_callbacks:
        result = callback()
        if inspect.isawaitable(result):
            await result
