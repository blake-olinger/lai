"""Live subsystem mutators and response builders for isolated runtime hot-swapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.api._cors import merge_cors_allow_headers
from lai.api._runtime_config_sections import redact_webhook_url
from lai.api.schemas.system_config import (
    AlertingConfigResponse,
    CorsRuntimeConfigResponse,
    GuardrailsPolicyConfigResponse,
    LimitsRuntimeConfigResponse,
    OidcRuntimeConfigResponse,
    UpdatesRuntimeConfigResponse,
)
from lai.db.analytics_repo import SqliteUsageAnalyticsRepository
from lai.domain.auth import ApiScope
from lai.guardrails import (
    AllowAllGuardrailProvider,
    CompositeGuardrailPipeline,
    ShieldGemmaCpuProvider,
)
from lai.ingestion import LocalBlobStore
from lai.observability import (
    AlertManager,
    reconfigure_observability_alerting,
    resolve_smtp_recipients,
    set_runtime_log_level,
)
from lai.orchestrator.service import TurnOrchestrator
from lai.scheduler import InferenceSpooler
from lai.security import CompositeAuthenticator, OidcJwtAuthProvider, OidcProviderConfig

if TYPE_CHECKING:
    from lai.api.deps import AppServices
    from lai.config.settings import LaiSettings
    from lai.ports.guardrails import GuardrailProviderPort


def sync_orchestrator_settings(services: AppServices) -> None:
    """Refresh TurnOrchestrator's settings reference in place."""
    if isinstance(services.orchestrator, TurnOrchestrator):
        services.orchestrator.update_settings(services.settings)


def apply_live_alerting(services: AppServices) -> None:
    """Reconfigure active alert sinks, cooldown, and handler toggles in place."""
    if isinstance(services.alert_manager, AlertManager):
        reconfigure_observability_alerting(
            services.alert_manager,
            services.settings.logging,
            fallback_admin_emails=services.settings.oidc_admin_emails,
        )


def apply_live_analytics(services: AppServices) -> None:
    """Update SQLite analytics retention policy in place."""
    if isinstance(services.analytics_repo, SqliteUsageAnalyticsRepository):
        services.analytics_repo.set_retention_days(services.settings.analytics.retention_days)


def apply_live_oidc(services: AppServices) -> None:
    """Hot-swap OIDC audiences, allowlists, scopes, and leeway preserving JWKS cache."""
    if isinstance(services.authenticator, CompositeAuthenticator):
        provider = services.authenticator.get_provider("oidc")
        if isinstance(provider, OidcJwtAuthProvider):
            s = services.settings
            updated_cfg = OidcProviderConfig(
                enabled=True,
                issuer_url=s.oidc_issuer_url,
                jwks_url=s.oidc_jwks_url,
                audience=s.oidc_audience,
                audiences=s.resolved_oidc_audiences(),
                allowed_client_ids=s.oidc_allowed_client_ids,
                allowed_emails=s.oidc_allowed_emails,
                allowed_domains=s.oidc_allowed_domains,
                default_scopes=tuple(ApiScope(scope) for scope in s.oidc_default_scopes),
                admin_emails=s.oidc_admin_emails,
                admin_roles=s.oidc_admin_roles,
                admin_subjects=s.oidc_admin_subjects,
                allow_guest_users=s.oidc_allow_guest_users,
                require_email_verified=s.oidc_require_email_verified,
                identity_claims=s.oidc_identity_claims,
                leeway_seconds=s.oidc_leeway_seconds,
            )
            provider.reconfigure(updated_cfg)
    smtp = services.settings.logging.alert_smtp
    if smtp.enabled and smtp.use_admin_emails_fallback and not smtp.to_addresses:
        apply_live_alerting(services)


def apply_live_cors(services: AppServices) -> None:
    """Hot-swap dynamic ASGI CORS middleware origins and extra allow headers."""
    if services.cors_controller is not None:
        services.cors_controller.reconfigure(
            services.settings.cors_allowed_origins,
            services.settings.cors_extra_allow_headers,
        )


def apply_live_limits(services: AppServices) -> None:
    """Hot-swap rate limits, queue bounds, upload size, and log level in place."""
    s = services.settings
    services.rate_limiter.reconfigure(requests_per_minute=s.rate_limit_rpm)
    services.ip_rate_limiter.reconfigure(requests_per_minute=s.ip_rate_limit_rpm)
    if isinstance(services.spooler, InferenceSpooler):
        services.spooler.reconfigure(
            max_queued_jobs=s.max_queued_jobs,
            max_queued_jobs_per_principal=s.max_queued_jobs_per_principal,
        )
    if isinstance(services.blob_store, LocalBlobStore):
        services.blob_store.set_max_upload_bytes(s.max_upload_bytes)
    set_runtime_log_level(s.log_level)
    sync_orchestrator_settings(services)


def apply_live_guardrails(services: AppServices) -> None:
    """Hot-swap guardrail enforcement policy without restarting subprocesses."""
    cfg = services.settings.guardrails
    if isinstance(services.guardrail_pipeline, CompositeGuardrailPipeline):
        existing = services.guardrail_pipeline.providers
        only_passthrough = bool(existing) and all(
            isinstance(provider, AllowAllGuardrailProvider) for provider in existing
        )
        if cfg.enabled and only_passthrough:
            shield_guard = ShieldGemmaCpuProvider(cfg)
            new_providers: list[GuardrailProviderPort] = [shield_guard]
            services.guardrail_health_probe = shield_guard.health
            services.guardrail_pipeline.reconfigure(cfg, providers=new_providers)
        else:
            services.guardrail_pipeline.reconfigure(cfg)
    sync_orchestrator_settings(services)


def apply_live_updates(services: AppServices) -> None:
    """Sync live update settings reference onto the orchestrator."""
    sync_orchestrator_settings(services)


def build_alerting_response(settings: LaiSettings) -> AlertingConfigResponse:
    """Build redacted AlertingConfigResponse from current settings."""
    log_cfg = settings.logging
    smtp = log_cfg.alert_smtp
    recipients = resolve_smtp_recipients(smtp, settings.oidc_admin_emails)
    return AlertingConfigResponse(
        alert_cooldown_seconds=log_cfg.alert_cooldown_seconds,
        alert_on_timeouts=log_cfg.alert_on_timeouts,
        alert_on_errors=log_cfg.alert_on_errors,
        alert_webhook_enabled=log_cfg.alert_webhook_enabled,
        alert_webhook_configured=bool(log_cfg.alert_webhook_url),
        alert_webhook_target=redact_webhook_url(log_cfg.alert_webhook_url),
        alert_webhook_min_severity=log_cfg.alert_webhook_min_severity,
        alert_webhook_categories=list(log_cfg.alert_webhook_categories),
        smtp_enabled=smtp.enabled,
        smtp_host=smtp.host,
        smtp_port=smtp.port,
        smtp_tls_mode=smtp.tls_mode,
        smtp_username=smtp.username,
        smtp_password_configured=bool(smtp.password and smtp.password.strip()),
        smtp_from_address=smtp.from_address,
        smtp_to_addresses=list(smtp.to_addresses),
        smtp_use_admin_emails_fallback=smtp.use_admin_emails_fallback,
        effective_smtp_recipients=list(recipients),
        smtp_subject_prefix=smtp.subject_prefix,
        smtp_timeout_seconds=smtp.timeout_seconds,
        smtp_min_severity=smtp.min_severity,
        smtp_categories=list(smtp.categories),
    )


def build_oidc_response(settings: LaiSettings) -> OidcRuntimeConfigResponse:
    """Build OidcRuntimeConfigResponse from current settings."""
    providers = {provider.strip().lower() for provider in settings.auth_providers}
    enabled = settings.oidc_enabled or "oidc" in providers
    return OidcRuntimeConfigResponse(
        enabled=enabled,
        issuer_url=settings.oidc_issuer_url,
        jwks_url=settings.oidc_jwks_url,
        oidc_audiences=list(settings.resolved_oidc_audiences()),
        oidc_allowed_client_ids=list(settings.oidc_allowed_client_ids),
        oidc_allowed_emails=list(settings.oidc_allowed_emails),
        oidc_allowed_domains=list(settings.oidc_allowed_domains),
        oidc_admin_emails=list(settings.oidc_admin_emails),
        oidc_admin_roles=list(settings.oidc_admin_roles),
        oidc_admin_subjects=list(settings.oidc_admin_subjects),
        oidc_allow_guest_users=settings.oidc_allow_guest_users,
        oidc_default_scopes=list(settings.oidc_default_scopes),
        oidc_require_email_verified=settings.oidc_require_email_verified,
        oidc_identity_claims=list(settings.oidc_identity_claims),
        oidc_leeway_seconds=settings.oidc_leeway_seconds,
    )


def build_cors_response(settings: LaiSettings) -> CorsRuntimeConfigResponse:
    """Build CorsRuntimeConfigResponse from current settings."""
    return CorsRuntimeConfigResponse(
        cors_allowed_origins=list(settings.cors_allowed_origins),
        cors_extra_allow_headers=list(settings.cors_extra_allow_headers),
        effective_allow_headers=merge_cors_allow_headers(settings.cors_extra_allow_headers),
    )


def build_limits_response(settings: LaiSettings) -> LimitsRuntimeConfigResponse:
    """Build LimitsRuntimeConfigResponse from current settings."""
    return LimitsRuntimeConfigResponse(
        rate_limit_rpm=settings.rate_limit_rpm,
        ip_rate_limit_rpm=settings.ip_rate_limit_rpm,
        max_projects_per_principal=settings.max_projects_per_principal,
        max_storage_bytes_per_principal=settings.max_storage_bytes_per_principal,
        max_queued_jobs=settings.max_queued_jobs,
        max_queued_jobs_per_principal=settings.max_queued_jobs_per_principal,
        max_upload_bytes=settings.max_upload_bytes,
        keepalive_interval_seconds=settings.keepalive_interval_seconds,
        json_keepalive_threshold_seconds=settings.json_keepalive_threshold_seconds,
        trust_cloudflare_headers=settings.trust_cloudflare_headers,
        trusted_proxy_cidrs=list(settings.trusted_proxy_cidrs),
        proxy_ip_header=settings.proxy_ip_header,
        log_level=settings.log_level,
    )


def build_guardrails_response(settings: LaiSettings) -> GuardrailsPolicyConfigResponse:
    """Build GuardrailsPolicyConfigResponse from current settings."""
    cfg = settings.guardrails
    return GuardrailsPolicyConfigResponse(
        enabled=cfg.enabled,
        block_mode=cfg.block_mode,
        output_mode=cfg.output_mode,
        refusal_text=cfg.refusal_text,
        violation_threshold=cfg.violation_threshold,
        fail_open=cfg.fail_open,
        allow_runtime_disable=cfg.allow_runtime_disable,
        max_violation_threshold=cfg.max_violation_threshold,
        timeout_seconds=cfg.timeout_seconds,
    )


def build_updates_response(settings: LaiSettings) -> UpdatesRuntimeConfigResponse:
    """Build UpdatesRuntimeConfigResponse from current settings."""
    cfg = settings.updates
    return UpdatesRuntimeConfigResponse(
        enabled=cfg.enabled,
        github_repo=cfg.github_repo,
        releases_dir=str(cfg.releases_dir),
        current_symlink=str(cfg.current_symlink),
        backups_dir=str(cfg.backups_dir),
        service_name=cfg.service_name,
        health_url=cfg.health_url,
        health_timeout_seconds=cfg.health_timeout_seconds,
        health_poll_interval_seconds=cfg.health_poll_interval_seconds,
        keep_releases=cfg.keep_releases,
        auto_rollback=cfg.auto_rollback,
        allow_prerelease=cfg.allow_prerelease,
    )
