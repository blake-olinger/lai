"""Isolated per-subsystem runtime hot-swap coordinator and scoped YAML reloader."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, cast

from lai.api._runtime_config_apply import (
    apply_live_alerting,
    apply_live_analytics,
    apply_live_cors,
    apply_live_guardrails,
    apply_live_limits,
    apply_live_oidc,
    apply_live_updates,
    build_alerting_response,
    build_cors_response,
    build_guardrails_response,
    build_limits_response,
    build_oidc_response,
    build_updates_response,
)
from lai.api._runtime_config_reload import copy_section_from_fresh_settings
from lai.api._runtime_config_sections import (
    enforce_guardrail_boot_lock,
    extract_nonsecret_section_payload,
    validate_candidate_settings,
    validate_log_level_name,
    validate_scopes_list,
)
from lai.api.schemas.system_config import (
    ALL_RELOAD_SECTIONS,
    AlertingConfigResponse,
    AnalyticsConfigResponse,
    CorsRuntimeConfigResponse,
    GuardrailsPolicyConfigResponse,
    LimitsRuntimeConfigResponse,
    OidcRuntimeConfigResponse,
    ReloadSection,
    ScopedConfigReloadResponse,
    UpdateAlertingConfigRequest,
    UpdateAnalyticsConfigRequest,
    UpdateCorsConfigRequest,
    UpdateGuardrailsPolicyRequest,
    UpdateLimitsConfigRequest,
    UpdateOidcConfigRequest,
    UpdatesRuntimeConfigResponse,
    UpdateUpdatesConfigRequest,
)
from lai.config.settings import load_settings
from lai.domain.base import utc_now
from lai.observability import get_component_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.api.deps import AppServices

__all__ = [
    "build_alerting_response",
    "build_cors_response",
    "build_guardrails_response",
    "build_limits_response",
    "build_oidc_response",
    "build_updates_response",
    "reload_config_sections",
    "update_alerting_config",
    "update_analytics_config",
    "update_cors_config",
    "update_guardrails_config",
    "update_limits_config",
    "update_oidc_config",
    "update_updates_config",
]

_logger = get_component_logger("api.runtime_config")


async def _persist_section(services: AppServices, section: ReloadSection) -> None:
    if services.runtime_config_repo is None:
        return
    payload = extract_nonsecret_section_payload(services.settings, section)
    await services.runtime_config_repo.upsert_section(section, payload)


async def update_alerting_config(
    services: AppServices,
    body: UpdateAlertingConfigRequest,
) -> AlertingConfigResponse:
    """Validate, live-swap, and persist non-secret alerting settings."""
    data = services.settings.model_dump()
    log_data = dict(cast("dict[str, object]", data["logging"]))
    smtp_data = dict(cast("dict[str, object]", log_data["alert_smtp"]))
    fields = body.model_fields_set

    log_map: dict[str, object] = {
        "alert_cooldown_seconds": body.alert_cooldown_seconds,
        "alert_on_timeouts": body.alert_on_timeouts,
        "alert_on_errors": body.alert_on_errors,
        "alert_webhook_enabled": body.alert_webhook_enabled,
        "alert_webhook_min_severity": body.alert_webhook_min_severity,
        "alert_webhook_categories": body.alert_webhook_categories,
    }
    log_data.update({key: val for key, val in log_map.items() if key in fields and val is not None})

    smtp_map: dict[str, tuple[str, object]] = {
        "smtp_enabled": ("enabled", body.smtp_enabled),
        "smtp_host": ("host", body.smtp_host),
        "smtp_port": ("port", body.smtp_port),
        "smtp_tls_mode": ("tls_mode", body.smtp_tls_mode),
        "smtp_username": ("username", body.smtp_username),
        "smtp_from_address": ("from_address", body.smtp_from_address),
        "smtp_to_addresses": ("to_addresses", body.smtp_to_addresses),
        "smtp_use_admin_emails_fallback": (
            "use_admin_emails_fallback",
            body.smtp_use_admin_emails_fallback,
        ),
        "smtp_subject_prefix": ("subject_prefix", body.smtp_subject_prefix),
        "smtp_timeout_seconds": ("timeout_seconds", body.smtp_timeout_seconds),
        "smtp_min_severity": ("min_severity", body.smtp_min_severity),
        "smtp_categories": ("categories", body.smtp_categories),
    }
    for req_field, (smtp_key, val) in smtp_map.items():
        if req_field in fields:
            smtp_data[smtp_key] = val

    log_data["alert_smtp"] = smtp_data
    data["logging"] = log_data
    services.settings = validate_candidate_settings(data)
    apply_live_alerting(services)
    await _persist_section(services, "alerting")
    _logger.info("runtime_config_updated", section="alerting")
    return build_alerting_response(services.settings)


async def update_analytics_config(
    services: AppServices,
    body: UpdateAnalyticsConfigRequest,
) -> AnalyticsConfigResponse:
    """Validate, live-swap, prune expired buckets, and persist analytics settings."""
    data = services.settings.model_dump()
    analytics_data = dict(cast("dict[str, object]", data["analytics"]))
    if body.enabled is not None:
        analytics_data["enabled"] = body.enabled
    if body.retention_days is not None:
        analytics_data["retention_days"] = body.retention_days
    data["analytics"] = analytics_data
    services.settings = validate_candidate_settings(data)
    apply_live_analytics(services)
    if services.analytics_repo is not None and body.retention_days is not None:
        cutoff = utc_now() - timedelta(days=services.settings.analytics.retention_days)
        await services.analytics_repo.prune_before(cutoff)
    await _persist_section(services, "analytics")
    _logger.info("runtime_config_updated", section="analytics")
    return AnalyticsConfigResponse(
        enabled=services.settings.analytics.enabled,
        retention_days=services.settings.analytics.retention_days,
    )


async def update_oidc_config(
    services: AppServices,
    body: UpdateOidcConfigRequest,
) -> OidcRuntimeConfigResponse:
    """Validate, live-swap OIDC policy preserving cached JWKS, and persist."""
    data = services.settings.model_dump()
    if body.oidc_audiences is not None:
        data["oidc_audiences"] = list(body.oidc_audiences)
        data["oidc_audience"] = None
    if body.oidc_allowed_emails is not None:
        data["oidc_allowed_emails"] = list(body.oidc_allowed_emails)
    if body.oidc_allowed_domains is not None:
        data["oidc_allowed_domains"] = list(body.oidc_allowed_domains)
    if body.oidc_admin_emails is not None:
        data["oidc_admin_emails"] = list(body.oidc_admin_emails)
    if body.oidc_default_scopes is not None:
        data["oidc_default_scopes"] = list(validate_scopes_list(body.oidc_default_scopes))
    if body.oidc_require_email_verified is not None:
        data["oidc_require_email_verified"] = body.oidc_require_email_verified
    if body.oidc_identity_claims is not None:
        data["oidc_identity_claims"] = list(body.oidc_identity_claims)
    if body.oidc_leeway_seconds is not None:
        data["oidc_leeway_seconds"] = body.oidc_leeway_seconds
    services.settings = validate_candidate_settings(data)
    apply_live_oidc(services)
    await _persist_section(services, "oidc")
    _logger.info("runtime_config_updated", section="oidc")
    return build_oidc_response(services.settings)


async def update_cors_config(
    services: AppServices,
    body: UpdateCorsConfigRequest,
) -> CorsRuntimeConfigResponse:
    """Validate, live-swap dynamic CORS middleware, and persist."""
    data = services.settings.model_dump()
    if body.cors_allowed_origins is not None:
        data["cors_allowed_origins"] = list(body.cors_allowed_origins)
    if body.cors_extra_allow_headers is not None:
        data["cors_extra_allow_headers"] = list(body.cors_extra_allow_headers)
    services.settings = validate_candidate_settings(data)
    apply_live_cors(services)
    await _persist_section(services, "cors")
    _logger.info("runtime_config_updated", section="cors")
    return build_cors_response(services.settings)


async def update_limits_config(
    services: AppServices,
    body: UpdateLimitsConfigRequest,
) -> LimitsRuntimeConfigResponse:
    """Validate, live-swap limits/quotas preserving active buckets/queues, and persist."""
    data = services.settings.model_dump()
    for field_name in (
        "rate_limit_rpm",
        "ip_rate_limit_rpm",
        "max_projects_per_principal",
        "max_storage_bytes_per_principal",
        "max_queued_jobs",
        "max_queued_jobs_per_principal",
        "max_upload_bytes",
        "keepalive_interval_seconds",
        "json_keepalive_threshold_seconds",
        "trust_cloudflare_headers",
        "proxy_ip_header",
    ):
        val = getattr(body, field_name)
        if val is not None:
            data[field_name] = val
    if body.trusted_proxy_cidrs is not None:
        data["trusted_proxy_cidrs"] = list(body.trusted_proxy_cidrs)
    if body.log_level is not None:
        normalized_level = validate_log_level_name(body.log_level)
        data["log_level"] = normalized_level
        log_data = dict(cast("dict[str, object]", data["logging"]))
        log_data["level"] = normalized_level
        data["logging"] = log_data
    services.settings = validate_candidate_settings(data)
    apply_live_limits(services)
    await _persist_section(services, "limits")
    _logger.info("runtime_config_updated", section="limits")
    return build_limits_response(services.settings)


async def update_guardrails_config(
    services: AppServices,
    body: UpdateGuardrailsPolicyRequest,
) -> GuardrailsPolicyConfigResponse:
    """Validate, enforce boot locks, live-swap guardrail policy in place, and persist."""
    data = services.settings.model_dump()
    gr_data = dict(cast("dict[str, object]", data["guardrails"]))
    for field_name in (
        "enabled",
        "block_mode",
        "output_mode",
        "refusal_text",
        "violation_threshold",
        "fail_open",
        "timeout_seconds",
    ):
        val = getattr(body, field_name)
        if val is not None:
            gr_data[field_name] = val
    data["guardrails"] = gr_data
    candidate = validate_candidate_settings(data)
    boot_guardrails = services.boot_guardrails_config or services.settings.guardrails
    enforce_guardrail_boot_lock(boot_guardrails, candidate.guardrails)
    services.settings = candidate
    apply_live_guardrails(services)
    await _persist_section(services, "guardrails")
    _logger.info("runtime_config_updated", section="guardrails")
    return build_guardrails_response(services.settings)


async def update_updates_config(
    services: AppServices,
    body: UpdateUpdatesConfigRequest,
) -> UpdatesRuntimeConfigResponse:
    """Validate, live-swap non-privileged self-update settings, and persist."""
    data = services.settings.model_dump()
    upd_data = dict(cast("dict[str, object]", data["updates"]))
    for field_name in (
        "enabled",
        "health_timeout_seconds",
        "health_poll_interval_seconds",
        "keep_releases",
        "auto_rollback",
        "allow_prerelease",
    ):
        val = getattr(body, field_name)
        if val is not None:
            upd_data[field_name] = val
    data["updates"] = upd_data
    services.settings = validate_candidate_settings(data)
    apply_live_updates(services)
    await _persist_section(services, "updates")
    _logger.info("runtime_config_updated", section="updates")
    return build_updates_response(services.settings)


async def reload_config_sections(
    services: AppServices,
    sections: Sequence[ReloadSection],
) -> ScopedConfigReloadResponse:
    """Reload ONLY requested sections from YAML/env, clearing their SQLite overrides."""
    fresh_settings = load_settings(services.config_path)
    ordered = list(dict.fromkeys(sections))
    candidate = services.settings
    for section in ordered:
        candidate = copy_section_from_fresh_settings(candidate, fresh_settings, section)
    if "limits" in ordered:
        validate_log_level_name(candidate.log_level)
    if "guardrails" in ordered:
        boot_guardrails = services.boot_guardrails_config or services.settings.guardrails
        enforce_guardrail_boot_lock(boot_guardrails, candidate.guardrails)

    services.settings = candidate
    appliers = {
        "alerting": apply_live_alerting,
        "analytics": apply_live_analytics,
        "oidc": apply_live_oidc,
        "cors": apply_live_cors,
        "limits": apply_live_limits,
        "guardrails": apply_live_guardrails,
        "updates": apply_live_updates,
    }
    for section in ordered:
        appliers[section](services)
        if services.runtime_config_repo is not None:
            await services.runtime_config_repo.delete_section(section)

    unchanged: list[ReloadSection] = [sec for sec in ALL_RELOAD_SECTIONS if sec not in ordered]
    _logger.info("runtime_config_reloaded", sections=ordered)
    return ScopedConfigReloadResponse(
        reloaded_sections=ordered,
        unchanged_sections=unchanged,
        config_path=str(services.config_path) if services.config_path is not None else None,
    )
