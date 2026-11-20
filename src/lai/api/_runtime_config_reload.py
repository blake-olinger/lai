"""Scoped YAML/env section copy helpers and section key maps for runtime config reload."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lai.api._runtime_config_sections import validate_candidate_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from lai.api.schemas.system_config import ReloadSection
    from lai.config.settings import LaiSettings

ALERTING_RELOAD_KEYS: tuple[str, ...] = (
    "alert_webhook_url",
    "alert_webhook_enabled",
    "alert_webhook_min_severity",
    "alert_webhook_categories",
    "alert_smtp",
    "alert_cooldown_seconds",
    "alert_on_timeouts",
    "alert_on_errors",
)
OIDC_RELOAD_KEYS: tuple[str, ...] = (
    "oidc_audience",
    "oidc_audiences",
    "oidc_allowed_client_ids",
    "oidc_allowed_emails",
    "oidc_allowed_domains",
    "oidc_admin_emails",
    "oidc_admin_roles",
    "oidc_admin_subjects",
    "oidc_allow_guest_users",
    "oidc_default_scopes",
    "oidc_require_email_verified",
    "oidc_identity_claims",
    "oidc_leeway_seconds",
)
LIMITS_RELOAD_KEYS: tuple[str, ...] = (
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
    "trusted_proxy_cidrs",
    "proxy_ip_header",
    "log_level",
)
GUARDRAILS_RELOAD_KEYS: tuple[str, ...] = (
    "enabled",
    "block_mode",
    "output_mode",
    "refusal_text",
    "violation_threshold",
    "fail_open",
    "timeout_seconds",
)


def _copy_alerting(data: dict[str, object], fresh: LaiSettings) -> None:
    cur_log = dict(cast("dict[str, object]", data["logging"]))
    fresh_log = fresh.logging.model_dump()
    for key in ALERTING_RELOAD_KEYS:
        cur_log[key] = fresh_log[key]
    data["logging"] = cur_log


def _copy_analytics(data: dict[str, object], fresh: LaiSettings) -> None:
    data["analytics"] = fresh.analytics.model_dump()


def _copy_oidc(data: dict[str, object], fresh: LaiSettings) -> None:
    for key in OIDC_RELOAD_KEYS:
        data[key] = getattr(fresh, key)


def _copy_cors(data: dict[str, object], fresh: LaiSettings) -> None:
    data["cors_allowed_origins"] = fresh.cors_allowed_origins
    data["cors_extra_allow_headers"] = fresh.cors_extra_allow_headers


def _copy_limits(data: dict[str, object], fresh: LaiSettings) -> None:
    for key in LIMITS_RELOAD_KEYS:
        data[key] = getattr(fresh, key)
    cur_log = dict(cast("dict[str, object]", data["logging"]))
    cur_log["level"] = fresh.logging.level if fresh.logging.level != "INFO" else fresh.log_level
    data["logging"] = cur_log


def _copy_updates(data: dict[str, object], fresh: LaiSettings) -> None:
    data["updates"] = fresh.updates.model_dump()


def _copy_guardrails(data: dict[str, object], fresh: LaiSettings) -> None:
    cur_gr = dict(cast("dict[str, object]", data["guardrails"]))
    fresh_gr = fresh.guardrails.model_dump()
    for key in GUARDRAILS_RELOAD_KEYS:
        cur_gr[key] = fresh_gr[key]
    data["guardrails"] = cur_gr


_SECTION_COPIERS: dict[ReloadSection, Callable[[dict[str, object], LaiSettings], None]] = {
    "alerting": _copy_alerting,
    "analytics": _copy_analytics,
    "oidc": _copy_oidc,
    "cors": _copy_cors,
    "limits": _copy_limits,
    "guardrails": _copy_guardrails,
    "updates": _copy_updates,
}


def copy_section_from_fresh_settings(
    current: LaiSettings,
    fresh: LaiSettings,
    section: ReloadSection,
) -> LaiSettings:
    """Overlay only the requested section from freshly loaded YAML/env settings onto current."""
    data = current.model_dump()
    _SECTION_COPIERS[section](data, fresh)
    return validate_candidate_settings(data)
