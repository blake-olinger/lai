"""Pure per-section candidate validation, redaction, and non-secret persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from pydantic import ValidationError

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.errors import ValidationFailedError
from lai.observability import get_component_logger

if TYPE_CHECKING:
    from lai.api.schemas.system_config import ReloadSection

_logger = get_component_logger("api.runtime_config")
_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def redact_webhook_url(url: str | None) -> str | None:
    """Return scheme://netloc/*** without exposing webhook secret path tokens."""
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/***"
    return "***"


def validate_log_level_name(level: str) -> str:
    """Normalize and validate an explicit Python logging level name."""
    normalized = level.strip().upper()
    if normalized not in _VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(_VALID_LOG_LEVELS))
        msg = f"Invalid log_level {level!r}; expected one of: {allowed}."
        raise ValidationFailedError(msg, details={"field": "log_level"})
    return normalized


def validate_scopes_list(scopes: list[str]) -> tuple[str, ...]:
    """Validate that every configured OIDC default scope is a recognized ApiScope."""
    validated: list[str] = []
    for raw in scopes:
        try:
            scope = ApiScope(raw.strip())
        except ValueError as exc:
            msg = f"Invalid ApiScope value {raw!r} in oidc_default_scopes."
            raise ValidationFailedError(msg, details={"field": "oidc_default_scopes"}) from exc
        validated.append(scope.value)
    return tuple(dict.fromkeys(validated))


def validate_candidate_settings(data: dict[str, object]) -> LaiSettings:
    """Validate a full candidate LaiSettings mapping fail-closed, raising ValidationFailedError."""
    try:
        return LaiSettings.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors()
        summary = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'config'}: {err['msg']}"
            for err in errors
        )
        msg = f"Invalid runtime configuration update: {summary}"
        raise ValidationFailedError(msg, details={"error_count": str(len(errors))}) from exc


def extract_nonsecret_section_payload(
    settings: LaiSettings,
    section: ReloadSection,
) -> dict[str, object]:
    """Serialize only non-secret settings for one section to persist in SQLite."""
    if section == "alerting":
        smtp = settings.logging.alert_smtp
        return {
            "alert_cooldown_seconds": settings.logging.alert_cooldown_seconds,
            "alert_on_timeouts": settings.logging.alert_on_timeouts,
            "alert_on_errors": settings.logging.alert_on_errors,
            "alert_webhook_enabled": settings.logging.alert_webhook_enabled,
            "alert_webhook_min_severity": settings.logging.alert_webhook_min_severity.value,
            "alert_webhook_categories": [
                cat.value for cat in settings.logging.alert_webhook_categories
            ],
            "alert_smtp": {
                "enabled": smtp.enabled,
                "host": smtp.host,
                "port": smtp.port,
                "tls_mode": smtp.tls_mode.value,
                "username": smtp.username,
                "from_address": smtp.from_address,
                "to_addresses": list(smtp.to_addresses),
                "use_admin_emails_fallback": smtp.use_admin_emails_fallback,
                "subject_prefix": smtp.subject_prefix,
                "timeout_seconds": smtp.timeout_seconds,
                "min_severity": smtp.min_severity.value,
                "categories": [cat.value for cat in smtp.categories],
            },
        }
    if section == "analytics":
        return {
            "enabled": settings.analytics.enabled,
            "retention_days": settings.analytics.retention_days,
        }
    if section == "oidc":
        return {
            "oidc_audiences": list(settings.resolved_oidc_audiences()),
            "oidc_allowed_emails": list(settings.oidc_allowed_emails),
            "oidc_allowed_domains": list(settings.oidc_allowed_domains),
            "oidc_admin_emails": list(settings.oidc_admin_emails),
            "oidc_default_scopes": list(settings.oidc_default_scopes),
            "oidc_leeway_seconds": settings.oidc_leeway_seconds,
        }
    if section == "cors":
        return {
            "cors_allowed_origins": list(settings.cors_allowed_origins),
            "cors_extra_allow_headers": list(settings.cors_extra_allow_headers),
        }
    if section == "limits":
        return {
            "rate_limit_rpm": settings.rate_limit_rpm,
            "ip_rate_limit_rpm": settings.ip_rate_limit_rpm,
            "max_projects_per_principal": settings.max_projects_per_principal,
            "max_storage_bytes_per_principal": settings.max_storage_bytes_per_principal,
            "max_queued_jobs": settings.max_queued_jobs,
            "max_queued_jobs_per_principal": settings.max_queued_jobs_per_principal,
            "max_upload_bytes": settings.max_upload_bytes,
            "keepalive_interval_seconds": settings.keepalive_interval_seconds,
            "json_keepalive_threshold_seconds": settings.json_keepalive_threshold_seconds,
            "trust_cloudflare_headers": settings.trust_cloudflare_headers,
            "log_level": settings.log_level,
        }
    return {
        "enabled": settings.guardrails.enabled,
        "block_mode": settings.guardrails.block_mode.value,
        "refusal_text": settings.guardrails.refusal_text,
        "violation_threshold": settings.guardrails.violation_threshold,
        "fail_open": settings.guardrails.fail_open,
        "timeout_seconds": settings.guardrails.timeout_seconds,
    }


_ALERTING_KEYS = (
    "alert_cooldown_seconds",
    "alert_on_timeouts",
    "alert_on_errors",
    "alert_webhook_enabled",
    "alert_webhook_min_severity",
    "alert_webhook_categories",
)
_OIDC_KEYS = (
    "oidc_audiences",
    "oidc_allowed_emails",
    "oidc_allowed_domains",
    "oidc_admin_emails",
    "oidc_default_scopes",
    "oidc_leeway_seconds",
)
_LIMITS_KEYS = (
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
    "log_level",
)
_GUARDRAIL_KEYS = (
    "enabled",
    "block_mode",
    "refusal_text",
    "violation_threshold",
    "fail_open",
    "timeout_seconds",
)


def _overlay_alerting_dict(merged: dict[str, object], payload: Mapping[str, object]) -> None:
    logging_data = dict(cast("dict[str, object]", merged["logging"]))
    smtp_data = dict(cast("dict[str, object]", logging_data["alert_smtp"]))
    logging_data.update({key: payload[key] for key in _ALERTING_KEYS if key in payload})
    raw_smtp = payload.get("alert_smtp")
    if isinstance(raw_smtp, Mapping):
        smtp_override = cast("Mapping[str, object]", raw_smtp)
        smtp_data.update({k: v for k, v in smtp_override.items() if k != "password"})
    logging_data["alert_smtp"] = smtp_data
    merged["logging"] = logging_data


def _overlay_nested_dict(
    merged: dict[str, object],
    container_key: str,
    keys: tuple[str, ...],
    payload: Mapping[str, object],
) -> None:
    nested = dict(cast("dict[str, object]", merged[container_key]))
    nested.update({key: payload[key] for key in keys if key in payload})
    merged[container_key] = nested


def _overlay_limits_dict(merged: dict[str, object], payload: Mapping[str, object]) -> None:
    merged.update({key: payload[key] for key in _LIMITS_KEYS if key in payload})
    if "log_level" in payload:
        logging_data = dict(cast("dict[str, object]", merged["logging"]))
        logging_data["level"] = payload["log_level"]
        merged["logging"] = logging_data


def _apply_section_override_dict(
    base_data: dict[str, object],
    section: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Overlay one persisted non-secret section dict onto a serialized LaiSettings dict."""
    merged = dict(base_data)
    if section == "alerting":
        _overlay_alerting_dict(merged, payload)
    elif section == "analytics":
        _overlay_nested_dict(merged, "analytics", ("enabled", "retention_days"), payload)
    elif section == "oidc":
        merged.update({key: payload[key] for key in _OIDC_KEYS if key in payload})
        if "oidc_audiences" in payload:
            merged["oidc_audience"] = None
    elif section == "cors":
        merged.update(
            {
                key: payload[key]
                for key in ("cors_allowed_origins", "cors_extra_allow_headers")
                if key in payload
            }
        )
    elif section == "limits":
        _overlay_limits_dict(merged, payload)
    elif section == "guardrails":
        _overlay_nested_dict(merged, "guardrails", _GUARDRAIL_KEYS, payload)
    return merged


def merge_persisted_runtime_overrides(
    base_settings: LaiSettings,
    persisted_sections: Mapping[str, Mapping[str, object]],
) -> LaiSettings:
    """Apply persisted non-secret SQLite section overrides onto YAML/env base_settings at boot."""
    current = base_settings
    for section, payload in persisted_sections.items():
        candidate_data = _apply_section_override_dict(current.model_dump(), section, payload)
        try:
            current = validate_candidate_settings(candidate_data)
        except ValidationFailedError as exc:
            _logger.warning(
                "persisted_runtime_override_skipped",
                section=section,
                error=exc.message,
            )
    return current


def copy_section_from_fresh_settings(
    current: LaiSettings,
    fresh: LaiSettings,
    section: ReloadSection,
) -> LaiSettings:
    """Overlay only the requested section from freshly loaded YAML/env settings onto current."""
    data = current.model_dump()
    if section == "alerting":
        cur_log = dict(cast("dict[str, object]", data["logging"]))
        fresh_log = fresh.logging.model_dump()
        for key in (
            "alert_webhook_url",
            "alert_webhook_enabled",
            "alert_webhook_min_severity",
            "alert_webhook_categories",
            "alert_smtp",
            "alert_cooldown_seconds",
            "alert_on_timeouts",
            "alert_on_errors",
        ):
            cur_log[key] = fresh_log[key]
        data["logging"] = cur_log
    elif section == "analytics":
        data["analytics"] = fresh.analytics.model_dump()
    elif section == "oidc":
        for key in (
            "oidc_audience",
            "oidc_audiences",
            "oidc_allowed_emails",
            "oidc_allowed_domains",
            "oidc_admin_emails",
            "oidc_default_scopes",
            "oidc_leeway_seconds",
        ):
            data[key] = getattr(fresh, key)
    elif section == "cors":
        data["cors_allowed_origins"] = fresh.cors_allowed_origins
        data["cors_extra_allow_headers"] = fresh.cors_extra_allow_headers
    elif section == "limits":
        for key in (
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
            "log_level",
        ):
            data[key] = getattr(fresh, key)
        cur_log = dict(cast("dict[str, object]", data["logging"]))
        cur_log["level"] = fresh.logging.level if fresh.logging.level != "INFO" else fresh.log_level
        data["logging"] = cur_log
    else:
        cur_gr = dict(cast("dict[str, object]", data["guardrails"]))
        fresh_gr = fresh.guardrails.model_dump()
        for key in (
            "enabled",
            "block_mode",
            "refusal_text",
            "violation_threshold",
            "fail_open",
            "timeout_seconds",
        ):
            cur_gr[key] = fresh_gr[key]
        data["guardrails"] = cur_gr
    return validate_candidate_settings(data)
