"""Pure per-section candidate validation, redaction, and non-secret persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from pydantic import ValidationError

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.errors import ValidationFailedError
from lai.domain.guardrails import (
    MIN_LOCKED_TIMEOUT_SECONDS,
    GuardrailConfig,
    validate_guardrail_runtime_update,
)
from lai.domain.observability import AlertCategory
from lai.observability import get_component_logger

if TYPE_CHECKING:
    from lai.api.schemas.system_config import ReloadSection

_logger = get_component_logger("api.runtime_config")
_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

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
    "trusted_proxy_cidrs",
    "proxy_ip_header",
    "log_level",
)
_GUARDRAIL_KEYS = (
    "enabled",
    "block_mode",
    "output_mode",
    "refusal_text",
    "violation_threshold",
    "fail_open",
    "timeout_seconds",
)
_UPDATES_KEYS = (
    "enabled",
    "health_timeout_seconds",
    "health_poll_interval_seconds",
    "keep_releases",
    "auto_rollback",
    "allow_prerelease",
)


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


def enforce_guardrail_boot_lock(
    boot_config: GuardrailConfig,
    candidate: GuardrailConfig,
    *,
    enforce_min_timeout: bool = False,
) -> None:
    """Enforce boot-only guardrail locks, emitting ERROR + SECURITY_ALERT on rejection."""
    rejected: ValidationFailedError | None = None
    try:
        validate_guardrail_runtime_update(boot_config, candidate)
    except ValidationFailedError as exc:
        rejected = exc
    if rejected is not None:
        _logger.error(
            "guardrail_runtime_lock_rejected",
            reason=rejected.message,
            field=rejected.details.get("field", ""),
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        raise rejected
    if enforce_min_timeout and candidate.timeout_seconds < MIN_LOCKED_TIMEOUT_SECONDS:
        msg = f"Guardrail timeout_seconds must be >= {MIN_LOCKED_TIMEOUT_SECONDS}s via runtime API."
        raise ValidationFailedError(msg, details={"field": "timeout_seconds"})


def _extract_alerting_payload(settings: LaiSettings) -> dict[str, object]:
    log_cfg = settings.logging
    return {
        "alert_cooldown_seconds": log_cfg.alert_cooldown_seconds,
        "alert_on_timeouts": log_cfg.alert_on_timeouts,
        "alert_on_errors": log_cfg.alert_on_errors,
        "alert_webhook_enabled": log_cfg.alert_webhook_enabled,
        "alert_webhook_min_severity": log_cfg.alert_webhook_min_severity.value,
        "alert_webhook_categories": [cat.value for cat in log_cfg.alert_webhook_categories],
        "alert_smtp": log_cfg.alert_smtp.model_dump(mode="json", exclude={"password"}),
    }


def _extract_limits_payload(settings: LaiSettings) -> dict[str, object]:
    payload = {key: getattr(settings, key) for key in _LIMITS_KEYS}
    payload["trusted_proxy_cidrs"] = list(settings.trusted_proxy_cidrs)
    return payload


def extract_nonsecret_section_payload(
    settings: LaiSettings,
    section: ReloadSection,
) -> dict[str, object]:
    """Serialize only non-secret settings for one section to persist in SQLite."""
    payloads: dict[ReloadSection, dict[str, object]] = {
        "alerting": _extract_alerting_payload(settings),
        "analytics": {
            "enabled": settings.analytics.enabled,
            "retention_days": settings.analytics.retention_days,
        },
        "oidc": {
            **{
                key: list(getattr(settings, key))
                if isinstance(getattr(settings, key), tuple)
                else getattr(settings, key)
                for key in _OIDC_KEYS
            },
            "oidc_audiences": list(settings.resolved_oidc_audiences()),
        },
        "cors": {
            "cors_allowed_origins": list(settings.cors_allowed_origins),
            "cors_extra_allow_headers": list(settings.cors_extra_allow_headers),
        },
        "limits": _extract_limits_payload(settings),
        "updates": {key: getattr(settings.updates, key) for key in _UPDATES_KEYS},
        "guardrails": {
            **{key: getattr(settings.guardrails, key) for key in _GUARDRAIL_KEYS},
            "block_mode": settings.guardrails.block_mode.value,
            "output_mode": settings.guardrails.output_mode.value,
        },
    }
    return payloads[section]


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


def _overlay_oidc_dict(merged: dict[str, object], payload: Mapping[str, object]) -> None:
    merged.update({key: payload[key] for key in _OIDC_KEYS if key in payload})
    if "oidc_audiences" in payload:
        merged["oidc_audience"] = None


def _overlay_cors_dict(merged: dict[str, object], payload: Mapping[str, object]) -> None:
    for key in ("cors_allowed_origins", "cors_extra_allow_headers"):
        if key in payload:
            merged[key] = payload[key]


def _apply_section_override_dict(
    base_data: dict[str, object],
    section: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Overlay one persisted non-secret section dict onto a serialized LaiSettings dict."""
    merged = dict(base_data)
    nested_specs: dict[str, tuple[str, tuple[str, ...]]] = {
        "analytics": ("analytics", ("enabled", "retention_days")),
        "guardrails": ("guardrails", _GUARDRAIL_KEYS),
        "updates": ("updates", _UPDATES_KEYS),
    }
    direct_handlers = {
        "alerting": _overlay_alerting_dict,
        "oidc": _overlay_oidc_dict,
        "cors": _overlay_cors_dict,
        "limits": _overlay_limits_dict,
    }
    if section in nested_specs:
        container_key, keys = nested_specs[section]
        _overlay_nested_dict(merged, container_key, keys, payload)
    elif section in direct_handlers:
        direct_handlers[section](merged, payload)
    return merged


def clear_smtp_password_if_target_changed(
    smtp_data: dict[str, object],
    *,
    boot_smtp_host: str | None,
    boot_smtp_port: int,
) -> bool:
    """Clear boot SMTP password in memory if host or port differs from the boot target."""
    raw_host, raw_port = smtp_data.get("host"), smtp_data.get("port")
    cand_host = raw_host.strip().lower() if isinstance(raw_host, str) else ""
    boot_host = (boot_smtp_host or "").strip().lower()
    cand_port = raw_port if isinstance(raw_port, int) and not isinstance(raw_port, bool) else 587
    if (cand_host, cand_port) == (boot_host, boot_smtp_port):
        return False
    raw_pw = smtp_data.get("password")
    if isinstance(raw_pw, str) and raw_pw.strip():
        smtp_data["password"] = None
        _logger.warning(
            "smtp_password_cleared_on_target_change",
            boot_host=boot_host,
            boot_port=boot_smtp_port,
            new_host=cand_host,
            new_port=cand_port,
        )
        return True
    return False


def merge_persisted_runtime_overrides(
    base_settings: LaiSettings,
    persisted_sections: Mapping[str, Mapping[str, object]],
    *,
    boot_guardrails_config: GuardrailConfig | None = None,
    boot_smtp_host: str | None = None,
    boot_smtp_port: int | None = None,
) -> LaiSettings:
    """Apply persisted non-secret SQLite section overrides onto YAML/env base_settings at boot."""
    current = base_settings
    boot_guardrails = boot_guardrails_config or base_settings.guardrails
    boot_smtp = base_settings.logging.alert_smtp
    ref_host = boot_smtp.host if boot_smtp_host is None else boot_smtp_host
    ref_port = boot_smtp.port if boot_smtp_port is None else boot_smtp_port
    for section, payload in persisted_sections.items():
        candidate_data = _apply_section_override_dict(current.model_dump(), section, payload)
        if section == "alerting":
            log_d = cast("dict[str, object]", candidate_data["logging"])
            clear_smtp_password_if_target_changed(
                cast("dict[str, object]", log_d["alert_smtp"]),
                boot_smtp_host=ref_host,
                boot_smtp_port=ref_port,
            )
        try:
            validated = validate_candidate_settings(candidate_data)
            if section == "guardrails":
                enforce_guardrail_boot_lock(boot_guardrails, validated.guardrails)
            current = validated
        except ValidationFailedError as exc:
            _logger.warning(
                "persisted_runtime_override_skipped", section=section, error=exc.message
            )
    return current
