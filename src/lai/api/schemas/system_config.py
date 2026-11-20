"""API schema DTOs for live alerting, analytics, and isolated runtime config hot-swapping."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.guardrails import GuardrailBlockMode, GuardrailOutputMode
from lai.domain.observability import (
    AlertCategory,
    AlertSeverity,
    AlertSinkDeliveryResult,
    SmtpTlsMode,
)

ProxyIpHeader = Literal["cf-connecting-ip", "x-forwarded-for", "x-azure-clientip"]
ReloadSection = Literal["alerting", "analytics", "oidc", "cors", "limits", "guardrails", "updates"]
ALL_RELOAD_SECTIONS: tuple[ReloadSection, ...] = (
    "alerting",
    "analytics",
    "oidc",
    "cors",
    "limits",
    "guardrails",
    "updates",
)


class AlertingConfigResponse(StrictModel):
    """Redacted live operational alerting configuration (never returns secrets)."""

    alert_cooldown_seconds: float = Field(ge=0.0)
    alert_on_timeouts: bool
    alert_on_errors: bool
    alert_webhook_enabled: bool
    alert_webhook_configured: bool
    alert_webhook_target: str | None = None
    alert_webhook_min_severity: AlertSeverity
    alert_webhook_categories: list[AlertCategory] = Field(default_factory=list)
    smtp_enabled: bool
    smtp_host: str | None = None
    smtp_port: int = Field(ge=1, le=65535)
    smtp_tls_mode: SmtpTlsMode
    smtp_username: str | None = None
    smtp_password_configured: bool
    smtp_from_address: str | None = None
    smtp_to_addresses: list[str] = Field(default_factory=list)
    smtp_use_admin_emails_fallback: bool
    effective_smtp_recipients: list[str] = Field(default_factory=list)
    smtp_subject_prefix: str
    smtp_timeout_seconds: float = Field(gt=0.0, le=60.0)
    smtp_min_severity: AlertSeverity
    smtp_categories: list[AlertCategory] = Field(default_factory=list)


class UpdateAlertingConfigRequest(StrictModel):
    """Partial update for live non-secret alerting settings (secrets stay YAML/env-only)."""

    alert_cooldown_seconds: float | None = Field(default=None, ge=0.0)
    alert_on_timeouts: bool | None = None
    alert_on_errors: bool | None = None
    alert_webhook_enabled: bool | None = None
    alert_webhook_min_severity: AlertSeverity | None = None
    alert_webhook_categories: list[AlertCategory] | None = None
    smtp_enabled: bool | None = None
    smtp_host: str | None = None
    smtp_port: int | None = Field(default=None, ge=1, le=65535)
    smtp_tls_mode: SmtpTlsMode | None = None
    smtp_username: str | None = None
    smtp_from_address: str | None = None
    smtp_to_addresses: list[str] | None = None
    smtp_use_admin_emails_fallback: bool | None = None
    smtp_subject_prefix: str | None = Field(default=None, min_length=1, max_length=80)
    smtp_timeout_seconds: float | None = Field(default=None, gt=0.0, le=60.0)
    smtp_min_severity: AlertSeverity | None = None
    smtp_categories: list[AlertCategory] | None = None


class TestAlertRequest(StrictModel):
    """Payload for dispatching a synthetic test alert through active alert sinks."""

    message: str = Field(
        default="Synthetic pre-flight test alert from LAI.",
        min_length=1,
        max_length=500,
    )
    severity: AlertSeverity = AlertSeverity.CRITICAL
    category: AlertCategory = AlertCategory.APPLICATION_ERROR
    bypass_filters: bool = True


class TestAlertResponse(StrictModel):
    """Per-sink delivery results from POST /v1/system/alerting/test."""

    alert_id: str = Field(min_length=1)
    delivered_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    results: list[AlertSinkDeliveryResult] = Field(default_factory=list)


class AnalyticsConfigResponse(StrictModel):
    """Live configuration for built-in endpoint usage analytics."""

    enabled: bool
    retention_days: int = Field(ge=1, le=3650)


class UpdateAnalyticsConfigRequest(StrictModel):
    """Partial live update for endpoint usage analytics collection and retention."""

    enabled: bool | None = None
    retention_days: int | None = Field(default=None, ge=1, le=3650)


class OidcRuntimeConfigResponse(StrictModel):
    """Live OIDC authorization policy settings."""

    enabled: bool
    issuer_url: str
    jwks_url: str
    oidc_audiences: list[str] = Field(default_factory=list)
    oidc_allowed_emails: list[str] = Field(default_factory=list)
    oidc_allowed_domains: list[str] = Field(default_factory=list)
    oidc_admin_emails: list[str] = Field(default_factory=list)
    oidc_default_scopes: list[str] = Field(default_factory=list)
    oidc_require_email_verified: bool
    oidc_identity_claims: list[str] = Field(default_factory=list)
    oidc_leeway_seconds: float = Field(ge=0.0, le=300.0)


class UpdateOidcConfigRequest(StrictModel):
    """Partial live update for OIDC audiences, allowlists, scopes, claims, and leeway."""

    oidc_audiences: list[str] | None = None
    oidc_allowed_emails: list[str] | None = None
    oidc_allowed_domains: list[str] | None = None
    oidc_admin_emails: list[str] | None = None
    oidc_default_scopes: list[str] | None = None
    oidc_require_email_verified: bool | None = None
    oidc_identity_claims: list[str] | None = None
    oidc_leeway_seconds: float | None = Field(default=None, ge=0.0, le=300.0)


class CorsRuntimeConfigResponse(StrictModel):
    """Live browser CORS origin and allowed-header policy."""

    cors_allowed_origins: list[str] = Field(default_factory=list)
    cors_extra_allow_headers: list[str] = Field(default_factory=list)
    effective_allow_headers: list[str] = Field(default_factory=list)


class UpdateCorsConfigRequest(StrictModel):
    """Partial live update for CORS allowed origins and extra request headers."""

    cors_allowed_origins: list[str] | None = None
    cors_extra_allow_headers: list[str] | None = None


class LimitsRuntimeConfigResponse(StrictModel):
    """Live rate limits, per-principal quotas, proxy trust, upload limits, keepalive, and logs."""

    rate_limit_rpm: int = Field(ge=1)
    ip_rate_limit_rpm: int = Field(ge=1)
    max_projects_per_principal: int = Field(ge=1)
    max_storage_bytes_per_principal: int = Field(ge=1)
    max_queued_jobs: int = Field(ge=1)
    max_queued_jobs_per_principal: int = Field(ge=1)
    max_upload_bytes: int = Field(ge=1)
    keepalive_interval_seconds: float = Field(gt=0.0)
    json_keepalive_threshold_seconds: float = Field(gt=0.0)
    trust_cloudflare_headers: bool
    trusted_proxy_cidrs: list[str] = Field(default_factory=list)
    proxy_ip_header: ProxyIpHeader
    log_level: str


class UpdateLimitsConfigRequest(StrictModel):
    """Partial live update for limits, quotas, proxy trust, keepalive, and log level."""

    rate_limit_rpm: int | None = Field(default=None, ge=1)
    ip_rate_limit_rpm: int | None = Field(default=None, ge=1)
    max_projects_per_principal: int | None = Field(default=None, ge=1)
    max_storage_bytes_per_principal: int | None = Field(default=None, ge=1)
    max_queued_jobs: int | None = Field(default=None, ge=1)
    max_queued_jobs_per_principal: int | None = Field(default=None, ge=1)
    max_upload_bytes: int | None = Field(default=None, ge=1)
    keepalive_interval_seconds: float | None = Field(default=None, gt=0.0)
    json_keepalive_threshold_seconds: float | None = Field(default=None, gt=0.0)
    trust_cloudflare_headers: bool | None = None
    trusted_proxy_cidrs: list[str] | None = None
    proxy_ip_header: ProxyIpHeader | None = None
    log_level: str | None = None


class GuardrailsPolicyConfigResponse(StrictModel):
    """Live ingress and egress ShieldGemma guardrail enforcement policy."""

    enabled: bool
    block_mode: GuardrailBlockMode
    output_mode: GuardrailOutputMode
    refusal_text: str
    violation_threshold: float = Field(ge=0.0, le=1.0)
    fail_open: bool
    allow_runtime_disable: bool
    max_violation_threshold: float = Field(ge=0.0, le=1.0)
    timeout_seconds: float = Field(gt=0.0)


class UpdateGuardrailsPolicyRequest(StrictModel):
    """Partial live update for guardrail enforcement policy without restarting subprocesses."""

    enabled: bool | None = None
    block_mode: GuardrailBlockMode | None = None
    output_mode: GuardrailOutputMode | None = None
    refusal_text: str | None = Field(default=None, min_length=1)
    violation_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fail_open: bool | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)


class UpdatesRuntimeConfigResponse(StrictModel):
    """Live self-update and rollback orchestration settings."""

    enabled: bool
    github_repo: str
    releases_dir: str
    current_symlink: str
    backups_dir: str
    service_name: str
    health_url: str
    health_timeout_seconds: float = Field(gt=0.0, le=600.0)
    health_poll_interval_seconds: float = Field(gt=0.0, le=30.0)
    keep_releases: int = Field(ge=1, le=20)
    auto_rollback: bool
    allow_prerelease: bool


class UpdateUpdatesConfigRequest(StrictModel):
    """Partial live update for non-privileged self-update scheduling and rollback toggles."""

    enabled: bool | None = None
    health_timeout_seconds: float | None = Field(default=None, gt=0.0, le=600.0)
    health_poll_interval_seconds: float | None = Field(default=None, gt=0.0, le=30.0)
    keep_releases: int | None = Field(default=None, ge=1, le=20)
    auto_rollback: bool | None = None
    allow_prerelease: bool | None = None


class ScopedConfigReloadRequest(StrictModel):
    """Explicit list of configuration sections to reload from YAML and environment variables."""

    sections: list[ReloadSection] = Field(min_length=1)


class ScopedConfigReloadResponse(StrictModel):
    """Summary of sections reloaded from YAML/env versus left untouched."""

    reloaded_sections: list[ReloadSection] = Field(default_factory=list)
    unchanged_sections: list[ReloadSection] = Field(default_factory=list)
    config_path: str | None = None
