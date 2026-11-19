"""Observability alert and rotating logging configuration domain models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_alert_id


class AlertSeverity(StrEnum):
    """Severity level for automated operational alerts."""

    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(StrEnum):
    """Failure classification attached to automated alert events."""

    TIMEOUT = "timeout"
    FATAL_EXCEPTION = "fatal_exception"
    SUBPROCESS_CRASH = "subprocess_crash"
    STORAGE_FAILURE = "storage_failure"
    SECURITY_ALERT = "security_alert"
    APPLICATION_ERROR = "application_error"


class SmtpTlsMode(StrEnum):
    """Transport security mode used when connecting to an outbound SMTP server."""

    STARTTLS = "starttls"
    SSL = "ssl"
    NONE = "none"


class SmtpAlertConfig(StrictModel):
    """Fully configurable outbound SMTP email alert sink settings."""

    enabled: bool = False
    host: str | None = None
    port: int = Field(default=587, ge=1, le=65535)
    tls_mode: SmtpTlsMode = SmtpTlsMode.STARTTLS
    username: str | None = None
    password: str | None = None
    from_address: str | None = None
    to_addresses: tuple[str, ...] = ()
    use_admin_emails_fallback: bool = True
    subject_prefix: str = Field(default="[LAI Alert]", min_length=1, max_length=80)
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=60.0)
    min_severity: AlertSeverity = AlertSeverity.ERROR
    categories: tuple[AlertCategory, ...] = ()


class AlertSinkDeliveryResult(StrictModel):
    """Per-sink delivery outcome returned when sending synthetic test alerts."""

    sink_name: str = Field(min_length=1)
    delivered: bool
    skipped: bool = False
    error: str | None = None


class AlertEvent(StrictModel):
    """Structured alert emitted on fatal errors, crashes, or timeouts."""

    id: str = Field(default_factory=new_alert_id)
    severity: AlertSeverity
    category: AlertCategory
    component: str = Field(min_length=1)
    message: str
    request_id: str | None = None
    job_id: str | None = None
    session_id: str | None = None
    project_id: str | None = None
    exception_type: str | None = None
    traceback_excerpt: str | None = None
    details: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class LoggingConfig(StrictModel):
    """24-hour rotating logging and alert delivery configuration."""

    log_dir: Path = Path("data/logs")
    level: str = "INFO"
    json_console: bool = False
    enable_file_logs: bool = True
    rotation_when: str = "midnight"
    rotation_interval: int = Field(default=1, ge=1)
    retention_days: int = Field(default=14, ge=1)
    error_retention_days: int = Field(default=30, ge=1)
    compress_rotated: bool = True
    alert_webhook_url: str | None = None
    alert_webhook_enabled: bool = True
    alert_webhook_min_severity: AlertSeverity = AlertSeverity.ERROR
    alert_webhook_categories: tuple[AlertCategory, ...] = ()
    alert_smtp: SmtpAlertConfig = Field(default_factory=SmtpAlertConfig)
    alert_cooldown_seconds: float = Field(default=60.0, ge=0.0)
    alert_on_timeouts: bool = True
    alert_on_errors: bool = True
