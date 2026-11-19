"""24-hour rotating structured logging, stream writers, alerting, and scrubbing."""

from __future__ import annotations

from lai.observability.alerts import AlertingLogHandler, AlertManager, WebhookAlertSink
from lai.observability.logging import (
    bind_log_context,
    bind_request_id,
    configure_observability,
    get_component_logger,
    get_request_id,
    reconfigure_observability_alerting,
    reset_request_id,
    set_request_id,
    set_runtime_log_level,
)
from lai.observability.rotation import (
    DailyRotatingStreamWriter,
    SafeTimedRotatingFileHandler,
    create_daily_rotating_handler,
)
from lai.observability.scrubber import (
    PromptAuditLogger,
    fingerprint_prompt,
    scrub_sensitive_text,
)
from lai.observability.smtp import (
    FilteredAlertSink,
    SmtpAlertSink,
    build_alert_email,
    build_alert_sinks,
    deliver_test_alert,
    resolve_smtp_recipients,
)

__all__ = [
    "AlertManager",
    "AlertingLogHandler",
    "DailyRotatingStreamWriter",
    "FilteredAlertSink",
    "PromptAuditLogger",
    "SafeTimedRotatingFileHandler",
    "SmtpAlertSink",
    "WebhookAlertSink",
    "bind_log_context",
    "bind_request_id",
    "build_alert_email",
    "build_alert_sinks",
    "configure_observability",
    "create_daily_rotating_handler",
    "deliver_test_alert",
    "fingerprint_prompt",
    "get_component_logger",
    "get_request_id",
    "reconfigure_observability_alerting",
    "reset_request_id",
    "resolve_smtp_recipients",
    "scrub_sensitive_text",
    "set_request_id",
    "set_runtime_log_level",
]
