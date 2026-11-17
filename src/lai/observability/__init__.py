"""24-hour rotating structured logging, stream writers, and automated alerting."""

from __future__ import annotations

from lai.observability.alerts import AlertingLogHandler, AlertManager, WebhookAlertSink
from lai.observability.logging import (
    bind_log_context,
    bind_request_id,
    configure_observability,
    get_component_logger,
    get_request_id,
    reset_request_id,
    set_request_id,
)
from lai.observability.rotation import (
    DailyRotatingStreamWriter,
    SafeTimedRotatingFileHandler,
    create_daily_rotating_handler,
)

__all__ = [
    "AlertManager",
    "AlertingLogHandler",
    "DailyRotatingStreamWriter",
    "SafeTimedRotatingFileHandler",
    "WebhookAlertSink",
    "bind_log_context",
    "bind_request_id",
    "configure_observability",
    "create_daily_rotating_handler",
    "get_component_logger",
    "get_request_id",
    "reset_request_id",
    "set_request_id",
]
