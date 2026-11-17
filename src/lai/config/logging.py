"""Structured logging configuration wrapping lai.observability."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from lai.domain.observability import LoggingConfig
from lai.observability.logging import (
    bind_log_context,
    bind_request_id,
    configure_observability,
    get_component_logger,
    get_request_id,
    reset_request_id,
    set_request_id,
)

if TYPE_CHECKING:
    from lai.observability.alerts import AlertManager

__all__ = [
    "bind_log_context",
    "bind_request_id",
    "configure_logging",
    "get_component_logger",
    "get_logger",
    "get_request_id",
    "reset_request_id",
    "set_request_id",
]


def configure_logging(
    *,
    level: str = "INFO",
    json_logs: bool = False,
    logging_config: LoggingConfig | None = None,
) -> AlertManager:
    """Configure structlog and stdlib logging with request_id correlation."""
    config = logging_config or LoggingConfig(
        level=level,
        json_console=json_logs,
        enable_file_logs=False,
    )
    return configure_observability(config)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger instance."""
    return structlog.stdlib.get_logger(name)
