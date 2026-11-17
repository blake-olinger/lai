"""Structured logging, context propagation, and daily rotating observability setup."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

import structlog

from lai.observability.alerts import AlertingLogHandler, AlertManager, WebhookAlertSink
from lai.observability.rotation import create_daily_rotating_handler

if TYPE_CHECKING:
    from collections.abc import Generator

    from lai.domain.observability import LoggingConfig
    from lai.ports.observability import AlertSinkPort


class _UnsetType:
    """Sentinel distinguishing omitted context parameters from explicit None."""


_UNSET = _UnsetType()

_REQUEST_ID_VAR: ContextVar[str | None] = ContextVar("lai_request_id", default=None)
_JOB_ID_VAR: ContextVar[str | None] = ContextVar("lai_job_id", default=None)
_SESSION_ID_VAR: ContextVar[str | None] = ContextVar("lai_session_id", default=None)
_PROJECT_ID_VAR: ContextVar[str | None] = ContextVar("lai_project_id", default=None)
_PRINCIPAL_ID_VAR: ContextVar[str | None] = ContextVar("lai_principal_id", default=None)
_COMPONENT_VAR: ContextVar[str | None] = ContextVar("lai_component", default=None)

_CONTEXT_VARS: tuple[tuple[str, ContextVar[str | None]], ...] = (
    ("request_id", _REQUEST_ID_VAR),
    ("job_id", _JOB_ID_VAR),
    ("session_id", _SESSION_ID_VAR),
    ("project_id", _PROJECT_ID_VAR),
    ("principal_id", _PRINCIPAL_ID_VAR),
    ("component", _COMPONENT_VAR),
)


def get_request_id() -> str | None:
    """Return the active request_id bound to the current async context."""
    return _REQUEST_ID_VAR.get()


def set_request_id(request_id: str | None) -> Token[str | None]:
    """Bind request_id into the current context and return the reset token."""
    return _REQUEST_ID_VAR.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request_id using a ContextVar token."""
    _REQUEST_ID_VAR.reset(token)


@contextmanager
def bind_request_id(request_id: str | None) -> Generator[None]:
    """Temporarily bind request_id for the duration of a context block."""
    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)


@contextmanager
def bind_log_context(  # noqa: PLR0913
    *,
    request_id: str | _UnsetType | None = _UNSET,
    job_id: str | _UnsetType | None = _UNSET,
    session_id: str | _UnsetType | None = _UNSET,
    project_id: str | _UnsetType | None = _UNSET,
    principal_id: str | _UnsetType | None = _UNSET,
    component: str | _UnsetType | None = _UNSET,
) -> Generator[None]:
    """Temporarily bind structured correlation IDs into ContextVars."""
    updates: tuple[tuple[ContextVar[str | None], str | _UnsetType | None], ...] = (
        (_REQUEST_ID_VAR, request_id),
        (_JOB_ID_VAR, job_id),
        (_SESSION_ID_VAR, session_id),
        (_PROJECT_ID_VAR, project_id),
        (_PRINCIPAL_ID_VAR, principal_id),
        (_COMPONENT_VAR, component),
    )
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    for var, value in updates:
        if not isinstance(value, _UnsetType):
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _inject_log_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Attach bound correlation IDs to each structured log record."""
    del logger, method_name
    for key, var in _CONTEXT_VARS:
        value = var.get()
        if value is not None and key not in event_dict:
            event_dict[key] = value
    return event_dict


def _build_processor_formatter(
    foreign_pre_chain: list[structlog.types.Processor],
    renderer: structlog.types.Processor,
) -> structlog.stdlib.ProcessorFormatter:
    """Construct a structlog ProcessorFormatter for stdlib handlers."""
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=foreign_pre_chain,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def _close_existing_handlers(root_logger: logging.Logger) -> None:
    """Flush, close, and clear existing handlers attached to root_logger."""
    for handler in list(root_logger.handlers):
        handler.flush()
        handler.close()
    root_logger.handlers.clear()


def configure_observability(
    config: LoggingConfig,
    alert_manager: AlertManager | None = None,
) -> AlertManager:
    """Configure structlog, console/file 24h rotating handlers, and alerting."""
    sinks: tuple[AlertSinkPort, ...] = (
        (WebhookAlertSink(config.alert_webhook_url),)
        if config.alert_webhook_url is not None
        else ()
    )
    manager = alert_manager or AlertManager(
        sinks=sinks,
        cooldown_seconds=config.alert_cooldown_seconds,
        log_dir=config.log_dir if config.enable_file_logs else None,
        retention_days=config.error_retention_days,
        compress_rotated=config.compress_rotated,
    )

    numeric_level = getattr(logging, config.level.upper(), logging.INFO)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_log_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    console_renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if config.json_console
        else structlog.dev.ConsoleRenderer()
    )
    json_renderer = structlog.processors.JSONRenderer()

    root_logger = logging.getLogger()
    _close_existing_handlers(root_logger)
    root_logger.setLevel(numeric_level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_build_processor_formatter(shared_processors, console_renderer))
    root_logger.addHandler(console_handler)

    if config.enable_file_logs:
        all_logs_handler = create_daily_rotating_handler(
            config.log_dir / "lai.jsonl",
            level=numeric_level,
            retention_days=config.retention_days,
            compress=config.compress_rotated,
            when=config.rotation_when,
            interval=config.rotation_interval,
        )
        all_logs_handler.setFormatter(_build_processor_formatter(shared_processors, json_renderer))
        root_logger.addHandler(all_logs_handler)

        error_logs_handler = create_daily_rotating_handler(
            config.log_dir / "errors.jsonl",
            level=logging.ERROR,
            retention_days=config.error_retention_days,
            compress=config.compress_rotated,
            when=config.rotation_when,
            interval=config.rotation_interval,
        )
        error_logs_handler.setFormatter(
            _build_processor_formatter(shared_processors, json_renderer)
        )
        root_logger.addHandler(error_logs_handler)

    alert_handler = AlertingLogHandler(manager, alert_on_timeouts=config.alert_on_timeouts)
    root_logger.addHandler(alert_handler)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    return manager


def get_component_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger pre-tagged with component name."""
    return structlog.stdlib.get_logger(f"lai.{component}").bind(component=component)
