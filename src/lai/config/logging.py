"""Structured logging configuration with request_id context propagation."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Generator

_REQUEST_ID_VAR: ContextVar[str | None] = ContextVar("lai_request_id", default=None)


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


def _inject_request_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Attach the active request_id to each structured log record when set."""
    del logger, method_name
    request_id = _REQUEST_ID_VAR.get()
    if request_id is not None and "request_id" not in event_dict:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(*, level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog and stdlib logging with request_id correlation."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger instance."""
    return structlog.stdlib.get_logger(name)
