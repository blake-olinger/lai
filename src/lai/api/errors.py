"""FastAPI exception handlers translating domain and unexpected errors into JSON envelopes."""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, cast

from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from lai.api.schemas.common import ErrorDetail, ErrorResponse
from lai.domain.errors import (
    GuardrailBlockedError,
    LaiError,
    RateLimitExceededError,
    ToolApprovalRequiredError,
)
from lai.domain.observability import AlertCategory
from lai.observability.logging import bind_log_context, get_request_id

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from starlette.responses import Response

_fatal_logger = logging.getLogger("lai.api")
_DEFAULT_QUEUE_RETRY_AFTER = "5"


def _resolve_request_id(request: Request) -> str | None:
    """Resolve the active request_id from ContextVar or ASGI request state."""
    from_ctx = get_request_id()
    if from_ctx:
        return from_ctx
    from_state = getattr(request.state, "request_id", None)
    return str(from_state) if isinstance(from_state, str) and from_state else None


def _build_error_headers(
    request_id: str | None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build standard error response headers including X-Request-Id."""
    headers: dict[str, str] = {}
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    if extra:
        headers.update(extra)
    return headers


def _extract_lai_details_and_headers(exc: LaiError) -> tuple[dict[str, str], dict[str, str]]:
    """Extract enriched error details and status-specific HTTP headers for a LaiError."""
    details = dict(exc.details)
    headers: dict[str, str] = {}

    if isinstance(exc, RateLimitExceededError):
        retry_seconds = (
            max(1, math.ceil(exc.retry_after_seconds)) if exc.retry_after_seconds is not None else 1
        )
        headers["Retry-After"] = str(retry_seconds)
        details.setdefault("retry_after_seconds", str(retry_seconds))
    elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        headers["Retry-After"] = details.get("retry_after_seconds", _DEFAULT_QUEUE_RETRY_AFTER)
    elif isinstance(exc, ToolApprovalRequiredError):
        details.setdefault("approval_id", exc.approval_id)
        details.setdefault("tool_name", exc.tool_name)
    elif isinstance(exc, GuardrailBlockedError):
        headers["X-LAI-Guardrail-Blocked"] = "true"
        if exc.violated_policy:
            headers["X-LAI-Guardrail-Policy"] = exc.violated_policy
            details.setdefault("violated_policy", exc.violated_policy)
        direction = details.get("direction")
        if direction:
            headers["X-LAI-Guardrail-Direction"] = direction

    return details, headers


def _error_json_response(
    status_code: int,
    detail: ErrorDetail,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a JSONResponse adhering to ErrorResponse."""
    payload = ErrorResponse(error=detail)
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=_build_error_headers(detail.request_id, headers),
    )


async def handle_lai_error(request: Request, exc: Exception) -> Response:
    """Translate any LaiError subclass into a structured JSON error envelope."""
    lai_exc = cast("LaiError", exc)
    request_id = _resolve_request_id(request)
    details, extra_headers = _extract_lai_details_and_headers(lai_exc)
    return _error_json_response(
        lai_exc.status_code,
        ErrorDetail(
            code=lai_exc.code,
            message=lai_exc.message,
            request_id=request_id,
            details=details,
        ),
        headers=extra_headers,
    )


async def handle_request_validation_error(request: Request, exc: Exception) -> Response:
    """Translate FastAPI/Pydantic request validation errors into 422 JSON envelopes."""
    val_exc = cast("RequestValidationError", exc)
    request_id = _resolve_request_id(request)
    first_errors = val_exc.errors()
    details: dict[str, str] = {"error_count": str(len(first_errors))}
    if first_errors:
        loc = ".".join(str(part) for part in first_errors[0].get("loc", ()))
        if loc:
            details["first_location"] = loc
    return _error_json_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorDetail(
            code="validation_failed",
            message="Request validation failed.",
            request_id=request_id,
            details=details,
        ),
    )


async def handle_http_exception(request: Request, exc: Exception) -> Response:
    """Translate Starlette HTTPException into the standard JSON error envelope."""
    http_exc = cast("StarletteHTTPException", exc)
    request_id = _resolve_request_id(request)
    return _error_json_response(
        http_exc.status_code,
        ErrorDetail(
            code="http_error",
            message=str(http_exc.detail),
            request_id=request_id,
        ),
        headers=dict(http_exc.headers) if http_exc.headers else None,
    )


async def handle_unexpected_exception(request: Request, exc: Exception) -> Response:
    """Log critical FATAL_EXCEPTION alert and return a sanitized 500 JSON envelope."""
    request_id = _resolve_request_id(request)
    # Emit a JSON payload string via stdlib logger so AlertingLogHandler._extract_payload
    # receives alert_category and request_id directly while ProcessorFormatter still records it.
    alert_payload = json.dumps(
        {
            "event": "unhandled_api_exception",
            "component": "api",
            "alert_category": AlertCategory.FATAL_EXCEPTION.value,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__,
        }
    )
    with bind_log_context(request_id=request_id):
        _fatal_logger.critical(alert_payload, exc_info=exc)
    return _error_json_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorDetail(
            code="internal_error",
            message="An unexpected internal server error occurred.",
            request_id=request_id,
        ),
    )


def serialize_post_header_error_json(
    exc: Exception,
    *,
    request_id: str | None,
) -> bytes:
    """Serialize an error occurring after keepalive whitespace headers were already committed."""
    if isinstance(exc, LaiError):
        details, _ = _extract_lai_details_and_headers(exc)
        details.setdefault("status_code", str(exc.status_code))
        payload = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                request_id=request_id,
                details=details,
            )
        )
        return payload.model_dump_json().encode("utf-8")

    alert_payload = json.dumps(
        {
            "event": "unhandled_post_header_json_exception",
            "component": "api",
            "alert_category": AlertCategory.FATAL_EXCEPTION.value,
            "request_id": request_id,
            "exception_type": type(exc).__name__,
        }
    )
    with bind_log_context(request_id=request_id):
        _fatal_logger.critical(alert_payload, exc_info=exc)
    payload = ErrorResponse(
        error=ErrorDetail(
            code="internal_error",
            message="An unexpected internal server error occurred.",
            request_id=request_id,
            details={"status_code": "500"},
        )
    )
    return payload.model_dump_json().encode("utf-8")


def register_exception_handlers(app: FastAPI) -> None:
    """Register all domain, validation, HTTP, and global catch-all exception handlers."""
    app.add_exception_handler(LaiError, handle_lai_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_exception)
