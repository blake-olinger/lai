"""ASGI request context middleware assigning X-Request-Id and binding log context."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, cast

from lai.observability.logging import bind_log_context, get_component_logger

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HEADER_REQUEST_ID = b"x-request-id"
_MAX_REQUEST_ID_LEN = 128
_MS_PER_SECOND = 1000.0
_logger = get_component_logger("api")


def _sanitize_request_id(raw_value: bytes | None) -> str:
    """Validate incoming X-Request-Id or mint a fresh req_ identifier."""
    if raw_value is not None:
        decoded = raw_value.decode("ascii", errors="ignore").strip()
        if (
            decoded
            and len(decoded) <= _MAX_REQUEST_ID_LEN
            and "\r" not in decoded
            and "\n" not in decoded
        ):
            return decoded
    return f"req_{uuid.uuid4().hex}"


def _extract_incoming_request_id(scope: Scope) -> str:
    """Read X-Request-Id from ASGI headers or generate a new correlation ID."""
    raw_headers = cast("list[tuple[bytes, bytes]]", scope.get("headers", []))
    for key, value in raw_headers:
        if key.lower() == _HEADER_REQUEST_ID:
            return _sanitize_request_id(value)
    return _sanitize_request_id(None)


def _inject_response_request_id(message: Message, request_id: str) -> None:
    """Attach X-Request-Id header to http.response.start if not already present."""
    headers = list(cast("list[tuple[bytes, bytes]]", message.get("headers", [])))
    if not any(key.lower() == _HEADER_REQUEST_ID for key, _ in headers):
        headers.append((_HEADER_REQUEST_ID, request_id.encode("ascii", errors="replace")))
        message["headers"] = headers


class RequestContextMiddleware:
    """Pure ASGI middleware propagating X-Request-Id without buffering SSE streams."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind request_id into ASGI state and structlog context for HTTP requests."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _extract_incoming_request_id(scope)
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        status_code: int | None = None
        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                raw_status = message.get("status")
                if isinstance(raw_status, int):
                    status_code = raw_status
                _inject_response_request_id(message, request_id)
            await send(message)

        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", ""))
        with bind_log_context(request_id=request_id, component="api"):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                duration_ms = round((time.perf_counter() - started) * _MS_PER_SECOND, 2)
                _logger.info(
                    "http_request_completed",
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
