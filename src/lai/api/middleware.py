"""ASGI request context middleware assigning X-Request-Id and binding log context."""

from __future__ import annotations

import re
import time
import uuid
from typing import TYPE_CHECKING, cast

from lai.api._analytics import (
    extract_principal_ids,
    extract_route_template,
    record_http_analytics,
)
from lai.observability.logging import bind_log_context, get_component_logger

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HEADER_REQUEST_ID = b"x-request-id"
_HEADER_ORIGIN = b"origin"
_HEADER_ACRM = b"access-control-request-method"
_HEADER_ACRH = b"access-control-request-headers"
_HEADER_CACHE_CONTROL = b"cache-control"
_CACHE_CONTROL_NO_STORE = b"no-store"
_HTTP_BAD_REQUEST = 400
_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
)
_MAX_REQUEST_ID_LEN = 128
_REQUEST_ID_PATTERN = re.compile(rf"[A-Za-z0-9._:-]{{1,{_MAX_REQUEST_ID_LEN}}}")
_MS_PER_SECOND = 1000.0
_logger = get_component_logger("api")


def _sanitize_request_id(raw_value: bytes | None) -> str:
    """Validate incoming X-Request-Id as strict ASCII token or mint a fresh req_ ID."""
    if raw_value is not None:
        try:
            decoded = raw_value.decode("ascii", errors="strict")
        except UnicodeDecodeError:
            decoded = ""
        candidate = decoded.strip(" \t")
        if _REQUEST_ID_PATTERN.fullmatch(candidate) is not None:
            return candidate
    return f"req_{uuid.uuid4().hex}"


def _extract_incoming_request_id(scope: Scope) -> str:
    """Read X-Request-Id from ASGI headers or generate a new correlation ID."""
    raw_headers = cast("list[tuple[bytes, bytes]]", scope.get("headers", []))
    for key, value in raw_headers:
        if key.lower() == _HEADER_REQUEST_ID:
            return _sanitize_request_id(value)
    return _sanitize_request_id(None)


def _extract_cors_preflight_headers(scope: Scope) -> tuple[str | None, str | None, str | None]:
    """Extract (origin, requested_method, requested_headers) from a CORS preflight scope."""
    raw_headers = cast("list[tuple[bytes, bytes]]", scope.get("headers", []))
    origin: str | None = None
    requested_method: str | None = None
    requested_headers: str | None = None
    for key, value in raw_headers:
        lowered = key.lower()
        if lowered == _HEADER_ORIGIN:
            origin = value.decode("latin-1", errors="replace").strip() or None
        elif lowered == _HEADER_ACRM:
            requested_method = value.decode("latin-1", errors="replace").strip() or None
        elif lowered == _HEADER_ACRH:
            requested_headers = value.decode("latin-1", errors="replace").strip() or None
    return origin, requested_method, requested_headers


def _log_rejected_cors_preflight(
    preflight: tuple[str | None, str | None, str | None] | None,
    status_code: int | None,
    path: str,
) -> None:
    """Emit a structured warning when a CORS preflight OPTIONS request is rejected (400)."""
    if (
        preflight is not None
        and status_code == _HTTP_BAD_REQUEST
        and preflight[0] is not None
        and preflight[1] is not None
    ):
        _logger.warning(
            "cors_preflight_rejected",
            path=path,
            origin=preflight[0],
            requested_method=preflight[1],
            requested_headers=preflight[2],
        )


def _inject_response_headers(message: Message, request_id: str) -> None:
    """Attach X-Request-Id and HTTP security headers without clobbering route headers."""
    headers = list(cast("list[tuple[bytes, bytes]]", message.get("headers", [])))
    existing_names = {key.lower() for key, _ in headers}
    if _HEADER_REQUEST_ID not in existing_names:
        headers.append((_HEADER_REQUEST_ID, request_id.encode("ascii", errors="replace")))
        existing_names.add(_HEADER_REQUEST_ID)
    for header_name, header_value in _SECURITY_HEADERS:
        if header_name not in existing_names:
            headers.append((header_name, header_value))
            existing_names.add(header_name)
    if _HEADER_CACHE_CONTROL not in existing_names:
        headers.append((_HEADER_CACHE_CONTROL, _CACHE_CONTROL_NO_STORE))
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
                _inject_response_headers(message, request_id)
            await send(message)

        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", ""))
        preflight = _extract_cors_preflight_headers(scope) if method == "OPTIONS" else None
        with bind_log_context(request_id=request_id, component="api"):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                duration_ms = round((time.perf_counter() - started) * _MS_PER_SECOND, 2)
                route = extract_route_template(scope)
                principal_id, oauth_client_id = extract_principal_ids(scope)
                _log_rejected_cors_preflight(preflight, status_code, path)
                await record_http_analytics(
                    scope,
                    method=method,
                    route=route,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    principal_id=principal_id,
                    oauth_client_id=oauth_client_id,
                )
                _logger.info(
                    "http_request_completed",
                    method=method,
                    path=path,
                    route=route,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    principal_id=principal_id,
                    oauth_client_id=oauth_client_id,
                )
