"""Pure ASGI request body size limit middleware preventing unauthenticated JSON OOM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lai.api.schemas.common import ErrorDetail, ErrorResponse
from lai.domain.errors import PayloadTooLargeError
from lai.observability.logging import get_request_id

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HTTP_PAYLOAD_TOO_LARGE = 413
_MULTIPART_FRAMING_ALLOWANCE_BYTES = 1_048_576
_UPLOAD_MULTIPART_PATH = "/v1/files"
_UPLOAD_STREAM_PATH = "/v1/files/stream"


def _parse_content_length(scope: Scope) -> int | None:
    """Extract non-negative Content-Length integer from ASGI headers if present."""
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.lower() == b"content-length":
            try:
                parsed = int(raw_value.decode("ascii").strip())
            except (UnicodeDecodeError, ValueError):
                return None
            return parsed if parsed >= 0 else None
    return None


async def _send_413_json(send: Send, *, max_bytes: int, request_id: str | None) -> None:
    """Emit a structured 413 Payload Too Large JSON ErrorResponse over ASGI send."""
    payload = ErrorResponse(
        error=ErrorDetail(
            code="payload_too_large",
            message=f"Request body exceeds maximum allowed size of {max_bytes} bytes.",
            request_id=request_id,
            details={"max_body_bytes": str(max_bytes)},
        )
    )
    body = payload.model_dump_json().encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if request_id:
        headers.append((b"x-request-id", request_id.encode("ascii", errors="ignore")))
    await send(
        {"type": "http.response.start", "status": _HTTP_PAYLOAD_TOO_LARGE, "headers": headers}
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class RequestBodySizeLimitMiddleware:
    """Enforce tight JSON body caps and upload fast-fail limits before body buffering."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_json_body_bytes: int = 2_097_152,
        max_upload_bytes: int = 262_144_000,
    ) -> None:
        self._app = app
        self._default_max_json_body_bytes = max_json_body_bytes
        self._default_max_upload_bytes = max_upload_bytes

    def _resolve_limits(self, scope: Scope) -> tuple[int, int]:
        app_obj: Any = scope.get("app")
        services: Any = getattr(getattr(app_obj, "state", None), "services", None)
        settings: Any = getattr(services, "settings", None)
        if settings is not None:
            return (
                int(getattr(settings, "max_json_body_bytes", self._default_max_json_body_bytes)),
                int(getattr(settings, "max_upload_bytes", self._default_max_upload_bytes)),
            )
        return self._default_max_json_body_bytes, self._default_max_upload_bytes

    def _resolve_route_limit(self, scope: Scope) -> int:
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "")).rstrip("/") or "/"
        max_json, max_upload = self._resolve_limits(scope)
        is_upload_route = method == "POST" and path in {
            _UPLOAD_MULTIPART_PATH,
            _UPLOAD_STREAM_PATH,
        }
        if not is_upload_route:
            return max_json
        return (
            max_upload
            if path == _UPLOAD_STREAM_PATH
            else max_upload + _MULTIPART_FRAMING_ALLOWANCE_BYTES
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject oversized HTTP bodies on Content-Length or streamed chunk accumulation."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        limit = self._resolve_route_limit(scope)
        content_length = _parse_content_length(scope)
        if content_length is not None and content_length > limit:
            await _send_413_json(send, max_bytes=limit, request_id=get_request_id())
            return

        await self._enforce_streamed_limit(scope, receive, send, limit=limit)

    async def _enforce_streamed_limit(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        limit: int,
    ) -> None:
        received_bytes = 0
        exceeded = False
        response_started = False

        async def _limited_receive() -> Message:
            nonlocal received_bytes, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    exceeded = True
                    msg = f"Request body exceeds maximum allowed size of {limit} bytes."
                    raise PayloadTooLargeError(msg, details={"max_body_bytes": str(limit)})
            return message

        async def _tracked_send(message: Message) -> None:
            nonlocal response_started
            if exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, _limited_receive, _tracked_send)
        except PayloadTooLargeError:
            exceeded = True
        if exceeded and not response_started:
            await _send_413_json(send, max_bytes=limit, request_id=get_request_id())
