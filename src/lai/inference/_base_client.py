"""Shared persistent async HTTP transport for OpenAI-compatible inference providers."""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from typing import TYPE_CHECKING

import httpx

from lai.domain.errors import InferenceBackendError
from lai.domain.observability import AlertCategory
from lai.inference._wire import (
    SseStreamProcessor,
    parse_complete_response,
    parse_json_mapping,
    serialize_chat_request,
)
from lai.observability.logging import get_component_logger
from lai.ports.inference import InferenceProviderPort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping
    from types import TracebackType

    import structlog

    from lai.domain.inference import (
        BackendHealth,
        InferenceChunk,
        InferenceRequest,
        InferenceResponse,
    )

_HTTP_ERROR_THRESHOLD = 400
_MAX_ERROR_BODY_CHARS = 400
_MS_PER_SECOND = 1000.0
_DEFAULT_MAX_CONNECTIONS = 32
_DEFAULT_KEEPALIVE_CONNECTIONS = 16


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


def _record_ttft(
    model: str,
    start: float,
    ttft_ms: float | None,
    chunk: InferenceChunk,
) -> float | None:
    """Log TTFT once on the first non-empty content, thinking, or tool-call delta."""
    if ttft_ms is not None:
        return ttft_ms
    if not (chunk.content_delta or chunk.thinking_delta or chunk.tool_calls):
        return None
    measured = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
    _logger().info("inference_first_token", model=model, ttft_ms=measured)
    return measured


async def _ensure_stream_ok(response: httpx.Response, model: str, url: str) -> None:
    """Raise InferenceBackendError when a streaming response returns HTTP >= 400."""
    if response.status_code < _HTTP_ERROR_THRESHOLD:
        return
    body = (await response.aread()).decode("utf-8", errors="replace")
    _logger().error(
        "inference_stream_http_error",
        model=model,
        status_code=response.status_code,
        endpoint=url,
    )
    msg = f"Inference backend returned HTTP {response.status_code}: {body[:_MAX_ERROR_BODY_CHARS]}"
    raise InferenceBackendError(msg)


async def _iter_sse_frames(
    response: httpx.Response,
    processor: SseStreamProcessor,
) -> AsyncGenerator[InferenceChunk]:
    """Yield parsed InferenceChunks from an open HTTP SSE response stream."""
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        chunk = processor.process_data(line[5:].strip())
        if chunk is not None:
            yield chunk
        if processor.done:
            break

    if not processor.done:
        tail = processor.flush_remaining()
        if tail is not None:
            yield tail


class BaseOpenAIProvider(InferenceProviderPort):
    """Shared async HTTP transport for OpenAI-compatible chat completion backends."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        base_url: str,
        completions_path: str = "/v1/chat/completions",
        timeout_seconds: float = 120.0,
        health_timeout_seconds: float = 5.0,
        headers: Mapping[str, str] | None = None,
        include_enable_thinking_kwarg: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._completions_path = completions_path
        self._timeout_seconds = timeout_seconds
        self._health_timeout_seconds = health_timeout_seconds
        self._headers = dict(headers) if headers is not None else {}
        self._include_enable_thinking_kwarg = include_enable_thinking_kwarg
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=_DEFAULT_MAX_CONNECTIONS,
                max_keepalive_connections=_DEFAULT_KEEPALIVE_CONNECTIONS,
            ),
        )

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        clean = path if path.startswith("/") else f"/{path}"
        return f"{self._base_url}{clean}"

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient if owned by this provider."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> BaseOpenAIProvider:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close owned HTTP client on context exit."""
        del exc_type, exc, tb
        await self.aclose()

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a non-streaming inference request and return full response."""
        url = self._url(self._completions_path)
        payload = serialize_chat_request(
            request,
            stream=False,
            include_enable_thinking_kwarg=self._include_enable_thinking_kwarg,
        )
        start = time.perf_counter()
        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            _logger().exception(
                "inference_request_timeout",
                model=request.model,
                endpoint=url,
                timeout_seconds=self._timeout_seconds,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = f"Inference completion timed out after {self._timeout_seconds}s: {exc}"
            raise InferenceBackendError(msg) from exc
        except httpx.HTTPError as exc:
            _logger().exception("inference_request_failed", model=request.model, endpoint=url)
            msg = f"Inference HTTP error calling {url}: {exc}"
            raise InferenceBackendError(msg) from exc

        if response.status_code >= _HTTP_ERROR_THRESHOLD:
            excerpt = response.text[:_MAX_ERROR_BODY_CHARS]
            _logger().error(
                "inference_backend_http_error",
                model=request.model,
                status_code=response.status_code,
                endpoint=url,
            )
            msg = f"Inference backend returned HTTP {response.status_code}: {excerpt}"
            raise InferenceBackendError(msg)

        data = parse_json_mapping(response.text, context="chat completion response")
        result = parse_complete_response(data)
        latency_ms = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
        _logger().info(
            "inference_completed",
            model=request.model,
            latency_ms=latency_ms,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            tool_call_count=len(result.tool_calls),
        )
        return result

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        """Stream incremental inference chunks from the backend."""
        url = self._url(self._completions_path)
        payload = serialize_chat_request(
            request,
            stream=True,
            include_enable_thinking_kwarg=self._include_enable_thinking_kwarg,
        )
        processor = SseStreamProcessor()
        start = time.perf_counter()
        ttft_ms: float | None = None
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout_seconds,
            ) as response:
                await _ensure_stream_ok(response, request.model, url)
                async with aclosing(_iter_sse_frames(response, processor)) as frames:
                    async for chunk in frames:
                        ttft_ms = _record_ttft(request.model, start, ttft_ms, chunk)
                        yield chunk
        except asyncio.CancelledError:
            _logger().info("inference_stream_cancelled", model=request.model, endpoint=url)
            raise
        except GeneratorExit:
            _logger().info("inference_stream_closed", model=request.model, endpoint=url)
            raise
        except httpx.TimeoutException as exc:
            _logger().exception(
                "inference_stream_timeout",
                model=request.model,
                endpoint=url,
                timeout_seconds=self._timeout_seconds,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = f"Inference stream timed out after {self._timeout_seconds}s: {exc}"
            raise InferenceBackendError(msg) from exc
        except httpx.HTTPError as exc:
            _logger().exception("inference_stream_failed", model=request.model, endpoint=url)
            msg = f"Inference stream HTTP error calling {url}: {exc}"
            raise InferenceBackendError(msg) from exc

        total_ms = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
        _logger().info(
            "inference_stream_completed",
            model=request.model,
            ttft_ms=ttft_ms,
            latency_ms=total_ms,
        )

    async def health(self) -> BackendHealth:
        """Query current readiness and slot status of the inference backend."""
        msg = "Subclasses must implement health()."
        raise NotImplementedError(msg)
