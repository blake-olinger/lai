"""Persistent HTTP provider for the local supervised llama-server backend."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import httpx

from lai.domain.errors import ValidationFailedError
from lai.domain.inference import BackendHealth
from lai.inference._base_client import BaseOpenAIProvider
from lai.inference._wire import as_mapping, as_optional_str, to_int

if TYPE_CHECKING:
    from collections.abc import Mapping

_MIN_PORT = 1
_MAX_PORT = 65535
_HTTP_OK = 200
_HTTP_SERVICE_UNAVAILABLE = 503
_MS_PER_SECOND = 1000.0


def _try_parse_json_dict(raw_text: str) -> Mapping[str, object] | None:
    """Parse JSON response body as mapping when possible."""
    if not raw_text.strip():
        return None
    try:
        parsed: object = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return as_mapping(parsed)


def _extract_llama_error(payload: Mapping[str, object] | None, fallback: str) -> str:
    """Extract error message from llama-server JSON error payloads."""
    if payload is None:
        return fallback
    err = as_mapping(payload.get("error"))
    if err is not None:
        msg = as_optional_str(err.get("message"))
        if msg:
            return msg
    status = as_optional_str(payload.get("status"))
    return status or fallback


class LlamaCppProvider(BaseOpenAIProvider):
    """InferenceProviderPort client targeting local llama-server over HTTP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8081,
        *,
        timeout_seconds: float = 120.0,
        health_timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not host.strip():
            msg = "LlamaCppProvider host must not be empty."
            raise ValidationFailedError(msg, details={"host": host})
        if port < _MIN_PORT or port > _MAX_PORT:
            msg = f"LlamaCppProvider port must be between {_MIN_PORT} and {_MAX_PORT}: {port}"
            raise ValidationFailedError(msg, details={"port": str(port)})

        self._host = host
        self._port = port
        super().__init__(
            base_url=f"http://{host}:{port}",
            completions_path="/v1/chat/completions",
            timeout_seconds=timeout_seconds,
            health_timeout_seconds=health_timeout_seconds,
            include_enable_thinking_kwarg=True,
            client=client,
        )

    async def health(self) -> BackendHealth:
        """Query llama-server /health endpoint and return readiness snapshot."""
        url = self._url("/health")
        start = time.perf_counter()
        try:
            response = await self._client.get(url, timeout=self._health_timeout_seconds)
        except httpx.HTTPError as exc:
            latency_ms = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
            return BackendHealth(
                healthy=False,
                status="unreachable",
                latency_ms=latency_ms,
                error=str(exc),
            )

        latency_ms = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
        payload = _try_parse_json_dict(response.text)

        if response.status_code == _HTTP_OK:
            status = (
                as_optional_str(payload.get("status")) if payload is not None else None
            ) or "ok"
            slots_idle = max(0, to_int(payload.get("slots_idle"), default=0)) if payload else 0
            slots_busy = (
                max(0, to_int(payload.get("slots_processing"), default=0)) if payload else 0
            )
            slots_total = (
                max(0, to_int(payload.get("slots_total"), default=slots_idle + slots_busy))
                if payload
                else 0
            )
            model_loaded = (
                as_optional_str(payload.get("model_loaded"))
                or as_optional_str(payload.get("model"))
                if payload is not None
                else None
            )
            return BackendHealth(
                healthy=status.lower() == "ok",
                status=status,
                model_loaded=model_loaded,
                slots_total=slots_total,
                slots_idle=slots_idle,
                latency_ms=latency_ms,
            )

        error_msg = _extract_llama_error(payload, f"HTTP {response.status_code}")
        status_label = (
            "starting" if response.status_code == _HTTP_SERVICE_UNAVAILABLE else "unhealthy"
        )
        return BackendHealth(
            healthy=False,
            status=status_label,
            latency_ms=latency_ms,
            error=error_msg,
        )
