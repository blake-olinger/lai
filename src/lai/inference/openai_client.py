"""OpenAI-compatible remote/Ollama inference provider client."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import httpx

from lai.domain.errors import ValidationFailedError
from lai.domain.inference import BackendHealth
from lai.inference._base_client import BaseOpenAIProvider
from lai.inference._wire import as_list, as_mapping, as_optional_str

if TYPE_CHECKING:
    from collections.abc import Mapping

_HTTP_ERROR_THRESHOLD = 400
_MS_PER_SECOND = 1000.0


def _normalize_openai_v1_base_url(base_url: str) -> str:
    """Normalize an external OpenAI or Ollama base URL to its /v1 root."""
    cleaned = base_url.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        msg = f"OpenAI-compatible base_url must start with http:// or https://: {base_url!r}"
        raise ValidationFailedError(msg, details={"base_url": base_url})
    if not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def _extract_first_model_id(raw_text: str) -> str | None:
    """Extract the first model id from an OpenAI /v1/models JSON payload."""
    try:
        parsed: object = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    mapping = as_mapping(parsed)
    if mapping is None:
        return None
    models = as_list(mapping.get("data"))
    if not models:
        return None
    first = as_mapping(models[0])
    return as_optional_str(first.get("id")) if first is not None else None


class OpenAICompatProvider(BaseOpenAIProvider):
    """InferenceProviderPort client for external OpenAI or Ollama /v1 endpoints."""

    def __init__(  # noqa: PLR0913
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        health_timeout_seconds: float = 5.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        v1_base = _normalize_openai_v1_base_url(base_url)
        merged_headers = dict(headers) if headers is not None else {}
        if api_key:
            merged_headers["Authorization"] = f"Bearer {api_key}"
        super().__init__(
            base_url=v1_base,
            completions_path="/chat/completions",
            timeout_seconds=timeout_seconds,
            health_timeout_seconds=health_timeout_seconds,
            headers=merged_headers,
            include_enable_thinking_kwarg=False,
            client=client,
        )

    async def health(self) -> BackendHealth:
        """Probe /v1/models to verify connectivity and authentication."""
        url = self._url("/models")
        start = time.perf_counter()
        try:
            response = await self._client.get(
                url,
                headers=self._headers,
                timeout=self._health_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            latency_ms = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
            return BackendHealth(
                healthy=False,
                status="unreachable",
                latency_ms=latency_ms,
                error=str(exc),
            )

        latency_ms = round((time.perf_counter() - start) * _MS_PER_SECOND, 2)
        if response.status_code >= _HTTP_ERROR_THRESHOLD:
            return BackendHealth(
                healthy=False,
                status="unhealthy",
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}",
            )

        return BackendHealth(
            healthy=True,
            status="ok",
            model_loaded=_extract_first_model_id(response.text),
            latency_ms=latency_ms,
        )
