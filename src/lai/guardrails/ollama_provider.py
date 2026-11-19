"""CPU-only ShieldGemma-2B guardrail provider backed by Ollama /api/generate."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import httpx

from lai.domain.errors import InferenceBackendError
from lai.domain.guardrails import GuardrailConfig, GuardrailDecision, GuardrailVerdict
from lai.domain.inference import BackendHealth
from lai.domain.observability import AlertCategory
from lai.guardrails.prompts import (
    DEFAULT_SAFETY_PRINCIPLES,
    build_shieldgemma_prompt,
    compute_yes_probability,
    parse_shieldgemma_verdict,
)
from lai.observability.logging import get_component_logger
from lai.observability.scrubber import fingerprint_prompt, scrub_sensitive_text
from lai.ports.guardrails import GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.guardrails import SafetyPrinciple

_HTTP_OK = 200
_MS_PER_SECOND = 1000.0
_N_PREDICT = 4
_TOP_LOGPROBS = 5
_STOP_TOKENS: tuple[str, ...] = ("\n", "<end_of_turn>")


def _normalize_ollama_url(base_url: str) -> str:
    return base_url.strip().rstrip("/").removesuffix("/v1")


def _parse_json_dict(response: httpx.Response) -> dict[str, object] | None:
    try:
        parsed: object = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    raw_map = cast("dict[object, object]", parsed)
    return {str(key): value for key, value in raw_map.items()}


def _extract_model_names(payload: dict[str, object]) -> set[str]:
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return set()
    names: set[str] = set()
    for item in cast("list[object]", raw_models):
        if not isinstance(item, dict):
            continue
        item_map = cast("dict[object, object]", item)
        name = item_map.get("name") or item_map.get("model")
        if isinstance(name, str) and name:
            names.add(name)
            names.add(name.split(":", maxsplit=1)[0])
    return names


class OllamaShieldGemmaProvider(GuardrailProviderPort):
    """Pre-flight ShieldGemma-2B safety classifier running on Ollama with num_gpu=0."""

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        *,
        principles: Sequence[SafetyPrinciple] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or GuardrailConfig(backend="ollama")
        self._principles = (
            tuple(principles) if principles is not None else DEFAULT_SAFETY_PRINCIPLES
        )
        self._client = client
        self._log = get_component_logger("guardrails")

    @property
    def name(self) -> str:
        """Return the identifier for this Ollama-backed guardrail provider."""
        return "shieldgemma-2b-ollama"

    @property
    def base_url(self) -> str:
        """Return the base HTTP URL for the Ollama daemon."""
        return _normalize_ollama_url(self._config.ollama_base_url)

    def reconfigure(self, config: GuardrailConfig) -> None:
        """Update guardrail policy parameters in place without restarting Ollama."""
        self._config = config

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        """Evaluate prompt across safety principles using CPU-pinned ShieldGemma."""
        if self._client is not None:
            return await self._evaluate_with_client(self._client, prompt)
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await self._evaluate_with_client(client, prompt)

    async def health(self) -> BackendHealth:
        """Check Ollama daemon reachability and confirm guardrail model availability."""
        if self._client is not None:
            return await self._health_with_client(self._client)
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await self._health_with_client(client)

    async def _evaluate_with_client(
        self,
        client: httpx.AsyncClient,
        prompt: str,
    ) -> GuardrailDecision:
        started = time.perf_counter()
        prompt_sha256 = fingerprint_prompt(prompt)
        prompt_chars = len(prompt)
        scrubbed = scrub_sensitive_text(
            prompt,
            max_chars=self._config.max_logged_prompt_chars,
        )
        last_raw = ""

        for principle in self._principles:
            violated, raw_output = await self._evaluate_principle(
                client=client,
                prompt=prompt,
                principle=principle,
                prompt_sha256=prompt_sha256,
            )
            last_raw = raw_output
            if violated:
                elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
                return GuardrailDecision(
                    verdict=GuardrailVerdict.BLOCK,
                    provider=self.name,
                    violated_policy=principle.name,
                    reason=f"Prompt violated safety principle '{principle.name}'.",
                    raw_output=raw_output,
                    latency_ms=elapsed_ms,
                    prompt_sha256=prompt_sha256,
                    prompt_chars=prompt_chars,
                    scrubbed_prompt=scrubbed,
                )

        elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            raw_output=last_raw,
            latency_ms=elapsed_ms,
            prompt_sha256=prompt_sha256,
            prompt_chars=prompt_chars,
            scrubbed_prompt=scrubbed,
        )

    async def _evaluate_principle(
        self,
        *,
        client: httpx.AsyncClient,
        prompt: str,
        principle: SafetyPrinciple,
        prompt_sha256: str,
    ) -> tuple[bool, str]:
        payload: dict[str, object] = {
            "model": self._config.ollama_model,
            "prompt": build_shieldgemma_prompt(prompt, principle),
            "raw": True,
            "stream": False,
            "keep_alive": -1,
            "logprobs": True,
            "top_logprobs": _TOP_LOGPROBS,
            "options": {
                "num_gpu": 0,
                "use_mmap": True,
                "num_ctx": self._config.ctx_size,
                "num_thread": self._config.cpu_threads,
                "temperature": 0.0,
                "num_predict": _N_PREDICT,
                "stop": list(_STOP_TOKENS),
            },
        }
        try:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            self._log.error(
                "shieldgemma_ollama_timeout",
                provider=self.name,
                principle=principle.name,
                timeout_seconds=self._config.timeout_seconds,
                prompt_sha256=prompt_sha256,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = f"Ollama ShieldGemma request timed out after {self._config.timeout_seconds}s"
            raise InferenceBackendError(msg) from exc
        except httpx.HTTPError as exc:
            self._log.error(
                "shieldgemma_ollama_http_error",
                provider=self.name,
                principle=principle.name,
                error=str(exc),
                prompt_sha256=prompt_sha256,
            )
            msg = f"Ollama ShieldGemma request failed: {exc}"
            raise InferenceBackendError(msg) from exc

        return self._interpret_response(response, principle, prompt_sha256)

    def _interpret_response(
        self,
        response: httpx.Response,
        principle: SafetyPrinciple,
        prompt_sha256: str,
    ) -> tuple[bool, str]:
        data = _parse_json_dict(response)
        raw_text = data.get("response") if data is not None else None
        if not isinstance(raw_text, str):
            msg = "Ollama ShieldGemma response missing string 'response' field."
            raise InferenceBackendError(msg)

        yes_prob = compute_yes_probability(data.get("logprobs") if data is not None else None)
        if yes_prob is not None:
            return yes_prob >= self._config.violation_threshold, f"{raw_text} (p={yes_prob:.3f})"

        try:
            return parse_shieldgemma_verdict(raw_text), raw_text
        except ValueError as exc:
            self._log.error(
                "shieldgemma_ollama_ambiguous",
                provider=self.name,
                principle=principle.name,
                raw_output=raw_text,
                prompt_sha256=prompt_sha256,
            )
            msg = f"Ambiguous Ollama ShieldGemma verdict for '{principle.name}': {raw_text!r}"
            raise InferenceBackendError(msg) from exc

    async def _health_with_client(self, client: httpx.AsyncClient) -> BackendHealth:
        started = time.perf_counter()
        try:
            response = await client.get(f"{self.base_url}/api/tags")
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
            return BackendHealth(
                healthy=False,
                status="timeout",
                model_loaded=self._config.ollama_model,
                latency_ms=elapsed_ms,
                error=str(exc),
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
            return BackendHealth(
                healthy=False,
                status="unreachable",
                model_loaded=self._config.ollama_model,
                latency_ms=elapsed_ms,
                error=str(exc),
            )

        elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
        if response.status_code != _HTTP_OK:
            return BackendHealth(
                healthy=False,
                status=f"http_{response.status_code}",
                model_loaded=self._config.ollama_model,
                latency_ms=elapsed_ms,
                error=response.text.strip() or f"HTTP {response.status_code}",
            )

        payload = _parse_json_dict(response) or {}
        names = _extract_model_names(payload)
        target = self._config.ollama_model
        target_short = target.split(":", maxsplit=1)[0]
        if target not in names and target_short not in names:
            return BackendHealth(
                healthy=False,
                status="model_missing",
                model_loaded=target,
                latency_ms=elapsed_ms,
                error=f"Ollama guardrail model {target!r} is not installed.",
            )
        return BackendHealth(
            healthy=True,
            status="ok",
            model_loaded=target,
            slots_total=self._config.parallel_slots,
            slots_idle=self._config.parallel_slots,
            latency_ms=elapsed_ms,
        )
