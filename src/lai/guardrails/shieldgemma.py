"""CPU-only ShieldGemma-2B guardrail provider talking to llama-server /completion."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import httpx

from lai.domain.errors import InferenceBackendError
from lai.domain.guardrails import (
    GuardrailConfig,
    GuardrailDecision,
    GuardrailDirection,
    GuardrailVerdict,
)
from lai.domain.inference import BackendHealth
from lai.domain.observability import AlertCategory
from lai.guardrails.prompts import (
    DEFAULT_SAFETY_PRINCIPLES,
    OUTPUT_SAFETY_PRINCIPLES,
    build_shieldgemma_prompt,
    build_shieldgemma_response_prompt,
    compute_yes_probability,
    parse_shieldgemma_verdict,
)
from lai.observability.logging import get_component_logger
from lai.observability.scrubber import fingerprint_prompt, scrub_sensitive_text
from lai.ports.guardrails import GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lai.domain.guardrails import SafetyPrinciple

_HTTP_OK = 200
_MS_PER_SECOND = 1000.0
_N_PREDICT = 4
_N_PROBS = 5
_STOP_TOKENS: tuple[str, ...] = ("\n", "<end_of_turn>")
_HEALTHY_STATUSES: frozenset[str] = frozenset({"ok", "ready", "healthy"})


def _parse_json_object(response: httpx.Response) -> dict[str, object] | None:
    """Decode an HTTP response body into a string-keyed dict if valid JSON object."""
    try:
        parsed: object = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    raw_map = cast("dict[object, object]", parsed)
    return {str(key): value for key, value in raw_map.items()}


def _non_negative_int(value: object) -> int:
    """Coerce integer counts from backend health JSON safely."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


class ShieldGemmaCpuProvider(GuardrailProviderPort):
    """Ingress and egress ShieldGemma-2B safety classifier backed by CPU-only llama-server."""

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        *,
        principles: Sequence[SafetyPrinciple] | None = None,
        output_principles: Sequence[SafetyPrinciple] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or GuardrailConfig()
        self._principles = (
            tuple(principles) if principles is not None else DEFAULT_SAFETY_PRINCIPLES
        )
        self._output_principles = (
            tuple(output_principles) if output_principles is not None else OUTPUT_SAFETY_PRINCIPLES
        )
        self._client = client
        self._log = get_component_logger("guardrails")

    @property
    def name(self) -> str:
        """Return the identifier for this guardrail provider."""
        return "shieldgemma-2b-cpu"

    @property
    def base_url(self) -> str:
        """Return the base HTTP URL for the ShieldGemma llama-server instance."""
        return f"http://{self._config.host}:{self._config.port}"

    def reconfigure(self, config: GuardrailConfig) -> None:
        """Update guardrail policy parameters in place without restarting llama-server."""
        self._config = config

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        """Evaluate prompt sequentially across ingress principles, short-circuiting on violation."""
        return await self._run_with_client(
            subject_text=prompt,
            principles=self._principles,
            direction=GuardrailDirection.INGRESS,
            prompt_builder=lambda principle: build_shieldgemma_prompt(prompt, principle),
        )

    async def evaluate_response(self, prompt: str, response: str) -> GuardrailDecision:
        """Evaluate model response sequentially across output principles."""
        return await self._run_with_client(
            subject_text=response,
            principles=self._output_principles,
            direction=GuardrailDirection.EGRESS,
            prompt_builder=lambda principle: build_shieldgemma_response_prompt(
                prompt, response, principle
            ),
        )

    async def health(self) -> BackendHealth:
        """Query ShieldGemma llama-server /health readiness and latency."""
        if self._client is not None:
            return await self._health_with_client(self._client)
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await self._health_with_client(client)

    async def _run_with_client(
        self,
        *,
        subject_text: str,
        principles: tuple[SafetyPrinciple, ...],
        direction: GuardrailDirection,
        prompt_builder: Callable[[SafetyPrinciple], str],
    ) -> GuardrailDecision:
        if self._client is not None:
            return await self._evaluate_principles(
                self._client, subject_text, principles, direction, prompt_builder
            )
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await self._evaluate_principles(
                client, subject_text, principles, direction, prompt_builder
            )

    async def _evaluate_principles(
        self,
        client: httpx.AsyncClient,
        subject_text: str,
        principles: tuple[SafetyPrinciple, ...],
        direction: GuardrailDirection,
        prompt_builder: Callable[[SafetyPrinciple], str],
    ) -> GuardrailDecision:
        """Run sequential principle checks using an active httpx.AsyncClient."""
        started = time.perf_counter()
        text_sha256 = fingerprint_prompt(subject_text)
        text_chars = len(subject_text)
        scrubbed = scrub_sensitive_text(
            subject_text, max_chars=self._config.max_logged_prompt_chars
        )
        last_raw_output = ""
        max_yes_prob: float | None = None
        noun = "Response" if direction is GuardrailDirection.EGRESS else "Prompt"

        for principle in principles:
            violated, raw_output, yes_prob = await self._evaluate_principle(
                client=client,
                rendered_prompt=prompt_builder(principle),
                principle=principle,
                prompt_sha256=text_sha256,
            )
            last_raw_output = raw_output
            if yes_prob is not None:
                max_yes_prob = yes_prob if max_yes_prob is None else max(max_yes_prob, yes_prob)
            if violated:
                return GuardrailDecision(
                    verdict=GuardrailVerdict.BLOCK,
                    provider=self.name,
                    direction=direction,
                    violated_policy=principle.name,
                    violation_score=yes_prob,
                    reason=f"{noun} violated safety principle '{principle.name}'.",
                    raw_output=raw_output,
                    latency_ms=(time.perf_counter() - started) * _MS_PER_SECOND,
                    prompt_sha256=text_sha256,
                    prompt_chars=text_chars,
                    scrubbed_prompt=scrubbed,
                )

        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            direction=direction,
            violation_score=max_yes_prob,
            raw_output=last_raw_output,
            latency_ms=(time.perf_counter() - started) * _MS_PER_SECOND,
            prompt_sha256=text_sha256,
            prompt_chars=text_chars,
            scrubbed_prompt=scrubbed,
        )

    async def _evaluate_principle(
        self,
        *,
        client: httpx.AsyncClient,
        rendered_prompt: str,
        principle: SafetyPrinciple,
        prompt_sha256: str,
    ) -> tuple[bool, str, float | None]:
        """Execute a single ShieldGemma /completion classification call with n_probs."""
        payload = {
            "prompt": rendered_prompt,
            "n_predict": _N_PREDICT,
            "n_probs": _N_PROBS,
            "temperature": 0.0,
            "stop": list(_STOP_TOKENS),
        }
        try:
            response = await client.post(f"{self.base_url}/completion", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            self._log.error(
                "shieldgemma_request_timeout",
                provider=self.name,
                principle=principle.name,
                timeout_seconds=self._config.timeout_seconds,
                prompt_sha256=prompt_sha256,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = f"ShieldGemma guardrail request timed out after {self._config.timeout_seconds}s"
            raise InferenceBackendError(
                msg,
                details={"provider": self.name, "principle": principle.name},
            ) from exc
        except httpx.HTTPError as exc:
            self._log.error(
                "shieldgemma_http_error",
                provider=self.name,
                principle=principle.name,
                error=str(exc),
                prompt_sha256=prompt_sha256,
            )
            msg = f"ShieldGemma guardrail request failed: {exc}"
            raise InferenceBackendError(
                msg,
                details={"provider": self.name, "principle": principle.name},
            ) from exc

        return self._parse_completion_response(response, principle, prompt_sha256)

    def _parse_completion_response(
        self,
        response: httpx.Response,
        principle: SafetyPrinciple,
        prompt_sha256: str,
    ) -> tuple[bool, str, float | None]:
        """Calibrate violation via completion_probabilities or strict Yes/No fallback."""
        data = _parse_json_object(response)
        raw_content = data.get("content") if data is not None else None
        if not isinstance(raw_content, str):
            msg = "ShieldGemma response missing string 'content' field."
            raise InferenceBackendError(
                msg,
                details={"provider": self.name, "principle": principle.name},
            )

        yes_prob = compute_yes_probability(
            data.get("completion_probabilities") if data is not None else None
        )
        if yes_prob is not None:
            return yes_prob >= self._config.violation_threshold, raw_content, yes_prob

        try:
            violated = parse_shieldgemma_verdict(raw_content)
        except ValueError as exc:
            self._log.error(
                "shieldgemma_ambiguous_verdict",
                provider=self.name,
                principle=principle.name,
                raw_output=raw_content,
                prompt_sha256=prompt_sha256,
            )
            msg = f"Ambiguous ShieldGemma verdict for '{principle.name}': {raw_content!r}"
            raise InferenceBackendError(
                msg,
                details={"provider": self.name, "principle": principle.name},
            ) from exc
        return violated, raw_content, None

    async def _health_with_client(self, client: httpx.AsyncClient) -> BackendHealth:
        """Execute GET /health and map HTTP/transport outcomes to BackendHealth."""
        started = time.perf_counter()
        try:
            response = await client.get(f"{self.base_url}/health")
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
            return BackendHealth(
                healthy=False,
                status="timeout",
                model_loaded=self._config.hf_filename,
                latency_ms=elapsed_ms,
                error=str(exc),
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
            return BackendHealth(
                healthy=False,
                status="unreachable",
                model_loaded=self._config.hf_filename,
                latency_ms=elapsed_ms,
                error=str(exc),
            )

        elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
        payload = _parse_json_object(response) or {}
        if response.status_code != _HTTP_OK:
            error_obj = payload.get("error")
            error_msg = (
                str(error_obj)
                if error_obj is not None
                else (response.text.strip() or f"HTTP {response.status_code}")
            )
            return BackendHealth(
                healthy=False,
                status=f"http_{response.status_code}",
                model_loaded=self._config.hf_filename,
                latency_ms=elapsed_ms,
                error=error_msg,
            )

        status_val = str(payload.get("status", "ok"))
        slots_idle = _non_negative_int(payload.get("slots_idle"))
        slots_processing = _non_negative_int(payload.get("slots_processing"))
        slots_total = _non_negative_int(payload.get("slots_total")) or (
            slots_idle + slots_processing
        )
        return BackendHealth(
            healthy=status_val.lower() in _HEALTHY_STATUSES,
            status=status_val,
            model_loaded=self._config.hf_filename,
            slots_total=slots_total,
            slots_idle=slots_idle,
            latency_ms=elapsed_ms,
        )
