"""CPU-only ShieldGemma-2B guardrail provider talking to llama-server /completion."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

from lai.domain.guardrails import (
    GuardrailConfig,
    GuardrailDecision,
    GuardrailDirection,
    GuardrailVerdict,
)
from lai.domain.inference import BackendHealth
from lai.guardrails._chunking import (
    bisect_text,
    clip_prompt_context,
    compute_slot_text_budget,
    estimate_conservative_tokens,
    split_overlapping_windows,
)
from lai.guardrails._shieldgemma_http import (
    build_completion_payload,
    is_context_size_error,
    parse_completion_response,
    parse_health_response,
    raise_http_error,
    raise_timeout_error,
)
from lai.guardrails.prompts import (
    DEFAULT_SAFETY_PRINCIPLES,
    OUTPUT_SAFETY_PRINCIPLES,
    build_shieldgemma_prompt,
    build_shieldgemma_response_prompt,
)
from lai.observability.logging import get_component_logger
from lai.observability.scrubber import fingerprint_prompt, scrub_sensitive_text
from lai.ports.guardrails import GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lai.domain.guardrails import SafetyPrinciple

_MS_PER_SECOND = 1000.0
_MAX_BISECTION_DEPTH = 4
_MIN_EGRESS_BUDGET = 256


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
        """Evaluate prompt across sliding windows and ingress principles."""
        budget = compute_slot_text_budget(self._config.ctx_size, self._config.parallel_slots)
        windows = split_overlapping_windows(prompt, max_tokens=budget)
        return await self._run_with_client(
            subject_text=prompt,
            windows=windows,
            principles=self._principles,
            direction=GuardrailDirection.INGRESS,
            prompt_builder=build_shieldgemma_prompt,
        )

    async def evaluate_response(self, prompt: str, response: str) -> GuardrailDecision:
        """Evaluate model response across sliding windows and output principles."""
        budget = compute_slot_text_budget(self._config.ctx_size, self._config.parallel_slots)
        prompt_budget = max(_MIN_EGRESS_BUDGET, budget // 3)
        clipped_prompt = clip_prompt_context(prompt, max_tokens=prompt_budget)
        prompt_tokens = estimate_conservative_tokens(clipped_prompt)
        response_budget = max(_MIN_EGRESS_BUDGET, budget - prompt_tokens)
        windows = split_overlapping_windows(response, max_tokens=response_budget)
        return await self._run_with_client(
            subject_text=response,
            windows=windows,
            principles=self._output_principles,
            direction=GuardrailDirection.EGRESS,
            prompt_builder=lambda chunk, principle: build_shieldgemma_response_prompt(
                clipped_prompt, chunk, principle
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
        windows: Sequence[str],
        principles: tuple[SafetyPrinciple, ...],
        direction: GuardrailDirection,
        prompt_builder: Callable[[str, SafetyPrinciple], str],
    ) -> GuardrailDecision:
        if self._client is not None:
            return await self._evaluate_windows(
                self._client, subject_text, windows, principles, direction, prompt_builder
            )
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await self._evaluate_windows(
                client, subject_text, windows, principles, direction, prompt_builder
            )

    async def _evaluate_windows(  # noqa: PLR0913, PLR0917
        self,
        client: httpx.AsyncClient,
        subject_text: str,
        windows: Sequence[str],
        principles: tuple[SafetyPrinciple, ...],
        direction: GuardrailDirection,
        prompt_builder: Callable[[str, SafetyPrinciple], str],
    ) -> GuardrailDecision:
        """Run sequential window x principle checks, short-circuiting on first BLOCK."""
        started = time.perf_counter()
        text_sha256 = fingerprint_prompt(subject_text)
        text_chars = len(subject_text)
        scrubbed = scrub_sensitive_text(
            subject_text, max_chars=self._config.max_logged_prompt_chars
        )
        last_raw_output = ""
        max_yes_prob: float | None = None
        noun = "Response" if direction is GuardrailDirection.EGRESS else "Prompt"

        for window in windows:
            for principle in principles:
                violated, raw_output, yes_prob = await self._evaluate_chunk(
                    client=client,
                    chunk_text=window,
                    principle=principle,
                    prompt_builder=prompt_builder,
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

    async def _evaluate_chunk(  # noqa: PLR0913
        self,
        *,
        client: httpx.AsyncClient,
        chunk_text: str,
        principle: SafetyPrinciple,
        prompt_builder: Callable[[str, SafetyPrinciple], str],
        prompt_sha256: str,
        depth: int = 0,
    ) -> tuple[bool, str, float | None]:
        """Execute ShieldGemma /completion with recursive bisection on HTTP 400 ctx overflow."""
        payload = build_completion_payload(prompt_builder(chunk_text, principle))
        try:
            response = await client.post(f"{self.base_url}/completion", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise raise_timeout_error(
                exc,
                provider_name=self.name,
                principle_name=principle.name,
                timeout_seconds=self._config.timeout_seconds,
                prompt_sha256=prompt_sha256,
                log=self._log,
            ) from exc
        except httpx.HTTPStatusError as exc:
            if depth < _MAX_BISECTION_DEPTH and is_context_size_error(exc.response):
                halves = bisect_text(chunk_text)
                if halves is not None:
                    self._log.warning(
                        "shieldgemma_context_overflow_bisect",
                        provider=self.name,
                        principle=principle.name,
                        chunk_chars=len(chunk_text),
                        depth=depth,
                        prompt_sha256=prompt_sha256,
                    )
                    return await self._evaluate_bisected(
                        client=client,
                        halves=halves,
                        principle=principle,
                        prompt_builder=prompt_builder,
                        prompt_sha256=prompt_sha256,
                        depth=depth + 1,
                    )
            raise raise_http_error(
                exc,
                provider_name=self.name,
                principle_name=principle.name,
                prompt_sha256=prompt_sha256,
                log=self._log,
            ) from exc
        except httpx.HTTPError as exc:
            raise raise_http_error(
                exc,
                provider_name=self.name,
                principle_name=principle.name,
                prompt_sha256=prompt_sha256,
                log=self._log,
            ) from exc

        return parse_completion_response(
            response,
            provider_name=self.name,
            principle_name=principle.name,
            violation_threshold=self._config.violation_threshold,
            prompt_sha256=prompt_sha256,
            log=self._log,
        )

    async def _evaluate_bisected(  # noqa: PLR0913
        self,
        *,
        client: httpx.AsyncClient,
        halves: tuple[str, str],
        principle: SafetyPrinciple,
        prompt_builder: Callable[[str, SafetyPrinciple], str],
        prompt_sha256: str,
        depth: int,
    ) -> tuple[bool, str, float | None]:
        """Evaluate left and right bisected chunks, short-circuiting on violation."""
        left_violated, left_raw, left_prob = await self._evaluate_chunk(
            client=client,
            chunk_text=halves[0],
            principle=principle,
            prompt_builder=prompt_builder,
            prompt_sha256=prompt_sha256,
            depth=depth,
        )
        if left_violated:
            return True, left_raw, left_prob
        right_violated, right_raw, right_prob = await self._evaluate_chunk(
            client=client,
            chunk_text=halves[1],
            principle=principle,
            prompt_builder=prompt_builder,
            prompt_sha256=prompt_sha256,
            depth=depth,
        )
        combined_prob: float | None = right_prob
        if left_prob is not None:
            combined_prob = left_prob if right_prob is None else max(left_prob, right_prob)
        return right_violated, right_raw, combined_prob

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
        return parse_health_response(
            response,
            model_loaded=self._config.hf_filename,
            elapsed_ms=elapsed_ms,
        )
