"""Internal prompt audit context and decision construction helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

from lai.domain.guardrails import GuardrailDecision, GuardrailVerdict
from lai.observability.logging import get_request_id
from lai.observability.scrubber import fingerprint_prompt, scrub_sensitive_text

_MS_PER_SECOND = 1000.0
_PREVIEW_MAX_CHARS = 160


@dataclass(frozen=True, slots=True)
class PromptCheckContext:
    """Correlation metadata and pre-scrubbed prompt representation for one check."""

    started: float
    prompt_sha256: str
    prompt_chars: int
    scrubbed: str
    preview: str
    request_id: str | None
    session_id: str | None
    project_id: str | None
    principal_id: str | None

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        prompt: str,
        *,
        max_chars: int,
        request_id: str | None,
        session_id: str | None,
        project_id: str | None,
        principal_id: str | None,
    ) -> PromptCheckContext:
        """Compute prompt digest, scrubbed strings, and effective correlation IDs."""
        return cls(
            started=time.perf_counter(),
            prompt_sha256=fingerprint_prompt(prompt),
            prompt_chars=len(prompt),
            scrubbed=scrub_sensitive_text(prompt, max_chars=max_chars),
            preview=scrub_sensitive_text(prompt, max_chars=min(max_chars, _PREVIEW_MAX_CHARS)),
            request_id=request_id if request_id is not None else get_request_id(),
            session_id=session_id,
            project_id=project_id,
            principal_id=principal_id,
        )

    def elapsed_ms(self) -> float:
        """Return elapsed wall time in milliseconds since evaluation started."""
        return (time.perf_counter() - self.started) * _MS_PER_SECOND

    def make_decision(
        self,
        *,
        verdict: GuardrailVerdict,
        provider: str,
        violated_policy: str | None = None,
        reason: str | None = None,
    ) -> GuardrailDecision:
        """Construct a GuardrailDecision stamped with scrubbed context fields."""
        return GuardrailDecision(
            verdict=verdict,
            provider=provider,
            violated_policy=violated_policy,
            reason=reason,
            latency_ms=self.elapsed_ms(),
            prompt_sha256=self.prompt_sha256,
            prompt_chars=self.prompt_chars,
            scrubbed_prompt=self.scrubbed,
        )

    def normalize(self, decision: GuardrailDecision) -> GuardrailDecision:
        """Ensure canonical SHA-256, char count, scrubbed prompt, and total latency."""
        return decision.model_copy(
            update={
                "latency_ms": self.elapsed_ms(),
                "prompt_sha256": self.prompt_sha256,
                "prompt_chars": self.prompt_chars,
                "scrubbed_prompt": self.scrubbed,
            },
        )

    def log_fields(self) -> dict[str, str | None]:
        """Return correlation and scrubbed preview fields for structured logging."""
        return {
            "prompt_sha256": self.prompt_sha256,
            "scrubbed_prompt": self.preview,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "principal_id": self.principal_id,
        }
