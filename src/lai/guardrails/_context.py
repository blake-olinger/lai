"""Internal prompt/response audit context, audit/log emission, and analytics helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.analytics import GuardrailUsageEvent
from lai.domain.guardrails import (
    GuardrailDecision,
    GuardrailDirection,
    GuardrailVerdict,
    PromptAuditEntry,
)
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_request_id
from lai.observability.scrubber import fingerprint_prompt, scrub_sensitive_text

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

    from lai.observability.scrubber import PromptAuditLogger
    from lai.ports.analytics import UsageAnalyticsPort

_MS_PER_SECOND = 1000.0
_PREVIEW_MAX_CHARS = 160


@dataclass(frozen=True, slots=True)
class PromptCheckContext:
    """Correlation metadata and pre-scrubbed text representation for one guardrail check."""

    started: float
    prompt_sha256: str
    prompt_chars: int
    scrubbed: str
    preview: str
    direction: GuardrailDirection
    request_id: str | None
    session_id: str | None
    project_id: str | None
    principal_id: str | None

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        text: str,
        *,
        max_chars: int,
        direction: GuardrailDirection = GuardrailDirection.INGRESS,
        request_id: str | None,
        session_id: str | None,
        project_id: str | None,
        principal_id: str | None,
    ) -> PromptCheckContext:
        """Compute text digest, scrubbed strings, direction, and effective correlation IDs."""
        return cls(
            started=time.perf_counter(),
            prompt_sha256=fingerprint_prompt(text),
            prompt_chars=len(text),
            scrubbed=scrub_sensitive_text(text, max_chars=max_chars),
            preview=scrub_sensitive_text(text, max_chars=min(max_chars, _PREVIEW_MAX_CHARS)),
            direction=direction,
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
        violation_score: float | None = None,
        reason: str | None = None,
    ) -> GuardrailDecision:
        """Construct a GuardrailDecision stamped with direction and scrubbed context fields."""
        return GuardrailDecision(
            verdict=verdict,
            provider=provider,
            direction=self.direction,
            violated_policy=violated_policy,
            violation_score=violation_score,
            reason=reason,
            latency_ms=self.elapsed_ms(),
            prompt_sha256=self.prompt_sha256,
            prompt_chars=self.prompt_chars,
            scrubbed_prompt=self.scrubbed,
        )

    def normalize(self, decision: GuardrailDecision) -> GuardrailDecision:
        """Ensure canonical direction, SHA-256, char count, scrubbed text, and total latency."""
        return decision.model_copy(
            update={
                "direction": self.direction,
                "latency_ms": self.elapsed_ms(),
                "prompt_sha256": self.prompt_sha256,
                "prompt_chars": self.prompt_chars,
                "scrubbed_prompt": self.scrubbed,
            },
        )

    def log_fields(self) -> dict[str, str | None]:
        """Return correlation, direction, and scrubbed preview fields for structured logging."""
        return {
            "direction": self.direction.value,
            "prompt_sha256": self.prompt_sha256,
            "scrubbed_prompt": self.preview,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "principal_id": self.principal_id,
        }


def record_audit_and_log(
    audit_logger: PromptAuditLogger,
    logger: BoundLogger,
    decision: GuardrailDecision,
    ctx: PromptCheckContext,
    *,
    alert_emitted: bool,
) -> None:
    """Persist scrubbed PromptAuditEntry and emit structured component log."""
    audit_logger.log_entry(
        PromptAuditEntry(
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            project_id=ctx.project_id,
            principal_id=ctx.principal_id,
            direction=decision.direction,
            verdict=decision.verdict,
            provider=decision.provider,
            violated_policy=decision.violated_policy,
            violation_score=decision.violation_score,
            latency_ms=decision.latency_ms,
            prompt_sha256=decision.prompt_sha256,
            prompt_chars=decision.prompt_chars,
            scrubbed_prompt=decision.scrubbed_prompt,
        ),
    )
    if alert_emitted:
        return
    fields: dict[str, object] = {
        "verdict": decision.verdict.value,
        "provider": decision.provider,
        "violated_policy": decision.violated_policy,
        "violation_score": decision.violation_score,
        "latency_ms": decision.latency_ms,
        **ctx.log_fields(),
    }
    if decision.verdict is GuardrailVerdict.BLOCK:
        logger.warning(
            "guardrail_ruling",
            alert_category=AlertCategory.SECURITY_ALERT.value,
            **fields,
        )
        return
    logger.info("guardrail_ruling", **fields)


async def record_guardrail_analytics(
    analytics_repo: UsageAnalyticsPort | None,
    logger: BoundLogger,
    decision: GuardrailDecision,
    *,
    fail_closed: bool,
) -> None:
    """Record guardrail evaluation efficacy without allowing storage faults to fail requests."""
    if analytics_repo is None:
        return
    event = GuardrailUsageEvent(
        direction=decision.direction,
        verdict=decision.verdict,
        policy=decision.violated_policy,
        fail_closed=fail_closed,
        latency_ms=max(0.0, decision.latency_ms),
    )
    try:
        await analytics_repo.record_guardrail(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "guardrail_analytics_record_failed",
            direction=decision.direction.value,
            verdict=decision.verdict.value,
            error=str(exc),
        )
