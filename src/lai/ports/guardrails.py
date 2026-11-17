"""Pre-flight safety guardrail provider, pipeline, and supervisor protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lai.domain.guardrails import GuardrailConfig, GuardrailDecision
    from lai.domain.inference import BackendHealth


@runtime_checkable
class GuardrailProviderPort(Protocol):
    """Model-backed or rule-backed prompt safety evaluator."""

    @property
    def name(self) -> str:
        """Return the identifier for this guardrail provider."""
        ...

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        """Evaluate raw prompt text and return a structured guardrail decision."""
        ...


@runtime_checkable
class GuardrailPipelinePort(Protocol):
    """Pre-flight prompt check coordinator handling fail-open/closed and auditing."""

    async def check_prompt(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> GuardrailDecision:
        """Evaluate a prompt, write a scrubbed audit entry, and enforce policy."""
        ...


@runtime_checkable
class GuardrailSupervisorPort(Protocol):
    """Lifecycle management contract for the supervised guardrail server process."""

    async def ensure_running(self, config: GuardrailConfig) -> BackendHealth:
        """Start or verify the guardrail inference process for config."""
        ...

    async def stop(self) -> None:
        """Gracefully terminate the supervised guardrail process."""
        ...

    async def health(self) -> BackendHealth:
        """Return process and endpoint health status."""
        ...
