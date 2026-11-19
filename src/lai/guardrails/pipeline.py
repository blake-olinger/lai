"""Composite pre-flight guardrail pipeline with fail-open/closed policy and auditing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from lai.domain.errors import GuardrailBlockedError, InferenceBackendError
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailVerdict,
    PromptAuditEntry,
)
from lai.domain.observability import AlertCategory
from lai.guardrails._context import PromptCheckContext
from lai.guardrails.passthrough import AllowAllGuardrailProvider
from lai.guardrails.shieldgemma import ShieldGemmaCpuProvider
from lai.observability.logging import get_component_logger
from lai.observability.scrubber import PromptAuditLogger
from lai.ports.guardrails import GuardrailPipelinePort

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from lai.ports.guardrails import GuardrailProviderPort

_DEFAULT_AUDIT_RETENTION_DAYS = 30
_BACKEND_ERRORS = (InferenceBackendError, httpx.HTTPError, TimeoutError, ValueError)


class CompositeGuardrailPipeline(GuardrailPipelinePort):
    """Coordinates guardrail providers, fail-open/closed handling, and scrubbed auditing."""

    def __init__(  # noqa: PLR0913
        self,
        config: GuardrailConfig | None = None,
        *,
        providers: Sequence[GuardrailProviderPort] | None = None,
        audit_logger: PromptAuditLogger | None = None,
        log_dir: Path | None = None,
        retention_days: int = _DEFAULT_AUDIT_RETENTION_DAYS,
        compress_rotated: bool = True,
    ) -> None:
        self._config = config or GuardrailConfig()
        if providers is None:
            default_provider: GuardrailProviderPort = (
                ShieldGemmaCpuProvider(self._config)
                if self._config.enabled
                else AllowAllGuardrailProvider(
                    max_logged_prompt_chars=self._config.max_logged_prompt_chars,
                )
            )
            self._providers: list[GuardrailProviderPort] = [default_provider]
        else:
            self._providers = list(providers)

        if audit_logger is not None:
            self._audit_logger = audit_logger
            self._owns_audit_logger = False
        else:
            self._audit_logger = PromptAuditLogger(
                log_dir or Path("data/logs"),
                retention_days=retention_days,
                compress_rotated=compress_rotated,
            )
            self._owns_audit_logger = True
        self._log = get_component_logger("guardrails")

    @property
    def providers(self) -> tuple[GuardrailProviderPort, ...]:
        """Return a snapshot of registered guardrail providers."""
        return tuple(self._providers)

    def register_provider(self, provider: GuardrailProviderPort) -> None:
        """Append a guardrail provider to the sequential evaluation chain."""
        self._providers.append(provider)

    def reconfigure(
        self,
        config: GuardrailConfig,
        *,
        providers: Sequence[GuardrailProviderPort] | None = None,
    ) -> None:
        """Hot-swap guardrail policy (and optionally providers) in place."""
        self._config = config
        if providers is not None:
            self._providers = list(providers)
            return
        for provider in self._providers:
            reconfigure_fn = getattr(provider, "reconfigure", None)
            if callable(reconfigure_fn):
                reconfigure_fn(config)

    def close(self) -> None:
        """Close the owned rotating PromptAuditLogger if created by this pipeline."""
        if self._owns_audit_logger:
            self._audit_logger.close()

    def __enter__(self) -> CompositeGuardrailPipeline:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close owned audit logger resources on exit."""
        del exc_type, exc, tb
        self.close()

    async def check_prompt(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> GuardrailDecision:
        """Evaluate prompt across providers, write scrubbed audit entry, and enforce policy."""
        ctx = PromptCheckContext.build(
            prompt,
            max_chars=self._config.max_logged_prompt_chars,
            request_id=request_id,
            session_id=session_id,
            project_id=project_id,
            principal_id=principal_id,
        )
        decision, alert_emitted = await self._evaluate_chain(prompt, ctx)
        self._record_audit_and_log(decision, ctx, alert_emitted=alert_emitted)

        if (
            decision.verdict is GuardrailVerdict.BLOCK
            and self._config.block_mode is GuardrailBlockMode.HTTP_ERROR
        ):
            raise GuardrailBlockedError(
                self._config.refusal_text,
                violated_policy=decision.violated_policy,
                details={"provider": decision.provider, "prompt_sha256": ctx.prompt_sha256},
            )
        return decision

    async def _evaluate_chain(
        self,
        prompt: str,
        ctx: PromptCheckContext,
    ) -> tuple[GuardrailDecision, bool]:
        """Run enabled providers sequentially with fail-open/closed error handling."""
        if not self._config.enabled:
            passthrough = AllowAllGuardrailProvider(
                max_logged_prompt_chars=self._config.max_logged_prompt_chars,
            )
            return ctx.normalize(await passthrough.evaluate(prompt)), False

        if not self._providers:
            return self._handle_unconfigured(ctx)

        last_allow: GuardrailDecision | None = None
        fail_open_note: str | None = None
        last_provider_name = self._providers[0].name

        for provider in self._providers:
            last_provider_name = provider.name
            try:
                raw = await provider.evaluate(prompt)
            except _BACKEND_ERRORS as exc:
                if not self._config.fail_open:
                    decision = self._fail_closed(
                        ctx,
                        event="guardrail_fail_closed",
                        provider_name=provider.name,
                        violated_policy="guardrail_backend_unavailable",
                        reason=f"Guardrail backend failed with fail_open=False: {exc}",
                        error=exc,
                    )
                    return decision, True
                fail_open_note = f"fail_open ({provider.name}): {exc}"
                self._log.warning(
                    "guardrail_provider_failed_open",
                    provider=provider.name,
                    error=str(exc),
                    **ctx.log_fields(),
                )
                continue

            normalized = ctx.normalize(raw)
            if normalized.verdict is GuardrailVerdict.BLOCK:
                return normalized, False
            last_allow = normalized

        if last_allow is not None:
            return last_allow, False
        return ctx.make_decision(
            verdict=GuardrailVerdict.ALLOW,
            provider=last_provider_name,
            reason=fail_open_note or "fail_open_fallback",
        ), False

    def _handle_unconfigured(self, ctx: PromptCheckContext) -> tuple[GuardrailDecision, bool]:
        """Handle enabled=True with zero registered providers according to fail_open."""
        if self._config.fail_open:
            return ctx.make_decision(
                verdict=GuardrailVerdict.ALLOW,
                provider="pipeline",
                reason="No guardrail providers registered (fail_open=True).",
            ), False
        return self._fail_closed(
            ctx,
            event="guardrail_unconfigured_fail_closed",
            provider_name="pipeline",
            violated_policy="guardrail_unconfigured",
            reason="No guardrail providers registered while fail_open=False.",
        ), True

    def _fail_closed(  # noqa: PLR0913
        self,
        ctx: PromptCheckContext,
        *,
        event: str,
        provider_name: str,
        violated_policy: str,
        reason: str,
        error: BaseException | None = None,
    ) -> GuardrailDecision:
        """Build a fail-closed BLOCK decision and emit a SECURITY_ALERT log."""
        decision = ctx.make_decision(
            verdict=GuardrailVerdict.BLOCK,
            provider=provider_name,
            violated_policy=violated_policy,
            reason=reason,
        )
        extra = (
            {"exception_type": type(error).__name__, "error": str(error)}
            if error is not None
            else {}
        )
        self._log.error(
            event,
            verdict=decision.verdict.value,
            provider=provider_name,
            violated_policy=violated_policy,
            latency_ms=decision.latency_ms,
            alert_category=AlertCategory.SECURITY_ALERT.value,
            **extra,
            **ctx.log_fields(),
        )
        return decision

    def _record_audit_and_log(
        self,
        decision: GuardrailDecision,
        ctx: PromptCheckContext,
        *,
        alert_emitted: bool,
    ) -> None:
        """Persist scrubbed PromptAuditEntry and emit structured component log."""
        self._audit_logger.log_entry(
            PromptAuditEntry(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                project_id=ctx.project_id,
                principal_id=ctx.principal_id,
                verdict=decision.verdict,
                provider=decision.provider,
                violated_policy=decision.violated_policy,
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
            "latency_ms": decision.latency_ms,
            **ctx.log_fields(),
        }
        if decision.verdict is GuardrailVerdict.BLOCK:
            self._log.warning(
                "guardrail_ruling",
                alert_category=AlertCategory.SECURITY_ALERT.value,
                **fields,
            )
            return
        self._log.info("guardrail_ruling", **fields)
