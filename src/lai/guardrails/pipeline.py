"""Composite ingress and egress guardrail pipeline with fail-open/closed, audit, and analytics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from lai.domain.errors import GuardrailBlockedError, InferenceBackendError, ValidationFailedError
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailDirection,
    GuardrailOutputMode,
    GuardrailVerdict,
    validate_guardrail_runtime_update,
)
from lai.domain.observability import AlertCategory
from lai.guardrails._context import (
    PromptCheckContext,
    record_audit_and_log,
    record_guardrail_analytics,
)
from lai.guardrails.passthrough import AllowAllGuardrailProvider
from lai.guardrails.shieldgemma import ShieldGemmaCpuProvider
from lai.observability.logging import get_component_logger
from lai.observability.scrubber import PromptAuditLogger
from lai.ports.guardrails import GuardrailPipelinePort

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from types import TracebackType

    from lai.ports.analytics import UsageAnalyticsPort
    from lai.ports.guardrails import GuardrailProviderPort

_DEFAULT_AUDIT_RETENTION_DAYS = 30
_BACKEND_ERRORS = (InferenceBackendError, httpx.HTTPError, TimeoutError, ValueError)


class CompositeGuardrailPipeline(GuardrailPipelinePort):
    """Coordinates guardrail providers, boot locks, fail-open/closed, auditing, and analytics."""

    def __init__(  # noqa: PLR0913
        self,
        config: GuardrailConfig | None = None,
        *,
        providers: Sequence[GuardrailProviderPort] | None = None,
        audit_logger: PromptAuditLogger | None = None,
        analytics_repo: UsageAnalyticsPort | None = None,
        log_dir: Path | None = None,
        retention_days: int = _DEFAULT_AUDIT_RETENTION_DAYS,
        compress_rotated: bool = True,
    ) -> None:
        self._config = config or GuardrailConfig()
        self._boot_config = self._config
        self._analytics_repo = analytics_repo
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

    @property
    def boot_config(self) -> GuardrailConfig:
        """Return the immutable startup GuardrailConfig governing runtime lock policies."""
        return self._boot_config

    def register_provider(self, provider: GuardrailProviderPort) -> None:
        """Append a guardrail provider to the sequential evaluation chain."""
        self._providers.append(provider)

    def reconfigure(
        self,
        config: GuardrailConfig,
        *,
        providers: Sequence[GuardrailProviderPort] | None = None,
    ) -> None:
        """Hot-swap guardrail policy in place after enforcing boot-only security locks."""
        try:
            validate_guardrail_runtime_update(self._boot_config, config)
        except ValidationFailedError as exc:
            self._log.error(
                "guardrail_runtime_lock_rejected",
                reason=exc.message,
                field=exc.details.get("field", ""),
                alert_category=AlertCategory.SECURITY_ALERT.value,
            )
            raise

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

    async def check_prompt(  # noqa: PLR0913
        self,
        prompt: str,
        *,
        direction: GuardrailDirection = GuardrailDirection.INGRESS,
        request_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> GuardrailDecision:
        """Evaluate a prompt or tool output, write audit/analytics, and enforce policy."""
        ctx = PromptCheckContext.build(
            prompt,
            max_chars=self._config.max_logged_prompt_chars,
            direction=direction,
            request_id=request_id,
            session_id=session_id,
            project_id=project_id,
            principal_id=principal_id,
        )
        decision, fail_closed = await self._evaluate_chain(
            ctx,
            passthrough_eval=lambda p: p.evaluate(prompt),
            provider_eval=lambda p: p.evaluate(prompt),
            enabled=self._config.enabled,
        )
        record_audit_and_log(
            self._audit_logger, self._log, decision, ctx, alert_emitted=fail_closed
        )
        await record_guardrail_analytics(
            self._analytics_repo, self._log, decision, fail_closed=fail_closed
        )
        if (
            decision.verdict is GuardrailVerdict.BLOCK
            and self._config.block_mode is GuardrailBlockMode.HTTP_ERROR
        ):
            raise GuardrailBlockedError(
                self._config.refusal_text,
                violated_policy=decision.violated_policy,
                details={
                    "provider": decision.provider,
                    "prompt_sha256": ctx.prompt_sha256,
                    "direction": direction.value,
                },
            )
        return decision

    async def check_response(  # noqa: PLR0913
        self,
        prompt: str,
        response: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> GuardrailDecision:
        """Evaluate a model response according to output_mode (BUFFERED, AUDIT_ONLY, or OFF)."""
        ctx = PromptCheckContext.build(
            response,
            max_chars=self._config.max_logged_prompt_chars,
            direction=GuardrailDirection.EGRESS,
            request_id=request_id,
            session_id=session_id,
            project_id=project_id,
            principal_id=principal_id,
        )
        egress_active = (
            self._config.enabled and self._config.output_mode is not GuardrailOutputMode.OFF
        )
        decision, fail_closed = await self._evaluate_chain(
            ctx,
            passthrough_eval=lambda p: p.evaluate_response(prompt, response),
            provider_eval=lambda p: p.evaluate_response(prompt, response),
            enabled=egress_active,
        )
        record_audit_and_log(
            self._audit_logger, self._log, decision, ctx, alert_emitted=fail_closed
        )
        await record_guardrail_analytics(
            self._analytics_repo, self._log, decision, fail_closed=fail_closed
        )
        if (
            self._config.output_mode is GuardrailOutputMode.BUFFERED
            and decision.verdict is GuardrailVerdict.BLOCK
            and self._config.block_mode is GuardrailBlockMode.HTTP_ERROR
        ):
            raise GuardrailBlockedError(
                self._config.refusal_text,
                violated_policy=decision.violated_policy,
                details={
                    "provider": decision.provider,
                    "prompt_sha256": ctx.prompt_sha256,
                    "direction": GuardrailDirection.EGRESS.value,
                },
            )
        return decision

    async def _evaluate_chain(
        self,
        ctx: PromptCheckContext,
        *,
        passthrough_eval: Callable[[AllowAllGuardrailProvider], Awaitable[GuardrailDecision]],
        provider_eval: Callable[[GuardrailProviderPort], Awaitable[GuardrailDecision]],
        enabled: bool,
    ) -> tuple[GuardrailDecision, bool]:
        """Run enabled providers sequentially with fail-open/closed error handling."""
        if not enabled:
            passthrough = AllowAllGuardrailProvider(
                max_logged_prompt_chars=self._config.max_logged_prompt_chars,
            )
            return ctx.normalize(await passthrough_eval(passthrough)), False

        if not self._providers:
            return self._handle_unconfigured(ctx)

        last_allow: GuardrailDecision | None = None
        fail_open_note: str | None = None
        last_provider_name = self._providers[0].name

        for provider in self._providers:
            last_provider_name = provider.name
            try:
                raw = await provider_eval(provider)
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
