"""Buffered and audit-only output egress guardrail gate for orchestrator streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.chat import (
    ContentDeltaEvent,
    ContextMode,
    Message,
    MessageRole,
    OrchestratorEvent,
    ThinkingDeltaEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnOutput,
)
from lai.domain.errors import GuardrailBlockedError
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailDirection,
    GuardrailOutputMode,
    GuardrailVerdict,
)
from lai.observability.logging import get_component_logger, get_request_id
from lai.orchestrator._stream import TurnOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from lai.config.settings import LaiSettings
    from lai.ports.guardrails import GuardrailPipelinePort


@dataclass(frozen=True, slots=True)
class EgressContext:
    """Correlation and configuration context for one turn's output egress scan."""

    settings: LaiSettings
    guardrail_pipeline: GuardrailPipelinePort | None
    user_prompt: str
    session_id: str
    project_id: str
    principal_id: str


@dataclass(frozen=True, slots=True)
class EgressGateResult:
    """Result of evaluating model output against configured egress guardrails."""

    blocked: bool
    refusal_text: str = ""
    violated_policy: str | None = None


def should_buffer_egress(settings: LaiSettings) -> bool:
    """Return True when content/thinking deltas must be buffered pending ShieldGemma egress."""
    return (
        settings.guardrails.enabled
        and settings.guardrails.output_mode is GuardrailOutputMode.BUFFERED
    )


def is_buffered_delta(event: OrchestratorEvent) -> bool:
    """Return True for content/thinking deltas held back in BUFFERED egress mode."""
    return isinstance(event, (ContentDeltaEvent, ThinkingDeltaEvent))


def combine_egress_text(content: str, thinking: str) -> str:
    """Combine model thinking and content for unified egress safety evaluation."""
    parts = [part.strip() for part in (thinking, content) if part and part.strip()]
    return "\n\n".join(parts)


async def evaluate_turn_egress(
    ctx: EgressContext,
    outcome: TurnOutcome,
) -> EgressGateResult:
    """Evaluate combined model output according to BUFFERED, AUDIT_ONLY, or OFF policy."""
    cfg = ctx.settings.guardrails
    if (
        ctx.guardrail_pipeline is None
        or not cfg.enabled
        or cfg.output_mode is GuardrailOutputMode.OFF
    ):
        return EgressGateResult(blocked=False)

    combined = combine_egress_text(outcome.content, outcome.thinking)
    if not combined:
        return EgressGateResult(blocked=False)

    decision = await ctx.guardrail_pipeline.check_response(
        ctx.user_prompt,
        combined,
        request_id=get_request_id(),
        session_id=ctx.session_id,
        project_id=ctx.project_id,
        principal_id=ctx.principal_id,
    )
    if cfg.output_mode is GuardrailOutputMode.AUDIT_ONLY:
        return EgressGateResult(blocked=False)

    if decision.verdict is GuardrailVerdict.BLOCK:
        get_component_logger("orchestrator").warning(
            "turn_egress_guardrail_blocked",
            violated_policy=decision.violated_policy,
            block_mode=cfg.block_mode.value,
            direction=GuardrailDirection.EGRESS.value,
        )
        if cfg.block_mode is GuardrailBlockMode.HTTP_ERROR:
            raise GuardrailBlockedError(
                cfg.refusal_text,
                violated_policy=decision.violated_policy,
                details={
                    "provider": decision.provider,
                    "direction": GuardrailDirection.EGRESS.value,
                },
            )
        return EgressGateResult(
            blocked=True,
            refusal_text=cfg.refusal_text,
            violated_policy=decision.violated_policy,
        )
    return EgressGateResult(blocked=False)


async def finalize_stream_with_egress(
    *,
    stream: AsyncGenerator[OrchestratorEvent | TurnOutcome],
    ctx: EgressContext,
    context_mode: ContextMode,
    persist_assistant: Callable[[Message, TurnOutcome], Awaitable[None]],
) -> AsyncGenerator[OrchestratorEvent]:
    """Buffer deltas when required, run egress gate, persist safe output, and emit completion."""
    buffer_deltas = should_buffer_egress(ctx.settings)
    buffered_deltas: list[OrchestratorEvent] = []
    outcome: TurnOutcome | None = None

    async for item in stream:
        if isinstance(item, TurnOutcome):
            outcome = item
        elif buffer_deltas and is_buffered_delta(item):
            buffered_deltas.append(item)
        else:
            yield item

    resolved = outcome or TurnOutcome(content="", thinking="", citations=(), usage=TokenUsage())
    egress = await evaluate_turn_egress(ctx, resolved)
    if egress.blocked:
        refusal_msg = Message(
            session_id=ctx.session_id,
            project_id=ctx.project_id,
            role=MessageRole.ASSISTANT,
            content=egress.refusal_text,
            usage=resolved.usage,
        )
        await persist_assistant(refusal_msg, resolved)
        yield ContentDeltaEvent(delta=egress.refusal_text)
        yield TurnCompletedEvent(
            output=TurnOutput(
                session_id=ctx.session_id,
                message=refusal_msg,
                context_mode_used=context_mode,
                citations=(),
                usage=resolved.usage,
                guardrail_blocked=True,
                guardrail_policy=egress.violated_policy,
                guardrail_direction=GuardrailDirection.EGRESS.value,
            )
        )
        return

    for delta_event in buffered_deltas:
        yield delta_event

    assistant_msg = Message(
        session_id=ctx.session_id,
        project_id=ctx.project_id,
        role=MessageRole.ASSISTANT,
        content=resolved.content,
        thinking=resolved.thinking or None,
        citations=resolved.citations,
        tool_calls=(
            (resolved.pending_tool_call,) if resolved.pending_tool_call is not None else ()
        ),
        usage=resolved.usage,
    )
    await persist_assistant(assistant_msg, resolved)
    yield TurnCompletedEvent(
        output=TurnOutput(
            session_id=ctx.session_id,
            message=assistant_msg,
            context_mode_used=context_mode,
            citations=resolved.citations,
            usage=resolved.usage,
            pending_approval_id=resolved.pending_approval_id,
            pending_client_tool_call=resolved.pending_client_tool_call,
        )
    )
