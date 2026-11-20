"""Buffered and audit-only output egress guardrail gate for orchestrator streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.chat import (
    ApprovalRequiredEvent,
    ClientToolCallEvent,
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
from lai.domain.inference import InferenceToolCall
from lai.observability.logging import get_component_logger, get_request_id
from lai.orchestrator._stream import TurnOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

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
    revoke_approval: Callable[[str, str], Awaitable[None]] | None = None


@dataclass(frozen=True, slots=True)
class EgressGateResult:
    """Result of evaluating model output against configured egress guardrails."""

    blocked: bool
    refusal_text: str = ""
    violated_policy: str | None = None


def should_buffer_egress(settings: LaiSettings) -> bool:
    """Return True when content/thinking/tool events must be buffered pending egress scan."""
    return (
        settings.guardrails.enabled
        and settings.guardrails.output_mode is GuardrailOutputMode.BUFFERED
    )


def is_buffered_delta(event: OrchestratorEvent) -> bool:
    """Return True for content, thinking, client-tool, or approval events held back in BUFFERED."""
    return isinstance(
        event,
        (ContentDeltaEvent, ThinkingDeltaEvent, ClientToolCallEvent, ApprovalRequiredEvent),
    )


def combine_egress_text(
    content: str,
    thinking: str,
    tool_calls: Sequence[InferenceToolCall] = (),
) -> str:
    """Combine model thinking, content, and serialized tool calls for egress safety evaluation."""
    parts = [part.strip() for part in (thinking, content) if part and part.strip()]
    seen_calls: set[tuple[str, str, str]] = set()
    for tc in tool_calls:
        key = (tc.id, tc.name, tc.arguments_json)
        if key in seen_calls:
            continue
        seen_calls.add(key)
        parts.append(f"tool_call {tc.name}: {tc.arguments_json}")
    return "\n\n".join(parts)


def _collect_outcome_tool_calls(outcome: TurnOutcome) -> tuple[InferenceToolCall, ...]:
    """Gather all tool calls proposed by the model across all turn iterations."""
    collected: list[InferenceToolCall] = []
    for inter_msg in outcome.intermediate_messages:
        if inter_msg.role is MessageRole.ASSISTANT and inter_msg.tool_calls:
            collected.extend(inter_msg.tool_calls)
    collected.extend(outcome.assistant_tool_calls)
    if outcome.pending_tool_call is not None:
        collected.append(outcome.pending_tool_call)
    client_tc = outcome.pending_client_tool_call
    if client_tc is not None:
        collected.append(
            InferenceToolCall(
                id=client_tc.tool_call_id,
                name=client_tc.tool_name,
                arguments_json=client_tc.arguments_json,
            )
        )
    return tuple(collected)


def _collect_outcome_content(
    outcome: TurnOutcome,
    extra_buffered_content: Sequence[str] = (),
) -> str:
    """Gather final, intermediate assistant, and buffered content so no text escapes egress."""
    parts = [
        inter_msg.content.strip()
        for inter_msg in outcome.intermediate_messages
        if inter_msg.role is MessageRole.ASSISTANT and inter_msg.content.strip()
    ]
    if outcome.content.strip():
        parts.append(outcome.content.strip())
    buffered_joined = "".join(extra_buffered_content).strip()
    if buffered_joined and buffered_joined not in "\n\n".join(parts):
        parts.append(buffered_joined)
    return "\n\n".join(dict.fromkeys(parts))


def _collect_outcome_thinking(
    outcome: TurnOutcome,
    extra_buffered_thinking: Sequence[str] = (),
) -> str:
    """Gather outcome thinking plus any buffered ThinkingDeltaEvent text."""
    parts: list[str] = []
    if outcome.thinking.strip():
        parts.append(outcome.thinking.strip())
    buffered_joined = "".join(extra_buffered_thinking).strip()
    if buffered_joined and buffered_joined not in "\n\n".join(parts):
        parts.append(buffered_joined)
    return "\n\n".join(dict.fromkeys(parts))


async def _revoke_pending_approvals(ctx: EgressContext, outcome: TurnOutcome) -> None:
    """Revoke any pending or leased tool approvals created during a turn blocked on egress."""
    if ctx.revoke_approval is None:
        return
    approval_ids: list[str] = []
    if outcome.pending_approval_id:
        approval_ids.append(outcome.pending_approval_id)
    if (
        outcome.pending_client_tool_call is not None
        and outcome.pending_client_tool_call.approval_id
    ):
        approval_ids.append(outcome.pending_client_tool_call.approval_id)
    for approval_id in dict.fromkeys(approval_ids):
        await ctx.revoke_approval(approval_id, "blocked_by_egress_guardrail")


async def evaluate_turn_egress(
    ctx: EgressContext,
    outcome: TurnOutcome,
    *,
    extra_buffered_content: Sequence[str] = (),
    extra_buffered_thinking: Sequence[str] = (),
) -> EgressGateResult:
    """Evaluate combined model output according to BUFFERED, AUDIT_ONLY, or OFF policy."""
    cfg = ctx.settings.guardrails
    if (
        ctx.guardrail_pipeline is None
        or not cfg.enabled
        or cfg.output_mode is GuardrailOutputMode.OFF
    ):
        return EgressGateResult(blocked=False)

    combined = combine_egress_text(
        _collect_outcome_content(outcome, extra_buffered_content),
        _collect_outcome_thinking(outcome, extra_buffered_thinking),
        _collect_outcome_tool_calls(outcome),
    )
    if not combined:
        return EgressGateResult(blocked=False)

    try:
        decision = await ctx.guardrail_pipeline.check_response(
            ctx.user_prompt,
            combined,
            request_id=get_request_id(),
            session_id=ctx.session_id,
            project_id=ctx.project_id,
            principal_id=ctx.principal_id,
        )
    except BaseException:
        await _revoke_pending_approvals(ctx, outcome)
        raise

    if cfg.output_mode is GuardrailOutputMode.AUDIT_ONLY:
        return EgressGateResult(blocked=False)

    if decision.verdict is GuardrailVerdict.BLOCK:
        await _revoke_pending_approvals(ctx, outcome)
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
    buffered_content = [ev.delta for ev in buffered_deltas if isinstance(ev, ContentDeltaEvent)]
    buffered_thinking = [ev.delta for ev in buffered_deltas if isinstance(ev, ThinkingDeltaEvent)]
    egress = await evaluate_turn_egress(
        ctx,
        resolved,
        extra_buffered_content=buffered_content,
        extra_buffered_thinking=buffered_thinking,
    )
    if egress.blocked:
        refusal_msg = Message(
            session_id=ctx.session_id,
            project_id=ctx.project_id,
            role=MessageRole.ASSISTANT,
            content=egress.refusal_text,
            usage=resolved.usage,
        )
        blocked_outcome = TurnOutcome(
            content="",
            thinking="",
            citations=(),
            usage=resolved.usage,
        )
        await persist_assistant(refusal_msg, blocked_outcome)
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

    resolved_tool_calls = resolved.assistant_tool_calls or (
        (resolved.pending_tool_call,) if resolved.pending_tool_call is not None else ()
    )
    assistant_msg = Message(
        session_id=ctx.session_id,
        project_id=ctx.project_id,
        role=MessageRole.ASSISTANT,
        content=resolved.content,
        thinking=resolved.thinking or None,
        citations=resolved.citations,
        tool_calls=resolved_tool_calls,
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
