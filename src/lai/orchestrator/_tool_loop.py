"""Shared bounded dual-plane tool-calling loop for Retrieval and Approval Resume."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lai.domain.chat import (
    ApprovalRequiredEvent,
    Citation,
    ClientToolCallEvent,
    MessageRole,
    OrchestratorEvent,
    ProgressEvent,
    TokenUsage,
)
from lai.domain.errors import (
    ClientToolDispatchError,
    ToolApprovalRequiredError,
)
from lai.domain.inference import InferenceMessage, InferenceRequest, InferenceToolCall
from lai.observability.logging import get_component_logger
from lai.orchestrator._helpers import add_token_usage
from lai.orchestrator._retrieval_search import (
    build_retrieval_citations,
    guard_server_tool_output,
    invoke_single_tool,
)
from lai.orchestrator._stream import (
    StreamedInferenceAccumulator,
    TurnOutcome,
    stream_leased_inference,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from lai.domain.files import FileRecord, ScoredChunk
    from lai.domain.inference import InferenceToolDef, ModelProfile
    from lai.domain.scheduler import LeaseRequest
    from lai.domain.tools import ToolExecutionContext, ToolSpec
    from lai.orchestrator._retrieval_search import RetrievalDeps

MAX_TOOL_ITERATIONS = 4
_ARG_PREVIEW_CHARS = 300
_DEFERRED_CLIENT_TOOL_JSON = json.dumps(
    {
        "success": False,
        "error": "deferred_one_client_tool_per_turn",
        "message": "Only one client tool can execute per pause; re-invoke if still needed.",
    }
)


@dataclass(slots=True)
class ToolLoopParams:
    """Bundled configuration for running a bounded tool-calling inference loop."""

    deps: RetrievalDeps
    conversation: list[InferenceMessage]
    tool_defs: tuple[InferenceToolDef, ...]
    tool_ctx: ToolExecutionContext
    client_tools_map: dict[str, ToolSpec]
    lease_req: LeaseRequest
    model_profile: ModelProfile
    max_tokens: int
    temperature: float
    enable_thinking: bool
    target_files: Sequence[FileRecord]
    ranked_hits: Sequence[ScoredChunk]
    scan_tool_outputs: bool = False


@dataclass(slots=True)
class _LoopState:
    all_content: list[str] = field(default_factory=list)
    all_thinking: list[str] = field(default_factory=list)
    tool_citations: list[Citation] = field(default_factory=list)
    completed_messages: list[InferenceMessage] = field(default_factory=list)
    total_usage: TokenUsage = field(default_factory=TokenUsage)


def _build_server_pause(
    params: ToolLoopParams,
    state: _LoopState,
    exc: ToolApprovalRequiredError,
    tc: InferenceToolCall,
) -> tuple[ApprovalRequiredEvent, TurnOutcome]:
    event = ApprovalRequiredEvent(
        approval_id=exc.approval_id,
        tool_name=exc.tool_name,
        arguments_preview=tc.arguments_json[:_ARG_PREVIEW_CHARS],
        reason=exc.message,
    )
    pause_msg = f"Tool '{exc.tool_name}' requires approval (approval_id={exc.approval_id})."
    outcome = TurnOutcome(
        content=pause_msg,
        thinking="".join(state.all_thinking),
        citations=build_retrieval_citations(
            pause_msg, params.target_files, params.ranked_hits, state.tool_citations
        ),
        usage=state.total_usage,
        pending_approval_id=exc.approval_id,
    )
    return event, outcome


def _build_client_pause(  # noqa: PLR0913
    params: ToolLoopParams,
    state: _LoopState,
    *,
    exc: ClientToolDispatchError,
    tc: InferenceToolCall,
    acc: StreamedInferenceAccumulator,
    batch_tools: Sequence[InferenceMessage],
) -> tuple[ApprovalRequiredEvent | None, ClientToolCallEvent, TurnOutcome]:
    approval_ev = (
        ApprovalRequiredEvent(
            approval_id=exc.approval_id,
            tool_name=exc.tool_name,
            arguments_preview=tc.arguments_json[:_ARG_PREVIEW_CHARS],
            reason=exc.message,
        )
        if exc.requires_approval
        else None
    )
    client_ev = ClientToolCallEvent(
        approval_id=exc.approval_id,
        tool_call_id=exc.tool_call_id,
        tool_name=exc.tool_name,
        arguments_json=exc.arguments_json,
        requires_approval=exc.requires_approval,
        expires_at=exc.expires_at,
    )
    outcome = TurnOutcome(
        content=acc.content,
        thinking="".join(state.all_thinking),
        citations=build_retrieval_citations(
            acc.content, params.target_files, params.ranked_hits, state.tool_citations
        ),
        usage=state.total_usage,
        pending_approval_id=exc.approval_id,
        pending_client_tool_call=client_ev,
        assistant_tool_calls=tuple(acc.tool_calls),
        intermediate_messages=tuple(state.completed_messages),
        post_assistant_tool_messages=tuple(batch_tools),
    )
    return approval_ev, client_ev, outcome


async def _stream_iteration(
    params: ToolLoopParams,
    *,
    tools: tuple[InferenceToolDef, ...],
    acc: StreamedInferenceAccumulator,
) -> AsyncGenerator[OrchestratorEvent]:
    request = InferenceRequest(
        model=params.model_profile.name,
        messages=tuple(params.conversation),
        tools=tools,
        temperature=params.temperature,
        top_p=params.model_profile.top_p,
        max_tokens=params.max_tokens,
        stream=True,
        enable_thinking=params.enable_thinking,
    )
    async for event in stream_leased_inference(
        spooler=params.deps.spooler,
        inference_provider=params.deps.inference_provider,
        lease_request=params.lease_req,
        inference_request=request,
        acc=acc,
    ):
        yield event


def _is_client_tool(params: ToolLoopParams, tool_name: str) -> bool:
    """Return True when tool_name resolves to a client tool rather than a server plugin."""
    if params.deps.tool_registry is not None and params.deps.tool_registry.get(tool_name):
        return False
    return tool_name in params.client_tools_map


def _make_deferred_client_msg(tc: InferenceToolCall) -> InferenceMessage:
    return InferenceMessage(
        role=MessageRole.TOOL,
        content=_DEFERRED_CLIENT_TOOL_JSON,
        name=tc.name,
        tool_call_id=tc.id,
    )


async def _run_client_tools_phase(
    params: ToolLoopParams,
    client_calls: Sequence[InferenceToolCall],
    tool_msgs_by_id: dict[str, InferenceMessage],
) -> AsyncGenerator[OrchestratorEvent | tuple[ClientToolDispatchError, InferenceToolCall]]:
    """Dispatch the first client tool call and defer sibling client tools cleanly."""
    executor = params.deps.tool_executor
    if executor is None:
        return
    log = get_component_logger("orchestrator")
    dispatched: tuple[ClientToolDispatchError, InferenceToolCall] | None = None
    for tc in client_calls:
        if dispatched is not None:
            tool_msgs_by_id[tc.id] = _make_deferred_client_msg(tc)
            continue
        yield ProgressEvent(stage="tool_call", message=f"Running tool {tc.name}.")
        call_ctx = params.tool_ctx.model_copy(update={"tool_call_id": tc.id})
        try:
            raw_output, _ = await invoke_single_tool(
                executor, tc, call_ctx, client_tools=params.client_tools_map
            )
        except ClientToolDispatchError as exc:
            log.info(
                "client_tool_dispatched",
                approval_id=exc.approval_id,
                tool_name=exc.tool_name,
                requires_approval=exc.requires_approval,
            )
            dispatched = (exc, tc)
            continue
        tool_msgs_by_id[tc.id] = InferenceMessage(
            role=MessageRole.TOOL, content=raw_output, name=tc.name, tool_call_id=tc.id
        )
    if dispatched is not None:
        yield dispatched


async def _run_tool_batch(
    params: ToolLoopParams,
    state: _LoopState,
    acc: StreamedInferenceAccumulator,
) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
    log = get_component_logger("orchestrator")
    executor = params.deps.tool_executor
    if executor is None:
        return

    assistant_turn = InferenceMessage(
        role=MessageRole.ASSISTANT,
        content=acc.content,
        tool_calls=tuple(acc.tool_calls),
    )
    params.conversation.append(assistant_turn)
    tool_msgs_by_id: dict[str, InferenceMessage] = {}
    server_calls = [tc for tc in acc.tool_calls if not _is_client_tool(params, tc.name)]
    client_calls = [tc for tc in acc.tool_calls if _is_client_tool(params, tc.name)]

    # Execute all server tools first so a server failure or approval pause never leaks
    # an active SQLite client lease for a sibling client tool in the same batch.
    for tc in server_calls:
        yield ProgressEvent(stage="tool_call", message=f"Running tool {tc.name}.")
        call_ctx = params.tool_ctx.model_copy(update={"tool_call_id": tc.id})
        try:
            raw_output, raw_citation = await invoke_single_tool(
                executor, tc, call_ctx, client_tools=params.client_tools_map
            )
        except ToolApprovalRequiredError as exc:
            log.info("turn_approval_required", approval_id=exc.approval_id, tool_name=exc.tool_name)
            ev, outcome = _build_server_pause(params, state, exc, tc)
            yield ev
            yield outcome
            return
        tool_output, citation = await guard_server_tool_output(
            params.deps,
            params.tool_ctx,
            tc,
            raw_output,
            raw_citation,
            scan_tool_outputs=params.scan_tool_outputs,
        )
        if citation is not None:
            state.tool_citations.append(citation)
        tool_msgs_by_id[tc.id] = InferenceMessage(
            role=MessageRole.TOOL, content=tool_output, name=tc.name, tool_call_id=tc.id
        )

    dispatched_client: tuple[ClientToolDispatchError, InferenceToolCall] | None = None
    async for item in _run_client_tools_phase(params, client_calls, tool_msgs_by_id):
        if isinstance(item, tuple):
            dispatched_client = item
        else:
            yield item

    batch_tools = [tool_msgs_by_id[tc.id] for tc in acc.tool_calls if tc.id in tool_msgs_by_id]
    if dispatched_client is not None:
        dispatch_exc, dispatch_tc = dispatched_client
        appr_ev, client_ev, outcome = _build_client_pause(
            params, state, exc=dispatch_exc, tc=dispatch_tc, acc=acc, batch_tools=batch_tools
        )
        if appr_ev is not None:
            yield appr_ev
        yield client_ev
        yield outcome
        return

    params.conversation.extend(batch_tools)
    state.completed_messages.extend((assistant_turn, *batch_tools))


async def run_tool_loop(params: ToolLoopParams) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
    """Run bounded tool iterations, pausing cleanly on server approval or client dispatch."""
    state = _LoopState()
    needs_final_stream = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        acc = StreamedInferenceAccumulator()
        async for event in _stream_iteration(params, tools=params.tool_defs, acc=acc):
            yield event

        state.all_content.extend(acc.content_parts)
        state.all_thinking.extend(acc.thinking_parts)
        state.total_usage = add_token_usage(state.total_usage, acc.usage)

        if not acc.tool_calls or params.deps.tool_executor is None:
            needs_final_stream = False
            break

        paused = False
        async for item in _run_tool_batch(params, state, acc):
            yield item
            if isinstance(item, TurnOutcome):
                paused = True
        if paused:
            return
        needs_final_stream = iteration == MAX_TOOL_ITERATIONS - 1

    if needs_final_stream:
        final_acc = StreamedInferenceAccumulator()
        async for event in _stream_iteration(params, tools=(), acc=final_acc):
            yield event
        state.all_content.extend(final_acc.content_parts)
        state.all_thinking.extend(final_acc.thinking_parts)
        state.total_usage = add_token_usage(state.total_usage, final_acc.usage)

    final_text = "".join(state.all_content)
    yield TurnOutcome(
        content=final_text,
        thinking="".join(state.all_thinking),
        citations=build_retrieval_citations(
            final_text, params.target_files, params.ranked_hits, state.tool_citations
        ),
        usage=state.total_usage,
    )
