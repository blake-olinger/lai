"""Bounded internal read-tool loop execution for Retrieval mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.chat import (
    ApprovalRequiredEvent,
    Citation,
    MessageRole,
    OrchestratorEvent,
    ProgressEvent,
    TokenUsage,
)
from lai.domain.errors import ToolApprovalRequiredError
from lai.domain.inference import InferenceMessage, InferenceRequest
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.domain.tools import ToolExecutionContext
from lai.observability.logging import get_component_logger, get_request_id
from lai.orchestrator._helpers import add_token_usage, build_inference_tool_defs
from lai.orchestrator._retrieval_search import (
    RetrievalDeps,
    RetrievalPromptParams,
    build_initial_retrieval_messages,
    build_retrieval_citations,
    invoke_single_tool,
    run_hybrid_search,
)
from lai.orchestrator._stream import (
    StreamedInferenceAccumulator,
    TurnOutcome,
    stream_leased_inference,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import Message, TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.inference import ModelProfile

MAX_TOOL_ITERATIONS = 4
_ARG_PREVIEW_CHARS = 300


async def run_retrieval_stream(  # noqa: PLR0913
    deps: RetrievalDeps,
    *,
    turn_input: TurnInput,
    principal: AuthPrincipal,
    model_profile: ModelProfile,
    target_files: Sequence[FileRecord],
    session_system_prompt: str | None = None,
    existing_messages: Sequence[Message] = (),
) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
    """Execute Retrieval search, bounded tool loop, and final streamed answer."""
    log = get_component_logger("orchestrator")
    yield ProgressEvent(stage="retrieval", message="Searching indexed project files.")
    ranked_hits = await run_hybrid_search(
        deps,
        project_id=turn_input.project_id,
        query=turn_input.user_message,
        target_files=target_files,
    )
    tools_enabled = turn_input.enable_tools and deps.tool_executor is not None
    tool_defs = build_inference_tool_defs(deps.tool_registry, enable_tools=tools_enabled)
    max_tokens = (
        turn_input.max_tokens if turn_input.max_tokens is not None else model_profile.max_tokens
    )
    temperature = (
        turn_input.temperature if turn_input.temperature is not None else model_profile.temperature
    )
    conversation = build_initial_retrieval_messages(
        RetrievalPromptParams(
            turn_input=turn_input,
            target_files=target_files,
            ranked_hits=ranked_hits,
            session_system_prompt=session_system_prompt,
            existing_messages=existing_messages,
            model_profile=model_profile,
            tools=tool_defs,
            max_output_tokens=max_tokens,
        )
    )
    tool_ctx = ToolExecutionContext(
        project_id=turn_input.project_id,
        session_id=turn_input.session_id,
        principal_id=principal.principal_id,
        principal_scopes=principal.scopes,
        allowed_file_ids=tuple(f.id for f in target_files) if turn_input.attached_file_ids else (),
        request_id=get_request_id(),
    )
    lease_req = LeaseRequest(
        session_id=turn_input.session_id,
        project_id=turn_input.project_id,
        principal_id=principal.principal_id,
        priority=JobPriority.INTERACTIVE,
    )

    all_content: list[str] = []
    all_thinking: list[str] = []
    tool_citations: list[Citation] = []
    total_usage = TokenUsage()
    needs_final_stream = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        acc = StreamedInferenceAccumulator()
        request = InferenceRequest(
            model=model_profile.name,
            messages=tuple(conversation),
            tools=tool_defs,
            temperature=temperature,
            top_p=model_profile.top_p,
            max_tokens=max_tokens,
            stream=True,
            enable_thinking=turn_input.enable_thinking,
        )
        async for event in stream_leased_inference(
            spooler=deps.spooler,
            inference_provider=deps.inference_provider,
            lease_request=lease_req,
            inference_request=request,
            acc=acc,
        ):
            yield event

        all_content.extend(acc.content_parts)
        all_thinking.extend(acc.thinking_parts)
        total_usage = add_token_usage(total_usage, acc.usage)

        if not acc.tool_calls or deps.tool_executor is None:
            needs_final_stream = False
            break

        conversation.append(
            InferenceMessage(
                role=MessageRole.ASSISTANT,
                content=acc.content,
                tool_calls=tuple(acc.tool_calls),
            )
        )
        for tc in acc.tool_calls:
            yield ProgressEvent(stage="tool_call", message=f"Running tool {tc.name}.")
            try:
                tool_output, citation = await invoke_single_tool(deps.tool_executor, tc, tool_ctx)
            except ToolApprovalRequiredError as exc:
                log.info(
                    "turn_approval_required",
                    approval_id=exc.approval_id,
                    tool_name=exc.tool_name,
                )
                yield ApprovalRequiredEvent(
                    approval_id=exc.approval_id,
                    tool_name=exc.tool_name,
                    arguments_preview=tc.arguments_json[:_ARG_PREVIEW_CHARS],
                    reason=exc.message,
                )
                pause_msg = (
                    f"Tool '{exc.tool_name}' requires approval (approval_id={exc.approval_id})."
                )
                yield TurnOutcome(
                    content=pause_msg,
                    thinking="".join(all_thinking),
                    citations=build_retrieval_citations(
                        pause_msg, target_files, ranked_hits, tool_citations
                    ),
                    usage=total_usage,
                    pending_approval_id=exc.approval_id,
                )
                return

            if citation is not None:
                tool_citations.append(citation)
            conversation.append(
                InferenceMessage(
                    role=MessageRole.TOOL,
                    content=tool_output,
                    name=tc.name,
                    tool_call_id=tc.id,
                )
            )
        needs_final_stream = iteration == MAX_TOOL_ITERATIONS - 1

    if needs_final_stream:
        final_acc = StreamedInferenceAccumulator()
        final_req = InferenceRequest(
            model=model_profile.name,
            messages=tuple(conversation),
            tools=(),
            temperature=temperature,
            top_p=model_profile.top_p,
            max_tokens=max_tokens,
            stream=True,
            enable_thinking=turn_input.enable_thinking,
        )
        async for event in stream_leased_inference(
            spooler=deps.spooler,
            inference_provider=deps.inference_provider,
            lease_request=lease_req,
            inference_request=final_req,
            acc=final_acc,
        ):
            yield event
        all_content.extend(final_acc.content_parts)
        all_thinking.extend(final_acc.thinking_parts)
        total_usage = add_token_usage(total_usage, final_acc.usage)

    final_text = "".join(all_content)
    yield TurnOutcome(
        content=final_text,
        thinking="".join(all_thinking),
        citations=build_retrieval_citations(final_text, target_files, ranked_hits, tool_citations),
        usage=total_usage,
    )
