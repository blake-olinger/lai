"""Client-tool approval resume continuation flow using JobPriority.APPROVAL_RESUME."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.chat import (
    ContextMode,
    InferenceToolCall,
    Message,
    MessageRole,
    OrchestratorEvent,
    ProgressEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnOutput,
)
from lai.domain.errors import AuthorizationError, NotFoundError
from lai.domain.inference import InferenceMessage
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.domain.tools import PermissionTier, ToolExecutionContext, ToolExecutionTarget
from lai.ingestion.detector import estimate_tokens
from lai.observability.logging import bind_log_context, get_component_logger, get_request_id
from lai.orchestrator._helpers import (
    build_inference_tool_defs,
    build_trimmed_history,
    resolve_model_profile,
    resolve_target_files,
)
from lai.orchestrator._stream import TurnOutcome
from lai.orchestrator._tool_loop import ToolLoopParams, run_tool_loop
from lai.orchestrator.prompts import build_system_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

    from lai.config.settings import LaiSettings
    from lai.domain.auth import AuthPrincipal
    from lai.domain.inference import ModelProfile
    from lai.domain.tools import ToolApprovalRecord, ToolResult, ToolSpec
    from lai.orchestrator._retrieval_search import RetrievalDeps
    from lai.ports.repositories import ClientAppRepositoryPort, SessionRepositoryPort
    from lai.ports.scheduler import SessionGatePort

SessionClientToolsEntry = tuple[str, tuple["ToolSpec", ...], tuple[PermissionTier, ...]]


@dataclass(slots=True)
class ResumeDeps:
    """Dependencies required to resume a session turn after client tool execution."""

    settings: LaiSettings
    session_repo: SessionRepositoryPort
    session_gate: SessionGatePort
    retrieval: RetrievalDeps
    client_app_repo: ClientAppRepositoryPort | None = None
    default_profile: ModelProfile | None = None


def _format_tool_result_content(tool_result: ToolResult) -> str:
    if tool_result.success:
        return tool_result.output
    payload: dict[str, object] = {
        "success": False,
        "error": tool_result.error or "tool execution failed",
    }
    if tool_result.output:
        payload["output"] = tool_result.output
    return json.dumps(payload)


async def _resolve_resume_client_tools(
    deps: ResumeDeps,
    approval: ToolApprovalRecord,
    session_tools_cache: Mapping[str, SessionClientToolsEntry],
) -> tuple[tuple[ToolSpec, ...], tuple[PermissionTier, ...]]:
    """Resolve active client tools and allowed tiers for continuation turns."""
    client_app_id = approval.client_app_id
    if not client_app_id:
        return (), ()

    cached = session_tools_cache.get(approval.session_id)
    cached_for_app = cached if cached is not None and cached[0] == client_app_id else None

    if deps.client_app_repo is None:
        return (cached_for_app[1], cached_for_app[2]) if cached_for_app is not None else ((), ())

    app = await deps.client_app_repo.get_by_id(client_app_id)
    if app is None or not app.is_active():
        return (), ()
    if app.allowed_project_ids and approval.project_id not in app.allowed_project_ids:
        return (), ()

    server_names: set[str] = (
        {spec.name for spec in deps.retrieval.tool_registry.list_specs()}
        if deps.retrieval.tool_registry is not None
        else set()
    )
    merged: dict[str, ToolSpec] = {}
    candidates = (*app.registered_tools, *(cached_for_app[1] if cached_for_app is not None else ()))
    for spec in candidates:
        if spec.name in server_names or spec.permission_tier not in app.allowed_tiers:
            continue
        merged[spec.name] = spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT})
    return tuple(merged.values()), app.allowed_tiers


async def _append_resume_messages(
    session_repo: SessionRepositoryPort,
    *,
    session_id: str,
    project_id: str,
    approval: ToolApprovalRecord,
    tool_result: ToolResult,
) -> list[Message]:
    existing = await session_repo.list_messages(session_id, limit=200)
    effective_call_id = approval.tool_call_id or approval.id
    tail = existing[-1] if existing else None
    has_tail_call = (
        tail is not None
        and tail.role is MessageRole.ASSISTANT
        and any(tc.id == effective_call_id for tc in tail.tool_calls)
    )
    if not has_tail_call:
        call_msg = Message(
            session_id=session_id,
            project_id=project_id,
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(
                InferenceToolCall(
                    id=effective_call_id,
                    name=approval.tool_name,
                    arguments_json=approval.arguments_json,
                ),
            ),
        )
        await session_repo.add_message(call_msg)
        existing.append(call_msg)

    tool_msg = Message(
        session_id=session_id,
        project_id=project_id,
        role=MessageRole.TOOL,
        content=_format_tool_result_content(tool_result),
        tool_call_id=effective_call_id,
        tool_name=approval.tool_name,
    )
    await session_repo.add_message(tool_msg)
    existing.append(tool_msg)
    return existing


async def run_client_tool_resume_stream(
    deps: ResumeDeps,
    approval: ToolApprovalRecord,
    tool_result: ToolResult,
    principal: AuthPrincipal,
    *,
    session_tools_cache: Mapping[str, SessionClientToolsEntry],
) -> AsyncGenerator[OrchestratorEvent]:
    """Acquire session lock, append TOOL result, lease APPROVAL_RESUME, and stream answer."""
    log = get_component_logger("orchestrator")
    with bind_log_context(
        session_id=approval.session_id,
        project_id=approval.project_id,
        principal_id=principal.principal_id,
        component="orchestrator",
    ):
        async with deps.session_gate.acquire(approval.session_id):
            session = await deps.session_repo.get_session(approval.session_id)
            if session is None:
                msg = f"Session {approval.session_id!r} not found."
                raise NotFoundError(msg, details={"session_id": approval.session_id})
            if session.project_id != approval.project_id:
                msg = f"Session {session.id!r} does not belong to project {approval.project_id!r}."
                raise AuthorizationError(msg, details={"session_id": session.id})

            existing = await _append_resume_messages(
                deps.session_repo,
                session_id=session.id,
                project_id=session.project_id,
                approval=approval,
                tool_result=tool_result,
            )
            client_tools, client_allowed_tiers = await _resolve_resume_client_tools(
                deps,
                approval,
                session_tools_cache,
            )
            target_files = await resolve_target_files(
                deps.retrieval.file_repo,
                session.project_id,
                approval.allowed_file_ids,
            )
            profile = resolve_model_profile(deps.settings, session, deps.default_profile)
            tool_defs = build_inference_tool_defs(
                deps.retrieval.tool_registry,
                enable_tools=deps.retrieval.tool_executor is not None,
                client_tools=client_tools,
                client_app_id=approval.client_app_id,
            )
            system_prompt = build_system_prompt(
                session.system_prompt,
                has_files=bool(target_files),
                has_tools=bool(tool_defs),
            )
            reserved = estimate_tokens(system_prompt) + 64
            history = build_trimmed_history(
                existing,
                profile=profile,
                reserved_prompt_tokens=reserved,
                max_output_tokens=profile.max_tokens,
            )
            conversation = [
                InferenceMessage(role=MessageRole.SYSTEM, content=system_prompt),
                *history,
            ]
            tool_ctx = ToolExecutionContext(
                project_id=session.project_id,
                session_id=session.id,
                principal_id=principal.principal_id,
                principal_scopes=principal.scopes,
                allowed_file_ids=approval.allowed_file_ids,
                request_id=get_request_id(),
                client_app_id=approval.client_app_id,
                client_allowed_tiers=client_allowed_tiers,
            )
            client_tools_map = (
                {spec.name: spec for spec in client_tools} if approval.client_app_id else {}
            )
            lease_req = LeaseRequest(
                session_id=session.id,
                project_id=session.project_id,
                principal_id=principal.principal_id,
                priority=JobPriority.APPROVAL_RESUME,
            )

            log.info(
                "client_tool_resume_started",
                approval_id=approval.id,
                tool_name=approval.tool_name,
            )
            yield ProgressEvent(
                stage="approval_resume",
                message="Resuming turn after client tool execution.",
            )
            outcome: TurnOutcome | None = None
            async for item in run_tool_loop(
                ToolLoopParams(
                    deps=deps.retrieval,
                    conversation=conversation,
                    tool_defs=tool_defs,
                    tool_ctx=tool_ctx,
                    client_tools_map=client_tools_map,
                    lease_req=lease_req,
                    model_profile=profile,
                    max_tokens=profile.max_tokens,
                    temperature=profile.temperature,
                    enable_thinking=True,
                    target_files=target_files,
                    ranked_hits=(),
                )
            ):
                if isinstance(item, TurnOutcome):
                    outcome = item
                else:
                    yield item

            resolved = outcome or TurnOutcome(
                content="",
                thinking="",
                citations=(),
                usage=TokenUsage(),
            )
            assistant_msg = Message(
                session_id=session.id,
                project_id=session.project_id,
                role=MessageRole.ASSISTANT,
                content=resolved.content,
                thinking=resolved.thinking or None,
                citations=resolved.citations,
                tool_calls=(
                    (resolved.pending_tool_call,) if resolved.pending_tool_call is not None else ()
                ),
                usage=resolved.usage,
            )
            for inter_msg in resolved.intermediate_messages:
                await deps.session_repo.add_message(
                    Message(
                        session_id=session.id,
                        project_id=session.project_id,
                        role=inter_msg.role,
                        content=inter_msg.content,
                        tool_calls=inter_msg.tool_calls,
                        tool_call_id=inter_msg.tool_call_id,
                        tool_name=inter_msg.name,
                        created_at=assistant_msg.created_at,
                    )
                )
            await deps.session_repo.add_message(assistant_msg)

            yield TurnCompletedEvent(
                output=TurnOutput(
                    session_id=session.id,
                    message=assistant_msg,
                    context_mode_used=ContextMode.RETRIEVAL,
                    citations=resolved.citations,
                    usage=resolved.usage,
                    pending_approval_id=resolved.pending_approval_id,
                    pending_client_tool_call=resolved.pending_client_tool_call,
                )
            )
