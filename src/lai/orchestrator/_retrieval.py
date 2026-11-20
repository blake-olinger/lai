"""Bounded dual-plane Retrieval mode orchestration entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.auth import ApiScope
from lai.domain.chat import OrchestratorEvent, ProgressEvent
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.domain.tools import ToolExecutionContext
from lai.observability.logging import get_request_id
from lai.orchestrator._helpers import build_inference_tool_defs
from lai.orchestrator._retrieval_search import (
    RetrievalDeps,
    RetrievalPromptParams,
    build_initial_retrieval_messages,
    run_hybrid_search,
)
from lai.orchestrator._tool_loop import MAX_TOOL_ITERATIONS, ToolLoopParams, run_tool_loop

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import Message, TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.inference import ModelProfile
    from lai.orchestrator._stream import TurnOutcome

__all__ = ["MAX_TOOL_ITERATIONS", "run_retrieval_stream"]


async def run_retrieval_stream(  # noqa: PLR0913
    deps: RetrievalDeps,
    *,
    turn_input: TurnInput,
    principal: AuthPrincipal,
    model_profile: ModelProfile,
    target_files: Sequence[FileRecord],
    session_system_prompt: str | None = None,
    existing_messages: Sequence[Message] = (),
    scan_tool_outputs: bool = False,
) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
    """Execute Retrieval search, bounded tool loop, and final streamed answer."""
    yield ProgressEvent(stage="retrieval", message="Searching indexed project files.")
    ranked_hits = await run_hybrid_search(
        deps,
        project_id=turn_input.project_id,
        query=turn_input.user_message,
        target_files=target_files,
    )
    tools_enabled = turn_input.enable_tools and deps.tool_executor is not None
    tool_defs = build_inference_tool_defs(
        deps.tool_registry,
        enable_tools=tools_enabled,
        client_tools=turn_input.client_tools,
        client_app_id=turn_input.client_app_id,
    )
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
        client_app_id=turn_input.client_app_id,
        client_allowed_tiers=turn_input.client_allowed_tiers,
    )
    client_tools_map = (
        {spec.name: spec for spec in turn_input.client_tools} if turn_input.client_app_id else {}
    )
    lease_req = LeaseRequest(
        session_id=turn_input.session_id,
        project_id=turn_input.project_id,
        principal_id=principal.principal_id,
        priority=JobPriority.INTERACTIVE,
        bypass_principal_quota=principal.has_scope(ApiScope.ADMIN),
    )
    async for item in run_tool_loop(
        ToolLoopParams(
            deps=deps,
            conversation=conversation,
            tool_defs=tool_defs,
            tool_ctx=tool_ctx,
            client_tools_map=client_tools_map,
            lease_req=lease_req,
            model_profile=model_profile,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=turn_input.enable_thinking,
            target_files=target_files,
            ranked_hits=ranked_hits,
            scan_tool_outputs=scan_tool_outputs,
        )
    ):
        yield item
