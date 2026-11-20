"""Hybrid search, message assembly, and tool invocation helpers for Retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from lai.domain.chat import Citation, MessageRole
from lai.domain.errors import (
    ClientToolDispatchError,
    GuardrailBlockedError,
    LaiError,
    StorageFailureError,
    ToolApprovalRequiredError,
)
from lai.domain.guardrails import GuardrailDirection, GuardrailVerdict
from lai.domain.inference import InferenceMessage
from lai.ingestion.detector import estimate_tokens
from lai.ingestion.rrf import reciprocal_rank_fusion
from lai.observability.logging import get_component_logger
from lai.orchestrator._helpers import (
    compute_history_budget,
    extract_citations,
    parse_tool_arguments,
    trim_history_to_budget,
)
from lai.orchestrator.prompts import (
    build_system_prompt,
    format_retrieval_context,
    wrap_untrusted_data,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lai.domain.chat import Message, TurnInput
    from lai.domain.files import FileRecord, ScoredChunk
    from lai.domain.inference import InferenceToolCall, InferenceToolDef, ModelProfile
    from lai.domain.tools import ToolExecutionContext, ToolSpec
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.scheduler import InferenceSpoolerPort
    from lai.ports.storage import EmbedderPort, VectorStorePort
    from lai.ports.tools import ToolExecutorPort, ToolRegistryPort

_SEARCH_FETCH_LIMIT = 20
_FUSED_CONTEXT_LIMIT = 12


@dataclass(slots=True)
class RetrievalDeps:
    """Injected ports needed by the Retrieval execution pipeline."""

    file_repo: FileRepositoryPort
    inference_provider: InferenceProviderPort
    spooler: InferenceSpoolerPort
    tool_registry: ToolRegistryPort | None = None
    tool_executor: ToolExecutorPort | None = None
    embedder: EmbedderPort | None = None
    vector_store: VectorStorePort | None = None
    guardrail_pipeline: GuardrailPipelinePort | None = None


@dataclass(slots=True)
class RetrievalPromptParams:
    """Bundled inputs for building initial Retrieval messages."""

    turn_input: TurnInput
    target_files: Sequence[FileRecord]
    ranked_hits: Sequence[ScoredChunk]
    session_system_prompt: str | None
    existing_messages: Sequence[Message]
    model_profile: ModelProfile
    tools: Sequence[InferenceToolDef]
    max_output_tokens: int


def _filter_scoped_hits(
    hits: Sequence[ScoredChunk],
    *,
    project_id: str,
    scoped_file_ids: frozenset[str] | None,
) -> list[ScoredChunk]:
    """Enforce project_id and attached_file_ids scoping on search results."""
    return [
        hit
        for hit in hits
        if hit.chunk.project_id == project_id
        and (scoped_file_ids is None or hit.chunk.file_id in scoped_file_ids)
    ]


async def run_hybrid_search(
    deps: RetrievalDeps,
    *,
    project_id: str,
    query: str,
    target_files: Sequence[FileRecord],
) -> list[ScoredChunk]:
    """Run FTS5 + optional vector similarity fused via Reciprocal Rank Fusion."""
    scoped_list = [record.id for record in target_files] if target_files else None
    scoped_set = frozenset(scoped_list) if scoped_list is not None else None

    raw_fts = await deps.file_repo.search_fts(
        project_id,
        query,
        file_ids=scoped_list,
        limit=_SEARCH_FETCH_LIMIT,
    )
    fts_hits = _filter_scoped_hits(raw_fts, project_id=project_id, scoped_file_ids=scoped_set)

    if deps.embedder is not None and deps.vector_store is not None:
        query_vector = await deps.embedder.embed_query(query)
        raw_vec = await deps.vector_store.search_vectors(
            project_id,
            query_vector,
            file_ids=scoped_list,
            limit=_SEARCH_FETCH_LIMIT,
        )
        vec_hits = _filter_scoped_hits(raw_vec, project_id=project_id, scoped_file_ids=scoped_set)
        return reciprocal_rank_fusion(fts_hits, vec_hits, limit=_FUSED_CONTEXT_LIMIT)

    return sorted(fts_hits, key=lambda item: item.score, reverse=True)[:_FUSED_CONTEXT_LIMIT]


def extract_read_citation(tool_name: str, output_json: str) -> Citation | None:
    """Extract a provenance Citation from a successful read_file_lines JSON output."""
    if tool_name != "read_file_lines":
        return None
    try:
        parsed: object = json.loads(output_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    payload = cast("dict[str, object]", parsed)
    file_id = payload.get("file_id")
    filename = payload.get("filename")
    start_line = payload.get("start_line")
    end_line = payload.get("end_line")
    if (
        isinstance(file_id, str)
        and file_id
        and isinstance(filename, str)
        and filename
        and isinstance(start_line, int)
        and start_line >= 1
    ):
        resolved_end = (
            end_line if isinstance(end_line, int) and end_line >= start_line else start_line
        )
        return Citation(
            file_id=file_id,
            filename=filename,
            start_line=start_line,
            end_line=resolved_end,
        )
    return None


def build_initial_retrieval_messages(params: RetrievalPromptParams) -> list[InferenceMessage]:
    """Assemble system prompt, budgeted history, and retrieved context user turn."""
    filenames_by_id = {record.id: record.filename for record in params.target_files}
    retrieval_ctx = format_retrieval_context(
        params.target_files,
        params.ranked_hits,
        filenames_by_id,
    )
    system_prompt = build_system_prompt(
        params.session_system_prompt,
        has_files=bool(params.target_files),
        has_tools=bool(params.tools),
    )
    user_payload = f"{retrieval_ctx}\n\nUser request:\n{params.turn_input.user_message}"
    reserved = estimate_tokens(system_prompt) + estimate_tokens(user_payload) + 32
    history_budget = compute_history_budget(
        params.model_profile,
        reserved_prompt_tokens=reserved,
        max_output_tokens=params.max_output_tokens,
    )
    history = trim_history_to_budget(params.existing_messages, max_history_tokens=history_budget)
    return [
        InferenceMessage(role=MessageRole.SYSTEM, content=system_prompt),
        *history,
        InferenceMessage(role=MessageRole.USER, content=user_payload),
    ]


async def invoke_single_tool(
    executor: ToolExecutorPort,
    tool_call: InferenceToolCall,
    context: ToolExecutionContext,
    *,
    client_tools: Mapping[str, ToolSpec] | None = None,
) -> tuple[str, Citation | None]:
    """Parse arguments and execute one guarded tool call, formatting recoverable errors."""
    try:
        arguments = parse_tool_arguments(tool_call.arguments_json)
        result = await executor.invoke(
            tool_call.name,
            arguments,
            context,
            client_tools=client_tools,
        )
    except (ToolApprovalRequiredError, ClientToolDispatchError, StorageFailureError):
        raise
    except LaiError as exc:
        if exc.status_code >= 500:  # noqa: PLR2004
            raise
        get_component_logger("orchestrator").warning(
            "retrieval_tool_error",
            tool_name=tool_call.name,
            error=str(exc),
        )
        return json.dumps({"success": False, "error": str(exc)}), None
    except (ValueError, TypeError) as exc:
        get_component_logger("orchestrator").warning(
            "retrieval_tool_error",
            tool_name=tool_call.name,
            error=str(exc),
        )
        return json.dumps({"success": False, "error": str(exc)}), None

    citation = extract_read_citation(tool_call.name, result.output) if result.success else None
    payload = (
        result.output if result.success else json.dumps({"success": False, "error": result.error})
    )
    return payload, citation


def build_retrieval_citations(
    final_text: str,
    target_files: Sequence[FileRecord],
    ranked_hits: Sequence[ScoredChunk],
    tool_citations: Sequence[Citation],
) -> tuple[Citation, ...]:
    """Resolve explicit citations from answer text or fallback to tool/chunk provenance."""
    filenames_by_id = {record.id: record.filename for record in target_files}
    hit_fallbacks = [
        Citation(
            file_id=hit.chunk.file_id,
            filename=filenames_by_id.get(hit.chunk.file_id, hit.chunk.file_id),
            chunk_id=hit.chunk.id,
            start_line=hit.chunk.start_line,
            end_line=hit.chunk.end_line,
        )
        for hit in ranked_hits
    ]
    return extract_citations(
        final_text,
        target_files,
        retrieved_chunks=ranked_hits,
        fallback_citations=(*tool_citations, *hit_fallbacks),
    )


async def guard_server_tool_output(  # noqa: PLR0913
    deps: RetrievalDeps,
    tool_ctx: ToolExecutionContext,
    tc: InferenceToolCall,
    tool_output: str,
    citation: Citation | None,
    *,
    scan_tool_outputs: bool,
) -> tuple[str, Citation | None]:
    """Scan server tool output with TOOL_OUTPUT guardrails and apply untrusted spotlighting."""
    pipeline = deps.guardrail_pipeline
    if scan_tool_outputs and pipeline is not None and tool_output.strip():
        violated_policy: str | None = None
        blocked = False
        try:
            decision = await pipeline.check_prompt(
                tool_output,
                direction=GuardrailDirection.TOOL_OUTPUT,
                request_id=tool_ctx.request_id,
                session_id=tool_ctx.session_id,
                project_id=tool_ctx.project_id,
                principal_id=tool_ctx.principal_id,
            )
            if decision.verdict is GuardrailVerdict.BLOCK:
                blocked = True
                violated_policy = decision.violated_policy
        except GuardrailBlockedError as exc:
            blocked = True
            violated_policy = exc.violated_policy
        if blocked:
            get_component_logger("orchestrator").warning(
                "server_tool_output_guardrail_blocked",
                tool_name=tc.name,
                violated_policy=violated_policy,
            )
            safe_error = json.dumps(
                {
                    "success": False,
                    "error": "tool_output_blocked_by_guardrail",
                    "violated_policy": violated_policy,
                }
            )
            return safe_error, None
    return wrap_untrusted_data(tool_output, source=f"tool:{tc.name}"), citation
