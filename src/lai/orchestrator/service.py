"""Main TurnOrchestrator implementing OrchestratorPort across all context modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.chat import (
    ContextMode,
    Message,
    MessageRole,
    OrchestratorEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnOutput,
)
from lai.domain.errors import InferenceBackendError
from lai.ingestion.detector import estimate_tokens
from lai.observability.logging import bind_log_context, get_component_logger
from lai.orchestrator._direct import DirectDeps, run_direct_stream
from lai.orchestrator._helpers import build_trimmed_history, persist_turn_if_session
from lai.orchestrator._resume import (
    ResumeDeps,
    SessionClientToolsEntry,
    run_client_tool_resume_stream,
)
from lai.orchestrator._retrieval import run_retrieval_stream
from lai.orchestrator._retrieval_search import RetrievalDeps
from lai.orchestrator._stream import TurnOutcome
from lai.orchestrator._turn_prep import (
    PreparedTurnContext,
    TurnPrepDeps,
    handle_guardrail_block,
    load_authorized_session,
    prepare_allowed_turn,
    scan_turn_guardrails,
)
from lai.orchestrator.aggregator import MapReduceAggregator
from lai.ports.orchestrator import OrchestratorPort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from lai.config.settings import LaiSettings
    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import Session, TurnInput
    from lai.domain.inference import ModelProfile
    from lai.domain.tools import ToolApprovalRecord, ToolResult
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.repositories import (
        ClientAppRepositoryPort,
        FileRepositoryPort,
        SessionRepositoryPort,
    )
    from lai.ports.scheduler import InferenceSpoolerPort, SessionGatePort
    from lai.ports.storage import BlobStorePort, EmbedderPort, VectorStorePort
    from lai.ports.tools import ToolExecutorPort, ToolRegistryPort

_MAX_CACHED_SESSION_TOOLS = 1024


class TurnOrchestrator(OrchestratorPort):
    """Coordinates pre-flight guardrails, session locks, routing, tools, and GPU spooler."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        settings: LaiSettings,
        session_repo: SessionRepositoryPort,
        file_repo: FileRepositoryPort,
        blob_store: BlobStorePort,
        inference_provider: InferenceProviderPort,
        spooler: InferenceSpoolerPort,
        session_gate: SessionGatePort,
        guardrail_pipeline: GuardrailPipelinePort,
        tool_registry: ToolRegistryPort | None = None,
        tool_executor: ToolExecutorPort | None = None,
        embedder: EmbedderPort | None = None,
        vector_store: VectorStorePort | None = None,
        model_profile: ModelProfile | None = None,
        client_app_repo: ClientAppRepositoryPort | None = None,
    ) -> None:
        self._settings = settings
        self._session_repo = session_repo
        self._session_gate = session_gate
        self._guardrail_pipeline = guardrail_pipeline
        self._log = get_component_logger("orchestrator")
        self._prep_deps = TurnPrepDeps(
            settings=settings,
            session_repo=session_repo,
            file_repo=file_repo,
            default_profile=model_profile,
        )
        self._direct_deps = DirectDeps(
            file_repo=file_repo,
            blob_store=blob_store,
            inference_provider=inference_provider,
            spooler=spooler,
        )
        self._retrieval_deps = RetrievalDeps(
            file_repo=file_repo,
            inference_provider=inference_provider,
            spooler=spooler,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            embedder=embedder,
            vector_store=vector_store,
        )
        self._resume_deps = ResumeDeps(
            settings=settings,
            session_repo=session_repo,
            session_gate=session_gate,
            retrieval=self._retrieval_deps,
            client_app_repo=client_app_repo,
            default_profile=model_profile,
        )
        self._session_client_tools: dict[str, SessionClientToolsEntry] = {}
        self._aggregator = MapReduceAggregator(
            blob_store=blob_store,
            inference_provider=inference_provider,
            spooler=spooler,
            file_repo=file_repo,
            window_tokens=settings.map_window_tokens,
            overlap_tokens=settings.map_window_overlap_tokens,
        )

    async def run_turn_stream(
        self,
        turn_input: TurnInput,
        principal: AuthPrincipal,
    ) -> AsyncGenerator[OrchestratorEvent]:
        """Execute a turn and yield streaming OrchestratorEvents ending in TurnCompletedEvent."""
        with bind_log_context(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            principal_id=principal.principal_id,
            component="orchestrator",
        ):
            async with self._session_gate.acquire(turn_input.session_id):
                session = await load_authorized_session(self._session_repo, turn_input)
                blocked = await scan_turn_guardrails(
                    self._guardrail_pipeline,
                    turn_input,
                    principal_id=principal.principal_id,
                )
                if blocked is not None:
                    async for event in handle_guardrail_block(
                        self._prep_deps, turn_input, blocked, session=session
                    ):
                        yield event
                    return
                async for event in self._execute_allowed_turn(
                    turn_input, principal, session=session
                ):
                    yield event

    async def run_turn(
        self,
        turn_input: TurnInput,
        principal: AuthPrincipal,
    ) -> TurnOutput:
        """Execute a turn to completion and return the final TurnOutput."""
        completed_output: TurnOutput | None = None
        async for event in self.run_turn_stream(turn_input, principal):
            if isinstance(event, TurnCompletedEvent):
                completed_output = event.output
        if completed_output is None:
            msg = "Turn stream ended without emitting TurnCompletedEvent."
            raise InferenceBackendError(msg)
        return completed_output

    async def resume_client_tool_stream(
        self,
        approval: ToolApprovalRecord,
        tool_result: ToolResult,
        principal: AuthPrincipal,
    ) -> AsyncGenerator[OrchestratorEvent]:
        """Resume a paused turn after client tool execution and yield streaming events."""
        async for event in run_client_tool_resume_stream(
            self._resume_deps,
            approval,
            tool_result,
            principal,
            session_tools_cache=self._session_client_tools,
        ):
            yield event

    async def resume_client_tool(
        self,
        approval: ToolApprovalRecord,
        tool_result: ToolResult,
        principal: AuthPrincipal,
    ) -> TurnOutput:
        """Resume a paused turn after client tool execution and return TurnOutput."""
        completed_output: TurnOutput | None = None
        async for event in self.resume_client_tool_stream(approval, tool_result, principal):
            if isinstance(event, TurnCompletedEvent):
                completed_output = event.output
        if completed_output is None:
            msg = "Resume stream ended without emitting TurnCompletedEvent."
            raise InferenceBackendError(msg)
        return completed_output

    def _remember_session_client_tools(self, turn_input: TurnInput) -> None:
        if not turn_input.client_app_id or not turn_input.client_tools:
            return
        if len(self._session_client_tools) >= _MAX_CACHED_SESSION_TOOLS:
            del self._session_client_tools[next(iter(self._session_client_tools))]
        self._session_client_tools[turn_input.session_id] = (
            turn_input.client_app_id,
            turn_input.client_tools,
            turn_input.client_allowed_tiers,
        )

    async def _execute_allowed_turn(
        self,
        turn_input: TurnInput,
        principal: AuthPrincipal,
        *,
        session: Session | None,
    ) -> AsyncGenerator[OrchestratorEvent]:
        self._remember_session_client_tools(turn_input)
        prep = await prepare_allowed_turn(self._prep_deps, turn_input, session)
        self._log.info(
            "turn_started",
            context_mode=prep.context_mode.value,
            file_count=len(prep.target_files),
            has_session=prep.session is not None,
        )

        outcome: TurnOutcome | None = None
        async for item in self._dispatch_mode(
            prep=prep, turn_input=turn_input, principal=principal
        ):
            if isinstance(item, TurnOutcome):
                outcome = item
            else:
                yield item

        resolved = outcome or TurnOutcome(content="", thinking="", citations=(), usage=TokenUsage())
        assistant_msg = Message(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=MessageRole.ASSISTANT,
            content=resolved.content,
            thinking=resolved.thinking or None,
            citations=resolved.citations,
            tool_calls=(
                (resolved.pending_tool_call,) if resolved.pending_tool_call is not None else ()
            ),
            usage=resolved.usage,
        )
        await persist_turn_if_session(
            self._session_repo,
            prep.session,
            turn_input,
            turn_input.user_message,
            assistant_msg,
            intermediate_messages=resolved.intermediate_messages,
        )

        self._log.info(
            "turn_completed",
            context_mode=prep.context_mode.value,
            prompt_tokens=resolved.usage.prompt_tokens,
            completion_tokens=resolved.usage.completion_tokens,
            citation_count=len(resolved.citations),
        )
        yield TurnCompletedEvent(
            output=TurnOutput(
                session_id=turn_input.session_id,
                message=assistant_msg,
                context_mode_used=prep.context_mode,
                citations=resolved.citations,
                usage=resolved.usage,
                pending_approval_id=resolved.pending_approval_id,
                pending_client_tool_call=resolved.pending_client_tool_call,
            )
        )

    async def _dispatch_mode(
        self,
        *,
        prep: PreparedTurnContext,
        turn_input: TurnInput,
        principal: AuthPrincipal,
    ) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
        if prep.context_mode is ContextMode.DIRECT:
            stream = run_direct_stream(
                self._direct_deps,
                turn_input=turn_input,
                principal=principal,
                profile=prep.profile,
                target_files=prep.target_files,
                session_system_prompt=prep.session_prompt,
                existing_messages=prep.existing_messages,
            )
        elif prep.context_mode is ContextMode.MAP_REDUCE:
            max_out = (
                turn_input.max_tokens
                if turn_input.max_tokens is not None
                else prep.profile.max_tokens
            )
            trimmed = build_trimmed_history(
                prep.existing_messages,
                profile=prep.profile,
                reserved_prompt_tokens=estimate_tokens(turn_input.user_message) + 2048,
                max_output_tokens=max_out,
            )
            stream = self._aggregator.run_stream(
                turn_input=turn_input,
                principal=principal,
                model_profile=prep.profile,
                files=prep.target_files,
                session_system_prompt=prep.session_prompt,
                history_messages=trimmed,
            )
        else:
            stream = run_retrieval_stream(
                self._retrieval_deps,
                turn_input=turn_input,
                principal=principal,
                model_profile=prep.profile,
                target_files=prep.target_files,
                session_system_prompt=prep.session_prompt,
                existing_messages=prep.existing_messages,
            )

        async for item in stream:
            yield item
