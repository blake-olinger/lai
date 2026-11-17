"""Main TurnOrchestrator implementing OrchestratorPort across all context modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.chat import (
    ContentDeltaEvent,
    ContextMode,
    Message,
    MessageRole,
    OrchestratorEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnOutput,
)
from lai.domain.errors import (
    AuthorizationError,
    GuardrailBlockedError,
    InferenceBackendError,
)
from lai.domain.guardrails import GuardrailBlockMode, GuardrailVerdict
from lai.ingestion.detector import estimate_tokens
from lai.observability.logging import bind_log_context, get_component_logger, get_request_id
from lai.orchestrator._direct import DirectDeps, run_direct_stream
from lai.orchestrator._helpers import (
    build_trimmed_history,
    persist_turn_if_session,
    resolve_model_profile,
    resolve_target_files,
)
from lai.orchestrator._retrieval import run_retrieval_stream
from lai.orchestrator._retrieval_search import RetrievalDeps
from lai.orchestrator._stream import TurnOutcome
from lai.orchestrator.aggregator import MapReduceAggregator
from lai.orchestrator.classifier import classify_context_mode
from lai.ports.orchestrator import OrchestratorPort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from lai.config.settings import LaiSettings
    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import Session, TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.guardrails import GuardrailDecision
    from lai.domain.inference import ModelProfile
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.repositories import FileRepositoryPort, SessionRepositoryPort
    from lai.ports.scheduler import InferenceSpoolerPort, SessionGatePort
    from lai.ports.storage import BlobStorePort, EmbedderPort, VectorStorePort
    from lai.ports.tools import ToolExecutorPort, ToolRegistryPort


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
    ) -> None:
        self._settings = settings
        self._session_repo = session_repo
        self._file_repo = file_repo
        self._session_gate = session_gate
        self._guardrail_pipeline = guardrail_pipeline
        self._model_profile = model_profile
        self._log = get_component_logger("orchestrator")
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
            decision = await self._guardrail_pipeline.check_prompt(
                turn_input.user_message,
                request_id=get_request_id(),
                session_id=turn_input.session_id,
                project_id=turn_input.project_id,
                principal_id=principal.principal_id,
            )
            async with self._session_gate.acquire(turn_input.session_id):
                if decision.verdict is GuardrailVerdict.BLOCK:
                    async for event in self._handle_guardrail_block(turn_input, decision):
                        yield event
                    return
                async for event in self._execute_allowed_turn(turn_input, principal):
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

    async def _load_authorized_session(self, turn_input: TurnInput) -> Session | None:
        """Fetch turn_input.session_id if present and verify project ownership."""
        session = await self._session_repo.get_session(turn_input.session_id)
        if session is not None and session.project_id != turn_input.project_id:
            msg = f"Session {session.id!r} does not belong to project {turn_input.project_id!r}."
            raise AuthorizationError(msg, details={"session_id": session.id})
        return session

    async def _handle_guardrail_block(
        self,
        turn_input: TurnInput,
        decision: GuardrailDecision,
    ) -> AsyncGenerator[OrchestratorEvent]:
        """Short-circuit blocked prompts via HTTP error or persisted refusal message."""
        self._log.warning(
            "turn_guardrail_blocked",
            violated_policy=decision.violated_policy,
            block_mode=self._settings.guardrails.block_mode.value,
        )
        if self._settings.guardrails.block_mode is GuardrailBlockMode.HTTP_ERROR:
            raise GuardrailBlockedError(
                self._settings.guardrails.refusal_text,
                violated_policy=decision.violated_policy,
                details={"provider": decision.provider},
            )

        refusal_text = self._settings.guardrails.refusal_text
        existing_session = await self._load_authorized_session(turn_input)
        assistant_msg = Message(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=MessageRole.ASSISTANT,
            content=refusal_text,
            usage=TokenUsage(),
        )
        await persist_turn_if_session(
            self._session_repo,
            existing_session,
            turn_input,
            decision.scrubbed_prompt,
            assistant_msg,
        )

        mode_used = (
            turn_input.context_mode
            if turn_input.context_mode is not ContextMode.AUTO
            else ContextMode.DIRECT
        )
        yield ContentDeltaEvent(delta=refusal_text)
        yield TurnCompletedEvent(
            output=TurnOutput(
                session_id=turn_input.session_id,
                message=assistant_msg,
                context_mode_used=mode_used,
                citations=(),
                usage=TokenUsage(),
                guardrail_blocked=True,
                guardrail_policy=decision.violated_policy,
            )
        )

    async def _execute_allowed_turn(
        self,
        turn_input: TurnInput,
        principal: AuthPrincipal,
    ) -> AsyncGenerator[OrchestratorEvent]:
        """Load session/files, route context mode, stream inference, and persist messages."""
        session = await self._load_authorized_session(turn_input)

        existing_messages = (
            await self._session_repo.list_messages(session.id, limit=200)
            if session is not None
            else []
        )
        profile = resolve_model_profile(self._settings, session, self._model_profile)
        target_files = await resolve_target_files(
            self._file_repo,
            turn_input.project_id,
            turn_input.attached_file_ids,
        )
        context_mode = classify_context_mode(
            turn_input.context_mode,
            turn_input.user_message,
            target_files,
            direct_budget_tokens=self._settings.direct_file_budget_tokens,
        )
        self._log.info(
            "turn_started",
            context_mode=context_mode.value,
            file_count=len(target_files),
            has_session=session is not None,
        )

        outcome: TurnOutcome | None = None
        async for item in self._dispatch_mode(
            context_mode=context_mode,
            turn_input=turn_input,
            principal=principal,
            session=session,
            existing_messages=existing_messages,
            profile=profile,
            target_files=target_files,
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
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=MessageRole.ASSISTANT,
            content=resolved.content,
            thinking=resolved.thinking or None,
            citations=resolved.citations,
            usage=resolved.usage,
        )
        await persist_turn_if_session(
            self._session_repo,
            session,
            turn_input,
            turn_input.user_message,
            assistant_msg,
        )

        self._log.info(
            "turn_completed",
            context_mode=context_mode.value,
            prompt_tokens=resolved.usage.prompt_tokens,
            completion_tokens=resolved.usage.completion_tokens,
            citation_count=len(resolved.citations),
        )
        yield TurnCompletedEvent(
            output=TurnOutput(
                session_id=turn_input.session_id,
                message=assistant_msg,
                context_mode_used=context_mode,
                citations=resolved.citations,
                usage=resolved.usage,
                pending_approval_id=resolved.pending_approval_id,
            )
        )

    async def _dispatch_mode(  # noqa: PLR0913
        self,
        *,
        context_mode: ContextMode,
        turn_input: TurnInput,
        principal: AuthPrincipal,
        session: Session | None,
        existing_messages: Sequence[Message],
        profile: ModelProfile,
        target_files: Sequence[FileRecord],
    ) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
        """Dispatch to Direct, Retrieval, or Map-Reduce execution strategy."""
        session_prompt = session.system_prompt if session is not None else None
        if context_mode is ContextMode.DIRECT:
            stream = run_direct_stream(
                self._direct_deps,
                turn_input=turn_input,
                principal=principal,
                profile=profile,
                target_files=target_files,
                session_system_prompt=session_prompt,
                existing_messages=existing_messages,
            )
        elif context_mode is ContextMode.MAP_REDUCE:
            max_out = (
                turn_input.max_tokens if turn_input.max_tokens is not None else profile.max_tokens
            )
            trimmed = build_trimmed_history(
                existing_messages,
                profile=profile,
                reserved_prompt_tokens=estimate_tokens(turn_input.user_message) + 2048,
                max_output_tokens=max_out,
            )
            stream = self._aggregator.run_stream(
                turn_input=turn_input,
                principal=principal,
                model_profile=profile,
                files=target_files,
                session_system_prompt=session_prompt,
                history_messages=trimmed,
            )
        else:
            stream = run_retrieval_stream(
                self._retrieval_deps,
                turn_input=turn_input,
                principal=principal,
                model_profile=profile,
                target_files=target_files,
                session_system_prompt=session_prompt,
                existing_messages=existing_messages,
            )

        async for item in stream:
            yield item
