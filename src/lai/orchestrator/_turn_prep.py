"""Pre-flight guardrail scanning, block handling, and turn context preparation."""

from __future__ import annotations

from dataclasses import dataclass
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
from lai.domain.errors import AuthorizationError, GuardrailBlockedError
from lai.domain.guardrails import GuardrailBlockMode, GuardrailVerdict
from lai.observability.logging import get_component_logger, get_request_id
from lai.orchestrator._helpers import (
    persist_turn_if_session,
    resolve_model_profile,
    resolve_target_files,
)
from lai.orchestrator.classifier import classify_context_mode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from lai.config.settings import LaiSettings
    from lai.domain.chat import Session, TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.guardrails import GuardrailDecision
    from lai.domain.inference import ModelProfile
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.repositories import FileRepositoryPort, SessionRepositoryPort


@dataclass(frozen=True, slots=True)
class TurnPrepDeps:
    """Dependencies required to prepare an allowed orchestrator turn."""

    settings: LaiSettings
    session_repo: SessionRepositoryPort
    file_repo: FileRepositoryPort
    default_profile: ModelProfile | None = None


@dataclass(frozen=True, slots=True)
class PreparedTurnContext:
    """Resolved session, authoritative history, system prompt, model profile, files, and mode."""

    session: Session | None
    existing_messages: list[Message]
    session_prompt: str | None
    profile: ModelProfile
    target_files: list[FileRecord]
    context_mode: ContextMode


async def load_authorized_session(
    session_repo: SessionRepositoryPort,
    turn_input: TurnInput,
) -> Session | None:
    """Fetch turn_input.session_id if present and verify project ownership."""
    session = await session_repo.get_session(turn_input.session_id)
    if session is not None and session.project_id != turn_input.project_id:
        msg = f"Session {session.id!r} does not belong to project {turn_input.project_id!r}."
        raise AuthorizationError(msg, details={"session_id": session.id})
    return session


async def scan_turn_guardrails(
    guardrail_pipeline: GuardrailPipelinePort,
    turn_input: TurnInput,
    *,
    principal_id: str,
) -> GuardrailDecision | None:
    """Scan all non-empty extra caller prompts and user_message; return first BLOCK decision."""
    prompts = [text for text in turn_input.extra_guardrail_prompts if text.strip()]
    prompts.append(turn_input.user_message)
    for prompt in prompts:
        decision = await guardrail_pipeline.check_prompt(
            prompt,
            request_id=get_request_id(),
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            principal_id=principal_id,
        )
        if decision.verdict is GuardrailVerdict.BLOCK:
            return decision
    return None


async def handle_guardrail_block(
    deps: TurnPrepDeps,
    turn_input: TurnInput,
    decision: GuardrailDecision,
    *,
    session: Session | None,
) -> AsyncGenerator[OrchestratorEvent]:
    """Short-circuit blocked prompts via HTTP error or persisted refusal message."""
    get_component_logger("orchestrator").warning(
        "turn_guardrail_blocked",
        violated_policy=decision.violated_policy,
        block_mode=deps.settings.guardrails.block_mode.value,
    )
    if deps.settings.guardrails.block_mode is GuardrailBlockMode.HTTP_ERROR:
        raise GuardrailBlockedError(
            deps.settings.guardrails.refusal_text,
            violated_policy=decision.violated_policy,
            details={"provider": decision.provider},
        )

    refusal_text = deps.settings.guardrails.refusal_text
    assistant_msg = Message(
        session_id=turn_input.session_id,
        project_id=turn_input.project_id,
        role=MessageRole.ASSISTANT,
        content=refusal_text,
        usage=TokenUsage(),
    )
    await persist_turn_if_session(
        deps.session_repo,
        session,
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


async def prepare_allowed_turn(
    deps: TurnPrepDeps,
    turn_input: TurnInput,
    session: Session | None,
) -> PreparedTurnContext:
    """Resolve session authority, prior turns, system prompt, model profile, files, and mode."""
    if session is not None:
        existing_messages = await deps.session_repo.list_messages(session.id, limit=200)
        session_prompt = (
            session.system_prompt if session.system_prompt is not None else turn_input.system_prompt
        )
    else:
        existing_messages = list(turn_input.prior_messages)
        session_prompt = turn_input.system_prompt

    profile = resolve_model_profile(
        deps.settings,
        session,
        deps.default_profile,
        requested_model=turn_input.model_profile,
    )
    target_files = await resolve_target_files(
        deps.file_repo,
        turn_input.project_id,
        turn_input.attached_file_ids,
    )
    context_mode = classify_context_mode(
        turn_input.context_mode,
        turn_input.user_message,
        target_files,
        direct_budget_tokens=deps.settings.direct_file_budget_tokens,
    )
    if (
        turn_input.context_mode is ContextMode.AUTO
        and turn_input.enable_tools
        and turn_input.client_tools
        and context_mode is ContextMode.DIRECT
    ):
        context_mode = ContextMode.RETRIEVAL
    return PreparedTurnContext(
        session=session,
        existing_messages=existing_messages,
        session_prompt=session_prompt,
        profile=profile,
        target_files=target_files,
        context_mode=context_mode,
    )
