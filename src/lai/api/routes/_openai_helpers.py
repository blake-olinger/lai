"""Request parsing, session/model resolution, and SSE formatting helpers for OpenAI routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from lai.api._tenancy import require_accessible_project, require_accessible_session
from lai.api.schemas.openai import (
    OpenAIChatChoice,
    OpenAIChatChoiceMessage,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponse,
    OpenAIDeltaPayload,
    OpenAIDeltaToolCall,
    OpenAIStreamChoice,
    OpenAIToolCall,
    OpenAIToolFunction,
    OpenAIUsage,
)
from lai.config.profiles import get_model_profile
from lai.domain.chat import (
    ApprovalRequiredEvent,
    ClientToolCallEvent,
    ContentDeltaEvent,
    ContextMode,
    KeepaliveEvent,
    Message,
    MessageRole,
    ProgressEvent,
    ThinkingDeltaEvent,
)
from lai.domain.errors import AuthorizationError, ValidationFailedError
from lai.domain.ids import new_session_id
from lai.orchestrator._helpers import resolve_model_profile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from lai.api.deps import AppServices
    from lai.api.schemas.openai import OpenAIChatMessageInput
    from lai.config.settings import LaiSettings
    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import OrchestratorEvent, Session, TurnOutput

_SYSTEM_ROLES = frozenset({"system", "developer"})
_HISTORY_ROLES = frozenset({"user", "assistant"})
_GUARDRAIL_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


@dataclass(frozen=True, slots=True)
class StreamMeta:
    """Correlation metadata shared across chunks and JSON completion responses."""

    completion_id: str
    created: int
    model_name: str


@dataclass(frozen=True, slots=True)
class ParsedOpenAIMessages:
    """Parsed OpenAI message array split into turn prompt, system prompt, and history."""

    user_message: str
    system_prompt: str | None
    prior_messages: tuple[Message, ...]
    extra_guardrail_prompts: tuple[str, ...]


def extract_message_text(msg_input: OpenAIChatMessageInput) -> str:
    """Extract plain text from string or OpenAI multimodal text-part list."""
    raw = msg_input.content
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    parts: list[str] = []
    for item in raw:
        if item.get("type") == "text":
            text_val = item.get("text")
            if isinstance(text_val, str) and text_val:
                parts.append(text_val)
    return "\n".join(parts)


def _find_target_message_index(entries: Sequence[tuple[str, str]]) -> int:
    """Return index of the latest non-empty user message, or latest non-empty fallback."""
    for idx in range(len(entries) - 1, -1, -1):
        role, text = entries[idx]
        if role == "user" and text:
            return idx
    for idx in range(len(entries) - 1, -1, -1):
        if entries[idx][1]:
            return idx
    msg = "At least one non-empty message content is required."
    raise ValidationFailedError(msg, details={"field": "messages"})


def parse_openai_messages(
    messages: Sequence[OpenAIChatMessageInput],
    *,
    session_id: str,
    project_id: str,
) -> ParsedOpenAIMessages:
    """Split OpenAI messages into system prompt, prior history, turn prompt, and scan texts."""
    entries: list[tuple[str, str]] = [
        (msg.role.strip().lower(), extract_message_text(msg).strip()) for msg in messages
    ]
    target_idx = _find_target_message_index(entries)
    user_message = entries[target_idx][1]

    system_parts = [text for role, text in entries if role in _SYSTEM_ROLES and text]
    system_prompt = "\n\n".join(system_parts) if system_parts else None

    prior_messages = tuple(
        Message(
            session_id=session_id,
            project_id=project_id,
            role=MessageRole(role),
            content=text,
        )
        for idx, (role, text) in enumerate(entries)
        if idx < target_idx and role in _HISTORY_ROLES and text
    )
    extra_guardrail_prompts = tuple(
        dict.fromkeys(
            text
            for idx, (role, text) in enumerate(entries)
            if idx != target_idx and role in _GUARDRAIL_ROLES and text and text != user_message
        )
    )
    return ParsedOpenAIMessages(
        user_message=user_message,
        system_prompt=system_prompt,
        prior_messages=prior_messages,
        extra_guardrail_prompts=extra_guardrail_prompts,
    )


def parse_file_ids(
    body_file_ids: list[str] | None,
    header_file_ids: str | None,
) -> tuple[str, ...]:
    """Resolve attached file IDs from request body or comma-separated X-LAI-File-Ids."""
    if body_file_ids is not None:
        return tuple(item.strip() for item in body_file_ids if item.strip())
    if header_file_ids:
        return tuple(item.strip() for item in header_file_ids.split(",") if item.strip())
    return ()


def parse_context_mode(
    body_mode: ContextMode | None,
    header_mode: str | None,
) -> ContextMode:
    """Resolve ContextMode from request body or X-LAI-Context-Mode header."""
    if body_mode is not None:
        return body_mode
    if header_mode is None or not header_mode.strip():
        return ContextMode.AUTO
    normalized = header_mode.strip().lower()
    try:
        return ContextMode(normalized)
    except ValueError as exc:
        msg = f"Invalid X-LAI-Context-Mode header: {header_mode!r}"
        raise ValidationFailedError(msg, details={"header": "X-LAI-Context-Mode"}) from exc


async def resolve_session_and_project(
    services: AppServices,
    *,
    principal: AuthPrincipal,
    explicit_session_id: str,
    explicit_project_id: str,
) -> tuple[str, str, Session | None, bool]:
    """Resolve (session_id, project_id, existing_session, has_owned_project) with isolation."""
    if not explicit_session_id:
        if explicit_project_id:
            await require_accessible_project(services, explicit_project_id, principal)
            return new_session_id(), explicit_project_id, None, True
        return new_session_id(), f"default:{principal.principal_id}", None, False

    if explicit_project_id:
        await require_accessible_project(services, explicit_project_id, principal)
    existing_session = await require_accessible_session(services, explicit_session_id, principal)
    if explicit_project_id and explicit_project_id != existing_session.project_id:
        msg = f"Session {existing_session.id!r} does not belong to project {explicit_project_id!r}."
        raise AuthorizationError(msg, details={"session_id": existing_session.id})
    return explicit_session_id, existing_session.project_id, existing_session, True


def resolve_openai_model(
    settings: LaiSettings,
    requested_model: str | None,
    existing_session: Session | None,
) -> tuple[str, str | None]:
    """Validate explicit model or resolve session/default model name."""
    if requested_model is not None:
        profile = get_model_profile(requested_model, settings.custom_profiles)
        return profile.name, requested_model
    profile = resolve_model_profile(settings, existing_session, None)
    return profile.name, None


def build_completion_response(
    output: TurnOutput,
    meta: StreamMeta,
) -> OpenAIChatCompletionResponse:
    """Convert TurnOutput into an OpenAI chat.completion JSON response."""
    client_tc = output.pending_client_tool_call
    tool_calls = (
        [
            OpenAIToolCall(
                id=client_tc.tool_call_id,
                function=OpenAIToolFunction(
                    name=client_tc.tool_name,
                    arguments=client_tc.arguments_json,
                ),
            )
        ]
        if client_tc is not None
        else None
    )
    finish_reason = "tool_calls" if client_tc is not None else "stop"
    return OpenAIChatCompletionResponse(
        id=meta.completion_id,
        created=meta.created,
        model=meta.model_name,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatChoiceMessage(
                    content=output.message.content,
                    reasoning_content=output.message.thinking,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=output.usage.prompt_tokens,
            completion_tokens=output.usage.completion_tokens,
            total_tokens=output.usage.total_tokens,
        ),
        lai_client_tool_call=client_tc,
    )


def make_chunk_sse(
    meta: StreamMeta,
    delta: OpenAIDeltaPayload,
    *,
    finish_reason: str | None = None,
    usage: OpenAIUsage | None = None,
    lai_client_tool_call: ClientToolCallEvent | None = None,
) -> str:
    """Serialize one OpenAI chat.completion.chunk SSE data frame."""
    chunk = OpenAIChatCompletionChunk(
        id=meta.completion_id,
        created=meta.created,
        model=meta.model_name,
        choices=[OpenAIStreamChoice(index=0, delta=delta, finish_reason=finish_reason)],
        usage=usage,
        lai_client_tool_call=lai_client_tool_call,
    )
    return f"data: {chunk.model_dump_json()}\n\n"


async def format_openai_sse(
    events: AsyncIterator[OrchestratorEvent],
    meta: StreamMeta,
) -> AsyncIterator[str]:
    """Translate OrchestratorEvents into OpenAI SSE chunks with : keepalive comments."""
    sent_role = False
    sent_content = False

    async for event in events:
        if isinstance(event, (KeepaliveEvent, ProgressEvent)):
            yield ": keepalive\n\n"
            continue

        role_val: Literal["assistant"] | None = "assistant" if not sent_role else None
        if isinstance(event, ThinkingDeltaEvent):
            sent_role = True
            yield make_chunk_sse(
                meta,
                OpenAIDeltaPayload(role=role_val, reasoning_content=event.delta),
            )
        elif isinstance(event, ContentDeltaEvent):
            sent_role = True
            sent_content = True
            yield make_chunk_sse(
                meta,
                OpenAIDeltaPayload(role=role_val, content=event.delta),
            )
        elif isinstance(event, ApprovalRequiredEvent):
            sent_role = True
            sent_content = True
            notice = f"\n[Tool approval required: {event.approval_id} ({event.tool_name})]\n"
            yield make_chunk_sse(
                meta,
                OpenAIDeltaPayload(role=role_val, content=notice),
            )
        elif isinstance(event, ClientToolCallEvent):
            sent_role = True
            sent_content = True
            yield make_chunk_sse(
                meta,
                OpenAIDeltaPayload(
                    role=role_val,
                    tool_calls=[
                        OpenAIDeltaToolCall(
                            index=0,
                            id=event.tool_call_id,
                            type="function",
                            function=OpenAIToolFunction(
                                name=event.tool_name,
                                arguments=event.arguments_json,
                            ),
                        )
                    ],
                ),
                lai_client_tool_call=event,
            )
        else:
            if not sent_content and event.output.message.content:
                yield make_chunk_sse(
                    meta,
                    OpenAIDeltaPayload(
                        role=role_val,
                        content=event.output.message.content,
                    ),
                )
            client_tc = event.output.pending_client_tool_call
            finish_reason = "tool_calls" if client_tc is not None else "stop"
            yield make_chunk_sse(
                meta,
                OpenAIDeltaPayload(),
                finish_reason=finish_reason,
                usage=OpenAIUsage(
                    prompt_tokens=event.output.usage.prompt_tokens,
                    completion_tokens=event.output.usage.completion_tokens,
                    total_tokens=event.output.usage.total_tokens,
                ),
            )

    yield "data: [DONE]\n\n"
