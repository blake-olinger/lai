"""OpenAI-compatible /v1/models and /v1/chat/completions endpoints."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lai.api._streaming import (
    build_guardrail_headers,
    guard_openai_sse_frames,
    prime_sse_stream,
)
from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.openai import (
    OpenAIChatChoice,
    OpenAIChatChoiceMessage,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatMessageInput,
    OpenAIDeltaPayload,
    OpenAIModelInfo,
    OpenAIModelListResponse,
    OpenAIStreamChoice,
    OpenAIUsage,
)
from lai.config.profiles import BUILTIN_PROFILES
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import (
    ApprovalRequiredEvent,
    ContentDeltaEvent,
    ContextMode,
    KeepaliveEvent,
    OrchestratorEvent,
    ProgressEvent,
    ThinkingDeltaEvent,
    TurnInput,
    TurnOutput,
)
from lai.domain.errors import ValidationFailedError
from lai.domain.ids import new_session_id
from lai.scheduler import await_or_keepalive_json

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter(tags=["openai"])

_DEFAULT_OPENAI_PROJECT_ID = "default"


def _extract_message_text(msg_input: OpenAIChatMessageInput) -> str:
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


def _extract_user_prompt(messages: list[OpenAIChatMessageInput]) -> str:
    """Resolve the prompt from the latest user message (or fallback to last message)."""
    for message in reversed(messages):
        if message.role == "user":
            text = _extract_message_text(message).strip()
            if text:
                return text
    for message in reversed(messages):
        text = _extract_message_text(message).strip()
        if text:
            return text
    msg = "At least one non-empty message content is required."
    raise ValidationFailedError(msg, details={"field": "messages"})


def _parse_file_ids(
    body_file_ids: list[str] | None,
    header_file_ids: str | None,
) -> tuple[str, ...]:
    """Resolve attached file IDs from request body or comma-separated X-LAI-File-Ids."""
    if body_file_ids is not None:
        return tuple(item.strip() for item in body_file_ids if item.strip())
    if header_file_ids:
        return tuple(item.strip() for item in header_file_ids.split(",") if item.strip())
    return ()


def _parse_context_mode(
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


def _build_completion_response(
    output: TurnOutput,
    *,
    completion_id: str,
    created: int,
    model_name: str,
) -> OpenAIChatCompletionResponse:
    """Convert TurnOutput into an OpenAI chat.completion JSON response."""
    return OpenAIChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatChoiceMessage(
                    content=output.message.content,
                    reasoning_content=output.message.thinking,
                ),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=output.usage.prompt_tokens,
            completion_tokens=output.usage.completion_tokens,
            total_tokens=output.usage.total_tokens,
        ),
    )


@dataclass(frozen=True, slots=True)
class _StreamMeta:
    """Correlation metadata shared across chunks of a single OpenAI SSE stream."""

    completion_id: str
    created: int
    model_name: str


def _make_chunk_sse(
    meta: _StreamMeta,
    delta: OpenAIDeltaPayload,
    *,
    finish_reason: str | None = None,
    usage: OpenAIUsage | None = None,
) -> str:
    """Serialize one OpenAI chat.completion.chunk SSE data frame."""
    chunk = OpenAIChatCompletionChunk(
        id=meta.completion_id,
        created=meta.created,
        model=meta.model_name,
        choices=[OpenAIStreamChoice(index=0, delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )
    return f"data: {chunk.model_dump_json()}\n\n"


async def _format_openai_sse(
    events: AsyncIterator[OrchestratorEvent],
    meta: _StreamMeta,
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
            yield _make_chunk_sse(
                meta,
                OpenAIDeltaPayload(role=role_val, reasoning_content=event.delta),
            )
        elif isinstance(event, ContentDeltaEvent):
            sent_role = True
            sent_content = True
            yield _make_chunk_sse(
                meta,
                OpenAIDeltaPayload(role=role_val, content=event.delta),
            )
        elif isinstance(event, ApprovalRequiredEvent):
            sent_role = True
            sent_content = True
            notice = f"\n[Tool approval required: {event.approval_id} ({event.tool_name})]\n"
            yield _make_chunk_sse(
                meta,
                OpenAIDeltaPayload(role=role_val, content=notice),
            )
        else:
            if not sent_content and event.output.message.content:
                yield _make_chunk_sse(
                    meta,
                    OpenAIDeltaPayload(
                        role=role_val,
                        content=event.output.message.content,
                    ),
                )
            yield _make_chunk_sse(
                meta,
                OpenAIDeltaPayload(),
                finish_reason="stop",
                usage=OpenAIUsage(
                    prompt_tokens=event.output.usage.prompt_tokens,
                    completion_tokens=event.output.usage.completion_tokens,
                    total_tokens=event.output.usage.total_tokens,
                ),
            )

    yield "data: [DONE]\n\n"


@router.get("/v1/models", response_model=OpenAIModelListResponse)
async def list_openai_models(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_READ))],
    services: Annotated[AppServices, Depends(get_services)],
) -> OpenAIModelListResponse:
    """List configured model profiles in OpenAI /v1/models format."""
    names: list[str] = [services.settings.default_model_profile]
    for name in (*BUILTIN_PROFILES.keys(), *services.settings.custom_profiles.keys()):
        if name not in names:
            names.append(name)
    return OpenAIModelListResponse(data=[OpenAIModelInfo(id=name) for name in names])


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def create_openai_chat_completion(  # noqa: PLR0913, PLR0917
    body: OpenAIChatCompletionRequest,
    request: Request,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
    x_lai_project_id: Annotated[str | None, Header(alias="X-LAI-Project-Id")] = None,
    x_lai_file_ids: Annotated[str | None, Header(alias="X-LAI-File-Ids")] = None,
    x_lai_session_id: Annotated[str | None, Header(alias="X-LAI-Session-Id")] = None,
    x_lai_context_mode: Annotated[str | None, Header(alias="X-LAI-Context-Mode")] = None,
) -> Response:
    """OpenAI-compatible chat completions supporting project/file context and keepalives."""
    project_id = (body.project_id or x_lai_project_id or _DEFAULT_OPENAI_PROJECT_ID).strip()
    session_id = (body.session_id or x_lai_session_id or "").strip() or new_session_id()
    file_ids = _parse_file_ids(body.file_ids, x_lai_file_ids)
    context_mode = _parse_context_mode(body.context_mode, x_lai_context_mode)
    user_message = _extract_user_prompt(body.messages)
    max_tokens = body.max_tokens if body.max_tokens is not None else body.max_completion_tokens

    turn_input = TurnInput(
        session_id=session_id,
        project_id=project_id or _DEFAULT_OPENAI_PROJECT_ID,
        user_message=user_message,
        attached_file_ids=file_ids,
        context_mode=context_mode,
        enable_tools=body.enable_tools,
        enable_thinking=body.enable_thinking,
        max_tokens=max_tokens,
        temperature=body.temperature,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = body.model or services.settings.default_model_profile

    if body.stream:
        primed = await prime_sse_stream(
            services.orchestrator.run_turn_stream(turn_input, principal),
            keepalive_interval_seconds=services.settings.keepalive_interval_seconds,
            is_disconnected=request.is_disconnected,
        )
        headers = {
            **build_guardrail_headers(
                guardrail_blocked=primed.guardrail_blocked,
                guardrail_policy=primed.guardrail_policy,
            ),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            guard_openai_sse_frames(
                _format_openai_sse(
                    primed.events,
                    _StreamMeta(
                        completion_id=completion_id,
                        created=created,
                        model_name=model_name,
                    ),
                )
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    outcome = await await_or_keepalive_json(
        services.orchestrator.run_turn(turn_input, principal),
        threshold_seconds=services.settings.json_keepalive_threshold_seconds,
        interval_seconds=services.settings.keepalive_interval_seconds,
        is_disconnected=request.is_disconnected,
        serializer=lambda output: (
            _build_completion_response(
                output,
                completion_id=completion_id,
                created=created,
                model_name=model_name,
            )
            .model_dump_json()
            .encode("utf-8")
        ),
    )
    if isinstance(outcome, TurnOutput):
        payload = _build_completion_response(
            outcome,
            completion_id=completion_id,
            created=created,
            model_name=model_name,
        )
        headers = build_guardrail_headers(
            guardrail_blocked=outcome.guardrail_blocked,
            guardrail_policy=outcome.guardrail_policy,
        )
        return JSONResponse(content=payload.model_dump(mode="json"), headers=headers)
    return StreamingResponse(outcome, media_type="application/json")
