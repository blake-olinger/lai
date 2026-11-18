"""OpenAI-compatible /v1/models and /v1/chat/completions endpoints."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lai.api._client_auth import resolve_turn_client_context
from lai.api._streaming import (
    build_guardrail_headers,
    guard_openai_sse_frames,
    prime_sse_stream,
)
from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.routes._openai_helpers import (
    StreamMeta,
    build_completion_response,
    format_openai_sse,
    parse_context_mode,
    parse_file_ids,
    parse_openai_messages,
    resolve_openai_model,
    resolve_session_and_project,
)
from lai.api.schemas.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelInfo,
    OpenAIModelListResponse,
)
from lai.config.profiles import BUILTIN_PROFILES
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import TurnInput, TurnOutput
from lai.scheduler import await_or_keepalive_json

router = APIRouter(tags=["openai"])


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
    x_lai_client_id: Annotated[str | None, Header(alias="X-LAI-Client-Id")] = None,
    x_lai_client_secret: Annotated[str | None, Header(alias="X-LAI-Client-Secret")] = None,
) -> Response:
    """OpenAI-compatible chat completions supporting project/file context and keepalives."""
    explicit_project_id = (body.project_id or x_lai_project_id or "").strip()
    explicit_session_id = (body.session_id or x_lai_session_id or "").strip()
    session_id, project_id, existing_session = await resolve_session_and_project(
        services,
        explicit_session_id=explicit_session_id,
        explicit_project_id=explicit_project_id,
    )
    client_ctx = await resolve_turn_client_context(
        services,
        project_id=project_id,
        requested_tools=body.client_tools,
        client_id=x_lai_client_id,
        client_secret=x_lai_client_secret,
    )
    resolved_model_name, turn_model_profile = resolve_openai_model(
        services.settings,
        body.model,
        existing_session,
    )
    file_ids = parse_file_ids(body.file_ids, x_lai_file_ids)
    context_mode = parse_context_mode(body.context_mode, x_lai_context_mode)
    parsed = parse_openai_messages(body.messages, session_id=session_id, project_id=project_id)
    max_tokens = body.max_tokens if body.max_tokens is not None else body.max_completion_tokens

    turn_input = TurnInput(
        session_id=session_id,
        project_id=project_id,
        user_message=parsed.user_message,
        attached_file_ids=file_ids,
        context_mode=context_mode,
        enable_tools=body.enable_tools,
        enable_thinking=body.enable_thinking,
        max_tokens=max_tokens,
        temperature=body.temperature,
        model_profile=turn_model_profile,
        system_prompt=parsed.system_prompt,
        prior_messages=parsed.prior_messages,
        extra_guardrail_prompts=parsed.extra_guardrail_prompts,
        client_tools=client_ctx.client_tools,
        client_app_id=client_ctx.client_app_id,
        client_allowed_tiers=client_ctx.client_allowed_tiers,
    )
    meta = StreamMeta(
        completion_id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model_name=resolved_model_name,
    )

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
            guard_openai_sse_frames(format_openai_sse(primed.events, meta)),
            media_type="text/event-stream",
            headers=headers,
        )

    outcome = await await_or_keepalive_json(
        services.orchestrator.run_turn(turn_input, principal),
        threshold_seconds=services.settings.json_keepalive_threshold_seconds,
        interval_seconds=services.settings.keepalive_interval_seconds,
        is_disconnected=request.is_disconnected,
        serializer=lambda output: (
            build_completion_response(output, meta).model_dump_json().encode("utf-8")
        ),
    )
    if isinstance(outcome, TurnOutput):
        payload = build_completion_response(outcome, meta)
        headers = build_guardrail_headers(
            guardrail_blocked=outcome.guardrail_blocked,
            guardrail_policy=outcome.guardrail_policy,
        )
        return JSONResponse(content=payload.model_dump(mode="json"), headers=headers)
    return StreamingResponse(outcome, media_type="application/json")
