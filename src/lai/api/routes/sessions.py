"""Chat session CRUD and streaming/keepalive turn message routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lai.api._client_auth import resolve_turn_client_context
from lai.api._streaming import (
    build_guardrail_headers,
    format_orchestrator_sse,
    guard_orchestrator_sse_frames,
    prime_sse_stream,
)
from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.common import DeleteResponse
from lai.api.schemas.sessions import (
    SessionCreateRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageRequest,
)
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import Session, TurnInput, TurnOutput
from lai.domain.errors import NotFoundError
from lai.scheduler import await_or_keepalive_json

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreateRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> Session:
    """Create a new chat session scoped to an existing project."""
    project = await services.project_repo.get(body.project_id)
    if project is None:
        msg = f"Project {body.project_id!r} not found."
        raise NotFoundError(msg, details={"project_id": body.project_id})

    session = Session(
        project_id=body.project_id,
        title=body.title,
        model_profile=body.model_profile or project.default_model_profile,
        system_prompt=body.system_prompt,
    )
    return await services.session_repo.create_session(session)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    project_id: Annotated[str, Query(min_length=1)],
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_READ))],
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SessionListResponse:
    """List chat sessions belonging to a project."""
    project = await services.project_repo.get(project_id)
    if project is None:
        msg = f"Project {project_id!r} not found."
        raise NotFoundError(msg, details={"project_id": project_id})

    items = await services.session_repo.list_sessions(project_id, limit=limit, offset=offset)
    return SessionListResponse(items=items, count=len(items))


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_READ))],
    services: Annotated[AppServices, Depends(get_services)],
) -> SessionDetailResponse:
    """Fetch session metadata along with its chronological message history."""
    session = await services.session_repo.get_session(session_id)
    if session is None:
        msg = f"Session {session_id!r} not found."
        raise NotFoundError(msg, details={"session_id": session_id})
    messages = await services.session_repo.list_messages(session_id)
    return SessionDetailResponse(session=session, messages=messages)


@router.delete("/{session_id}", response_model=DeleteResponse)
async def delete_session(
    session_id: str,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> DeleteResponse:
    """Delete a chat session and its messages."""
    deleted = await services.session_repo.delete_session(session_id)
    if not deleted:
        msg = f"Session {session_id!r} not found."
        raise NotFoundError(msg, details={"session_id": session_id})
    return DeleteResponse(id=session_id, deleted=True)


@router.post("/{session_id}/messages", response_model=TurnOutput)
async def post_session_message(  # noqa: PLR0913, PLR0917
    session_id: str,
    body: SessionMessageRequest,
    request: Request,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
    x_lai_client_id: Annotated[str | None, Header(alias="X-LAI-Client-Id")] = None,
    x_lai_client_secret: Annotated[str | None, Header(alias="X-LAI-Client-Secret")] = None,
) -> Response:
    """Run a user turn in session_id as JSON (with keepalive) or SSE stream."""
    session = await services.session_repo.get_session(session_id)
    if session is None:
        msg = f"Session {session_id!r} not found."
        raise NotFoundError(msg, details={"session_id": session_id})

    client_ctx = await resolve_turn_client_context(
        services,
        project_id=session.project_id,
        requested_tools=body.client_tools,
        client_id=x_lai_client_id,
        client_secret=x_lai_client_secret,
    )
    turn_input = TurnInput(
        session_id=session.id,
        project_id=session.project_id,
        user_message=body.content,
        attached_file_ids=body.attached_file_ids,
        context_mode=body.context_mode,
        enable_tools=body.enable_tools,
        enable_thinking=body.enable_thinking,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        client_tools=client_ctx.client_tools,
        client_app_id=client_ctx.client_app_id,
        client_allowed_tiers=client_ctx.client_allowed_tiers,
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
            guard_orchestrator_sse_frames(format_orchestrator_sse(primed.events)),
            media_type="text/event-stream",
            headers=headers,
        )

    outcome = await await_or_keepalive_json(
        services.orchestrator.run_turn(turn_input, principal),
        threshold_seconds=services.settings.json_keepalive_threshold_seconds,
        interval_seconds=services.settings.keepalive_interval_seconds,
        is_disconnected=request.is_disconnected,
    )
    if isinstance(outcome, TurnOutput):
        headers = build_guardrail_headers(
            guardrail_blocked=outcome.guardrail_blocked,
            guardrail_policy=outcome.guardrail_policy,
        )
        return JSONResponse(
            content=outcome.model_dump(mode="json"),
            headers=headers,
        )
    return StreamingResponse(outcome, media_type="application/json")
