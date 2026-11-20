"""Tool specification listing, approval resolution, and client result submission routes."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lai.api._analytics import track_turn_output, track_turn_stream
from lai.api._client_auth import require_authenticated_client_app
from lai.api._streaming import (
    build_guardrail_headers,
    format_orchestrator_sse,
    guard_orchestrator_sse_frames,
    prime_sse_stream,
)
from lai.api._tenancy import is_admin_principal, require_accessible_approval
from lai.api.deps import (
    AppServices,
    get_services,
    require_any_scope_dep,
    require_scope_dep,
)
from lai.api.errors import serialize_post_header_error_json
from lai.api.schemas.tools import (
    ResolveApprovalRequest,
    SubmitToolResultRequest,
    ToolListResponse,
)
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import Message, MessageRole, TurnOutput
from lai.domain.errors import AuthorizationError, GuardrailBlockedError, ValidationFailedError
from lai.domain.tools import ApprovalStatus, ToolApprovalRecord, ToolResult
from lai.observability.logging import get_request_id
from lai.scheduler import await_or_keepalive_json

router = APIRouter(tags=["tools"])


async def _append_terminal_tool_response_if_pending(
    services: AppServices,
    approval: ToolApprovalRecord,
    *,
    error_code: str,
    detail: str,
) -> None:
    """Append a synthetic TOOL error reply when an approval terminates without a result."""
    async with services.session_gate.acquire(approval.session_id):
        session = await services.session_repo.get_session(approval.session_id)
        if session is None:
            return
        recent = await services.session_repo.list_recent_messages(approval.session_id, limit=50)
        effective_call_id = approval.tool_call_id or approval.id
        idx = len(recent) - 1
        answered_ids: set[str] = set()
        while idx >= 0 and recent[idx].role is MessageRole.TOOL:
            tool_call_id = recent[idx].tool_call_id
            if tool_call_id:
                answered_ids.add(tool_call_id)
            idx -= 1
        if (
            idx >= 0
            and recent[idx].role is MessageRole.ASSISTANT
            and any(tc.id == effective_call_id for tc in recent[idx].tool_calls)
            and effective_call_id not in answered_ids
        ):
            payload = json.dumps({"success": False, "error": error_code, "message": detail})
            await services.session_repo.add_message(
                Message(
                    session_id=approval.session_id,
                    project_id=approval.project_id,
                    role=MessageRole.TOOL,
                    content=payload,
                    tool_call_id=effective_call_id,
                    tool_name=approval.tool_name,
                )
            )


@router.get("/v1/tools", response_model=ToolListResponse)
async def list_tools(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.TOOLS_SAFE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ToolListResponse:
    """List registered tool plugins and their permission tiers."""
    specs = services.tool_registry.list_specs()
    return ToolListResponse(items=specs, count=len(specs))


@router.get("/v1/tool-approvals/{approval_id}", response_model=ToolApprovalRecord)
async def get_tool_approval(
    approval_id: str,
    principal: Annotated[
        AuthPrincipal,
        Depends(require_any_scope_dep(ApiScope.TOOLS_APPROVE, ApiScope.TOOLS_SAFE)),
    ],
    services: Annotated[AppServices, Depends(get_services)],
) -> ToolApprovalRecord:
    """Fetch status and details for an accessible tool approval request."""
    return await require_accessible_approval(services, approval_id, principal)


@router.post("/v1/tool-approvals/{approval_id}/resolve", response_model=ToolApprovalRecord)
async def resolve_tool_approval(
    approval_id: str,
    body: ResolveApprovalRequest,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.TOOLS_APPROVE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ToolApprovalRecord:
    """Approve or deny a pending tool execution request (requires TOOLS_APPROVE)."""
    await require_accessible_approval(services, approval_id, principal)
    resolved = await services.tool_executor.approve_and_execute(
        approval_id=approval_id,
        decided_by=principal.principal_id,
        approved=body.approved,
        note=body.note,
    )
    if not body.approved:
        await _append_terminal_tool_response_if_pending(
            services,
            resolved,
            error_code="tool_approval_denied",
            detail=body.note or "Tool approval denied by operator.",
        )
    return resolved


@router.post("/v1/tool-approvals/{approval_id}/result", response_model=TurnOutput)
async def submit_tool_result(
    approval_id: str,
    body: SubmitToolResultRequest,
    request: Request,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> Response:
    """Submit a client-executed tool result and resume the paused session turn."""
    client_app = await require_authenticated_client_app(services, principal)
    existing = await require_accessible_approval(services, approval_id, principal)
    if client_app.allowed_project_ids and existing.project_id not in client_app.allowed_project_ids:
        msg = f"Client app {client_app.id!r} is not authorized for project {existing.project_id!r}."
        raise AuthorizationError(
            msg,
            details={"client_app_id": client_app.id, "project_id": existing.project_id},
        )

    raw_result = ToolResult(
        tool_name=existing.tool_name,
        success=body.success,
        output=body.output,
        error=body.error,
        duration_ms=body.duration_ms,
    )
    try:
        sanitized = await services.tool_executor.submit_client_result(
            approval_id,
            client_app.id,
            raw_result,
            principal_id=principal.principal_id,
            is_admin=is_admin_principal(principal),
        )
    except GuardrailBlockedError as exc:
        policy = exc.violated_policy or "policy"
        await _append_terminal_tool_response_if_pending(
            services,
            existing,
            error_code="tool_output_blocked_by_guardrail",
            detail=f"Client tool output blocked by safety guardrail ({policy}).",
        )
        raise
    except ValidationFailedError:
        refreshed = await services.tool_audit_repo.get_approval(approval_id)
        if refreshed is not None and refreshed.status is ApprovalStatus.EXPIRED:
            await _append_terminal_tool_response_if_pending(
                services,
                refreshed,
                error_code="tool_approval_expired",
                detail="Client tool execution lease expired before result submission.",
            )
        raise
    updated_approval = (await services.tool_audit_repo.get_approval(approval_id)) or existing

    if body.stream:
        primed = await prime_sse_stream(
            track_turn_stream(
                services.orchestrator.resume_client_tool_stream(
                    updated_approval, sanitized, principal
                ),
                services,
                principal,
            ),
            keepalive_interval_seconds=services.settings.keepalive_interval_seconds,
            is_disconnected=request.is_disconnected,
        )
        headers = {
            **build_guardrail_headers(
                guardrail_blocked=primed.guardrail_blocked,
                guardrail_policy=primed.guardrail_policy,
                guardrail_direction=primed.guardrail_direction,
            ),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            guard_orchestrator_sse_frames(format_orchestrator_sse(primed.events)),
            media_type="text/event-stream",
            headers=headers,
        )

    req_id = get_request_id()
    outcome = await await_or_keepalive_json(
        track_turn_output(
            services.orchestrator.resume_client_tool(updated_approval, sanitized, principal),
            services,
            principal,
        ),
        threshold_seconds=services.settings.json_keepalive_threshold_seconds,
        interval_seconds=services.settings.keepalive_interval_seconds,
        is_disconnected=request.is_disconnected,
        error_serializer=lambda exc: serialize_post_header_error_json(exc, request_id=req_id),
    )
    if isinstance(outcome, TurnOutput):
        headers = build_guardrail_headers(
            guardrail_blocked=outcome.guardrail_blocked,
            guardrail_policy=outcome.guardrail_policy,
            guardrail_direction=outcome.guardrail_direction,
        )
        return JSONResponse(content=outcome.model_dump(mode="json"), headers=headers)
    return StreamingResponse(outcome, media_type="application/json")
