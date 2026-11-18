"""Tool specification listing, approval resolution, and client result submission routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lai.api._client_auth import require_authenticated_client_app
from lai.api._streaming import (
    format_orchestrator_sse,
    guard_orchestrator_sse_frames,
    prime_sse_stream,
)
from lai.api.deps import (
    AppServices,
    get_services,
    require_any_scope_dep,
    require_scope_dep,
)
from lai.api.schemas.tools import (
    ResolveApprovalRequest,
    SubmitToolResultRequest,
    ToolListResponse,
)
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import TurnOutput
from lai.domain.errors import AuthorizationError, NotFoundError
from lai.domain.tools import ToolApprovalRecord, ToolResult
from lai.scheduler import await_or_keepalive_json

router = APIRouter(tags=["tools"])


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
    _principal: Annotated[
        AuthPrincipal,
        Depends(require_any_scope_dep(ApiScope.TOOLS_APPROVE, ApiScope.TOOLS_SAFE)),
    ],
    services: Annotated[AppServices, Depends(get_services)],
) -> ToolApprovalRecord:
    """Fetch status and details for a tool approval request."""
    record = await services.tool_audit_repo.get_approval(approval_id)
    if record is None:
        msg = f"Tool approval {approval_id!r} not found."
        raise NotFoundError(msg, details={"approval_id": approval_id})
    return record


@router.post("/v1/tool-approvals/{approval_id}/resolve", response_model=ToolApprovalRecord)
async def resolve_tool_approval(
    approval_id: str,
    body: ResolveApprovalRequest,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.TOOLS_APPROVE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ToolApprovalRecord:
    """Approve or deny a pending tool execution request (requires TOOLS_APPROVE)."""
    return await services.tool_executor.approve_and_execute(
        approval_id=approval_id,
        decided_by=principal.principal_id,
        approved=body.approved,
        note=body.note,
    )


@router.post("/v1/tool-approvals/{approval_id}/result", response_model=TurnOutput)
async def submit_tool_result(  # noqa: PLR0913, PLR0917
    approval_id: str,
    body: SubmitToolResultRequest,
    request: Request,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
    x_lai_client_id: Annotated[str | None, Header(alias="X-LAI-Client-Id")] = None,
    x_lai_client_secret: Annotated[str | None, Header(alias="X-LAI-Client-Secret")] = None,
) -> Response:
    """Submit a client-executed tool result and resume the paused session turn."""
    client_app = await require_authenticated_client_app(
        services, x_lai_client_id, x_lai_client_secret
    )
    existing = await services.tool_audit_repo.get_approval(approval_id)
    if existing is None:
        msg = f"Tool approval {approval_id!r} not found."
        raise NotFoundError(msg, details={"approval_id": approval_id})
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
    sanitized = await services.tool_executor.submit_client_result(
        approval_id,
        client_app.id,
        raw_result,
        principal_id=principal.principal_id,
    )
    updated_approval = (await services.tool_audit_repo.get_approval(approval_id)) or existing

    if body.stream:
        primed = await prime_sse_stream(
            services.orchestrator.resume_client_tool_stream(updated_approval, sanitized, principal),
            keepalive_interval_seconds=services.settings.keepalive_interval_seconds,
            is_disconnected=request.is_disconnected,
        )
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(
            guard_orchestrator_sse_frames(format_orchestrator_sse(primed.events)),
            media_type="text/event-stream",
            headers=headers,
        )

    outcome = await await_or_keepalive_json(
        services.orchestrator.resume_client_tool(updated_approval, sanitized, principal),
        threshold_seconds=services.settings.json_keepalive_threshold_seconds,
        interval_seconds=services.settings.keepalive_interval_seconds,
        is_disconnected=request.is_disconnected,
    )
    if isinstance(outcome, TurnOutput):
        return JSONResponse(content=outcome.model_dump(mode="json"))
    return StreamingResponse(outcome, media_type="application/json")
