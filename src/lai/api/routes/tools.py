"""Tool specification listing and approval resolution routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from lai.api.deps import (
    AppServices,
    get_services,
    require_any_scope_dep,
    require_scope_dep,
)
from lai.api.schemas.tools import ResolveApprovalRequest, ToolListResponse
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.errors import NotFoundError
from lai.domain.tools import ToolApprovalRecord

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
