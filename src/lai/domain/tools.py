"""Guarded tool specification, policy evaluation, approval, and audit models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from lai.domain.auth import ApiScope
from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_approval_id, new_audit_id


class PermissionTier(StrEnum):
    """Risk classification governing default tool execution policy."""

    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    NETWORK = "network"


class PolicyDecision(StrEnum):
    """Policy verdict produced before executing a tool call."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class ApprovalStatus(StrEnum):
    """Lifecycle status of an explicit human tool approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    EXECUTED = "executed"


class ToolSpec(StrictModel):
    """Declarative specification for a registered tool plugin."""

    name: str = Field(min_length=1)
    description: str
    permission_tier: PermissionTier
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    requires_approval_by_default: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class ToolExecutionContext(StrictModel):
    """Caller, project, and workspace confinement context for a tool call."""

    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    principal_scopes: tuple[ApiScope, ...] = ()
    allowed_file_ids: tuple[str, ...] = ()
    workspace_root: str | None = None
    approved_approval_id: str | None = None
    request_id: str | None = None


class ToolResult(StrictModel):
    """Normalized output returned by a tool execution."""

    tool_name: str = Field(min_length=1)
    success: bool
    output: str
    error: str | None = None
    truncated: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)


class ToolApprovalRecord(StrictModel):
    """Persisted approval gate record for a high-risk tool call."""

    id: str = Field(default_factory=new_approval_id)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments_json: str
    permission_tier: PermissionTier
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: str = Field(min_length=1)
    decided_by: str | None = None
    decision_note: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None


class ToolAuditRecord(StrictModel):
    """Immutable audit log entry capturing every evaluated tool invocation."""

    id: str = Field(default_factory=new_audit_id)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    permission_tier: PermissionTier
    policy_decision: PolicyDecision
    arguments_json: str
    success: bool
    output_preview: str = ""
    error: str | None = None
    approval_id: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=utc_now)
