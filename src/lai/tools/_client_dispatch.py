"""Client-plane tool dispatch and lease issuance."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, NoReturn

from lai.domain.base import utc_now
from lai.domain.errors import AuthorizationError, ClientToolDispatchError
from lai.domain.tools import (
    ApprovalStatus,
    PolicyDecision,
    ToolApprovalRecord,
    ToolExecutionTarget,
)
from lai.tools._audit import InvocationMeta, build_audit_record, normalize_optional_id

if TYPE_CHECKING:
    from lai.domain.tools import ToolExecutionContext, ToolSpec
    from lai.ports.repositories import ToolAuditRepositoryPort
    from lai.ports.tools import ToolPolicyPort


async def dispatch_client_tool(  # noqa: PLR0913
    *,
    spec: ToolSpec,
    arguments: dict[str, Any],
    arguments_json: str,
    context: ToolExecutionContext,
    policy: ToolPolicyPort,
    audit_repo: ToolAuditRepositoryPort,
    lease_ttl_seconds: float,
) -> NoReturn:
    """Evaluate client tool policy, persist approval/lease, and raise dispatch error."""
    decision = policy.evaluate(spec, arguments, context)
    meta = InvocationMeta(
        context=context,
        spec=spec,
        decision=decision,
        arguments_json=arguments_json,
        approval_id=normalize_optional_id(context.approved_approval_id),
    )

    if decision is PolicyDecision.DENY:
        await audit_repo.record_audit(
            build_audit_record(meta, success=False, error="policy_denied")
        )
        msg = f"Principal {context.principal_id!r} is denied access to client tool {spec.name!r}."
        raise AuthorizationError(
            msg,
            details={"tool_name": spec.name, "permission_tier": spec.permission_tier.value},
        )

    if meta.approval_id is not None:
        await audit_repo.record_audit(
            build_audit_record(meta, success=False, error="client_approval_replay_forbidden")
        )
        msg = f"Client tool {spec.name!r} cannot be invoked directly with an approval ID."
        raise AuthorizationError(
            msg,
            details={"tool_name": spec.name, "approval_id": meta.approval_id},
        )

    if decision is PolicyDecision.REQUIRE_APPROVAL:
        await _persist_pending_and_raise(
            spec=spec,
            arguments_json=arguments_json,
            context=context,
            decision=decision,
            audit_repo=audit_repo,
        )

    await _issue_client_lease_and_raise(
        spec=spec,
        arguments_json=arguments_json,
        context=context,
        decision=decision,
        audit_repo=audit_repo,
        lease_ttl_seconds=lease_ttl_seconds,
    )


async def _persist_pending_and_raise(
    *,
    spec: ToolSpec,
    arguments_json: str,
    context: ToolExecutionContext,
    decision: PolicyDecision,
    audit_repo: ToolAuditRepositoryPort,
) -> NoReturn:
    approval = ToolApprovalRecord(
        project_id=context.project_id,
        session_id=context.session_id,
        tool_name=spec.name,
        arguments_json=arguments_json,
        permission_tier=spec.permission_tier,
        reason=(
            f"Client tool {spec.name!r} ({spec.permission_tier.value}) requires explicit approval."
        ),
        status=ApprovalStatus.PENDING,
        requested_by=context.principal_id,
        execution_target=ToolExecutionTarget.CLIENT,
        client_app_id=context.client_app_id,
        tool_call_id=context.tool_call_id,
        allowed_file_ids=context.allowed_file_ids,
        principal_scopes=context.principal_scopes,
    )
    saved = await audit_repo.create_approval(approval)
    pending_meta = InvocationMeta(
        context=context,
        spec=spec,
        decision=decision,
        arguments_json=arguments_json,
        approval_id=saved.id,
    )
    await audit_repo.record_audit(
        build_audit_record(pending_meta, success=False, error="approval_required")
    )
    msg = f"Client tool {spec.name!r} requires explicit approval."
    raise ClientToolDispatchError(
        msg,
        approval_id=saved.id,
        tool_call_id=context.tool_call_id or saved.id,
        tool_name=spec.name,
        arguments_json=arguments_json,
        requires_approval=True,
        expires_at=None,
        details={"permission_tier": spec.permission_tier.value},
    )


async def _issue_client_lease_and_raise(  # noqa: PLR0913
    *,
    spec: ToolSpec,
    arguments_json: str,
    context: ToolExecutionContext,
    decision: PolicyDecision,
    audit_repo: ToolAuditRepositoryPort,
    lease_ttl_seconds: float,
) -> NoReturn:
    now = utc_now()
    expires_at = now + timedelta(seconds=lease_ttl_seconds)
    lease = ToolApprovalRecord(
        project_id=context.project_id,
        session_id=context.session_id,
        tool_name=spec.name,
        arguments_json=arguments_json,
        permission_tier=spec.permission_tier,
        reason=f"Client read tool {spec.name!r} granted execution lease.",
        status=ApprovalStatus.APPROVED,
        requested_by=context.principal_id,
        decided_by=context.principal_id,
        decided_at=now,
        execution_target=ToolExecutionTarget.CLIENT,
        client_app_id=context.client_app_id,
        tool_call_id=context.tool_call_id,
        allowed_file_ids=context.allowed_file_ids,
        principal_scopes=context.principal_scopes,
        expires_at=expires_at,
    )
    saved = await audit_repo.create_approval(lease)
    lease_meta = InvocationMeta(
        context=context,
        spec=spec,
        decision=decision,
        arguments_json=arguments_json,
        approval_id=saved.id,
    )
    await audit_repo.record_audit(
        build_audit_record(lease_meta, success=True, error="client_lease_issued")
    )
    msg = f"Client tool {spec.name!r} dispatched for client execution."
    raise ClientToolDispatchError(
        msg,
        approval_id=saved.id,
        tool_call_id=context.tool_call_id or saved.id,
        tool_name=spec.name,
        arguments_json=arguments_json,
        requires_approval=False,
        expires_at=expires_at,
        details={"permission_tier": spec.permission_tier.value},
    )
