"""Default scope- and approval-based tool execution policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lai.domain.auth import ApiScope
from lai.domain.tools import PermissionTier, PolicyDecision, ToolExecutionTarget
from lai.ports.tools import ToolPolicyPort

if TYPE_CHECKING:
    from lai.domain.tools import ToolExecutionContext, ToolSpec

_READ_ALLOWED_SCOPES: frozenset[ApiScope] = frozenset(
    {
        ApiScope.ADMIN,
        ApiScope.TOOLS_SAFE,
        ApiScope.CHAT_WRITE,
        ApiScope.FILES_READ,
    }
)

_PRIVILEGED_ALLOWED_SCOPES: frozenset[ApiScope] = frozenset(
    {
        ApiScope.ADMIN,
        ApiScope.TOOLS_EXEC,
    }
)

_TIER_REQUIRED_SCOPES: dict[PermissionTier, frozenset[ApiScope]] = {
    PermissionTier.READ: _READ_ALLOWED_SCOPES,
    PermissionTier.WRITE: _PRIVILEGED_ALLOWED_SCOPES,
    PermissionTier.EXEC: _PRIVILEGED_ALLOWED_SCOPES,
    PermissionTier.NETWORK: _PRIVILEGED_ALLOWED_SCOPES,
}


def _has_approval_id(approval_id: str | None) -> bool:
    return bool(approval_id and approval_id.strip())


def _evaluate_server_attachment(
    spec: ToolSpec,
    caller_scopes: frozenset[ApiScope],
    context: ToolExecutionContext,
) -> PolicyDecision:
    if spec.permission_tier is not PermissionTier.READ:
        return PolicyDecision.DENY
    if not (caller_scopes & _TIER_REQUIRED_SCOPES[PermissionTier.READ]):
        return PolicyDecision.DENY
    if spec.requires_approval_by_default:
        return (
            PolicyDecision.ALLOW
            if _has_approval_id(context.approved_approval_id)
            else PolicyDecision.REQUIRE_APPROVAL
        )
    return PolicyDecision.ALLOW


def _evaluate_client_tool(
    spec: ToolSpec,
    caller_scopes: frozenset[ApiScope],
    context: ToolExecutionContext,
) -> PolicyDecision:
    if not (context.client_app_id and context.client_app_id.strip()):
        return PolicyDecision.DENY
    if spec.permission_tier not in context.client_allowed_tiers:
        return PolicyDecision.DENY
    if not (caller_scopes & _TIER_REQUIRED_SCOPES[spec.permission_tier]):
        return PolicyDecision.DENY
    if spec.requires_approval_by_default or spec.permission_tier is not PermissionTier.READ:
        return (
            PolicyDecision.ALLOW
            if _has_approval_id(context.approved_approval_id)
            else PolicyDecision.REQUIRE_APPROVAL
        )
    return PolicyDecision.ALLOW


class DefaultToolPolicy(ToolPolicyPort):
    """Evaluates dual-plane tool permission tiers against principal, app, and approval state."""

    def evaluate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PolicyDecision:
        """Return ALLOW, REQUIRE_APPROVAL, or DENY for a tool invocation."""
        del arguments
        caller_scopes = frozenset(context.principal_scopes)
        if spec.execution_target is ToolExecutionTarget.CLIENT:
            return _evaluate_client_tool(spec, caller_scopes, context)
        return _evaluate_server_attachment(spec, caller_scopes, context)
