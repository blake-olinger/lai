"""Default scope- and approval-based tool execution policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lai.domain.auth import ApiScope
from lai.domain.tools import PermissionTier, PolicyDecision
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


class DefaultToolPolicy(ToolPolicyPort):
    """Evaluates tool permission tiers against principal scopes and approval state."""

    def evaluate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PolicyDecision:
        """Return ALLOW, REQUIRE_APPROVAL, or DENY for a tool invocation."""
        del arguments
        required_scopes = _TIER_REQUIRED_SCOPES[spec.permission_tier]
        caller_scopes = frozenset(context.principal_scopes)
        if not (caller_scopes & required_scopes):
            return PolicyDecision.DENY

        if spec.requires_approval_by_default:
            has_approval = bool(
                context.approved_approval_id and context.approved_approval_id.strip()
            )
            return PolicyDecision.ALLOW if has_approval else PolicyDecision.REQUIRE_APPROVAL

        return PolicyDecision.ALLOW
