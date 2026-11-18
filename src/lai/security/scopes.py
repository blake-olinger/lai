"""Hierarchical API scope expansion and enforcement helpers."""

from __future__ import annotations

from lai.domain.auth import ApiScope, AuthPrincipal, expand_scopes
from lai.domain.errors import AuthorizationError
from lai.observability.logging import get_component_logger

__all__ = ["expand_scopes", "has_effective_scope", "require_scope"]


def has_effective_scope(principal: AuthPrincipal, required: ApiScope) -> bool:
    """Return True when the principal holds `required` directly or via hierarchy."""
    return required in expand_scopes(principal.scopes)


def require_scope(principal: AuthPrincipal, required: ApiScope) -> None:
    """Raise AuthorizationError unless the principal holds the required effective scope."""
    if has_effective_scope(principal, required):
        return
    get_component_logger("security").warning(
        "authorization_denied",
        principal_id=principal.principal_id,
        required_scope=required.value,
        granted_scopes=[scope.value for scope in principal.scopes],
    )
    msg = f"Principal {principal.principal_id!r} lacks required scope {required.value!r}."
    raise AuthorizationError(
        msg,
        details={
            "principal_id": principal.principal_id,
            "required_scope": required.value,
        },
    )
