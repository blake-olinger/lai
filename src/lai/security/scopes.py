"""Hierarchical API scope expansion and enforcement helpers."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.errors import AuthorizationError
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

_IMPLIED_SCOPES: dict[ApiScope, tuple[ApiScope, ...]] = {
    ApiScope.ADMIN: tuple(ApiScope),
    ApiScope.PROJECTS_WRITE: (ApiScope.PROJECTS_READ,),
    ApiScope.FILES_WRITE: (ApiScope.FILES_READ,),
    ApiScope.CHAT_WRITE: (ApiScope.CHAT_READ,),
    ApiScope.TOOLS_EXEC: (ApiScope.TOOLS_SAFE,),
}


def expand_scopes(scopes: Iterable[ApiScope]) -> frozenset[ApiScope]:
    """Return the transitive closure of granted scopes under the scope hierarchy."""
    granted: set[ApiScope] = set()
    pending: deque[ApiScope] = deque(scopes)
    while pending:
        scope = pending.popleft()
        if scope in granted:
            continue
        granted.add(scope)
        for implied in _IMPLIED_SCOPES.get(scope, ()):
            if implied not in granted:
                pending.append(implied)
    return frozenset(granted)


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
