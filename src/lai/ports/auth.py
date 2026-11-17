"""Pluggable authentication provider and edge verifier protocols (PEP 544)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lai.domain.auth import AuthPrincipal, RequestCredentials


@runtime_checkable
class AuthProviderPort(Protocol):
    """Pluggable authentication strategy (API key, OIDC JWT, local CLI, etc.)."""

    @property
    def name(self) -> str:
        """Return the unique identifier for this auth provider."""
        ...

    def supports(self, credentials: RequestCredentials) -> bool:
        """Return True if this provider can evaluate the given credentials."""
        ...

    async def authenticate(self, credentials: RequestCredentials) -> AuthPrincipal:
        """Validate credentials and return the authenticated principal."""
        ...


@runtime_checkable
class EdgeVerifierPort(Protocol):
    """Optional network edge verification guard (e.g., Cloudflare Access)."""

    async def verify_edge(self, credentials: RequestCredentials) -> None:
        """Verify edge assertions or raise AuthenticationError/AuthorizationError."""
        ...


@runtime_checkable
class AuthenticatorPort(Protocol):
    """Composite authenticator dispatching across registered auth providers."""

    def register_provider(self, provider: AuthProviderPort) -> None:
        """Register an authentication provider strategy."""
        ...

    async def authenticate(self, credentials: RequestCredentials) -> AuthPrincipal:
        """Verify edge assertions and authenticate credentials via a matching provider."""
        ...
