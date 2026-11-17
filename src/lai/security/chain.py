"""Composite authenticator chaining optional edge verification with auth providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.errors import AuthenticationError
from lai.observability.logging import get_component_logger
from lai.ports.auth import AuthenticatorPort, AuthProviderPort, EdgeVerifierPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.auth import AuthPrincipal, RequestCredentials


class CompositeAuthenticator(AuthenticatorPort):
    """Run optional EdgeVerifierPort before dispatching to matching AuthProviderPorts."""

    def __init__(
        self,
        *,
        providers: Sequence[AuthProviderPort] = (),
        edge_verifier: EdgeVerifierPort | None = None,
    ) -> None:
        self._providers: list[AuthProviderPort] = []
        self._edge_verifier = edge_verifier
        for provider in providers:
            self.register_provider(provider)

    def register_provider(self, provider: AuthProviderPort) -> None:
        """Register or replace an authentication provider by unique name."""
        for index, existing in enumerate(self._providers):
            if existing.name == provider.name:
                self._providers[index] = provider
                return
        self._providers.append(provider)

    async def authenticate(self, credentials: RequestCredentials) -> AuthPrincipal:
        """Validate edge assertions and authenticate via the first matching provider."""
        if self._edge_verifier is not None:
            await self._edge_verifier.verify_edge(credentials)

        for provider in self._providers:
            if provider.supports(credentials):
                return await provider.authenticate(credentials)

        get_component_logger("security").warning(
            "authentication_failed",
            reason="no_matching_provider",
            scheme=credentials.authorization_scheme,
        )
        msg = "Missing or unsupported authentication credentials."
        raise AuthenticationError(msg)
