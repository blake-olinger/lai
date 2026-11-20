"""OIDC JWT authentication provider with JWKS TTL caching and claim enforcement."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import jwt
from pydantic import Field

from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal, RequestCredentials
from lai.domain.base import StrictModel
from lai.domain.errors import AuthenticationError
from lai.observability.logging import get_component_logger
from lai.ports.auth import AuthProviderPort
from lai.security._jwks import JwksCache, JwksCachePolicy, JwksFetchSpec, fetch_jwks_payload
from lai.security._oidc_claims import (
    resolve_oauth_client_id,
    resolve_principal_identity,
    resolve_scopes,
    validate_email_policy,
    validate_guest_policy,
)
from lai.security.keys import API_KEY_PREFIX

if TYPE_CHECKING:
    import httpx

JwksFetcher = Callable[[str], Awaitable[dict[str, Any]]]
_JWT_PART_COUNT = 3


class OidcProviderConfig(StrictModel):
    """Configuration for OIDC JWT validation and principal mapping."""

    enabled: bool = False
    issuer_url: str = "https://accounts.google.com"
    jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    audience: str | None = None
    audiences: tuple[str, ...] = ()
    allowed_client_ids: tuple[str, ...] = ()
    allowed_emails: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    default_scopes: tuple[ApiScope, ...] = (
        ApiScope.PROJECTS_READ,
        ApiScope.PROJECTS_WRITE,
        ApiScope.FILES_READ,
        ApiScope.FILES_WRITE,
        ApiScope.CHAT_READ,
        ApiScope.CHAT_WRITE,
        ApiScope.TOOLS_SAFE,
    )
    admin_emails: tuple[str, ...] = ()
    admin_roles: tuple[str, ...] = ()
    admin_subjects: tuple[str, ...] = ()
    allow_guest_users: bool = False
    algorithms: tuple[str, ...] = ("RS256", "ES256")
    jwks_cache_ttl_seconds: float = Field(default=3600.0, gt=0.0)
    jwks_timeout_seconds: float = Field(default=5.0, gt=0.0)
    min_jwks_refresh_interval_seconds: float = Field(default=30.0, ge=0.0)
    require_email_verified: bool = True
    identity_claims: tuple[str, ...] = ("email", "preferred_username", "upn")
    leeway_seconds: float = Field(default=0.0, ge=0.0)

    def resolved_audiences(self) -> tuple[str, ...]:
        """Return deduplicated, non-empty audiences from audiences and legacy audience."""
        items = [item.strip() for item in self.audiences if item.strip()]
        if self.audience is not None:
            legacy = self.audience.strip()
            if legacy:
                items.append(legacy)
        return tuple(dict.fromkeys(items))

    def resolved_allowed_client_ids(self) -> tuple[str, ...]:
        """Return deduplicated, non-empty calling OAuth client IDs (azp/appid)."""
        return tuple(
            dict.fromkeys(item.strip() for item in self.allowed_client_ids if item.strip())
        )


def _extract_jwt_bearer(credentials: RequestCredentials) -> str | None:
    """Return a 3-part JWT bearer token (excluding `lai_sk_` API keys)."""
    if (
        credentials.authorization_scheme is not None
        and credentials.authorization_scheme.lower() != "bearer"
    ):
        return None
    token = credentials.bearer_token
    if not token or token.startswith(API_KEY_PREFIX):
        return None
    parts = token.split(".")
    if len(parts) != _JWT_PART_COUNT or not all(parts):
        return None
    return token


class OidcJwtAuthProvider(AuthProviderPort):
    """Validate OIDC RS256/ES256 ID/access JWTs using TTL-cached JWKS keys."""

    def __init__(
        self,
        config: OidcProviderConfig,
        *,
        jwks_fetcher: JwksFetcher | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._jwks_fetcher = jwks_fetcher
        self._http_client = http_client
        self._jwks_cache = JwksCache(
            fetcher=self._fetch_jwks,
            policy=JwksCachePolicy(
                ttl_seconds=config.jwks_cache_ttl_seconds,
                min_refresh_interval_seconds=config.min_jwks_refresh_interval_seconds,
                can_fetch=True,
                log_prefix="oidc",
            ),
            clock=clock,
        )

    @property
    def name(self) -> str:
        """Return provider identifier."""
        return "oidc"

    def reconfigure(self, config: OidcProviderConfig) -> None:
        """Update OIDC policy in place, invalidating cached JWKS only if jwks_url changed."""
        if config.jwks_url != self._config.jwks_url:
            self._jwks_cache.invalidate()
        self._config = config

    def supports(self, credentials: RequestCredentials) -> bool:
        """Return True when OIDC is enabled and credentials carry a 3-part JWT."""
        if not self._config.enabled:
            return False
        return _extract_jwt_bearer(credentials) is not None

    async def authenticate(self, credentials: RequestCredentials) -> AuthPrincipal:
        """Verify JWT signature and OIDC claims, returning an AuthPrincipal."""
        if not self._config.enabled:
            msg = "OIDC authentication is disabled."
            raise AuthenticationError(msg)

        audiences = self._config.resolved_audiences()
        if not audiences:
            get_component_logger("security").warning(
                "oidc_auth_failed", reason="missing_audiences_config"
            )
            msg = "OIDC authentication requires at least one configured audience."
            raise AuthenticationError(msg)

        token = _extract_jwt_bearer(credentials)
        if token is None:
            get_component_logger("security").warning(
                "oidc_auth_failed", reason="missing_or_malformed_jwt"
            )
            msg = "Missing or malformed OIDC JWT."
            raise AuthenticationError(msg)

        alg, kid = self._read_unverified_header(token)
        signing_key = await self._resolve_signing_key(kid)
        claims = self._decode_and_verify(token, signing_key, alg, audiences)

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            get_component_logger("security").warning("oidc_auth_failed", reason="missing_subject")
            msg = "OIDC token missing subject claim."
            raise AuthenticationError(msg)

        resolved_client_id = resolve_oauth_client_id(
            claims,
            audiences=audiences,
            allowed_client_ids=self._config.resolved_allowed_client_ids(),
        )
        issuer = claims.get("iss")
        issuer_str = (
            issuer.strip()
            if isinstance(issuer, str) and issuer.strip()
            else self._config.issuer_url
        )
        validate_guest_policy(self._config, claims, issuer=issuer_str)
        email = validate_email_policy(self._config, claims)
        identity_key = resolve_principal_identity(claims)
        scopes = resolve_scopes(self._config, claims, email=email, identity_key=identity_key)

        return AuthPrincipal(
            principal_id=f"oidc:{issuer_str.rstrip('/')}:{identity_key}",
            auth_method=AuthMethod.OIDC,
            scopes=scopes,
            email=email,
            issuer=issuer_str,
            subject=subject.strip(),
            oauth_client_id=resolved_client_id,
        )

    def _read_unverified_header(self, token: str) -> tuple[str, str | None]:
        """Parse JWT header and enforce allowed asymmetric algorithms."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            get_component_logger("security").warning(
                "oidc_auth_failed", reason="invalid_jwt_header", error=str(exc)
            )
            msg = f"Invalid OIDC token header: {exc}"
            raise AuthenticationError(msg) from exc

        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in self._config.algorithms:
            get_component_logger("security").warning(
                "oidc_auth_failed", reason="unsupported_alg", alg=str(alg)
            )
            msg = f"Unsupported OIDC JWT algorithm: {alg!r}"
            raise AuthenticationError(msg)

        raw_kid = header.get("kid")
        kid = raw_kid if isinstance(raw_kid, str) and raw_kid else None
        return alg, kid

    async def _resolve_signing_key(self, kid: str | None) -> jwt.PyJWK:
        """Resolve signing key from cached JWKS, refreshing on expiry/unknown kid with cooldown."""
        return await self._jwks_cache.resolve_key(
            kid,
            error_prefix="Invalid OIDC JWK",
            missing_key_message="No matching JWKS key found for token.",
        )

    async def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch JWKS JSON from configured URL, logging timeouts as TIMEOUT alerts."""
        return await fetch_jwks_payload(
            JwksFetchSpec(
                url=self._config.jwks_url,
                timeout_seconds=self._config.jwks_timeout_seconds,
                label="OIDC",
                log_prefix="oidc",
            ),
            http_client=self._http_client,
            custom_fetcher=self._jwks_fetcher,
        )

    def _decode_and_verify(
        self,
        token: str,
        signing_key: jwt.PyJWK,
        algorithm: str,
        audiences: tuple[str, ...],
    ) -> dict[str, Any]:
        """Verify signature, exp, iss, sub, and aud using PyJWT."""
        if not audiences:
            msg = "OIDC authentication requires at least one configured audience."
            raise AuthenticationError(msg)
        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                issuer=self._config.issuer_url,
                audience=list(audiences),
                leeway=self._config.leeway_seconds,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                    "require": ["exp", "iss", "sub", "aud"],
                },
            )
        except jwt.PyJWTError as exc:
            get_component_logger("security").warning(
                "oidc_auth_failed",
                reason="jwt_verification_failed",
                error=str(exc),
                exception_type=type(exc).__name__,
            )
            msg = f"OIDC JWT verification failed: {exc}"
            raise AuthenticationError(msg) from exc
