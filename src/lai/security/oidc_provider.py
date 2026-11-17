"""OIDC JWT authentication provider with JWKS TTL caching and claim enforcement."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import httpx
import jwt
from pydantic import Field

from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal, RequestCredentials
from lai.domain.base import StrictModel
from lai.domain.errors import AuthenticationError, AuthorizationError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger
from lai.ports.auth import AuthProviderPort
from lai.security.keys import API_KEY_PREFIX

JwksFetcher = Callable[[str], Awaitable[dict[str, Any]]]
_JWT_PART_COUNT = 3


class OidcProviderConfig(StrictModel):
    """Configuration for OIDC JWT validation and principal mapping."""

    enabled: bool = False
    issuer_url: str = "https://accounts.google.com"
    jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    audience: str | None = None
    allowed_emails: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    default_scopes: tuple[ApiScope, ...] = (
        ApiScope.PROJECTS_READ,
        ApiScope.FILES_READ,
        ApiScope.FILES_WRITE,
        ApiScope.CHAT_READ,
        ApiScope.CHAT_WRITE,
        ApiScope.TOOLS_SAFE,
    )
    admin_emails: tuple[str, ...] = ()
    algorithms: tuple[str, ...] = ("RS256", "ES256")
    jwks_cache_ttl_seconds: float = Field(default=3600.0, gt=0.0)
    jwks_timeout_seconds: float = Field(default=5.0, gt=0.0)
    require_email_verified: bool = True
    leeway_seconds: float = Field(default=0.0, ge=0.0)


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


def _find_jwk_for_kid(jwks: Mapping[str, Any], kid: str | None) -> jwt.PyJWK | None:
    """Locate a signing PyJWK from a JWKS dictionary by `kid` (or sole key)."""
    raw_keys = jwks.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        return None
    keys_list = cast("list[object]", raw_keys)
    dict_keys = [cast("dict[str, Any]", item) for item in keys_list if isinstance(item, dict)]
    if not dict_keys:
        return None
    if kid is not None:
        for entry in dict_keys:
            if entry.get("kid") == kid:
                return jwt.PyJWK.from_dict(entry)
        return None
    if len(dict_keys) == 1:
        return jwt.PyJWK.from_dict(dict_keys[0])
    return None


def _validate_email_policy(config: OidcProviderConfig, claims: Mapping[str, Any]) -> str | None:
    """Enforce email verification and email/domain allowlists, returning normalized email."""
    raw_email = claims.get("email")
    email = raw_email.strip().lower() if isinstance(raw_email, str) and raw_email.strip() else None

    if config.require_email_verified and claims.get("email_verified") is not True:
        get_component_logger("security").warning("oidc_auth_failed", reason="email_not_verified")
        msg = "OIDC token email is not verified."
        raise AuthenticationError(msg)

    has_allowlist = bool(config.allowed_emails or config.allowed_domains)
    if not has_allowlist:
        return email

    if email is None:
        get_component_logger("security").warning(
            "oidc_auth_denied", reason="missing_email_for_allowlist"
        )
        msg = "OIDC token missing email required by allowlist."
        raise AuthorizationError(msg)

    allowed_emails = {item.strip().lower() for item in config.allowed_emails}
    allowed_domains = {item.strip().lower().lstrip("@") for item in config.allowed_domains}
    domain = email.partition("@")[2]
    if email in allowed_emails or (domain and domain in allowed_domains):
        return email

    get_component_logger("security").warning(
        "oidc_auth_denied", reason="email_not_allowed", email=email
    )
    msg = "OIDC caller email is not permitted."
    raise AuthorizationError(msg, details={"email": email})


def _resolve_scopes(config: OidcProviderConfig, email: str | None) -> tuple[ApiScope, ...]:
    """Map admin emails to ADMIN scope and other principals to default_scopes."""
    if email is not None:
        admin_emails = {item.strip().lower() for item in config.admin_emails}
        if email in admin_emails:
            return (ApiScope.ADMIN,)
    return config.default_scopes


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
        self._clock = clock
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_expires_at: float = 0.0
        self._jwks_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Return provider identifier."""
        return "oidc"

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

        token = _extract_jwt_bearer(credentials)
        if token is None:
            get_component_logger("security").warning(
                "oidc_auth_failed", reason="missing_or_malformed_jwt"
            )
            msg = "Missing or malformed OIDC JWT."
            raise AuthenticationError(msg)

        alg, kid = self._read_unverified_header(token)
        signing_key = await self._resolve_signing_key(kid)
        claims = self._decode_and_verify(token, signing_key, alg)

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            get_component_logger("security").warning("oidc_auth_failed", reason="missing_subject")
            msg = "OIDC token missing subject claim."
            raise AuthenticationError(msg)

        issuer = claims.get("iss")
        issuer_str = issuer if isinstance(issuer, str) else self._config.issuer_url
        email = _validate_email_policy(self._config, claims)
        scopes = _resolve_scopes(self._config, email)

        return AuthPrincipal(
            principal_id=f"oidc:{subject}",
            auth_method=AuthMethod.OIDC,
            scopes=scopes,
            email=email,
            issuer=issuer_str,
            subject=subject,
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
        """Resolve signing key from cached JWKS, refreshing on expiry or unknown kid."""
        now = self._clock()
        if self._jwks_cache is not None and now < self._jwks_expires_at:
            cached_key = _find_jwk_for_kid(self._jwks_cache, kid)
            if cached_key is not None:
                return cached_key

        async with self._jwks_lock:
            now = self._clock()
            if self._jwks_cache is not None and now < self._jwks_expires_at:
                cached_key = _find_jwk_for_kid(self._jwks_cache, kid)
                if cached_key is not None:
                    return cached_key

            jwks = await self._fetch_jwks()
            self._jwks_cache = jwks
            self._jwks_expires_at = self._clock() + self._config.jwks_cache_ttl_seconds

        refreshed_key = _find_jwk_for_kid(jwks, kid)
        if refreshed_key is None:
            get_component_logger("security").warning(
                "oidc_auth_failed", reason="unknown_kid", kid=kid
            )
            msg = "No matching JWKS key found for token."
            raise AuthenticationError(msg)
        return refreshed_key

    async def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch JWKS JSON from configured URL, logging timeouts as TIMEOUT alerts."""
        try:
            if self._jwks_fetcher is not None:
                payload = await self._jwks_fetcher(self._config.jwks_url)
            else:
                payload = await self._http_get_jwks()
        except (httpx.TimeoutException, TimeoutError) as exc:
            get_component_logger("security").error(
                "oidc_jwks_fetch_timeout",
                jwks_url=self._config.jwks_url,
                timeout_seconds=self._config.jwks_timeout_seconds,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = "Timed out fetching OIDC JWKS."
            raise AuthenticationError(msg) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            get_component_logger("security").error(
                "oidc_jwks_fetch_failed",
                jwks_url=self._config.jwks_url,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
            msg = f"Failed to fetch OIDC JWKS: {exc}"
            raise AuthenticationError(msg) from exc

        if not isinstance(payload.get("keys"), list):
            get_component_logger("security").error(
                "oidc_jwks_invalid_payload", jwks_url=self._config.jwks_url
            )
            msg = "Invalid OIDC JWKS payload: missing 'keys' list."
            raise AuthenticationError(msg)
        return payload

    async def _http_get_jwks(self) -> dict[str, Any]:
        """Fetch JWKS JSON via httpx using injected or short-lived AsyncClient."""
        if self._http_client is not None:
            response = await self._http_client.get(
                self._config.jwks_url,
                timeout=self._config.jwks_timeout_seconds,
            )
        else:
            async with httpx.AsyncClient(timeout=self._config.jwks_timeout_seconds) as client:
                response = await client.get(self._config.jwks_url)
        response.raise_for_status()
        parsed: object = response.json()
        if not isinstance(parsed, dict):
            msg = "JWKS endpoint did not return a JSON object."
            raise TypeError(msg)
        return cast("dict[str, Any]", parsed)

    def _decode_and_verify(
        self,
        token: str,
        signing_key: jwt.PyJWK,
        algorithm: str,
    ) -> dict[str, Any]:
        """Verify signature, exp, iss, and optional aud using PyJWT."""
        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                issuer=self._config.issuer_url,
                audience=self._config.audience,
                leeway=self._config.leeway_seconds,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": self._config.audience is not None,
                    "require": ["exp", "iss", "sub"],
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
