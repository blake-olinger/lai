"""Cloudflare Access edge verification guard implementing EdgeVerifierPort."""

from __future__ import annotations

import hmac
import time
from typing import TYPE_CHECKING, Any

import jwt
from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.errors import AuthenticationError
from lai.observability.logging import get_component_logger
from lai.ports.auth import EdgeVerifierPort
from lai.security._jwks import JwksCache, JwksCachePolicy, JwksFetchSpec, fetch_jwks_payload

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

    from lai.domain.auth import RequestCredentials

_HEADER_CLIENT_ID = "cf-access-client-id"
_HEADER_CLIENT_SECRET_HDR = "cf-access-client-secret"  # noqa: S105
_HEADER_JWT_ASSERTION = "cf-access-jwt-assertion"
_ALLOWED_EDGE_ALGORITHMS = frozenset({"RS256", "ES256"})


class CloudflareEdgeConfig(StrictModel):
    """Configuration for Cloudflare Access edge validation."""

    enabled: bool = False
    client_id: str | None = None
    client_secret: str | None = None
    team_domain: str | None = None
    audience: str | None = None
    jwks_timeout_seconds: float = Field(default=5.0, gt=0.0)
    jwks_cache_ttl_seconds: float = Field(default=300.0, gt=0.0)
    min_jwks_refresh_interval_seconds: float = Field(default=30.0, ge=0.0)


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Lowercase all header names for case-insensitive lookup."""
    return {key.lower(): value for key, value in headers.items()}


def _normalize_issuer(team_domain: str | None) -> str | None:
    """Build canonical Cloudflare Access issuer URL from team domain."""
    if not team_domain:
        return None
    stripped = team_domain.strip().rstrip("/")
    if not stripped:
        return None
    if stripped.startswith(("https://", "http://")):
        return stripped
    return f"https://{stripped}"


class CloudflareEdgeVerifier(EdgeVerifierPort):
    """Validate Cloudflare Access service tokens or JWT edge assertions."""

    def __init__(
        self,
        config: CloudflareEdgeConfig,
        *,
        jwks: Mapping[str, Any] | None = None,
        jwt_verifier: Callable[[str], None] | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._jwt_verifier = jwt_verifier
        self._http_client = http_client
        self._jwks_cache = JwksCache(
            fetcher=self._fetch_jwks,
            policy=JwksCachePolicy(
                ttl_seconds=config.jwks_cache_ttl_seconds,
                min_refresh_interval_seconds=config.min_jwks_refresh_interval_seconds,
                can_fetch=_normalize_issuer(config.team_domain) is not None,
                log_prefix="cloudflare",
            ),
            clock=clock,
            initial_jwks=jwks,
        )

    async def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch Cloudflare Access certs from `team_domain/cdn-cgi/access/certs`."""
        issuer = _normalize_issuer(self._config.team_domain)
        if issuer is None:
            msg = "Cloudflare team_domain must be configured to fetch JWKS."
            raise AuthenticationError(msg)
        certs_url = f"{issuer}/cdn-cgi/access/certs"
        return await fetch_jwks_payload(
            JwksFetchSpec(
                url=certs_url,
                timeout_seconds=self._config.jwks_timeout_seconds,
                label="Cloudflare Access",
                log_prefix="cloudflare",
            ),
            http_client=self._http_client,
        )

    async def refresh_jwks(self) -> None:
        """Fetch Cloudflare Access certs from `team_domain/cdn-cgi/access/certs`."""
        await self._jwks_cache.refresh()

    async def verify_edge(self, credentials: RequestCredentials) -> None:
        """Verify Cloudflare Access service token or edge JWT when enabled."""
        if not self._config.enabled:
            return

        headers = _normalize_headers(credentials.headers)
        provided_id = headers.get(_HEADER_CLIENT_ID)
        provided_secret = headers.get(_HEADER_CLIENT_SECRET_HDR)
        if provided_id is not None or provided_secret is not None:
            self._verify_service_token(provided_id, provided_secret)
            return

        assertion = headers.get(_HEADER_JWT_ASSERTION)
        if assertion:
            await self._verify_jwt_assertion(assertion)
            return

        get_component_logger("security").warning(
            "cloudflare_edge_rejected", reason="missing_edge_credentials"
        )
        msg = "Missing Cloudflare Access edge credentials."
        raise AuthenticationError(msg)

    def _verify_service_token(
        self,
        provided_id: str | None,
        provided_secret: str | None,
    ) -> None:
        """Verify CF-Access-Client-Id and CF-Access-Client-Secret in constant time."""
        expected_id = self._config.client_id
        expected_secret = self._config.client_secret
        if not expected_id or not expected_secret or not provided_id or not provided_secret:
            get_component_logger("security").warning(
                "cloudflare_edge_rejected", reason="invalid_service_token"
            )
            msg = "Invalid Cloudflare Access service token."
            raise AuthenticationError(msg)

        id_ok = hmac.compare_digest(provided_id.encode("utf-8"), expected_id.encode("utf-8"))
        secret_ok = hmac.compare_digest(
            provided_secret.encode("utf-8"),
            expected_secret.encode("utf-8"),
        )
        if not (id_ok and secret_ok):
            get_component_logger("security").warning(
                "cloudflare_edge_rejected", reason="service_token_mismatch"
            )
            msg = "Invalid Cloudflare Access service token."
            raise AuthenticationError(msg)

    def _read_unverified_header(self, assertion: str) -> tuple[str, str | None]:
        """Parse JWT header and require asymmetric RS256/ES256 edge signature."""
        try:
            header = jwt.get_unverified_header(assertion)
        except jwt.PyJWTError as exc:
            get_component_logger("security").warning(
                "cloudflare_edge_rejected",
                reason="invalid_jwt_header",
                error=str(exc),
            )
            msg = f"Invalid Cloudflare Access JWT assertion: {exc}"
            raise AuthenticationError(msg) from exc

        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in _ALLOWED_EDGE_ALGORITHMS:
            get_component_logger("security").warning(
                "cloudflare_edge_rejected",
                reason="unsupported_alg",
                alg=str(alg),
            )
            msg = f"Unsupported Cloudflare JWT algorithm: {alg!r}"
            raise AuthenticationError(msg)

        raw_kid = header.get("kid")
        kid = raw_kid if isinstance(raw_kid, str) and raw_kid else None
        return alg, kid

    async def _verify_jwt_assertion(self, assertion: str) -> None:
        """Validate `Cf-Access-Jwt-Assertion` via injected verifier or cached/on-demand JWKS."""
        if self._jwt_verifier is not None:
            self._jwt_verifier(assertion)
            return

        alg, kid = self._read_unverified_header(assertion)
        issuer = _normalize_issuer(self._config.team_domain)
        missing_msg = (
            "Cloudflare Access JWKS is not loaded to verify JWT assertion."
            if self._jwks_cache.jwks is None and issuer is None
            else "No matching Cloudflare Access JWK found."
        )
        signing_key = await self._jwks_cache.resolve_key(
            kid,
            error_prefix="Invalid Cloudflare Access JWK",
            missing_key_message=missing_msg,
        )

        try:
            jwt.decode(
                assertion,
                signing_key.key,
                algorithms=[alg],
                issuer=issuer,
                audience=self._config.audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": issuer is not None,
                    "verify_aud": self._config.audience is not None,
                    "require": ["exp"],
                },
            )
        except jwt.PyJWTError as exc:
            get_component_logger("security").warning(
                "cloudflare_edge_rejected",
                reason="invalid_jwt_assertion",
                error=str(exc),
            )
            msg = f"Invalid Cloudflare Access JWT assertion: {exc}"
            raise AuthenticationError(msg) from exc
