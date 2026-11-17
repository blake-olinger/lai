"""Cloudflare Access edge verification guard implementing EdgeVerifierPort."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING, Any, cast

import httpx
import jwt
from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.errors import AuthenticationError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger
from lai.ports.auth import EdgeVerifierPort

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lai.domain.auth import RequestCredentials

_HEADER_CLIENT_ID = "cf-access-client-id"
_HEADER_CLIENT_SECRET_HDR = "cf-access-client-secret"  # noqa: S105
_HEADER_JWT_ASSERTION = "cf-access-jwt-assertion"


class CloudflareEdgeConfig(StrictModel):
    """Configuration for Cloudflare Access edge validation."""

    enabled: bool = False
    client_id: str | None = None
    client_secret: str | None = None
    team_domain: str | None = None
    audience: str | None = None
    jwks_timeout_seconds: float = Field(default=5.0, gt=0.0)


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Lowercase all header names for case-insensitive lookup."""
    return {key.lower(): value for key, value in headers.items()}


def _normalize_issuer(team_domain: str | None) -> str | None:
    """Build canonical Cloudflare Access issuer URL from team domain."""
    if not team_domain:
        return None
    stripped = team_domain.strip().rstrip("/")
    if stripped.startswith(("https://", "http://")):
        return stripped
    return f"https://{stripped}"


def _find_edge_key(jwks: Mapping[str, Any], kid: str | None) -> jwt.PyJWK | None:
    """Locate a JWK matching `kid`, or the sole key if `kid` is absent."""
    raw_keys = jwks.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        return None
    keys_list = cast("list[object]", raw_keys)
    dict_keys = [cast("dict[str, Any]", item) for item in keys_list if isinstance(item, dict)]
    if kid is not None:
        for entry in dict_keys:
            if entry.get("kid") == kid:
                return jwt.PyJWK.from_dict(entry)
        return None
    if len(dict_keys) == 1:
        return jwt.PyJWK.from_dict(dict_keys[0])
    return None


class CloudflareEdgeVerifier(EdgeVerifierPort):
    """Validate Cloudflare Access service tokens or JWT edge assertions."""

    def __init__(
        self,
        config: CloudflareEdgeConfig,
        *,
        jwks: Mapping[str, Any] | None = None,
        jwt_verifier: Callable[[str], None] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._jwks: dict[str, Any] | None = dict(jwks) if jwks is not None else None
        self._jwt_verifier = jwt_verifier
        self._http_client = http_client

    async def refresh_jwks(self) -> None:
        """Fetch Cloudflare Access certs from `team_domain/cdn-cgi/access/certs`."""
        issuer = _normalize_issuer(self._config.team_domain)
        if issuer is None:
            msg = "Cloudflare team_domain must be configured to fetch JWKS."
            raise AuthenticationError(msg)
        certs_url = f"{issuer}/cdn-cgi/access/certs"
        try:
            if self._http_client is not None:
                response = await self._http_client.get(
                    certs_url, timeout=self._config.jwks_timeout_seconds
                )
            else:
                async with httpx.AsyncClient(timeout=self._config.jwks_timeout_seconds) as client:
                    response = await client.get(certs_url)
            response.raise_for_status()
            parsed: object = response.json()
        except (httpx.TimeoutException, TimeoutError) as exc:
            get_component_logger("security").error(
                "cloudflare_jwks_fetch_timeout",
                certs_url=certs_url,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = "Timed out fetching Cloudflare Access JWKS."
            raise AuthenticationError(msg) from exc
        except (httpx.HTTPError, ValueError) as exc:
            get_component_logger("security").error(
                "cloudflare_jwks_fetch_failed", certs_url=certs_url, error=str(exc)
            )
            msg = f"Failed to fetch Cloudflare Access JWKS: {exc}"
            raise AuthenticationError(msg) from exc

        if not isinstance(parsed, dict):
            msg = "Invalid Cloudflare Access JWKS payload."
            raise AuthenticationError(msg)
        parsed_dict = cast("dict[str, Any]", parsed)
        if not isinstance(parsed_dict.get("keys"), list):
            msg = "Invalid Cloudflare Access JWKS payload."
            raise AuthenticationError(msg)
        self._jwks = parsed_dict

    def verify_edge(self, credentials: RequestCredentials) -> None:
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
            self._verify_jwt_assertion(assertion)
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

    def _verify_jwt_assertion(self, assertion: str) -> None:
        """Validate `Cf-Access-Jwt-Assertion` via injected verifier or loaded JWKS."""
        if self._jwt_verifier is not None:
            self._jwt_verifier(assertion)
            return

        if self._jwks is None:
            get_component_logger("security").warning(
                "cloudflare_edge_rejected", reason="jwks_not_loaded"
            )
            msg = "Cloudflare Access JWKS is not loaded to verify JWT assertion."
            raise AuthenticationError(msg)

        try:
            header = jwt.get_unverified_header(assertion)
            raw_kid = header.get("kid")
            kid = raw_kid if isinstance(raw_kid, str) and raw_kid else None
            alg = header.get("alg")
            if not isinstance(alg, str) or alg not in {"RS256", "ES256"}:
                msg = f"Unsupported Cloudflare JWT algorithm: {alg!r}"
                raise AuthenticationError(msg)
            signing_key = _find_edge_key(self._jwks, kid)
            if signing_key is None:
                msg = "No matching Cloudflare Access JWK found."
                raise AuthenticationError(msg)
            issuer = _normalize_issuer(self._config.team_domain)
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
