"""Additional edge-case tests for OIDC HTTP JWKS, Cloudflare refresh, and DB rollback."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from lai.db import SqliteDatabase, SqliteProjectRepository, SqliteSessionRepository
from lai.domain import (
    AlertCategory,
    ApiKeyRecord,
    ApiScope,
    AuthMethod,
    LoggingConfig,
    RequestCredentials,
)
from lai.domain.chat import Message, MessageRole, Session
from lai.domain.errors import AuthenticationError, LaiError
from lai.domain.projects import Project
from lai.observability import configure_observability
from lai.security import (
    ApiKeyAuthProvider,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
    TokenBucketRateLimiter,
    generate_api_key,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

_ISSUER = "https://accounts.google.com"
_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_AUDIENCE = "lai-client"


def _rsa_key_and_jwk(kid: str = "k1") -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return priv, jwk


async def test_oidc_http_jwks_success_and_error_paths() -> None:
    """OidcJwtAuthProvider handles HTTP JWKS success, disabled mode, and bad payloads."""
    priv, jwk = _rsa_key_and_jwk("k1")
    token = jwt.encode(
        {
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "sub": "u_http",
            "email": "user@example.com",
            "email_verified": True,
            "exp": 9999999999,
        },
        priv,
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    disabled_provider = OidcJwtAuthProvider(OidcProviderConfig(enabled=False))
    assert not disabled_provider.supports(RequestCredentials(bearer_token=token))
    with pytest.raises(AuthenticationError):
        await disabled_provider.authenticate(RequestCredentials(bearer_token=token))

    cfg = OidcProviderConfig(
        enabled=True, issuer_url=_ISSUER, jwks_url=_JWKS_URL, audience=_AUDIENCE
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(_JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [jwk]}))
        provider = OidcJwtAuthProvider(cfg)
        principal = await provider.authenticate(RequestCredentials(bearer_token=token))
    assert principal.auth_method is AuthMethod.OIDC
    assert principal.subject == "u_http"

    with respx.mock(assert_all_called=True) as mock:
        mock.get(_JWKS_URL).mock(return_value=httpx.Response(200, json=["not-a-dict"]))
        bad_provider = OidcJwtAuthProvider(cfg)
        with pytest.raises(AuthenticationError, match="Failed to fetch"):
            await bad_provider.authenticate(RequestCredentials(bearer_token=token))


async def test_cloudflare_refresh_jwks_timeout_and_disabled_verifier() -> None:
    """CloudflareEdgeVerifier refreshes certs via HTTP, alerts on timeout, and skips disabled."""
    priv, jwk = _rsa_key_and_jwk("cf-1")
    await CloudflareEdgeVerifier(CloudflareEdgeConfig(enabled=False)).verify_edge(
        RequestCredentials()
    )

    cfg = CloudflareEdgeConfig(
        enabled=True,
        team_domain="team.cloudflareaccess.com",
        audience="cf-aud",
    )
    verifier = CloudflareEdgeVerifier(cfg)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://team.cloudflareaccess.com/cdn-cgi/access/certs").mock(
            return_value=httpx.Response(200, json={"keys": [jwk]})
        )
        await verifier.refresh_jwks()

    assertion = jwt.encode(
        {"iss": "https://team.cloudflareaccess.com", "aud": "cf-aud", "exp": 9999999999},
        priv,
        algorithm="RS256",
        headers={"kid": "cf-1"},
    )
    await verifier.verify_edge(RequestCredentials(headers={"Cf-Access-Jwt-Assertion": assertion}))

    manager = configure_observability(LoggingConfig(enable_file_logs=False, level="INFO"))
    try:
        with respx.mock(assert_all_called=True) as mock:
            mock.get("https://team.cloudflareaccess.com/cdn-cgi/access/certs").mock(
                side_effect=httpx.ReadTimeout("timeout")
            )
            with pytest.raises(AuthenticationError, match="Timed out"):
                await verifier.refresh_jwks()
        assert any(a.category is AlertCategory.TIMEOUT for a in manager.recent_alerts())
    finally:
        manager.close()
        logging.getLogger().handlers.clear()


async def test_composite_replacement_ratelimit_validation_and_session_delete(
    tmp_path: Path,
) -> None:
    """Verify provider replacement, rate limiter argument guards, and DB delete/rollback."""
    created = generate_api_key("first", (ApiScope.CHAT_READ,))
    chain = CompositeAuthenticator()

    class _StubRepo:
        async def create(self, record: ApiKeyRecord) -> ApiKeyRecord:
            return record

        async def get_by_id(self, key_id: str) -> ApiKeyRecord | None:
            del key_id
            return created.record

        async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
            return created.record if key_hash == created.record.key_hash else None

        async def list_all(self) -> list[ApiKeyRecord]:
            return [created.record]

        async def touch_last_used(self, key_id: str, used_at: datetime) -> None:
            del key_id, used_at

        async def revoke(self, key_id: str, revoked_at: datetime) -> bool:
            del key_id, revoked_at
            return True

    provider = ApiKeyAuthProvider(_StubRepo())
    chain.register_provider(provider)
    chain.register_provider(provider)  # replace existing provider cleanly
    principal = await chain.authenticate(RequestCredentials(bearer_token=created.plaintext_key))
    assert principal.key_id == created.record.id

    with pytest.raises(ValueError, match="requests_per_minute"):
        TokenBucketRateLimiter(requests_per_minute=0)
    with pytest.raises(ValueError, match="burst_capacity"):
        TokenBucketRateLimiter(burst_capacity=0)
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst_capacity=2)
    with pytest.raises(ValueError, match="Requested token cost"):
        await limiter.acquire("u1", tokens=5.0)

    async with SqliteDatabase(tmp_path / "edge.db") as db:
        projects = SqliteProjectRepository(db)
        sessions = SqliteSessionRepository(db)
        proj = await projects.create(Project(name="edge-proj"))
        ses = await sessions.create_session(Session(project_id=proj.id))
        await sessions.add_message(
            Message(session_id=ses.id, project_id=proj.id, role=MessageRole.USER, content="hi")
        )
        assert await sessions.delete_session(ses.id) is True
        assert await sessions.delete_session(ses.id) is False
        assert await sessions.list_messages(ses.id) == []

        with pytest.raises(LaiError):
            async with db.write_transaction() as conn:
                await conn.execute("INSERT INTO projects (id) VALUES ('bad');")
