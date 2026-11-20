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

from lai import security as sec
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
from lai.domain.errors import AuthenticationError, LaiError, ValidationFailedError
from lai.domain.ids import CLIENT_APP_ID_PREFIX
from lai.domain.projects import Project
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec
from lai.observability import configure_observability
from lai.security import (
    ApiKeyAuthProvider,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
    TokenBucketRateLimiter,
    build_client_app,
    generate_api_key,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

_ISSUER = "https://accounts.google.com"
_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_AUDIENCE = "lai-client"
_EXP_FUTURE = 9999999999


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
    assert principal.oauth_client_id == _AUDIENCE

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


def test_build_client_app_validation_and_secret_removal() -> None:
    """build_client_app validates inputs, deduplicates, coerces CLIENT target, and drops secrets."""
    for removed in (
        "CLIENT_APP_SECRET_PREFIX",
        "hash_client_app_secret",
        "verify_client_app_secret",
        "generate_client_app",
    ):
        assert not hasattr(sec, removed)

    tool_in = ToolSpec(
        name="run_local",
        description="Run local command",
        permission_tier=PermissionTier.EXEC,
        execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
    )
    app = build_client_app(
        "  Desktop Helper  ",
        "  12345.apps.googleusercontent.com  ",
        allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC, PermissionTier.READ),
        registered_tools=(tool_in,),
        allowed_project_ids=(" prj_1 ", "", "prj_1", "prj_2"),
    )
    assert app.id.startswith(CLIENT_APP_ID_PREFIX)
    assert (app.name, app.oauth_client_id) == ("Desktop Helper", "12345.apps.googleusercontent.com")
    assert app.allowed_tiers == (PermissionTier.READ, PermissionTier.EXEC)
    assert app.allowed_project_ids == ("prj_1", "prj_2")
    assert app.registered_tools[0].execution_target is ToolExecutionTarget.CLIENT
    assert app.is_active() is True

    with pytest.raises(ValidationFailedError, match="name"):
        build_client_app("   ", "client-1")
    with pytest.raises(ValidationFailedError, match="OAuth client ID"):
        build_client_app("app", "   ")
    with pytest.raises(ValidationFailedError, match="allowed_tiers"):
        build_client_app("bad-tier", "client-1", (PermissionTier.READ,), (tool_in,))

    dup_a = ToolSpec(name="dup", description="a", permission_tier=PermissionTier.READ)
    dup_b = ToolSpec(name="dup", description="b", permission_tier=PermissionTier.READ)
    with pytest.raises(ValidationFailedError, match="Duplicate"):
        build_client_app("dup-tools", "client-1", (PermissionTier.READ,), (dup_a, dup_b))


async def test_oidc_multi_audience_and_azp_client_binding() -> None:
    """OidcJwtAuthProvider enforces non-empty audiences and extracts azp/aud oauth_client_id."""
    priv, jwk = _rsa_key_and_jwk("k_multi")

    async def fetcher(_url: str) -> dict[str, Any]:
        return {"keys": [jwk]}

    empty_cfg = OidcProviderConfig(enabled=True, issuer_url=_ISSUER, jwks_url=_JWKS_URL)
    assert empty_cfg.resolved_audiences() == ()
    empty_provider = OidcJwtAuthProvider(empty_cfg, jwks_fetcher=fetcher)
    base_claims = {
        "iss": _ISSUER,
        "aud": "web-client",
        "sub": "u_multi",
        "email": "user@acme.test",
        "email_verified": True,
        "exp": _EXP_FUTURE,
    }
    tok = jwt.encode(base_claims, priv, algorithm="RS256", headers={"kid": "k_multi"})
    with pytest.raises(AuthenticationError, match="audience"):
        await empty_provider.authenticate(RequestCredentials(bearer_token=tok))

    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=_ISSUER,
        jwks_url=_JWKS_URL,
        audience="legacy-client",
        audiences=("  web-client  ", "", "legacy-client", "desktop-client"),
    )
    assert cfg.resolved_audiences() == ("web-client", "legacy-client", "desktop-client")
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=fetcher)

    # azp present and authorized overrides aud for oauth_client_id
    azp_tok = jwt.encode(
        {**base_claims, "aud": "web-client", "azp": " desktop-client "},
        priv,
        algorithm="RS256",
        headers={"kid": "k_multi"},
    )
    azp_principal = await provider.authenticate(RequestCredentials(bearer_token=azp_tok))
    assert azp_principal.oauth_client_id == "desktop-client"

    # aud as list picks first matching configured audience when azp is absent or None
    list_tok = jwt.encode(
        {**base_claims, "aud": ["other-aud", "legacy-client", "web-client"], "azp": None},
        priv,
        algorithm="RS256",
        headers={"kid": "k_multi"},
    )
    list_principal = await provider.authenticate(RequestCredentials(bearer_token=list_tok))
    assert list_principal.oauth_client_id == "legacy-client"

    # Unauthorized or non-string azp fails closed
    for bad_azp in ("rogue-client", "", 12345):
        bad_tok = jwt.encode(
            {**base_claims, "azp": bad_azp},
            priv,
            algorithm="RS256",
            headers={"kid": "k_multi"},
        )
        with pytest.raises(AuthenticationError):
            await provider.authenticate(RequestCredentials(bearer_token=bad_tok))


async def test_oidc_entra_id_identity_claims_and_appid_binding() -> None:
    """OidcJwtAuthProvider supports Entra ID preferred_username/upn and appid claims."""
    priv, jwk = _rsa_key_and_jwk("entra-k1")
    iss = "https://login.microsoftonline.com/tenant-123/v2.0"

    async def fetcher(_url: str) -> dict[str, Any]:
        return {"keys": [jwk]}

    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=iss,
        jwks_url="https://login.microsoftonline.com/tenant-123/discovery/v2.0/keys",
        audiences=("api://lai-prod", "entra-desktop", "entra-web"),
        allowed_domains=("contoso.com",),
        admin_emails=("boss@contoso.com",),
        require_email_verified=False,
    )
    assert cfg.identity_claims == ("email", "preferred_username", "upn")
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=fetcher)

    def _tok(**claims: object) -> RequestCredentials:
        body = {"iss": iss, "aud": "api://lai-prod", "sub": "oid-1", "exp": _EXP_FUTURE, **claims}
        tok = jwt.encode(body, priv, algorithm="RS256", headers={"kid": "entra-k1"})
        return RequestCredentials(bearer_token=tok)

    p1 = await provider.authenticate(
        _tok(email="  ", preferred_username="Boss@Contoso.com", appid=" entra-desktop ")
    )
    assert (p1.email, p1.oauth_client_id) == ("boss@contoso.com", "entra-desktop")
    assert p1.scopes == (ApiScope.ADMIN,)
    p2 = await provider.authenticate(
        _tok(preferred_username=123, upn="Dev@Contoso.com", azp="entra-web", appid="entra-desktop")
    )
    assert (p2.email, p2.oauth_client_id) == ("dev@contoso.com", "entra-web")
    assert p2.scopes == cfg.default_scopes
    for bad_appid in ("rogue-entra-app", "", 999):
        with pytest.raises(AuthenticationError, match="appid"):
            await provider.authenticate(_tok(preferred_username="dev@contoso.com", appid=bad_appid))
