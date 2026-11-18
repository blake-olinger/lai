"""End-to-end ASGI integration tests for pluggable API-key, Google OIDC, and CF edge auth."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta
from http import HTTPStatus as S
from typing import TYPE_CHECKING, Any

import httpx
import jwt
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain import chat as dc
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.observability import LoggingConfig
from lai.security import (
    ApiKeyAuthProvider,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
)
from tests._helpers import StubOrchestratorBase, bearer_headers, open_app_client
from tests.integration._helpers import mint_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from fastapi import FastAPI

_ISSUER = "https://accounts.google.com"
_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_AUDIENCE = "lai-e2e-client"
_CF_ID = "cf-edge-client-id"
_CF_DOMAIN = "team.cloudflareaccess.com"
_CF_CERTS_URL = f"https://{_CF_DOMAIN}/cdn-cgi/access/certs"
_CF_AUD = "cf-e2e-aud"
_RSA_KEY_SIZE = 2048
_RSA_PUBLIC_EXPONENT = 65537
_EXPECTED_RECOVERY_CALLS = 2


class _StubOrchestrator(StubOrchestratorBase):
    """Minimal orchestrator verifying authenticated principals reach route handlers."""

    async def run_turn(self, turn_input: dc.TurnInput, principal: AuthPrincipal) -> dc.TurnOutput:
        msg = dc.Message(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=dc.MessageRole.ASSISTANT,
            content=f"hello:{principal.principal_id}",
        )
        return dc.TurnOutput(
            session_id=turn_input.session_id,
            message=msg,
            context_mode_used=turn_input.context_mode,
            usage=dc.TokenUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        )

    async def run_turn_stream(
        self, turn_input: dc.TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[dc.OrchestratorEvent]:
        out = await self.run_turn(turn_input, principal)
        yield dc.ContentDeltaEvent(delta=out.message.content)
        yield dc.TurnCompletedEvent(output=out)


def _make_rsa_jwk(kid: str) -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    priv = rsa.generate_private_key(public_exponent=_RSA_PUBLIC_EXPONENT, key_size=_RSA_KEY_SIZE)
    jwk_dict: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk_dict.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return priv, jwk_dict


def _sign_jwt(
    private_key: rsa.RSAPrivateKey, *, kid: str = "google-k1", **claim_overrides: object
) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "sub": "google-sub-42",
        "email": "dev@acme.test",
        "email_verified": True,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def _build_oidc_settings(tmp_path: Path, cf_secret: str) -> LaiSettings:
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        auth_providers=("api_key", "oidc"),
        oidc_enabled=True,
        oidc_issuer_url=_ISSUER,
        oidc_jwks_url=_JWKS_URL,
        oidc_audience=_AUDIENCE,
        oidc_allowed_emails=("solo@partner.org",),
        oidc_allowed_domains=("acme.test",),
        oidc_admin_emails=("boss@acme.test",),
        cf_access_enabled=True,
        cf_access_client_id=_CF_ID,
        cf_access_client_secret=cf_secret,
    )


async def _mint_app_key(app: FastAPI, name: str, scopes: tuple[ApiScope, ...]) -> tuple[str, str]:
    return await mint_api_key(app.state.services, name, scopes)


async def _get_code(
    client: httpx.AsyncClient, hdrs: dict[str, str], path: str = "/v1/projects"
) -> int:
    return (await client.get(path, headers=hdrs)).status_code


async def _post_code(client: httpx.AsyncClient, hdrs: dict[str, str], name: str) -> int:
    return (await client.post("/v1/projects", headers=hdrs, json={"name": name})).status_code


async def test_composite_auth_api_key_google_oidc_and_cloudflare_edge(tmp_path: Path) -> None:
    """Verify API keys, RS256 Google OIDC tokens, and CF edge checks through CompositeAuth."""
    cf_secret = secrets.token_hex(16)
    app = create_app(_build_oidc_settings(tmp_path, cf_secret), orchestrator=_StubOrchestrator())
    rsa_priv, rsa_jwk = _make_rsa_jwk("google-k1")

    async with open_app_client(app) as client:
        plaintext, key_id = await _mint_app_key(
            app, "ci", (ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE)
        )
        cf_hdrs = {"CF-Access-Client-Id": _CF_ID, "CF-Access-Client-Secret": cf_secret}

        with respx.mock(assert_all_called=True) as mock:
            mock.get(_JWKS_URL).respond(S.OK, json={"keys": [rsa_jwk]})
            for extra in ({}, {"CF-Access-Client-Id": _CF_ID, "CF-Access-Client-Secret": "bad"}):
                assert await _get_code(client, bearer_headers(plaintext, **extra)) == S.UNAUTHORIZED

            key_hdrs = bearer_headers(plaintext, **cf_hdrs)
            assert await _get_code(client, key_hdrs) == S.OK
            oai_body = {"model": "gemma4-26b-a4b", "messages": [{"role": "user", "content": "hi"}]}
            chat_res = await client.post("/v1/chat/completions", headers=key_hdrs, json=oai_body)
            assert chat_res.status_code == S.OK
            assert key_id in chat_res.json()["choices"][0]["message"]["content"]
            assert await _get_code(client, key_hdrs, "/v1/system/alerts") == S.FORBIDDEN

            def _oidc(sub: str, email: str) -> dict[str, str]:
                return bearer_headers(_sign_jwt(rsa_priv, sub=sub, email=email), **cf_hdrs)

            dom_h, adm_h = _oidc("sub-dev", "dev@acme.test"), _oidc("sub-b", "boss@acme.test")
            for hdrs, path, code in (
                (dom_h, "/v1/projects", S.OK),
                (dom_h, "/v1/system/alerts", S.FORBIDDEN),
                (_oidc("sub-s", "solo@partner.org"), "/v1/projects", S.OK),
                (adm_h, "/v1/system/alerts", S.OK),
            ):
                assert await _get_code(client, hdrs, path) == code
            assert await _post_code(client, dom_h, "no") == S.FORBIDDEN
            assert await _post_code(client, adm_h, "adm") == S.CREATED

            past_exp = int((datetime.now(UTC) - timedelta(minutes=2)).timestamp())
            for tok, code in (
                (_sign_jwt(rsa_priv, exp=past_exp), S.UNAUTHORIZED),
                (_sign_jwt(rsa_priv, aud="other-audience"), S.UNAUTHORIZED),
                (_sign_jwt(rsa_priv, iss="https://evil.example"), S.UNAUTHORIZED),
                (_sign_jwt(rsa_priv, email_verified=False), S.UNAUTHORIZED),
                (_sign_jwt(rsa_priv, email="stranger@external.org"), S.FORBIDDEN),
            ):
                assert await _get_code(client, bearer_headers(tok, **cf_hdrs)) == code


async def test_oidc_custom_jwks_fetcher_integration(tmp_path: Path) -> None:
    """Verify injected jwks_fetcher works identically inside CompositeAuthenticator over ASGI."""
    cf_secret = secrets.token_hex(16)
    settings = _build_oidc_settings(tmp_path, cf_secret)
    svc = await build_app_services(settings, orchestrator=_StubOrchestrator())
    rsa_priv, rsa_jwk = _make_rsa_jwk("fetcher-k1")
    fetch_calls = 0

    async def fake_jwks_fetcher(_url: str) -> dict[str, Any]:
        nonlocal fetch_calls
        fetch_calls += 1
        return {"keys": [rsa_jwk]}

    oidc_cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=_ISSUER,
        jwks_url=_JWKS_URL,
        audience=_AUDIENCE,
        allowed_domains=("acme.test",),
        admin_emails=("boss@acme.test",),
    )
    p_oidc = OidcJwtAuthProvider(oidc_cfg, jwks_fetcher=fake_jwks_fetcher)
    cf_cfg = CloudflareEdgeConfig(enabled=True, client_id=_CF_ID, client_secret=cf_secret)
    svc.authenticator = CompositeAuthenticator(
        providers=[ApiKeyAuthProvider(svc.api_key_repo), p_oidc],
        edge_verifier=CloudflareEdgeVerifier(cf_cfg),
    )
    app = create_app(settings, services=svc)
    try:
        async with open_app_client(app) as client:
            tok = _sign_jwt(rsa_priv, kid="fetcher-k1", sub="sub-f", email="boss@acme.test")
            h = bearer_headers(
                tok, **{"CF-Access-Client-Id": _CF_ID, "CF-Access-Client-Secret": cf_secret}
            )
            assert (await _get_code(client, h, "/v1/system/alerts"), fetch_calls) == (S.OK, 1)
    finally:
        await close_app_services(svc)


async def test_cloudflare_dual_mode_prewarm_and_malformed_jwk_asgi(tmp_path: Path) -> None:
    """Verify dual-mode CF pre-warm, partial header 401, and bad JWK rejection over ASGI."""
    cf_priv, cf_jwk = _make_rsa_jwk("cf-dual-1")
    cf_secret = secrets.token_hex(16)
    cf_jwt = _sign_jwt(cf_priv, kid="cf-dual-1", iss=f"https://{_CF_DOMAIN}", aud=_CF_AUD)
    bad_jwt = _sign_jwt(cf_priv, kid="bad-cf", iss=f"https://{_CF_DOMAIN}", aud=_CF_AUD)
    cf_up = {"cf_access_team_domain": _CF_DOMAIN, "cf_access_aud": _CF_AUD}
    with respx.mock(assert_all_called=True) as mock:
        body = {"keys": [cf_jwk, {"kid": "bad-cf", "kty": "RSA"}]}
        certs = mock.get(_CF_CERTS_URL).respond(S.OK, json=body)
        cfg = _build_oidc_settings(tmp_path, cf_secret).model_copy(update=cf_up)
        app = create_app(cfg, orchestrator=_StubOrchestrator())
        async with open_app_client(app) as client:
            assert certs.call_count == 1
            key, _ = await _mint_app_key(app, "dual", (ApiScope.PROJECTS_READ,))
            for h, code in (
                (bearer_headers(key, **{"Cf-Access-Jwt-Assertion": cf_jwt}), S.OK),
                (bearer_headers(key, **{"CF-Access-Client-Id": _CF_ID}), S.UNAUTHORIZED),
                (bearer_headers(key, **{"Cf-Access-Jwt-Assertion": bad_jwt}), S.UNAUTHORIZED),
            ):
                assert await _get_code(client, h) == code


async def test_cloudflare_dual_mode_startup_recovery_retries_jwks(tmp_path: Path) -> None:
    """Verify Cloudflare JWKS startup failure recovers on the first authenticated request."""
    cf_priv, cf_jwk = _make_rsa_jwk("cf-dual-1")
    cf_secret = secrets.token_hex(16)
    cf_jwt = _sign_jwt(cf_priv, kid="cf-dual-1", iss=f"https://{_CF_DOMAIN}", aud=_CF_AUD)
    cf_up = {"cf_access_team_domain": _CF_DOMAIN, "cf_access_aud": _CF_AUD}
    with respx.mock(assert_all_called=True) as mock:
        resps = [
            httpx.Response(S.SERVICE_UNAVAILABLE),
            httpx.Response(S.OK, json={"keys": [cf_jwk]}),
        ]
        certs = mock.get(_CF_CERTS_URL).mock(side_effect=resps)
        rec_cfg = _build_oidc_settings(tmp_path, cf_secret).model_copy(update=cf_up)
        rec_app = create_app(rec_cfg, orchestrator=_StubOrchestrator())
        async with open_app_client(rec_app) as client:
            assert certs.call_count == 1
            key, _ = await _mint_app_key(rec_app, "rec", (ApiScope.PROJECTS_READ,))
            h_ok = bearer_headers(key, **{"Cf-Access-Jwt-Assertion": cf_jwt})
            assert await _get_code(client, h_ok) == S.OK
            assert certs.call_count == _EXPECTED_RECOVERY_CALLS


async def test_pre_auth_ip_rate_limit_blocks_before_oidc_and_ignores_spoofed_cf_ip(
    tmp_path: Path,
) -> None:
    """Enforce pre-auth per-IP 429 before OIDC JWKS fetch and ignore CF-Connecting-IP when off."""
    rsa_priv, _ = _make_rsa_jwk("ip-k1")
    oidc_jwt = _sign_jwt(rsa_priv, kid="ip-k1")
    no_cf = _build_oidc_settings(tmp_path, "").model_copy(
        update={"cf_access_enabled": False, "ip_rate_limit_rpm": 1}
    )
    with respx.mock(assert_all_called=False) as mock:
        jwks_route = mock.get(_JWKS_URL).respond(S.OK, json={"keys": []})
        app = create_app(no_cf, orchestrator=_StubOrchestrator())
        async with open_app_client(app) as client:
            key, _ = await _mint_app_key(app, "ip", (ApiScope.PROJECTS_READ,))
            assert await _get_code(client, bearer_headers(key)) == S.OK
            hdrs = bearer_headers(oidc_jwt, **{"CF-Connecting-IP": "203.0.113.99"})
            r_429 = await client.get("/v1/projects", headers=hdrs)
            assert (r_429.status_code, not jwks_route.called) == (S.TOO_MANY_REQUESTS, True)
            assert int(r_429.headers["Retry-After"]) >= 1


async def test_pre_auth_ip_rate_limit_uses_cf_connecting_ip_when_cf_enabled(
    tmp_path: Path,
) -> None:
    """Enforce per-IP 429 keyed by CF-Connecting-IP when Cloudflare Access is enabled."""
    cf_sec = secrets.token_hex(16)
    cf_cfg = _build_oidc_settings(tmp_path, cf_sec).model_copy(update={"ip_rate_limit_rpm": 1})
    cf_app = create_app(cf_cfg, orchestrator=_StubOrchestrator())
    async with open_app_client(cf_app) as client:
        key, _ = await _mint_app_key(cf_app, "cf-ip", (ApiScope.PROJECTS_READ,))
        base = {
            "CF-Access-Client-Id": _CF_ID,
            "CF-Access-Client-Secret": cf_sec,
        }
        cases = (
            ("198.51.100.1", S.OK),
            ("198.51.100.1", S.TOO_MANY_REQUESTS),
            ("198.51.100.2", S.OK),
        )
        for ip, exp in cases:
            assert (
                await _get_code(client, bearer_headers(key, **base, **{"CF-Connecting-IP": ip}))
                == exp
            )
