"""End-to-end ASGI integration tests for pluggable API-key, Google OIDC, and CF edge auth."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import (
    ContentDeltaEvent,
    Message,
    MessageRole,
    OrchestratorEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnInput,
    TurnOutput,
)
from lai.domain.observability import LoggingConfig
from lai.ports.orchestrator import OrchestratorPort
from lai.security import (
    ApiKeyAuthProvider,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
    generate_api_key,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator
    from pathlib import Path

_ISSUER = "https://accounts.google.com"
_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_AUDIENCE = "lai-e2e-client"
_CF_ID = "cf-edge-client-id"
_RSA_KEY_SIZE = 2048
_RSA_PUBLIC_EXPONENT = 65537


class _StubOrchestrator(OrchestratorPort):
    """Minimal orchestrator verifying authenticated principals reach route handlers."""

    async def run_turn(self, turn_input: TurnInput, principal: AuthPrincipal) -> TurnOutput:
        msg = Message(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=MessageRole.ASSISTANT,
            content=f"hello:{principal.principal_id}",
        )
        return TurnOutput(
            session_id=turn_input.session_id,
            message=msg,
            context_mode_used=turn_input.context_mode,
            usage=TokenUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        )

    async def run_turn_stream(
        self, turn_input: TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[OrchestratorEvent]:
        out = await self.run_turn(turn_input, principal)
        yield ContentDeltaEvent(delta=out.message.content)
        yield TurnCompletedEvent(output=out)


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _make_rsa_jwk(kid: str) -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    private_key = rsa.generate_private_key(
        public_exponent=_RSA_PUBLIC_EXPONENT,
        key_size=_RSA_KEY_SIZE,
    )
    jwk_dict: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk_dict.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return private_key, jwk_dict


def _sign_google_jwt(
    private_key: rsa.RSAPrivateKey,
    *,
    kid: str = "google-k1",
    **claim_overrides: object,
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


async def test_composite_auth_api_key_google_oidc_and_cloudflare_edge(tmp_path: Path) -> None:
    """Verify API keys, RS256 Google OIDC tokens, and CF edge checks through CompositeAuth."""
    cf_secret = secrets.token_hex(16)
    settings = _build_oidc_settings(tmp_path, cf_secret)
    app = create_app(settings, orchestrator=_StubOrchestrator())
    rsa_priv, rsa_jwk = _make_rsa_jwk("google-k1")

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        created_key = generate_api_key(
            "ci-key",
            (ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE),
        )
        await app.state.services.api_key_repo.create(created_key.record)
        cf_headers = {
            "CF-Access-Client-Id": _CF_ID,
            "CF-Access-Client-Secret": cf_secret,
        }

        with respx.mock(assert_all_called=True) as mock:
            mock.get(_JWKS_URL).respond(HTTPStatus.OK, json={"keys": [rsa_jwk]})

            # 1. Edge gate blocks missing or wrong Cloudflare service token headers
            no_cf = await client.get(
                "/v1/projects",
                headers={"Authorization": f"Bearer {created_key.plaintext_key}"},
            )
            assert no_cf.status_code == HTTPStatus.UNAUTHORIZED

            bad_cf = await client.get(
                "/v1/projects",
                headers={
                    "Authorization": f"Bearer {created_key.plaintext_key}",
                    "CF-Access-Client-Id": _CF_ID,
                    "CF-Access-Client-Secret": "wrong-secret",
                },
            )
            assert bad_cf.status_code == HTTPStatus.UNAUTHORIZED

            # 2. Valid CF edge + lai_sk_ Bearer API key authenticates with scoped permissions
            api_key_hdrs = {**cf_headers, "Authorization": f"Bearer {created_key.plaintext_key}"}
            assert (await client.get("/v1/projects", headers=api_key_hdrs)).status_code == (
                HTTPStatus.OK
            )
            chat_res = await client.post(
                "/v1/chat/completions",
                headers=api_key_hdrs,
                json={"model": "gemma4-26b-a4b", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert chat_res.status_code == HTTPStatus.OK
            assert created_key.record.id in chat_res.json()["choices"][0]["message"]["content"]
            assert (await client.get("/v1/system/alerts", headers=api_key_hdrs)).status_code == (
                HTTPStatus.FORBIDDEN
            )

            # 3. Domain-allowed RS256 Google OIDC JWT receives default_scopes (not ADMIN)
            domain_jwt = _sign_google_jwt(rsa_priv, sub="sub-dev", email="dev@acme.test")
            domain_hdrs = {**cf_headers, "Authorization": f"Bearer {domain_jwt}"}
            assert (await client.get("/v1/projects", headers=domain_hdrs)).status_code == (
                HTTPStatus.OK
            )
            assert (
                await client.post("/v1/projects", headers=domain_hdrs, json={"name": "nope"})
            ).status_code == HTTPStatus.FORBIDDEN
            assert (await client.get("/v1/system/alerts", headers=domain_hdrs)).status_code == (
                HTTPStatus.FORBIDDEN
            )

            # 4. Explicitly allowed email passes allowlist
            solo_jwt = _sign_google_jwt(rsa_priv, sub="sub-solo", email="solo@partner.org")
            solo_hdrs = {**cf_headers, "Authorization": f"Bearer {solo_jwt}"}
            assert (await client.get("/v1/projects", headers=solo_hdrs)).status_code == (
                HTTPStatus.OK
            )

            # 5. admin_emails maps directly to ADMIN scope
            admin_jwt = _sign_google_jwt(rsa_priv, sub="sub-boss", email="boss@acme.test")
            admin_hdrs = {**cf_headers, "Authorization": f"Bearer {admin_jwt}"}
            assert (await client.get("/v1/system/alerts", headers=admin_hdrs)).status_code == (
                HTTPStatus.OK
            )
            assert (
                await client.post("/v1/projects", headers=admin_hdrs, json={"name": "admin-prj"})
            ).status_code == HTTPStatus.CREATED

            # 6. Expired, wrong-aud, wrong-iss, unverified-email, and disallowed callers fail
            expired_exp = int((datetime.now(UTC) - timedelta(minutes=2)).timestamp())
            unauth_tokens = (
                _sign_google_jwt(rsa_priv, exp=expired_exp),
                _sign_google_jwt(rsa_priv, aud="other-audience"),
                _sign_google_jwt(rsa_priv, iss="https://evil.example"),
                _sign_google_jwt(rsa_priv, email_verified=False),
            )
            for bad_tok in unauth_tokens:
                bad_res = await client.get(
                    "/v1/projects",
                    headers={**cf_headers, "Authorization": f"Bearer {bad_tok}"},
                )
                assert bad_res.status_code == HTTPStatus.UNAUTHORIZED

            disallowed_jwt = _sign_google_jwt(rsa_priv, email="stranger@external.org")
            denied_res = await client.get(
                "/v1/projects",
                headers={**cf_headers, "Authorization": f"Bearer {disallowed_jwt}"},
            )
            assert denied_res.status_code == HTTPStatus.FORBIDDEN


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
    cf_cfg = CloudflareEdgeConfig(
        enabled=True,
        client_id=_CF_ID,
        client_secret=cf_secret,
    )
    svc.authenticator = CompositeAuthenticator(
        providers=[
            ApiKeyAuthProvider(svc.api_key_repo),
            OidcJwtAuthProvider(oidc_cfg, jwks_fetcher=fake_jwks_fetcher),
        ],
        edge_verifier=CloudflareEdgeVerifier(cf_cfg),
    )
    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            token = _sign_google_jwt(
                rsa_priv,
                kid="fetcher-k1",
                sub="sub-fetcher",
                email="boss@acme.test",
            )
            res = await client.get(
                "/v1/system/alerts",
                headers={
                    "Authorization": f"Bearer {token}",
                    "CF-Access-Client-Id": _CF_ID,
                    "CF-Access-Client-Secret": cf_secret,
                },
            )
            assert res.status_code == HTTPStatus.OK
            assert fetch_calls == 1
    finally:
        await close_app_services(svc)
