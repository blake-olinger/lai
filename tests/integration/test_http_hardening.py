"""Integration tests for HTTP hardening, security headers, CORS, CF-IP trust, and OIDC auth."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from http import HTTPStatus as S
from typing import TYPE_CHECKING

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import BackendHealth, InferenceChunk, InferenceRequest, InferenceResponse
from lai.domain.observability import LoggingConfig
from lai.domain.tools import PermissionTier
from lai.ports.inference import InferenceProviderPort
from tests.integration._helpers import (
    IntegrationSpec,
    configure_test_oidc,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
    register_client_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    import httpx
    from starlette.types import ASGIApp, Receive, Scope, Send


class _SimpleInference(InferenceProviderPort):
    """Minimal inference provider recording tool names passed into chat_stream."""

    def __init__(self) -> None:
        self.tool_names_history: list[tuple[str, ...]] = []

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="ok")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.tool_names_history.append(tuple(t.name for t in request.tools))
        yield InferenceChunk(
            content_delta="hardened-ok",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


def _hardening_settings(
    tmp_path: Path,
    *,
    cors_allowed_origins: tuple[str, ...] = (),
    trust_cloudflare_headers: bool = False,
    cf_access_enabled: bool = False,
    ip_rate_limit_rpm: int = 240,
) -> LaiSettings:
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
        cors_allowed_origins=cors_allowed_origins,
        trust_cloudflare_headers=trust_cloudflare_headers,
        cf_access_enabled=cf_access_enabled,
        ip_rate_limit_rpm=ip_rate_limit_rpm,
    )


def _assert_security_headers(headers: httpx.Headers, *, expected_cache_control: str) -> None:
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["permissions-policy"] == "geolocation=(), microphone=(), camera=()"
    assert headers["cache-control"] == expected_cache_control


async def test_disabled_docs_routes_and_security_headers_including_sse(tmp_path: Path) -> None:
    """Verify /docs, /redoc, /openapi.json return 404 and security/cache headers apply cleanly."""
    settings = _hardening_settings(tmp_path)
    async with open_services_harness(
        settings, IntegrationSpec(inference_provider=_SimpleInference())
    ) as h:
        hdrs = await mint_auth_headers(h.svc, "hdr-user", (ApiScope.ADMIN,))
        for disabled_path in ("/docs", "/redoc", "/openapi.json"):
            r_404 = await h.client.get(disabled_path)
            assert r_404.status_code == S.NOT_FOUND
            _assert_security_headers(r_404.headers, expected_cache_control="no-store")

        r_hp = await h.client.get("/v1/health")
        assert r_hp.status_code == S.OK
        _assert_security_headers(r_hp.headers, expected_cache_control="no-store")

        pid = await create_project(h.client, hdrs, "sec-hdrs")
        sid = await create_session(h.client, hdrs, pid)
        r_ses_sse = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=hdrs,
            json={"content": "hi", "stream": True},
        )
        assert r_ses_sse.status_code == S.OK
        _assert_security_headers(r_ses_sse.headers, expected_cache_control="no-cache")

        r_oai_sse = await h.client.post(
            "/v1/chat/completions",
            headers=hdrs,
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r_oai_sse.status_code == S.OK
        _assert_security_headers(r_oai_sse.headers, expected_cache_control="no-cache")


async def test_cors_preflight_allowed_vs_disallowed_origins_and_headers(tmp_path: Path) -> None:
    """Verify strict CORS origin matching, credentials=False, and outermost security headers."""
    allowed = "https://spa.example.com"
    settings = _hardening_settings(tmp_path, cors_allowed_origins=(allowed,))
    async with open_services_harness(settings) as h:
        r_preflight = await h.client.options(
            "/v1/projects",
            headers={
                "Origin": allowed,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type, X-LAI-Project-Id",
            },
        )
        assert r_preflight.status_code == S.OK
        assert r_preflight.headers["access-control-allow-origin"] == allowed
        assert "access-control-allow-credentials" not in r_preflight.headers
        assert "POST" in r_preflight.headers["access-control-allow-methods"]
        _assert_security_headers(r_preflight.headers, expected_cache_control="no-store")

        hdrs = await mint_auth_headers(h.svc, "cors-user", (ApiScope.PROJECTS_READ,))
        r_ok = await h.client.get("/v1/projects", headers={**hdrs, "Origin": allowed})
        assert r_ok.status_code == S.OK
        assert r_ok.headers["access-control-allow-origin"] == allowed
        exposed = r_ok.headers["access-control-expose-headers"]
        for expected in (
            "X-Request-Id",
            "Retry-After",
            "X-LAI-Guardrail-Blocked",
            "X-LAI-Guardrail-Policy",
        ):
            assert expected in exposed

        r_bad = await h.client.get(
            "/v1/projects", headers={**hdrs, "Origin": "https://evil.example"}
        )
        assert r_bad.status_code == S.OK
        assert "access-control-allow-origin" not in r_bad.headers


def _wrap_client_host_from_header(app: ASGIApp) -> ASGIApp:
    """ASGI test wrapper setting scope['client'] from X-Test-Peer-IP header."""

    async def _asgi(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            raw_hdrs = dict(scope.get("headers", ()))
            peer = raw_hdrs.get(b"x-test-peer-ip")
            if peer is not None:
                scope["client"] = (peer.decode("ascii"), 50000)
        await app(scope, receive, send)

    return _asgi


async def test_cf_connecting_ip_trusted_only_from_loopback_when_enabled(tmp_path: Path) -> None:
    """Verify CF-Connecting-IP is honored on 127.0.0.1/::1 and ignored from non-loopback."""
    settings_on = _hardening_settings(
        tmp_path / "on", trust_cloudflare_headers=True, ip_rate_limit_rpm=1
    )
    spec = IntegrationSpec(asgi_wrapper=_wrap_client_host_from_header)
    async with open_services_harness(settings_on, spec) as h:
        hdrs = await mint_auth_headers(h.svc, "ip-usr", (ApiScope.PROJECTS_READ,))
        for peer, cf_ip, expected in (
            ("127.0.0.1", "198.51.100.1", S.OK),
            ("127.0.0.1", "198.51.100.1", S.TOO_MANY_REQUESTS),
            ("127.0.0.1", "198.51.100.2", S.OK),
            ("::1", "198.51.100.3", S.OK),
            ("::1", "198.51.100.3", S.TOO_MANY_REQUESTS),
            ("::1", "198.51.100.4", S.OK),
            ("203.0.113.50", "198.51.100.10", S.OK),
            ("203.0.113.50", "198.51.100.11", S.TOO_MANY_REQUESTS),
        ):
            r = await h.client.get(
                "/v1/projects",
                headers={**hdrs, "X-Test-Peer-IP": peer, "CF-Connecting-IP": cf_ip},
            )
            assert r.status_code == expected

    settings_off = _hardening_settings(
        tmp_path / "off",
        trust_cloudflare_headers=False,
        cf_access_enabled=False,
        ip_rate_limit_rpm=1,
    )
    async with open_services_harness(settings_off, spec) as h:
        hdrs = await mint_auth_headers(h.svc, "ip-off", (ApiScope.PROJECTS_READ,))
        r1 = await h.client.get(
            "/v1/projects",
            headers={**hdrs, "X-Test-Peer-IP": "127.0.0.1", "CF-Connecting-IP": "198.51.100.20"},
        )
        r2 = await h.client.get(
            "/v1/projects",
            headers={**hdrs, "X-Test-Peer-IP": "127.0.0.1", "CF-Connecting-IP": "198.51.100.21"},
        )
        assert (r1.status_code, r2.status_code) == (S.OK, S.TOO_MANY_REQUESTS)


async def test_resolve_turn_client_context_fails_closed_on_client_tools(tmp_path: Path) -> None:
    """client_tools fail closed (401/403) without an active authorized OIDC client app."""
    inf = _SimpleInference()
    async with open_services_harness(
        _hardening_settings(tmp_path), IntegrationSpec(inference_provider=inf)
    ) as h:
        oidc = configure_test_oidc(
            h.svc,
            audiences=("unregistered-cid", "revoked-cid", "expired-cid", "scoped-cid"),
            default_scopes=(ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
        )
        h_api = await mint_auth_headers(
            h.svc, "api-user", (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE)
        )
        p_api = await create_project(h.client, h_api, "api-proj")
        s_api = await create_session(h.client, h_api, p_api)

        h_unreg = oidc.mint_headers(sub="oidc-u", aud="unregistered-cid")
        p_oidc = await create_project(h.client, h_unreg, "oidc-proj")
        p_other = await create_project(h.client, h_unreg, "oidc-other")
        s_oidc = await create_session(h.client, h_unreg, p_oidc)

        revoked = await register_client_app(
            h.svc,
            "revoked",
            oauth_client_id="revoked-cid",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(),
        )
        await h.svc.client_app_repo.revoke(revoked.id, datetime.now(UTC))
        await register_client_app(
            h.svc,
            "expired",
            oauth_client_id="expired-cid",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(),
            expires_at=datetime.now(UTC) - timedelta(seconds=5),
        )
        await register_client_app(
            h.svc,
            "scoped",
            oauth_client_id="scoped-cid",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(),
            allowed_project_ids=(p_other,),
        )

        client_tool = {
            "name": "ping_client",
            "description": "Ping",
            "permission_tier": "read",
            "execution_target": "client",
        }

        assert (
            await h.client.post(
                f"/v1/sessions/{s_api}/messages",
                headers=h_api,
                json={"content": "hi", "client_tools": [client_tool]},
            )
        ).status_code == S.UNAUTHORIZED
        assert (
            await h.client.post(
                "/v1/chat/completions",
                headers=h_api,
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "client_tools": [client_tool],
                },
            )
        ).status_code == S.UNAUTHORIZED

        for aud in ("unregistered-cid", "revoked-cid", "expired-cid", "scoped-cid"):
            hdr = oidc.mint_headers(sub="oidc-u", aud=aud)
            r_ses = await h.client.post(
                f"/v1/sessions/{s_oidc}/messages",
                headers=hdr,
                json={"content": "hi", "client_tools": [client_tool]},
            )
            assert r_ses.status_code == S.FORBIDDEN
            r_oai = await h.client.post(
                "/v1/chat/completions",
                headers=hdr,
                json={
                    "project_id": p_oidc,
                    "messages": [{"role": "user", "content": "hi"}],
                    "client_tools": [client_tool],
                },
            )
            assert r_oai.status_code == S.FORBIDDEN
            r_plain = await h.client.post(
                f"/v1/sessions/{s_oidc}/messages",
                headers=hdr,
                json={"content": "hi"},
            )
            assert r_plain.status_code == S.OK
            assert "ping_client" not in inf.tool_names_history[-1]
