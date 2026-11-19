"""Targeted boundary/mutation-killing tests for validators, OIDC, queue, deps, and headers."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from pydantic import ValidationError
from starlette.datastructures import Headers

from lai.api.deps import get_current_principal
from lai.api.middleware import RequestContextMiddleware
from lai.config._validators import PUBLIC_CONSUMER_EMAIL_DOMAINS
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal, RequestCredentials
from lai.domain.errors import AuthenticationError, QueueFullError, ValidationFailedError
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.scheduler.fair_queue import FairPriorityQueue
from lai.security.oidc_provider import OidcJwtAuthProvider, OidcProviderConfig

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.types import Message, Receive, Scope, Send

    from lai.api.deps import AppServices

_RSA_PUBLIC_EXPONENT = 65537
_RSA_KEY_SIZE = 2048
_BAD_AZP_INT = 123
_PRINCIPAL_QUOTA = 2
_ROUND_ROBIN_COUNT = 4


def test_validators_and_oidc_resolved_audiences_boundaries() -> None:
    """Cover fail-closed validators across OIDC provider alias, consumer domains, and CORS."""
    with pytest.raises(ValidationError, match="audience"):
        LaiSettings(auth_providers=(" OIDC ",), oidc_allowed_domains=("acme.test",))
    for consumer in PUBLIC_CONSUMER_EMAIL_DOMAINS:
        with pytest.raises(ValidationError, match="Consumer email domain"):
            LaiSettings(
                oidc_enabled=True,
                oidc_audiences=("aud-1",),
                oidc_allowed_domains=(f" @{consumer.upper()} ",),
            )
    for query_or_fragment in ("https://a.com?x=1", "https://a.com#f"):
        with pytest.raises(ValidationError, match="query and fragment"):
            LaiSettings(cors_allowed_origins=(query_or_fragment,))
    for bad_origin in (
        "",
        " https://a.com",
        "https://a.com ",
        "*",
        "ftp://a.com",
        "https://u:p@a.com",
        "https://a.com:bad",
        "https://a.com/",
        "https://a.com/sub",
    ):
        with pytest.raises(ValidationError):
            LaiSettings(cors_allowed_origins=(bad_origin,))
    for bad_header in ("", " ", "*", " X-Custom", "X-Custom ", "X:Bad", "X Bad"):
        with pytest.raises(ValidationError, match="Invalid CORS header name"):
            LaiSettings(cors_extra_allow_headers=(bad_header,))
    assert LaiSettings(
        cors_extra_allow_headers=("X-Custom-Trace", "OpenAI-Beta")
    ).cors_extra_allow_headers == ("X-Custom-Trace", "OpenAI-Beta")
    assert LaiSettings().oidc_leeway_seconds == 30.0
    for bad_leeway in (-1.0, 301.0):
        with pytest.raises(ValidationError):
            LaiSettings(oidc_leeway_seconds=bad_leeway)

    cfg = OidcProviderConfig(
        enabled=True, audience="aud-legacy", audiences=(" aud-1 ", " ", "aud-legacy", "aud-2")
    )
    assert cfg.resolved_audiences() == ("aud-1", "aud-legacy", "aud-2")
    assert ApiScope.PROJECTS_WRITE in OidcProviderConfig().default_scopes


def test_fair_queue_two_level_round_robin_and_principal_quota_boundaries() -> None:
    """Verify two-level (principal -> session) round-robin fairness and principal quota limits."""
    q = FairPriorityQueue(max_queued_jobs=10, max_queued_jobs_per_principal=_PRINCIPAL_QUOTA)
    with pytest.raises(ValidationFailedError):
        FairPriorityQueue(max_queued_jobs_per_principal=0)
    for jid, pid, sid in (
        ("a1", "pA", "s1"),
        ("a2", "pA", "s2"),
        ("b1", "pB", "sb"),
    ):
        q.enqueue(
            LeaseRequest(
                job_id=jid,
                session_id=sid,
                project_id="p",
                principal_id=pid,
                priority=JobPriority.INTERACTIVE,
            )
        )
    assert q.queued_count_for_principal("pA") == _PRINCIPAL_QUOTA
    with pytest.raises(QueueFullError) as exc_qf:
        q.enqueue(
            LeaseRequest(
                job_id="a3",
                session_id="s3",
                project_id="p",
                principal_id="pA",
                priority=JobPriority.INTERACTIVE,
            )
        )
    assert exc_qf.value.details == {"principal_id": "pA", "max_queued_jobs_per_principal": "2"}
    q.enqueue(
        LeaseRequest(
            job_id="a_bypass",
            session_id="s3",
            project_id="p",
            principal_id="pA",
            priority=JobPriority.INTERACTIVE,
            bypass_principal_quota=True,
        )
    )
    popped = [q.dequeue() for _ in range(_ROUND_ROBIN_COUNT)]
    assert [req.job_id for req in popped if req is not None] == [
        "a1",
        "b1",
        "a2",
        "a_bypass",
    ]
    assert q.queued_count_for_principal("pA") == 0


async def test_deps_loopback_cf_ip_and_middleware_security_headers() -> None:
    """Verify loopback-only CF-Connecting-IP rate-limit keys and SSE Cache-Control preservation."""
    seen_ip_keys: list[str] = []
    seen_user_keys: list[str] = []

    class _IpLimiter:
        async def acquire(self, key: str) -> None:
            seen_ip_keys.append(key)

    class _UserLimiter:
        async def acquire(self, key: str) -> None:
            seen_user_keys.append(key)

    class _AuthStub:
        async def authenticate(self, _creds: RequestCredentials) -> AuthPrincipal:
            return AuthPrincipal(
                principal_id="u1", auth_method=AuthMethod.API_KEY, scopes=(ApiScope.CHAT_READ,)
            )

    def _make_services(*, trust_cf: bool, cf_enabled: bool = False) -> AppServices:
        return cast(
            "AppServices",
            SimpleNamespace(
                settings=LaiSettings(
                    trust_cloudflare_headers=trust_cf,
                    cf_access_enabled=cf_enabled,
                    cf_access_client_id="cf-id" if cf_enabled else None,
                    cf_access_client_secret="cf-sec" if cf_enabled else None,
                ),
                ip_rate_limiter=_IpLimiter(),
                rate_limiter=_UserLimiter(),
                authenticator=_AuthStub(),
            ),
        )

    def _make_req(client_host: str | None, cf_ip: str | None) -> Request:
        hdrs = Headers({"cf-connecting-ip": cf_ip} if cf_ip is not None else {})
        client = SimpleNamespace(host=client_host) if client_host is not None else None
        return cast(
            "Request",
            SimpleNamespace(headers=hdrs, client=client, state=SimpleNamespace()),
        )

    await get_current_principal(
        _make_req("127.0.0.1", " 198.51.100.7 "), _make_services(trust_cf=True)
    )
    await get_current_principal(
        _make_req("::1", "198.51.100.8"), _make_services(trust_cf=False, cf_enabled=True)
    )
    await get_current_principal(
        _make_req("127.0.0.1", "198.51.100.7"), _make_services(trust_cf=False)
    )
    await get_current_principal(_make_req("127.0.0.1", "   "), _make_services(trust_cf=True))
    await get_current_principal(
        _make_req("10.0.0.5", "198.51.100.9"), _make_services(trust_cf=True)
    )
    await get_current_principal(_make_req(None, "198.51.100.9"), _make_services(trust_cf=True))
    assert seen_ip_keys == [
        "ip:198.51.100.7",
        "ip:198.51.100.8",
        "ip:127.0.0.1",
        "ip:127.0.0.1",
        "ip:10.0.0.5",
        "ip:unknown",
    ]
    assert seen_user_keys == ["u1"] * 6

    async def _run_mw(preset: list[tuple[bytes, bytes]]) -> dict[bytes, bytes]:
        sent: list[Message] = []

        async def _app(_s: Scope, _r: Receive, send: Send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": list(preset)})
            await send({"type": "http.response.body", "body": b""})

        async def _recv() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg: Message) -> None:
            sent.append(msg)

        scope: Scope = {"type": "http", "method": "GET", "path": "/x", "headers": []}
        await RequestContextMiddleware(_app)(scope, _recv, _send)
        return {k.lower(): v for k, v in cast("list[tuple[bytes, bytes]]", sent[0]["headers"])}

    sse_map = await _run_mw([(b"Cache-Control", b"no-cache")])
    assert sse_map[b"cache-control"] == b"no-cache"
    assert sse_map[b"x-content-type-options"] == b"nosniff"
    assert sse_map[b"x-frame-options"] == b"DENY"
    assert sse_map[b"referrer-policy"] == b"no-referrer"
    assert sse_map[b"permissions-policy"] == b"geolocation=(), microphone=(), camera=()"
    assert (await _run_mw([]))[b"cache-control"] == b"no-store"


async def test_oidc_provider_azp_and_multi_aud_claim_extraction() -> None:
    """Verify OIDC provider validates azp against audiences and extracts oauth_client_id."""
    priv = rsa.generate_private_key(public_exponent=_RSA_PUBLIC_EXPONENT, key_size=_RSA_KEY_SIZE)
    jwk_dict: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk_dict.update({"kid": "k1", "alg": "RS256", "use": "sig"})

    async def _jwks(_url: str) -> dict[str, Any]:
        return {"keys": [jwk_dict]}

    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url="https://accounts.google.com",
        jwks_url="https://www.googleapis.com/oauth2/v3/certs",
        audiences=("cid-web", "cid-desktop"),
        allowed_domains=("acme.test",),
    )
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=_jwks)
    now = datetime.now(UTC)

    def _tok(**extra: object) -> str:
        payload: dict[str, Any] = {
            "iss": cfg.issuer_url,
            "sub": "s1",
            "aud": "cid-web",
            "email": "u@acme.test",
            "email_verified": True,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            **extra,
        }
        return jwt.encode(payload, priv, algorithm="RS256", headers={"kid": "k1"})

    p_azp = await provider.authenticate(
        RequestCredentials(authorization_scheme="Bearer", bearer_token=_tok(azp=" cid-desktop "))
    )
    assert p_azp.oauth_client_id == "cid-desktop"

    p_list = await provider.authenticate(
        RequestCredentials(
            authorization_scheme="Bearer",
            bearer_token=_tok(aud=["other", "cid-desktop"], azp=None),
        )
    )
    assert p_list.oauth_client_id == "cid-desktop"

    for bad_azp in (_BAD_AZP_INT, "unlisted-azp"):
        with pytest.raises(AuthenticationError, match="azp"):
            await provider.authenticate(
                RequestCredentials(authorization_scheme="Bearer", bearer_token=_tok(azp=bad_azp))
            )

    slightly_expired = _tok(exp=int((now - timedelta(seconds=10)).timestamp()))
    with pytest.raises(AuthenticationError):
        await provider.authenticate(
            RequestCredentials(authorization_scheme="Bearer", bearer_token=slightly_expired)
        )
    tolerant = OidcJwtAuthProvider(
        cfg.model_copy(update={"leeway_seconds": 30.0}),
        jwks_fetcher=_jwks,
    )
    ok_leeway = await tolerant.authenticate(
        RequestCredentials(authorization_scheme="Bearer", bearer_token=slightly_expired)
    )
    assert ok_leeway.oauth_client_id == "cid-web"


async def test_middleware_logs_cors_preflight_rejected_only_on_400_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RequestContextMiddleware warns cors_preflight_rejected only on 400 CORS preflights."""
    warnings: list[tuple[str, dict[str, object]]] = []

    class _CapLog:
        def info(self, _event: str, **_kw: object) -> None:
            pass

        def warning(self, event: str, **kw: object) -> None:
            warnings.append((event, kw))

    monkeypatch.setattr("lai.api.middleware._logger", _CapLog())

    async def _run(method: str, status_code: int, headers: list[tuple[bytes, bytes]]) -> None:
        async def _app(_s: Scope, _r: Receive, send: Send) -> None:
            await send({"type": "http.response.start", "status": status_code, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def _recv() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(_m: Message) -> None:
            pass

        scope: Scope = {
            "type": "http",
            "method": method,
            "path": "/v1/projects",
            "headers": headers,
        }
        await RequestContextMiddleware(_app)(scope, _recv, _send)

    preflight_hdrs = [
        (b"Origin", b"https://spa.example.com"),
        (b"Access-Control-Request-Method", b"POST"),
        (b"Access-Control-Request-Headers", b"X-Forbidden"),
    ]
    await _run("OPTIONS", 400, preflight_hdrs)
    await _run("OPTIONS", 200, preflight_hdrs)
    await _run("GET", 400, preflight_hdrs)
    await _run("OPTIONS", 400, [(b"Access-Control-Request-Method", b"POST")])
    await _run("OPTIONS", 400, [(b"Origin", b"https://spa.example.com")])
    assert len(warnings) == 1
    assert warnings[0] == (
        "cors_preflight_rejected",
        {
            "path": "/v1/projects",
            "origin": "https://spa.example.com",
            "requested_method": "POST",
            "requested_headers": "X-Forbidden",
        },
    )
