"""Hardening tests for CF/OIDC JWKS cooldowns, stampede locks, malformed JWKs, and claims."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from pydantic import ValidationError

from lai.domain import AuthMethod, RequestCredentials
from lai.domain.errors import AuthenticationError, AuthorizationError
from lai.security import (
    API_KEY_PREFIX,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    OidcJwtAuthProvider,
    OidcProviderConfig,
)

_ISSUER = "https://accounts.google.com"
_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_AUDIENCE = "lai-hardening-aud"
_CF_DOMAIN = "team.cloudflareaccess.com"
_CF_CERTS_URL = f"https://{_CF_DOMAIN}/cdn-cgi/access/certs"
_CF_ISSUER = f"https://{_CF_DOMAIN}"
_HTTP_UNAVAILABLE = 503
_TIMEOUT = 5.0
_CF_TTL = 300.0
_OIDC_TTL = 3600.0
_COOLDOWN = 30.0
_TWO_FETCHES = 2
_THREE_FETCHES = 3
_STAMPEDE_N = 10


def _make_jwk(
    alg: str, kid: str
) -> tuple[rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey, dict[str, Any]]:
    if alg == "ES256":
        ec_priv = ec.generate_private_key(ec.SECP256R1())
        jwk: dict[str, Any] = json.loads(ECAlgorithm.to_jwk(ec_priv.public_key()))
        jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
        return ec_priv, jwk
    rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(rsa_priv.public_key()))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return rsa_priv, jwk


def _sign_token(
    key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    *,
    alg: str = "RS256",
    kid: str | None = "k1",
    drop: tuple[str, ...] = (),
    **overrides: object,
) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "sub": "sub_hardening",
        "email": "user@acme.test",
        "email_verified": True,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    claims.update(overrides)
    for claim in drop:
        claims.pop(claim, None)
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid} if kid is not None else None)


def _alg_none_token(claims: dict[str, Any], kid: str = "k1") -> str:
    hdr = base64.urlsafe_b64encode(json.dumps({"alg": "none", "kid": kid}).encode()).rstrip(b"=")
    pay = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"{hdr.decode()}.{pay.decode()}.sig"


def _cf_hdr(token: str, header: str = "Cf-Access-Jwt-Assertion") -> RequestCredentials:
    return RequestCredentials(headers={header: token})


def _bearer(token: str | None, scheme: str | None = None) -> RequestCredentials:
    return RequestCredentials(authorization_scheme=scheme, bearer_token=token)


async def _expect_cf_err(
    verifier: CloudflareEdgeVerifier, creds: RequestCredentials, match: str | None = None
) -> None:
    with pytest.raises(AuthenticationError, match=match):
        await verifier.verify_edge(creds)


async def _expect_oidc_err(
    provider: OidcJwtAuthProvider,
    creds: RequestCredentials,
    exc: type[Exception] = AuthenticationError,
    match: str | None = None,
) -> None:
    with pytest.raises(exc, match=match):
        await provider.authenticate(creds)


def _oidc_cfg(**updates: object) -> OidcProviderConfig:
    return OidcProviderConfig(
        enabled=True, issuer_url=_ISSUER, jwks_url=_JWKS_URL, audience=_AUDIENCE
    ).model_copy(update=updates)


def test_config_defaults_and_validation_bounds() -> None:
    """CloudflareEdgeConfig and OidcProviderConfig enforce defaults and validation bounds."""
    cf, od = CloudflareEdgeConfig(), OidcProviderConfig()
    assert (cf.enabled, cf.jwks_timeout_seconds) == (False, _TIMEOUT)
    assert (cf.jwks_cache_ttl_seconds, cf.min_jwks_refresh_interval_seconds) == (
        _CF_TTL,
        _COOLDOWN,
    )
    assert not CloudflareEdgeConfig(min_jwks_refresh_interval_seconds=0.0).enabled
    assert od.require_email_verified is True
    assert (od.enabled, od.algorithms) == (False, ("RS256", "ES256"))
    assert (od.jwks_cache_ttl_seconds, od.jwks_timeout_seconds) == (_OIDC_TTL, _TIMEOUT)
    assert (od.min_jwks_refresh_interval_seconds, od.leeway_seconds) == (_COOLDOWN, 0.0)
    assert OidcProviderConfig(min_jwks_refresh_interval_seconds=0.0).leeway_seconds == 0.0

    for k, v in (("jwks_cache_ttl_seconds", 0.0), ("min_jwks_refresh_interval_seconds", -0.1)):
        with pytest.raises(ValidationError):
            CloudflareEdgeConfig.model_validate({k: v})
        with pytest.raises(ValidationError):
            OidcProviderConfig.model_validate({k: v})
    with pytest.raises(ValidationError):
        OidcProviderConfig.model_validate({"leeway_seconds": -1.0})


async def test_cloudflare_service_tokens_headers_algs_and_issuer_variants() -> None:
    """CF verifier checks partial tokens, custom verifier, header errors, HS256/none, and ES256."""
    ec_priv, ec_jwk = _make_jwk("ES256", "ec-cf")
    secret = secrets.token_hex(8)
    cfg = CloudflareEdgeConfig(
        enabled=True,
        client_id="cf-id",
        client_secret=secret,
        team_domain="https://team.cloudflareaccess.com/",
        audience="cf-aud",
    )
    verifier = CloudflareEdgeVerifier(cfg, jwks={"keys": [ec_jwk]})

    def _cf_tok(kid: str | None = "ec-cf", drop: tuple[str, ...] = (), **kw: object) -> str:
        return _sign_token(ec_priv, alg="ES256", kid=kid, drop=drop, **kw)

    es_tok = _cf_tok(iss=_CF_ISSUER, aud="cf-aud")
    other_ec, _ = _make_jwk("ES256", "ec-cf")
    expired_ts = int((datetime.now(UTC) - timedelta(minutes=2)).timestamp())
    for h in (
        {"CF-Access-Client-Id": "cf-id", "Cf-Access-Jwt-Assertion": es_tok},
        {"CF-Access-Client-Secret": secret},
        {"CF-Access-Client-Id": "wrong", "CF-Access-Client-Secret": secret},
        {"Cf-Access-Jwt-Assertion": ""},
        {"Cf-Access-Jwt-Assertion": "not-a-jwt"},
    ):
        await _expect_cf_err(verifier, RequestCredentials(headers=h))
    hs_tok = jwt.encode({"exp": 9999999999}, secrets.token_hex(16), algorithm="HS256")
    for bad_jwt in (
        hs_tok,
        _alg_none_token({"exp": 9999999999}, kid="ec-cf"),
        _cf_tok(iss="https://evil.test", aud="cf-aud"),
        _cf_tok(iss=_CF_ISSUER, aud="cf-aud", exp=expired_ts),
        _sign_token(other_ec, alg="ES256", kid="ec-cf", iss=_CF_ISSUER, aud="cf-aud"),
        _cf_tok(iss=_CF_ISSUER, drop=("exp",)),
    ):
        await _expect_cf_err(verifier, _cf_hdr(bad_jwt))
    await verifier.verify_edge(_cf_hdr(es_tok, header="cf-access-jwt-assertion"))

    no_cfg = CloudflareEdgeVerifier(CloudflareEdgeConfig(enabled=True))
    svc_bad = {"CF-Access-Client-Id": "a", "CF-Access-Client-Secret": "b"}
    await _expect_cf_err(no_cfg, RequestCredentials(headers=svc_bad))
    await _expect_cf_err(no_cfg, _cf_hdr(es_tok), "not loaded")
    with pytest.raises(AuthenticationError, match="team_domain"):
        await no_cfg.refresh_jwks()

    dom_cfg = CloudflareEdgeConfig(enabled=True, team_domain="   ")
    no_dom = CloudflareEdgeVerifier(dom_cfg, jwks={"keys": [ec_jwk]})
    await no_dom.verify_edge(_cf_hdr(_cf_tok(kid="", drop=("iss", "aud"))))
    await _expect_cf_err(no_dom, _cf_hdr(_cf_tok(kid="missing")), "No matching Cloudflare")

    seen: list[str] = []
    cb = CloudflareEdgeVerifier(CloudflareEdgeConfig(enabled=True), jwt_verifier=seen.append)
    await cb.verify_edge(_cf_hdr("custom"))
    assert seen == ["custom"]


async def test_cloudflare_recovery_cooldown_ttl_malformed_jwk_and_stampede() -> None:
    """CF verifier supports cold load, 30s cooldown, TTL expiry, 401 on bad JWK, and stampede."""
    priv1, jwk1 = _make_jwk("RS256", "cf-1")
    priv2, jwk2 = _make_jwk("RS256", "cf-2")
    now, fetch_count = 100.0, 0
    payload: dict[str, Any] = {"keys": [jwk1, {"kid": "bad", "kty": "RSA"}]}

    def _certs(_req: httpx.Request) -> httpx.Response:
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 1:
            return httpx.Response(_HTTP_UNAVAILABLE, text="transient")
        return httpx.Response(200, json=payload)

    def _cf_signed(key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey, kid: str | None) -> str:
        return _sign_token(key, kid=kid, iss=_CF_ISSUER, aud="cf-aud")

    with respx.mock(assert_all_called=False) as mock:
        mock.get(_CF_CERTS_URL).mock(side_effect=_certs)
        cfg = CloudflareEdgeConfig(enabled=True, team_domain=_CF_DOMAIN, audience="cf-aud")
        async with httpx.AsyncClient() as http_client:
            verifier = CloudflareEdgeVerifier(cfg, http_client=http_client, clock=lambda: now)
            with pytest.raises(AuthenticationError):
                await verifier.refresh_jwks()

            tok1 = _cf_signed(priv1, "cf-1")
            await asyncio.gather(*[verifier.verify_edge(_cf_hdr(tok1)) for _ in range(_STAMPEDE_N)])
            assert fetch_count == _TWO_FETCHES
            await _expect_cf_err(verifier, _cf_hdr(_cf_signed(priv1, "bad")), "Invalid Cloudflare")

            for t_step in (100.0, 115.0, 129.9):
                now = t_step
                for unk_kid in (f"rand-{t_step}", None):
                    await _expect_cf_err(verifier, _cf_hdr(_cf_signed(priv2, unk_kid)))
            assert fetch_count == _TWO_FETCHES

            payload, now = {"keys": [jwk2]}, 130.0
            tok2 = _cf_signed(priv2, "cf-2")
            await verifier.verify_edge(_cf_hdr(tok2))
            assert fetch_count == _THREE_FETCHES

            inj = CloudflareEdgeVerifier(cfg, jwks={"keys": [jwk1]}, clock=lambda: now)
            await _expect_cf_err(inj, _cf_hdr(tok2))
            assert fetch_count == _THREE_FETCHES
            now = 160.0
            await inj.verify_edge(_cf_hdr(tok2))
            assert fetch_count == _THREE_FETCHES + 1


async def test_oidc_header_sub_email_domain_normalization_and_leeway() -> None:
    """OIDC checks bearer shapes, HS256/none, sub='', missing email, @domain, and leeway."""
    rsa_priv, rsa_jwk = _make_jwk("RS256", "k1")

    async def fetcher(_url: str) -> dict[str, Any]:
        return {"keys": [rsa_jwk]}

    cfg = _oidc_cfg(
        allowed_emails=(" VIP@Other.ORG ",),
        allowed_domains=(" @Acme.Test ",),
        admin_emails=(" Boss@Acme.Test ",),
        leeway_seconds=10.0,
    )
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=fetcher)
    assert provider.name == "oidc"
    assert OidcJwtAuthProvider(OidcProviderConfig()).supports(_bearer("a.b.c")) is False
    for bad_c in (
        _bearer(None),
        _bearer("a.b.c", scheme="Basic"),
        _bearer(f"{API_KEY_PREFIX}1.2.3"),
        *(_bearer(r) for r in ("a.b", "a.b.c.d", "a..c")),
    ):
        assert provider.supports(bad_c) is False
        await _expect_oidc_err(provider, bad_c)
    await _expect_oidc_err(provider, _bearer("bad.jwt.hdr"))

    hs_claims = {"iss": _ISSUER, "aud": _AUDIENCE, "sub": "s", "exp": 9999999999}
    hs_tok = jwt.encode(hs_claims, secrets.token_hex(16), algorithm="HS256")
    for tok in (
        _alg_none_token(hs_claims),
        hs_tok,
        *(_sign_token(rsa_priv, sub=s) for s in ("", "   ")),
        _sign_token(rsa_priv, drop=("sub",)),
    ):
        await _expect_oidc_err(provider, _bearer(tok))
    for tok in (
        _sign_token(rsa_priv, drop=("email",)),
        *(_sign_token(rsa_priv, email=e) for e in ("  ", "localuser")),
    ):
        await _expect_oidc_err(provider, _bearer(tok), AuthorizationError)

    recent_exp = int((datetime.now(UTC) - timedelta(seconds=5)).timestamp())
    admin_tok = _sign_token(rsa_priv, email="Boss@Acme.Test", exp=recent_exp)
    admin_p = await provider.authenticate(_bearer(admin_tok, scheme="bearer"))
    assert admin_p.auth_method is AuthMethod.OIDC
    assert (admin_p.email, len(admin_p.scopes)) == ("boss@acme.test", 1)
    vip = await provider.authenticate(_bearer(_sign_token(rsa_priv, email="vip@other.org")))
    assert vip.scopes == cfg.default_scopes

    strict_cfg = _oidc_cfg(require_email_verified=False)
    strict_p = OidcJwtAuthProvider(strict_cfg, jwks_fetcher=fetcher)
    await _expect_oidc_err(strict_p, _bearer(_sign_token(rsa_priv, drop=("aud",))))
    anon = _sign_token(rsa_priv, email_verified=False, drop=("email",))
    anon_p = await strict_p.authenticate(_bearer(anon))
    assert (anon_p.email, anon_p.oauth_client_id) == (None, _AUDIENCE)
    free = _sign_token(rsa_priv, kid="", email="free@any.org")
    assert (await strict_p.authenticate(_bearer(free))).email == "free@any.org"
    no_aud_p = OidcJwtAuthProvider(_oidc_cfg(audience=None), jwks_fetcher=fetcher)
    await _expect_oidc_err(no_aud_p, _bearer(admin_tok))


async def test_oidc_cooldown_stampede_multikey_malformed_jwk_and_http_client() -> None:
    """OIDC prevents unknown-kid flood within 30s, coalesces stampede, and wraps bad JWKs."""
    priv1, jwk1 = _make_jwk("RS256", "k1")
    priv2, jwk2 = _make_jwk("RS256", "k2")
    now, fetch_count = 50.0, 0
    jwks_doc: dict[str, Any] = {"keys": [jwk1, {"kid": "bad-jwk", "kty": "RSA"}]}

    async def _fetcher(_url: str) -> dict[str, Any]:
        nonlocal fetch_count
        fetch_count += 1
        return jwks_doc

    cfg = _oidc_cfg(allowed_domains=("acme.test",))
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=_fetcher, clock=lambda: now)

    tok1 = _sign_token(priv1, kid="k1")
    coros = [provider.authenticate(_bearer(tok1)) for _ in range(_STAMPEDE_N)]
    assert len(await asyncio.gather(*coros)) == _STAMPEDE_N
    assert fetch_count == 1
    await _expect_oidc_err(
        provider, _bearer(_sign_token(priv1, kid="bad-jwk")), match="Invalid OIDC JWK"
    )

    for t_step in (50.0, 65.0, 79.9):
        now = t_step
        for bad_kid in (f"rand-{t_step}", None):
            await _expect_oidc_err(provider, _bearer(_sign_token(priv2, kid=bad_kid)))
    assert fetch_count == 1

    jwks_doc, now = {"keys": [jwk2]}, 80.0
    res2 = await provider.authenticate(_bearer(_sign_token(priv2, kid="k2")))
    assert res2.subject == "sub_hardening"
    assert fetch_count == _TWO_FETCHES

    with respx.mock(assert_all_called=True) as mock:
        mock.get(_JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [jwk1]}))
        async with httpx.AsyncClient() as http_client:
            http_p = OidcJwtAuthProvider(cfg, http_client=http_client)
            assert (await http_p.authenticate(_bearer(tok1))).email == "user@acme.test"
