"""Unit tests for API keys, scope hierarchy, auth providers, Cloudflare edge, and rate limits."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from lai.domain import (
    AlertCategory,
    ApiKeyRecord,
    ApiScope,
    AuthMethod,
    AuthPrincipal,
    LoggingConfig,
    RequestCredentials,
)
from lai.domain.errors import AuthenticationError, AuthorizationError, RateLimitExceededError
from lai.observability import configure_observability
from lai.ports import AuthenticatorPort, AuthProviderPort, EdgeVerifierPort
from lai.security import (
    API_KEY_PREFIX,
    ApiKeyAuthProvider,
    CloudflareEdgeConfig,
    CloudflareEdgeVerifier,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
    TokenBucketRateLimiter,
    expand_scopes,
    generate_api_key,
    hash_api_key,
    require_scope,
    verify_key_hash,
)

_ISSUER = "https://accounts.google.com"
_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_AUDIENCE = "lai-client-id"
_HTTP_FORBIDDEN = 403
_HTTP_TOO_MANY = 429
_FETCH_TWICE = 2
_FETCH_THRICE = 3


class _MemoryApiKeyRepo:
    """In-memory ApiKeyRepositoryPort stub for isolated provider unit tests."""

    def __init__(self, records: list[ApiKeyRecord] | None = None) -> None:
        self._records: dict[str, ApiKeyRecord] = {r.id: r for r in (records or [])}

    async def create(self, record: ApiKeyRecord) -> ApiKeyRecord:
        self._records[record.id] = record
        return record

    async def get_by_id(self, key_id: str) -> ApiKeyRecord | None:
        return self._records.get(key_id)

    async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        return next((r for r in self._records.values() if r.key_hash == key_hash), None)

    async def list_all(self) -> list[ApiKeyRecord]:
        return list(self._records.values())

    async def touch_last_used(self, key_id: str, used_at: datetime) -> None:
        if rec := self._records.get(key_id):
            self._records[key_id] = rec.model_copy(update={"last_used_at": used_at})

    async def revoke(self, key_id: str, revoked_at: datetime) -> bool:
        if (rec := self._records.get(key_id)) is None:
            return False
        self._records[key_id] = rec.model_copy(update={"revoked_at": revoked_at})
        return True


def _make_jwk_pair(
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


def _sign_jwt(
    key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    *,
    alg: str = "RS256",
    kid: str | None = "rsa-1",
    **claim_overrides: object,
) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "sub": "sub_123",
        "email": "dev@example.com",
        "email_verified": True,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid} if kid else None)


def _bearer(token: str | None, scheme: str | None = None) -> RequestCredentials:
    return RequestCredentials(authorization_scheme=scheme, bearer_token=token)


def test_api_key_generation_hashing_and_constant_time_verify() -> None:
    """generate_api_key emits lai_sk_ token, prefix, and SHA-256 hash verified cleanly."""
    created = generate_api_key("ci-key", (ApiScope.PROJECTS_WRITE, ApiScope.PROJECTS_WRITE))
    assert created.plaintext_key.startswith(API_KEY_PREFIX)
    assert created.record.key_prefix == created.plaintext_key[:14]
    assert created.record.key_hash == hash_api_key(created.plaintext_key)
    assert created.record.key_hash != created.plaintext_key
    assert created.record.scopes == (ApiScope.PROJECTS_WRITE,)
    assert verify_key_hash(created.plaintext_key, created.record.key_hash)
    assert not verify_key_hash(created.plaintext_key + "x", created.record.key_hash)
    assert not verify_key_hash("", created.record.key_hash)
    assert not verify_key_hash(created.plaintext_key, "")


@pytest.mark.parametrize(
    ("granted", "allowed", "denied"),
    [
        ((ApiScope.ADMIN,), tuple(ApiScope), ()),
        ((ApiScope.PROJECTS_WRITE,), (ApiScope.PROJECTS_READ,), (ApiScope.FILES_READ,)),
        ((ApiScope.FILES_WRITE,), (ApiScope.FILES_READ,), (ApiScope.CHAT_READ,)),
        ((ApiScope.CHAT_WRITE,), (ApiScope.CHAT_READ,), (ApiScope.TOOLS_SAFE,)),
        ((ApiScope.TOOLS_EXEC,), (ApiScope.TOOLS_SAFE,), (ApiScope.TOOLS_APPROVE,)),
    ],
)
def test_scope_hierarchy_expansion_and_require_scope(
    granted: tuple[ApiScope, ...],
    allowed: tuple[ApiScope, ...],
    denied: tuple[ApiScope, ...],
) -> None:
    """Scope hierarchy implies read/safe scopes and rejects missing privileges."""
    principal = AuthPrincipal(principal_id="usr_1", auth_method=AuthMethod.API_KEY, scopes=granted)
    expanded = expand_scopes(granted)
    for scope in allowed:
        assert scope in expanded
        require_scope(principal, scope)
    for scope in denied:
        assert scope not in expanded
        with pytest.raises(AuthorizationError) as exc_info:
            require_scope(principal, scope)
        assert exc_info.value.status_code == _HTTP_FORBIDDEN


async def test_api_key_provider_auth_revocation_and_expiry() -> None:
    """ApiKeyAuthProvider authenticates active keys, touches last_used, and rejects stale keys."""
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    active = generate_api_key("active", (ApiScope.CHAT_WRITE,))
    revoked = generate_api_key("revoked", (ApiScope.CHAT_READ,))
    revoked_rec = revoked.record.model_copy(update={"revoked_at": now - timedelta(minutes=1)})
    expired = generate_api_key("expired", (ApiScope.CHAT_READ,), expires_at=now)
    repo = _MemoryApiKeyRepo([active.record, revoked_rec, expired.record])
    provider = ApiKeyAuthProvider(repo, clock=lambda: now)
    assert isinstance(provider, AuthProviderPort)
    assert provider.name == "api_key"

    valid_creds = _bearer(active.plaintext_key)
    jwt_like = "header.payload.sig"
    assert provider.supports(valid_creds)
    assert not provider.supports(_bearer(active.plaintext_key, scheme="Basic"))
    assert not provider.supports(_bearer(jwt_like))

    principal = await provider.authenticate(valid_creds)
    assert principal.auth_method is AuthMethod.API_KEY
    assert principal.key_id == active.record.id
    updated = await repo.get_by_id(active.record.id)
    assert updated is not None
    assert updated.last_used_at == now

    bad_tokens = (revoked.plaintext_key, expired.plaintext_key, f"{API_KEY_PREFIX}unknown123", None)
    for bad_token in bad_tokens:
        with pytest.raises(AuthenticationError):
            await provider.authenticate(_bearer(bad_token))


async def test_oidc_provider_rsa_ec_caching_rotation_and_policies() -> None:
    """OidcJwtAuthProvider validates RS256/ES256 JWTs, caches/rotates JWKS, and maps scopes."""
    rsa_priv1, rsa_jwk1 = _make_jwk_pair("RS256", "rsa-1")
    rsa_priv2, rsa_jwk2 = _make_jwk_pair("RS256", "rsa-2")
    ec_priv, ec_jwk = _make_jwk_pair("ES256", "ec-1")
    fetch_count = 0
    clock_now = 100.0
    jwks_payload: dict[str, Any] = {"keys": [rsa_jwk1, ec_jwk]}

    async def fake_fetcher(_url: str) -> dict[str, Any]:
        nonlocal fetch_count
        fetch_count += 1
        return jwks_payload

    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=_ISSUER,
        jwks_url=_JWKS_URL,
        audience=_AUDIENCE,
        allowed_emails=("allowed@other.org",),
        allowed_domains=("example.com",),
        admin_emails=("admin@example.com",),
        jwks_cache_ttl_seconds=60.0,
    )
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=fake_fetcher, clock=lambda: clock_now)

    rsa_token = _sign_jwt(rsa_priv1, alg="RS256", kid="rsa-1", email="admin@example.com")
    ec_token = _sign_jwt(ec_priv, alg="ES256", kid="ec-1", email="allowed@other.org")
    assert provider.supports(_bearer(rsa_token))
    assert not provider.supports(_bearer(f"{API_KEY_PREFIX}not_a_jwt"))

    admin_principal = await provider.authenticate(_bearer(rsa_token))
    assert admin_principal.auth_method is AuthMethod.OIDC
    assert admin_principal.scopes == (ApiScope.ADMIN,)
    assert admin_principal.issuer == _ISSUER
    assert admin_principal.subject == "sub_123"

    user_principal = await provider.authenticate(_bearer(ec_token))
    assert user_principal.scopes == cfg.default_scopes
    assert fetch_count == 1

    # Unknown kid refreshes after cooldown; TTL expiry triggers subsequent refresh.
    jwks_payload = {"keys": [rsa_jwk2]}
    clock_now = 130.0
    rotated_token = _sign_jwt(rsa_priv2, alg="RS256", kid="rsa-2", email="user@example.com")
    await provider.authenticate(_bearer(rotated_token))
    assert fetch_count == _FETCH_TWICE
    clock_now = 200.0
    await provider.authenticate(_bearer(rotated_token))
    assert fetch_count == _FETCH_THRICE


async def test_oidc_provider_claims_errors_and_timeout_alert() -> None:
    """OidcJwtAuthProvider rejects bad claims/algs and logs TIMEOUT alerts on JWKS timeout."""
    rsa_priv, rsa_jwk = _make_jwk_pair("RS256", "rsa-1")
    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=_ISSUER,
        jwks_url=_JWKS_URL,
        audience=_AUDIENCE,
        allowed_domains=("example.com",),
    )

    async def static_fetcher(_url: str) -> dict[str, Any]:
        return {"keys": [rsa_jwk]}

    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=static_fetcher)
    past_exp = int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())
    hs_key = secrets.token_hex(16)
    bad_tokens = (
        _sign_jwt(rsa_priv, exp=past_exp),
        _sign_jwt(rsa_priv, iss="https://evil.example"),
        _sign_jwt(rsa_priv, aud="wrong-aud"),
        _sign_jwt(rsa_priv, email_verified=False),
        jwt.encode({"iss": _ISSUER, "aud": _AUDIENCE, "sub": "s", "exp": 9999999999}, hs_key),
    )
    for tok in bad_tokens:
        with pytest.raises(AuthenticationError):
            await provider.authenticate(_bearer(tok))

    with pytest.raises(AuthorizationError):
        await provider.authenticate(_bearer(_sign_jwt(rsa_priv, email="stranger@blocked.org")))

    manager = configure_observability(LoggingConfig(enable_file_logs=False, level="INFO"))
    try:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(_JWKS_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
            http_provider = OidcJwtAuthProvider(cfg)
            with pytest.raises(AuthenticationError, match="Timed out"):
                await http_provider.authenticate(_bearer(_sign_jwt(rsa_priv)))
        assert any(alert.category is AlertCategory.TIMEOUT for alert in manager.recent_alerts())
    finally:
        manager.close()
        logging.getLogger().handlers.clear()


async def test_cloudflare_edge_and_composite_authenticator_chain() -> None:
    """CloudflareEdgeVerifier and CompositeAuthenticator compose edge + multi-provider auth."""
    rsa_priv, rsa_jwk = _make_jwk_pair("RS256", "cf-1")
    edge_cred = secrets.token_hex(16)
    cf_cfg = CloudflareEdgeConfig(
        enabled=True,
        client_id="cf-id",
        client_secret=edge_cred,
        team_domain="team.cloudflareaccess.com",
        audience="cf-aud",
    )
    verifier = CloudflareEdgeVerifier(cf_cfg, jwks={"keys": [rsa_jwk]})
    assert isinstance(verifier, EdgeVerifierPort)
    svc_hdrs = {"CF-Access-Client-Id": "cf-id", "CF-Access-Client-Secret": edge_cred}
    await verifier.verify_edge(RequestCredentials(headers=svc_hdrs))
    cf_jwt = _sign_jwt(rsa_priv, kid="cf-1", iss="https://team.cloudflareaccess.com", aud="cf-aud")
    await verifier.verify_edge(RequestCredentials(headers={"Cf-Access-Jwt-Assertion": cf_jwt}))

    for bad_hdrs in ({"CF-Access-Client-Id": "cf-id", "CF-Access-Client-Secret": "bad"}, {}):
        with pytest.raises(AuthenticationError):
            await verifier.verify_edge(RequestCredentials(headers=bad_hdrs))

    created = generate_api_key("chain-key", (ApiScope.FILES_READ,))
    chain = CompositeAuthenticator(
        providers=[ApiKeyAuthProvider(_MemoryApiKeyRepo([created.record]))],
        edge_verifier=verifier,
    )
    assert isinstance(chain, AuthenticatorPort)
    ok_creds = RequestCredentials(
        bearer_token=created.plaintext_key, headers={"Cf-Access-Jwt-Assertion": cf_jwt}
    )
    assert (await chain.authenticate(ok_creds)).key_id == created.record.id
    for bad_creds in (
        _bearer(created.plaintext_key),
        RequestCredentials(headers={"Cf-Access-Jwt-Assertion": cf_jwt}),
    ):
        with pytest.raises(AuthenticationError):
            await chain.authenticate(bad_creds)


async def test_token_bucket_rate_limiter_enforces_burst_refill_and_isolation() -> None:
    """TokenBucketRateLimiter enforces per-principal capacity and refills via monotonic clock."""
    now = 10.0
    limiter = TokenBucketRateLimiter(
        requests_per_minute=60, burst_capacity=2, clock=lambda: now, max_buckets=2
    )
    await limiter.acquire("usr_a")
    await limiter.check("usr_a")
    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.acquire("usr_a")
    assert exc_info.value.status_code == _HTTP_TOO_MANY
    assert exc_info.value.retry_after_seconds == pytest.approx(1.0)
    await limiter.acquire("usr_b")
    now = 11.0
    await limiter.acquire("usr_a")
    await limiter.acquire("usr_c")
