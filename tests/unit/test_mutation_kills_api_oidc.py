"""Mutation-killing tests for deps, RequestContextMiddleware, and OIDC JWT validation."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request

from lai.api.deps import extract_request_credentials
from lai.api.middleware import RequestContextMiddleware
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.errors import AuthenticationError
from lai.domain.observability import LoggingConfig
from lai.observability import configure_observability
from lai.security import OidcJwtAuthProvider, OidcProviderConfig

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from starlette.types import Message, Receive, Scope, Send

_HTTP_OK = 200
_FIFTY_MS = 50.0
_OIDC_ISS = "https://idp.mutation.test"
_OIDC_JWKS = f"{_OIDC_ISS}/.well-known/jwks.json"


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are flushed and closed."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _http_request(headers: list[tuple[bytes, bytes]]) -> Request:
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


async def test_deps_schedule_creds_and_middleware_duration_ms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert AppServices task return, scheme preservation, and middleware duration_ms."""
    sample_jwt = "sample-jwt-value"
    creds = extract_request_credentials(
        _http_request([(b"authorization", f"Bearer {sample_jwt}".encode("ascii"))]),
    )
    assert creds.authorization_scheme == "Bearer"
    assert creds.bearer_token == sample_jwt

    services = await build_app_services(LaiSettings(data_dir=tmp_path / "svc"))
    try:

        async def _job() -> str:
            return "bg_ok"

        task = services.schedule_background_task(_job(), name="unit-bg")
        assert isinstance(task, asyncio.Task)
        assert await task == "bg_ok"
    finally:
        await close_app_services(services)

    clock = [100.0]
    monkeypatch.setattr("lai.api.middleware.time.perf_counter", lambda: clock[0])
    log_dir = tmp_path / "mw_logs"
    manager = configure_observability(
        LoggingConfig(log_dir=log_dir, level="INFO", json_console=False, enable_file_logs=True),
    )
    try:

        async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
            del scope, receive
            clock[0] += 0.05
            await send({"type": "http.response.start", "status": _HTTP_OK, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def _recv() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg: Message) -> None:
            del msg

        await RequestContextMiddleware(_ok_app)(
            {"type": "http", "method": "GET", "path": "/health", "headers": []},
            _recv,
            _send,
        )
    finally:
        manager.close()
        for handler in list(logging.getLogger().handlers):
            handler.flush()
            handler.close()
        logging.getLogger().handlers.clear()

    mw_events = [
        json.loads(line)
        for line in (log_dir / "lai.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed = [row for row in mw_events if row.get("event") == "http_request_completed"]
    assert len(completed) == 1
    assert completed[0]["duration_ms"] == pytest.approx(_FIFTY_MS)


async def test_oidc_provider_enforces_signature_and_audience_verification() -> None:
    """Reject wrong-key RS256 signatures and mismatched aud claims."""
    good_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(good_key.public_key()))
    jwk.update({"kid": "k1", "alg": "RS256", "use": "sig"})

    now = datetime.now(UTC)
    forged_sig_jwt = jwt.encode(
        {
            "iss": _OIDC_ISS,
            "sub": "u1",
            "aud": "lai-api",
            "azp": "cid-web",
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        wrong_key,
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    bad_aud_jwt = jwt.encode(
        {
            "iss": _OIDC_ISS,
            "sub": "u1",
            "aud": "wrong-aud",
            "azp": "cid-web",
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        good_key,
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    provider = OidcJwtAuthProvider(
        OidcProviderConfig(
            enabled=True,
            issuer_url=_OIDC_ISS,
            jwks_url=_OIDC_JWKS,
            audience="lai-api",
            require_email_verified=False,
            jwks_cache_ttl_seconds=60.0,
        ),
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(_OIDC_JWKS).mock(return_value=httpx.Response(_HTTP_OK, json={"keys": [jwk]}))
        for raw_token in (forged_sig_jwt, bad_aud_jwt):
            creds = extract_request_credentials(
                _http_request([(b"authorization", f"Bearer {raw_token}".encode("ascii"))]),
            )
            with pytest.raises(AuthenticationError):
                await provider.authenticate(creds)
