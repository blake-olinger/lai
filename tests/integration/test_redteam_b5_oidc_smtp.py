"""Red-team verification proofs for B5 (Entra v2 OIDC hardening & SMTP secret guards)."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from pydantic import ValidationError

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, RequestCredentials
from lai.domain.errors import AuthenticationError, AuthorizationError
from lai.domain.observability import (
    AlertCategory,
    AlertEvent,
    AlertSeverity,
    LoggingConfig,
    SmtpAlertConfig,
    SmtpTlsMode,
)
from lai.observability import AlertManager
from lai.observability.smtp import SmtpAlertSink, deliver_test_alert
from lai.security import OidcJwtAuthProvider, OidcProviderConfig
from tests.integration._helpers import mint_auth_headers, open_services_harness

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

_ENTRA_ISSUER = "https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/v2.0"
_ENTRA_JWKS = (
    "https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/discovery/v2.0/keys"
)
_API_AUD = "api-app-client-id"
_SPA_ID = "web-spa-client-id"
_DESKTOP_ID = "desktop-app-client-id"
_TEST_SMTP_SECRET = "boot-secret-value"  # noqa: S105 - synthetic test fixture


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _rsa_pair() -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk_dict: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk_dict.update({"kid": "entra-k1", "alg": "RS256", "use": "sig"})
    return priv, jwk_dict


def _mint(priv: rsa.RSAPrivateKey, kid: str, extra: dict[str, Any]) -> str:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": _ENTRA_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
        **extra,
    }
    return jwt.encode(claims, priv, algorithm="RS256", headers={"kid": kid})


def _creds(token: str) -> RequestCredentials:
    return RequestCredentials(authorization_scheme="Bearer", bearer_token=token)


def _start_dummy_smtp_listener() -> tuple[
    socket.socket, int, threading.Event, threading.Thread, list[bytes]
]:
    received_bytes: list[bytes] = []
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.bind(("127.0.0.1", 0))
    srv_sock.listen(4)
    port = int(srv_sock.getsockname()[1])
    stop_evt = threading.Event()

    def _serve() -> None:
        srv_sock.settimeout(0.2)
        while not stop_evt.is_set():
            try:
                conn, _ = srv_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                conn.sendall(b"220 EVIL_BANNER_CANARY_987654 ESMTP\r\n")
                conn.settimeout(0.3)
                try:
                    chunk = conn.recv(4096)
                    if chunk:
                        received_bytes.append(chunk)
                    conn.sendall(b"550 SECRET_REMOTE_BANNER_LEAK\r\n")
                except OSError:
                    pass

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return srv_sock, port, stop_evt, thread, received_bytes


async def test_b5_entra_v2_oid_principal_split_aud_azp_admin_roles_and_guests() -> None:
    """B5(a)+(b)+(c)+(d): Entra oid stability, aud/azp split, verified admin, guest block."""
    priv, jwk_dict = _rsa_pair()
    kid = str(jwk_dict["kid"])

    async def _jwks(_url: str) -> dict[str, Any]:
        return {"keys": [jwk_dict]}

    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=_ENTRA_ISSUER,
        jwks_url=_ENTRA_JWKS,
        audiences=(_API_AUD,),
        allowed_client_ids=(_SPA_ID, _DESKTOP_ID),
        allowed_domains=("corp.example.com",),
        admin_roles=("LAI.Admin",),
        admin_emails=("boss@corp.example.com",),
        require_email_verified=False,
        allow_guest_users=False,
    )
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=_jwks)

    spa_tok = _mint(
        priv,
        kid,
        {
            "aud": _API_AUD,
            "azp": _SPA_ID,
            "sub": "pairwise-sub-spa",
            "oid": "oid-user-42",
            "email": "alice@corp.example.com",
            "roles": ["LAI.Admin"],
        },
    )
    desk_tok = _mint(
        priv,
        kid,
        {
            "aud": _API_AUD,
            "azp": _DESKTOP_ID,
            "sub": "pairwise-sub-desktop",
            "oid": "oid-user-42",
            "email": "alice@corp.example.com",
            "roles": ["LAI.Admin"],
        },
    )
    p_spa = await provider.authenticate(_creds(spa_tok))
    p_desk = await provider.authenticate(_creds(desk_tok))
    expected_pid = f"oidc:{_ENTRA_ISSUER}:oid-user-42"
    assert p_spa is not None
    assert p_desk is not None
    assert p_spa.principal_id == expected_pid
    assert p_desk.principal_id == expected_pid
    assert p_spa.oauth_client_id == _SPA_ID
    assert p_desk.oauth_client_id == _DESKTOP_ID
    assert p_spa.has_scope(ApiScope.ADMIN)

    id_tok = _mint(
        priv,
        kid,
        {"aud": _SPA_ID, "sub": "sub-1", "oid": "oid-1", "email": "alice@corp.example.com"},
    )
    with pytest.raises(AuthenticationError):
        await provider.authenticate(_creds(id_tok))

    base_boss = {
        "aud": _API_AUD,
        "azp": _SPA_ID,
        "sub": "sub-boss",
        "oid": "oid-boss",
        "email": "boss@corp.example.com",
    }
    p_unv = await provider.authenticate(_creds(_mint(priv, kid, base_boss)))
    assert p_unv is not None
    assert not p_unv.has_scope(ApiScope.ADMIN)

    p_ver = await provider.authenticate(_creds(_mint(priv, kid, {**base_boss, "xms_edov": True})))
    assert p_ver is not None
    assert p_ver.has_scope(ApiScope.ADMIN)

    for guest_claims in (
        {"acct": 1},
        {"idp": "https://sts.windows.net/99999999-9999-9999-9999-999999999999/"},
    ):
        guest_tok = _mint(
            priv,
            kid,
            {
                "aud": _API_AUD,
                "azp": _SPA_ID,
                "sub": "sub-g",
                "oid": "oid-g",
                "email": "guest@corp.example.com",
                **guest_claims,
            },
        )
        with pytest.raises(AuthorizationError):
            await provider.authenticate(_creds(guest_tok))


def test_b5_oidc_https_and_multitenant_validator_and_smtp_cleartext_validator() -> None:
    """B5(e)+(f): Reject HTTP OIDC, /common|/organizations|/consumers, and remote plaintext SMTP."""
    for bad_oidc in (
        {"oidc_issuer_url": "http://login.microsoftonline.com/tenant/v2.0"},
        {"oidc_jwks_url": "http://login.microsoftonline.com/tenant/discovery/v2.0/keys"},
        {"oidc_issuer_url": "https://login.microsoftonline.com/common/v2.0"},
        {"oidc_issuer_url": "https://login.microsoftonline.com/%63ommon/v2.0"},
        {"oidc_issuer_url": "https://login.microsoftonline.com/organizations/v2.0"},
        {"oidc_issuer_url": "https://login.microsoftonline.com/consumers/v2.0"},
        {"oidc_jwks_url": "https://login.microsoftonline.com/common/discovery/v2.0/keys"},
    ):
        with pytest.raises(ValidationError):
            LaiSettings(
                oidc_enabled=True,
                oidc_issuer_url=str(bad_oidc.get("oidc_issuer_url", _ENTRA_ISSUER)),
                oidc_jwks_url=str(bad_oidc.get("oidc_jwks_url", _ENTRA_JWKS)),
                oidc_audiences=(_API_AUD,),
                oidc_allowed_domains=("corp.example.com",),
            )

    with pytest.raises(ValidationError, match=r"(?i)plaintext|tls"):
        LaiSettings(
            logging=LoggingConfig(
                enable_file_logs=False,
                alert_smtp=SmtpAlertConfig(
                    enabled=True,
                    host="smtp.attacker.example",
                    port=25,
                    tls_mode=SmtpTlsMode.NONE,
                    username="ops",
                    password=_TEST_SMTP_SECRET,
                    from_address="alerts@corp.example.com",
                    to_addresses=("admin@corp.example.com",),
                ),
            )
        )


async def test_b5_smtp_retarget_clears_password_and_sanitizes_banner(tmp_path: Path) -> None:
    """B5(f): Retargeting SMTP clears boot password; remote NONE sends zero AUTH; banner clean."""
    srv_sock, port, stop_evt, thread, received_bytes = _start_dummy_smtp_listener()
    ev = AlertEvent(
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.SECURITY_ALERT,
        component="test",
        message="probe",
    )
    try:
        sink_remote = SmtpAlertSink(
            SmtpAlertConfig(
                enabled=True,
                host="smtp.evil.example",
                port=port,
                tls_mode=SmtpTlsMode.NONE,
                username="ops",
                password=_TEST_SMTP_SECRET,
                from_address="alerts@corp.example.com",
                to_addresses=("admin@corp.example.com",),
            )
        )
        with pytest.raises(OSError, match="plaintext"):
            await sink_remote.send_alert_strict(ev)
        assert not any(b"AUTH" in b for b in received_bytes)

        settings = LaiSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            bin_dir=tmp_path / "bin",
            logging=LoggingConfig(
                log_dir=tmp_path / "logs",
                enable_file_logs=False,
                alert_smtp=SmtpAlertConfig(
                    enabled=True,
                    host="smtp.corp.example.com",
                    port=587,
                    tls_mode=SmtpTlsMode.STARTTLS,
                    username="ops",
                    password=_TEST_SMTP_SECRET,
                    from_address="alerts@corp.example.com",
                    to_addresses=("admin@corp.example.com",),
                ),
            ),
        )
        async with open_services_harness(settings) as h:
            headers = await mint_auth_headers(h.svc, "admin", tuple(ApiScope))
            patch = await h.client.patch(
                "/v1/system/alerting",
                headers=headers,
                json={
                    "smtp_enabled": True,
                    "smtp_host": "127.0.0.1",
                    "smtp_port": port,
                    "smtp_tls_mode": "none",
                    "smtp_username": "ops",
                    "smtp_from_address": "alerts@corp.example.com",
                    "smtp_to_addresses": ["admin@corp.example.com"],
                },
            )
            assert patch.status_code == HTTPStatus.OK, patch.text
            assert h.svc.settings.logging.alert_smtp.password is None

            res_test = await h.client.post(
                "/v1/system/alerting/test", headers=headers, json={"message": "check"}
            )
            assert res_test.status_code == HTTPStatus.OK
            payload_text = json.dumps(res_test.json())
            assert "EVIL_BANNER_CANARY_987654" not in payload_text
            assert "SECRET_REMOTE_BANNER_LEAK" not in payload_text
            assert not any(b"AUTH" in b or _TEST_SMTP_SECRET.encode() in b for b in received_bytes)

        local_sink = SmtpAlertSink(
            SmtpAlertConfig(
                enabled=True,
                host="127.0.0.1",
                port=port,
                tls_mode=SmtpTlsMode.NONE,
                from_address="a@corp.example.com",
                to_addresses=("b@corp.example.com",),
            )
        )
        results = await deliver_test_alert(AlertManager(sinks=(local_sink,)), ev)
        assert results
        assert not results[0].delivered
        assert "SECRET_REMOTE_BANNER_LEAK" not in (results[0].error or "")
    finally:
        stop_evt.set()
        srv_sock.close()
        await asyncio.to_thread(thread.join, 2.0)
