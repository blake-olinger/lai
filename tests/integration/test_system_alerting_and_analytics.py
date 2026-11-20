"""Integration tests for SMTP alerting admin API, redaction, and endpoint usage analytics."""

from __future__ import annotations

import secrets
import smtplib
from http import HTTPStatus as S
from typing import TYPE_CHECKING, ClassVar

import respx

from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import BackendHealth, InferenceChunk, InferenceRequest, InferenceResponse
from lai.domain.observability import LoggingConfig, SmtpAlertConfig
from lai.ports.inference import InferenceProviderPort
from tests.integration._helpers import (
    IntegrationSpec,
    configure_test_oidc,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from email.message import EmailMessage
    from pathlib import Path
    from types import TracebackType

    import pytest


class _TokenInference(InferenceProviderPort):
    """Stub inference backend returning deterministic token usage."""

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(
            content="ok",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        yield InferenceChunk(
            content_delta="hello",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _StubSmtp:
    """Test double for smtplib.SMTP capturing sent EmailMessages."""

    sent: ClassVar[list[EmailMessage]] = []

    def __init__(self, host: str, port: int, timeout: float = 10.0) -> None:
        del host, port, timeout

    def __enter__(self) -> _StubSmtp:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb

    def ehlo(self) -> None:
        return None

    def starttls(self, *, context: object | None = None) -> None:
        del context

    def login(self, user: str, password: str) -> None:
        del user, password

    def send_message(self, msg: EmailMessage) -> None:
        type(self).sent.append(msg)


async def test_alerting_admin_api_redaction_test_delivery_and_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GET/PATCH /v1/system/alerting, secret safety, test delivery, and restart load."""
    _StubSmtp.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _StubSmtp)

    webhook_url = "https://hooks.example.test/services/SECRET_TOKEN"
    smtp_cred = secrets.token_hex(8)
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        oidc_admin_emails=("owner@example.test",),
        guardrails=GuardrailConfig(enabled=False),
        logging=LoggingConfig(
            log_dir=tmp_path / "data" / "logs",
            enable_file_logs=False,
            alert_webhook_url=webhook_url,
            alert_smtp=SmtpAlertConfig(
                enabled=False,
                host="smtp.example.test",
                username="mailer@example.test",
                password=smtp_cred,
            ),
        ),
    )

    async with open_services_harness(settings) as h:
        admin_hdrs = await mint_auth_headers(h.svc, "adm", (ApiScope.ADMIN,))
        user_hdrs = await mint_auth_headers(h.svc, "usr", (ApiScope.PROJECTS_READ,))

        for path in (
            "/v1/system/alerting",
            "/v1/system/analytics",
            "/v1/system/analytics/config",
            "/v1/system/config/oidc",
            "/v1/system/config/cors",
            "/v1/system/config/limits",
            "/v1/system/config/guardrails",
        ):
            assert (await h.client.get(path, headers=user_hdrs)).status_code == S.FORBIDDEN

        r_get = await h.client.get("/v1/system/alerting", headers=admin_hdrs)
        assert r_get.status_code == S.OK
        payload = r_get.json()
        assert payload["alert_webhook_configured"] is True
        assert payload["alert_webhook_target"] == "https://hooks.example.test/***"
        assert "SECRET_TOKEN" not in r_get.text
        assert smtp_cred not in r_get.text
        assert payload["smtp_password_configured"] is True
        assert payload["effective_smtp_recipients"] == ["owner@example.test"]

        r_bad = await h.client.patch(
            "/v1/system/alerting", headers=admin_hdrs, json={"smtp_enabled": True}
        )
        assert r_bad.status_code == S.UNPROCESSABLE_ENTITY

        r_patch = await h.client.patch(
            "/v1/system/alerting",
            headers=admin_hdrs,
            json={
                "smtp_enabled": True,
                "smtp_host": "smtp.example.test",
                "smtp_from_address": "alerts@example.test",
                "alert_cooldown_seconds": 15.0,
            },
        )
        assert r_patch.status_code == S.OK
        assert r_patch.json()["smtp_enabled"] is True

        assert h.svc.runtime_config_repo is not None
        stored = await h.svc.runtime_config_repo.get_section("alerting")
        assert stored is not None
        assert "alert_webhook_url" not in stored
        assert "password" not in str(stored)

        with respx.mock(assert_all_called=True) as mock:
            wh_route = mock.post(webhook_url).respond(S.NO_CONTENT)
            r_test = await h.client.post(
                "/v1/system/alerting/test",
                headers=admin_hdrs,
                json={"message": "Pre-flight check"},
            )
            assert r_test.status_code == S.OK
            body = r_test.json()
            assert (body["delivered_count"], body["failed_count"]) == (2, 0)
            assert wh_route.called
            assert len(_StubSmtp.sent) == 1
            assert "Pre-flight check" in str(_StubSmtp.sent[0]["Subject"])

    restarted = await build_app_services(settings)
    try:
        assert restarted.settings.logging.alert_smtp.enabled is True
        assert restarted.settings.logging.alert_smtp.host == "smtp.example.test"
        assert restarted.settings.logging.alert_smtp.password == smtp_cred
        assert restarted.settings.logging.alert_cooldown_seconds == 15.0
    finally:
        await close_app_services(restarted)


async def test_usage_analytics_endpoint_rollups_tokens_and_toggle(tmp_path: Path) -> None:
    """Verify normalized route rollups, opaque principals, token totals, and enable toggle."""
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "data" / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    async with open_services_harness(
        settings, IntegrationSpec(inference_provider=_TokenInference())
    ) as h:
        admin_hdrs = await mint_auth_headers(h.svc, "analytics-adm", (ApiScope.ADMIN,))
        oidc = configure_test_oidc(
            h.svc, audiences=("spa-client-1",), allowed_domains=("acme.test",)
        )
        user_hdrs = oidc.mint_headers(sub="user-42", email="alice@acme.test", aud="spa-client-1")

        p1 = await create_project(h.client, user_hdrs, "proj-1")
        p2 = await create_project(h.client, user_hdrs, "proj-2")
        assert (await h.client.get(f"/v1/projects/{p1}", headers=user_hdrs)).status_code == S.OK
        assert (await h.client.get(f"/v1/projects/{p2}", headers=user_hdrs)).status_code == S.OK
        assert (await h.client.get("/v1/nonexistent-probe-123")).status_code == S.NOT_FOUND

        s1 = await create_session(h.client, user_hdrs, p1)
        assert (
            await h.client.post(
                f"/v1/sessions/{s1}/messages",
                headers=user_hdrs,
                json={"content": "hello", "stream": False},
            )
        ).status_code == S.OK
        assert (
            await h.client.post(
                "/v1/chat/completions",
                headers=user_hdrs,
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
        ).status_code == S.OK

        r_report = await h.client.get("/v1/system/analytics?window_hours=24", headers=admin_hdrs)
        assert r_report.status_code == S.OK
        report = r_report.json()
        assert "alice@acme.test" not in r_report.text
        routes = {(e["method"], e["route"]): e for e in report["endpoints"]}
        assert routes[("GET", "/v1/projects/{project_id}")]["request_count"] == 2
        assert routes[("GET", "<unmatched>")]["status_4xx"] == 1
        assert report["total_tokens"] == 40
        assert report["unique_client_apps"] == 1
        principals = {p["principal_id"]: p for p in report["principals"]}
        assert principals[f"oidc:{oidc.issuer.rstrip('/')}:user-42"]["oauth_client_id"] == (
            "spa-client-1"
        )

        before_total = report["total_requests"]
        r_cfg = await h.client.patch(
            "/v1/system/analytics/config",
            headers=admin_hdrs,
            json={"enabled": False, "retention_days": 30},
        )
        assert r_cfg.status_code == S.OK
        assert r_cfg.json() == {"enabled": False, "retention_days": 30}
        await h.client.get("/v1/health")
        await h.client.get("/v1/health")
        after_report = (await h.client.get("/v1/system/analytics", headers=admin_hdrs)).json()
        assert after_report["total_requests"] <= before_total + 2
