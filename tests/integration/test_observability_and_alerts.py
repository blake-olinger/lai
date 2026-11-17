"""End-to-end integration tests for structured JSONL logs, alerting, and webhook delivery."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.orchestrator import TurnOrchestrator
from lai.ports.orchestrator import OrchestratorPort
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator
    from pathlib import Path

    from lai.api.deps import AppServices
    from lai.domain.chat import OrchestratorEvent, TurnInput, TurnOutput

_LLAMA_PORT = 18081
_SHIELD_PORT = 18082
_WEBHOOK_URL = "https://alerts.example.test/webhook"
_MIN_EXPECTED_ALERTS = 3


class _CrashingOrchestrator(OrchestratorPort):
    """Orchestrator stub raising an unhandled RuntimeError to verify 500 alerting."""

    async def run_turn(self, turn_input: TurnInput, principal: AuthPrincipal) -> TurnOutput:
        del turn_input, principal
        msg = "unhandled crash with sensitive token"
        raise RuntimeError(msg)

    async def run_turn_stream(
        self, turn_input: TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[OrchestratorEvent]:
        del turn_input, principal
        msg = "unhandled stream crash"
        raise RuntimeError(msg)
        yield  # pragma: no cover


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def _exercise_alerts_over_http(
    client: httpx.AsyncClient,
    svc: AppServices,
    headers: dict[str, str],
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        shield_route = mock.post(f"http://127.0.0.1:{_SHIELD_PORT}/completion").mock(
            side_effect=httpx.ReadTimeout("guardrail timed out")
        )
        llama_route = mock.post(f"http://127.0.0.1:{_LLAMA_PORT}/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("inference timed out")
        )
        webhook_route = mock.post(_WEBHOOK_URL).respond(HTTPStatus.NO_CONTENT)
        mock.get(f"http://127.0.0.1:{_LLAMA_PORT}/health").respond(
            HTTPStatus.OK, json={"status": "ok"}
        )
        mock.get(f"http://127.0.0.1:{_SHIELD_PORT}/health").respond(
            HTTPStatus.OK, json={"status": "ok"}
        )

        r_timeout = await client.post(
            "/v1/chat/completions",
            headers={**headers, "X-Request-Id": "req_timeout_1"},
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": "trigger timeout"}],
            },
        )
        assert r_timeout.status_code == HTTPStatus.BAD_GATEWAY
        assert shield_route.called
        assert llama_route.called

        svc.orchestrator = _CrashingOrchestrator()
        r_fatal = await client.post(
            "/v1/chat/completions",
            headers={**headers, "X-Request-Id": "req_fatal_500"},
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": "crash now"}],
            },
        )
        assert r_fatal.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert r_fatal.json()["error"]["code"] == "internal_error"
        assert r_fatal.json()["error"]["request_id"] == "req_fatal_500"

        await svc.alert_manager.flush()
        assert webhook_route.call_count >= _MIN_EXPECTED_ALERTS

        r_alerts = await client.get("/v1/system/alerts", headers=headers)
        assert r_alerts.status_code == HTTPStatus.OK
        alerts_payload = r_alerts.json()
        assert alerts_payload["count"] >= _MIN_EXPECTED_ALERTS
        categories = {item["category"] for item in alerts_payload["items"]}
        assert AlertCategory.TIMEOUT.value in categories
        assert AlertCategory.FATAL_EXCEPTION.value in categories
        components = {item["component"] for item in alerts_payload["items"]}
        assert {"guardrails", "inference", "api"}.issubset(components)

        r_status = await client.get("/v1/system/status", headers=headers)
        assert r_status.status_code == HTTPStatus.OK
        status_payload = r_status.json()
        assert status_payload["status"] == "ok"
        assert status_payload["recent_alert_count"] >= _MIN_EXPECTED_ALERTS


async def test_errors_timeouts_and_fatal_exceptions_surface_in_logs_webhook_and_api(
    tmp_path: Path,
) -> None:
    """Verify 500s and guardrail/inference timeouts write JSONL logs, fire webhooks, and surface."""
    data_dir = tmp_path / "data"
    log_dir = data_dir / "logs"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        llama_port=_LLAMA_PORT,
        logging=LoggingConfig(
            log_dir=log_dir,
            enable_file_logs=True,
            alert_webhook_url=_WEBHOOK_URL,
            alert_cooldown_seconds=0.0,
            alert_on_timeouts=True,
        ),
        guardrails=GuardrailConfig(
            enabled=True,
            fail_open=True,
            port=_SHIELD_PORT,
            timeout_seconds=0.5,
        ),
    )
    svc = await build_app_services(settings)
    assert isinstance(svc.orchestrator, TurnOrchestrator)
    admin_key = generate_api_key("obs-admin", (ApiScope.ADMIN,))
    await svc.api_key_repo.create(admin_key.record)
    headers = {"Authorization": f"Bearer {admin_key.plaintext_key}"}

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            await _exercise_alerts_over_http(client, svc, headers)
    finally:
        await close_app_services(svc)

    error_lines = [
        json.loads(line)
        for line in (log_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    alert_lines = [
        json.loads(line)
        for line in (log_dir / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(entry.get("event") == "shieldgemma_request_timeout" for entry in error_lines)
    assert any(entry.get("event") == "inference_stream_timeout" for entry in error_lines)
    assert any("unhandled_api_exception" in str(entry.get("event", "")) for entry in error_lines)

    assert len(alert_lines) >= _MIN_EXPECTED_ALERTS
    alert_cats = {entry["category"] for entry in alert_lines}
    assert AlertCategory.TIMEOUT.value in alert_cats
    assert AlertCategory.FATAL_EXCEPTION.value in alert_cats
