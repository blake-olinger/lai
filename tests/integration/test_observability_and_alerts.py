"""End-to-end integration tests for structured JSONL logs, alerting, and webhook delivery."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx
import respx

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.orchestrator import TurnOrchestrator
from tests._helpers import FailingEventStream, StubOrchestratorBase
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from lai.domain.chat import OrchestratorEvent, TurnInput, TurnOutput

_LLAMA_PORT = 18081
_SHIELD_PORT = 18082
_WEBHOOK_URL = "https://alerts.example.test/webhook"
_MIN_TIMEOUT_ALERTS = 2
_TWO = 2
_SANITIZED_ERR = "An unexpected internal server error occurred."
_CHAT_BODY: dict[str, object] = {
    "model": "gemma4-26b-a4b",
    "messages": [{"role": "user", "content": "crash now"}],
}


class _CrashingOrchestrator(StubOrchestratorBase):
    """Orchestrator stub raising unhandled RuntimeError directly or mid-stream."""

    def __init__(self, *, stream_prefix_events: Sequence[OrchestratorEvent] = ()) -> None:
        self._prefix = tuple(stream_prefix_events)

    async def run_turn(self, turn_input: TurnInput, principal: AuthPrincipal) -> TurnOutput:
        del turn_input, principal
        msg = "unhandled crash with sensitive token"
        raise RuntimeError(msg)

    def run_turn_stream(
        self, turn_input: TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[OrchestratorEvent]:
        del turn_input, principal
        if not self._prefix:
            return FailingEventStream(RuntimeError("unhandled stream crash token"))

        async def _midstream_fail() -> AsyncIterator[OrchestratorEvent]:
            for event in self._prefix:
                yield event
            msg = "midstream secret crash token"
            raise RuntimeError(msg)

        return _midstream_fail()


def _obs_settings(tmp_path: Path) -> tuple[LaiSettings, Path]:
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
    return settings, log_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


async def test_guardrail_and_inference_timeouts_emit_logs_webhooks_and_alerts(
    tmp_path: Path,
) -> None:
    """Verify guardrail and inference timeouts write JSONL logs, fire webhooks, and surface."""
    settings, log_dir = _obs_settings(tmp_path)
    async with open_services_harness(settings) as harness:
        assert isinstance(harness.svc.orchestrator, TurnOrchestrator)
        headers = await mint_auth_headers(harness.svc, "obs-timeout-admin", (ApiScope.ADMIN,))
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

            r_timeout = await harness.client.post(
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

            await harness.svc.alert_manager.flush()
            assert webhook_route.call_count >= _MIN_TIMEOUT_ALERTS

            alerts_payload = (await harness.client.get("/v1/system/alerts", headers=headers)).json()
            assert alerts_payload["count"] >= _MIN_TIMEOUT_ALERTS
            assert AlertCategory.TIMEOUT.value in {i["category"] for i in alerts_payload["items"]}
            assert {"guardrails", "inference"}.issubset(
                {i["component"] for i in alerts_payload["items"]}
            )
            status_payload = (await harness.client.get("/v1/system/status", headers=headers)).json()
            assert status_payload["status"] == "ok"
            assert status_payload["recent_alert_count"] >= _MIN_TIMEOUT_ALERTS

    error_lines = _read_jsonl(log_dir / "errors.jsonl")
    alert_lines = _read_jsonl(log_dir / "alerts.jsonl")
    assert any(entry.get("event") == "shieldgemma_request_timeout" for entry in error_lines)
    assert any(entry.get("event") == "inference_stream_timeout" for entry in error_lines)
    assert any(entry["category"] == AlertCategory.TIMEOUT.value for entry in alert_lines)


async def test_unhandled_500_exception_emits_fatal_alert_logs_and_webhook(
    tmp_path: Path,
) -> None:
    """Verify unhandled JSON route 500 writes JSONL logs, fires webhook, and records alert."""
    settings, log_dir = _obs_settings(tmp_path)
    spec = IntegrationSpec(orchestrator=_CrashingOrchestrator(), raise_app_exceptions=False)
    async with open_services_harness(settings, spec) as harness:
        headers = await mint_auth_headers(harness.svc, "obs-500-admin", (ApiScope.ADMIN,))
        with respx.mock(assert_all_called=True) as mock:
            webhook_route = mock.post(_WEBHOOK_URL).respond(HTTPStatus.NO_CONTENT)
            r_fatal = await harness.client.post(
                "/v1/chat/completions",
                headers={**headers, "X-Request-Id": "req_fatal_500"},
                json=_CHAT_BODY,
            )
            assert r_fatal.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            assert r_fatal.json()["error"]["code"] == "internal_error"
            assert r_fatal.json()["error"]["request_id"] == "req_fatal_500"
            assert "sensitive token" not in r_fatal.text

            await harness.svc.alert_manager.flush()
            assert webhook_route.call_count >= 1

            alerts = (await harness.client.get("/v1/system/alerts", headers=headers)).json()[
                "items"
            ]
            assert any(
                a["category"] == AlertCategory.FATAL_EXCEPTION.value
                and a["component"] == "api"
                and a["request_id"] == "req_fatal_500"
                for a in alerts
            )

    error_lines = _read_jsonl(log_dir / "errors.jsonl")
    alert_lines = _read_jsonl(log_dir / "alerts.jsonl")
    assert any("unhandled_api_exception" in str(entry.get("event", "")) for entry in error_lines)
    assert any(entry["category"] == AlertCategory.FATAL_EXCEPTION.value for entry in alert_lines)


async def test_sse_midstream_crash_emits_fatal_alert_and_sanitized_error_frame(
    tmp_path: Path,
) -> None:
    """Verify real SSE stream=True mid-stream crash emits FATAL_EXCEPTION and sanitized frame."""
    from lai.domain.chat import ProgressEvent  # noqa: PLC0415

    settings, log_dir = _obs_settings(tmp_path)
    orch = _CrashingOrchestrator(
        stream_prefix_events=(ProgressEvent(stage="inference", message="running"),)
    )
    spec = IntegrationSpec(orchestrator=orch, raise_app_exceptions=False)
    async with open_services_harness(settings, spec) as harness:
        headers = await mint_auth_headers(harness.svc, "obs-sse-admin", (ApiScope.ADMIN,))
        pid = await create_project(harness.client, headers, "obs-sse-proj")
        sid = await create_session(harness.client, headers, pid)
        with respx.mock(assert_all_called=True) as mock:
            webhook_route = mock.post(_WEBHOOK_URL).respond(HTTPStatus.NO_CONTENT)
            r_ses = await harness.client.post(
                f"/v1/sessions/{sid}/messages",
                headers={**headers, "X-Request-Id": "req_sse_ses_500"},
                json={"content": "crash midstream", "stream": True},
            )
            assert r_ses.status_code == HTTPStatus.OK
            assert "event: progress" in r_ses.text
            assert '"code":"internal_error"' in r_ses.text
            assert _SANITIZED_ERR in r_ses.text
            assert "midstream secret crash token" not in r_ses.text

            r_oai = await harness.client.post(
                "/v1/chat/completions",
                headers={**headers, "X-Request-Id": "req_sse_oai_500"},
                json={**_CHAT_BODY, "stream": True},
            )
            assert r_oai.status_code == HTTPStatus.OK
            assert '"type": "internal_error"' in r_oai.text
            assert _SANITIZED_ERR in r_oai.text
            assert r_oai.text.endswith("data: [DONE]\n\n")
            assert "midstream secret crash token" not in r_oai.text

            await harness.svc.alert_manager.flush()
            assert webhook_route.call_count >= _TWO

            alerts = (await harness.client.get("/v1/system/alerts", headers=headers)).json()[
                "items"
            ]
            cat = AlertCategory.FATAL_EXCEPTION.value
            fatal_ids = {a["request_id"] for a in alerts if a["category"] == cat}
            assert {"req_sse_ses_500", "req_sse_oai_500"}.issubset(fatal_ids)

    alert_lines = _read_jsonl(log_dir / "alerts.jsonl")
    assert any(entry["category"] == AlertCategory.FATAL_EXCEPTION.value for entry in alert_lines)
    assert any("unhandled_sse_exception" in str(entry.get("message", "")) for entry in alert_lines)


async def test_sse_pre_priming_crash_returns_500_and_fatal_alert(tmp_path: Path) -> None:
    """Verify FailingEventStream crash during SSE priming returns HTTP 500 and FATAL_EXCEPTION."""
    settings, _ = _obs_settings(tmp_path)
    spec = IntegrationSpec(orchestrator=_CrashingOrchestrator(), raise_app_exceptions=False)
    async with open_services_harness(settings, spec) as harness:
        headers = await mint_auth_headers(harness.svc, "obs-prime-admin", (ApiScope.ADMIN,))
        r_err = await harness.client.post(
            "/v1/chat/completions",
            headers={**headers, "X-Request-Id": "req_sse_prime_500"},
            json={**_CHAT_BODY, "stream": True},
        )
        assert r_err.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert r_err.json()["error"]["code"] == "internal_error"
        assert "unhandled stream crash token" not in r_err.text
        alerts = harness.svc.alert_manager.recent_alerts()
        assert any(
            a.category is AlertCategory.FATAL_EXCEPTION and a.request_id == "req_sse_prime_500"
            for a in alerts
        )
