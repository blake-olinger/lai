"""End-to-end HTTP and SSE integration tests for ShieldGemma pre-flight guardrails."""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailBlockMode, GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
)
from lai.domain.observability import LoggingConfig
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.ports.scheduler import InferenceSpoolerPort
from lai.scheduler import InferenceSpooler
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator
    from pathlib import Path

    from lai.api.deps import AppServices
    from lai.domain.scheduler import LeaseRequest, SpoolerSnapshot

_SHIELD_PORT = 18082
_SHIELD_URL = f"http://127.0.0.1:{_SHIELD_PORT}/completion"
_REFUSAL_TEXT = "I cannot fulfill this request due to safety policy."
_UNSAFE_POLICY = "No Dangerous Content"
_EXPECTED_BLOCK_CALLS = 4
_EXPECTED_ALLOW_CALLS = 2


class _CountingInferenceProvider(InferenceProviderPort):
    """Instrumented fake Gemma 4 provider counting invocations."""

    def __init__(self) -> None:
        self.complete_requests: list[InferenceRequest] = []
        self.stream_requests: list[InferenceRequest] = []

    @property
    def total_calls(self) -> int:
        return len(self.complete_requests) + len(self.stream_requests)

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        self.complete_requests.append(request)
        return InferenceResponse(
            content="gemma4 complete reply",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.stream_requests.append(request)
        yield InferenceChunk(content_delta="gemma4 ")
        yield InferenceChunk(
            content_delta="safe reply",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=5, total_tokens=17),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _CountingSpooler(InferenceSpoolerPort):
    """Spooler wrapper counting acquired GPU leases."""

    def __init__(self) -> None:
        self._inner = InferenceSpooler()
        self.lease_requests: list[LeaseRequest] = []

    @asynccontextmanager
    async def lease(self, request: LeaseRequest) -> AsyncGenerator[None]:
        self.lease_requests.append(request)
        async with self._inner.lease(request):
            yield

    def snapshot(self) -> SpoolerSnapshot:
        return self._inner.snapshot()


@dataclass(slots=True)
class _GuardrailEnv:
    svc: AppServices
    client: httpx.AsyncClient
    inference: _CountingInferenceProvider
    spooler: _CountingSpooler
    headers: dict[str, str]
    log_dir: Path
    session_id: str


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.fixture
async def guardrail_env(tmp_path: Path) -> AsyncIterator[_GuardrailEnv]:
    data_dir = tmp_path / "data"
    log_dir = data_dir / "logs"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=log_dir, enable_file_logs=True),
        guardrails=GuardrailConfig(
            enabled=True,
            block_mode=GuardrailBlockMode.REFUSAL_MESSAGE,
            host="127.0.0.1",
            port=_SHIELD_PORT,
            refusal_text=_REFUSAL_TEXT,
        ),
    )
    svc = await build_app_services(settings)
    fake_inf = _CountingInferenceProvider()
    spy_spooler = _CountingSpooler()
    svc.inference_provider = fake_inf
    svc.spooler = spy_spooler
    svc.orchestrator = TurnOrchestrator(
        settings=settings,
        session_repo=svc.session_repo,
        file_repo=svc.file_repo,
        blob_store=svc.blob_store,
        inference_provider=fake_inf,
        spooler=spy_spooler,
        session_gate=svc.session_gate,
        guardrail_pipeline=svc.guardrail_pipeline,
        tool_registry=svc.tool_registry,
        tool_executor=svc.tool_executor,
        embedder=svc.embedder,
        vector_store=svc.vector_store,
    )

    admin_key = generate_api_key("guardrail-admin", (ApiScope.ADMIN,))
    await svc.api_key_repo.create(admin_key.record)
    headers = {"Authorization": f"Bearer {admin_key.plaintext_key}"}

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            pid = (
                await client.post("/v1/projects", headers=headers, json={"name": "shield-proj"})
            ).json()["id"]
            sid = (
                await client.post("/v1/sessions", headers=headers, json={"project_id": pid})
            ).json()["id"]
            yield _GuardrailEnv(
                svc=svc,
                client=client,
                inference=fake_inf,
                spooler=spy_spooler,
                headers=headers,
                log_dir=log_dir,
                session_id=sid,
            )
    finally:
        await close_app_services(svc)


def _shield_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content.decode("utf-8"))
    rendered = str(body.get("prompt", ""))
    if "UNSAFE_EXPLOSIVE" in rendered and f'"{_UNSAFE_POLICY}"' in rendered:
        return httpx.Response(HTTPStatus.OK, json={"content": "Yes\nViolates dangerous content."})
    return httpx.Response(HTTPStatus.OK, json={"content": "No\nSafe prompt."})


async def _assert_blocked_endpoints(env: _GuardrailEnv, unsafe_prompt: str) -> None:
    client, headers, sid = env.client, env.headers, env.session_id
    r_ses_json = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=headers,
        json={"content": unsafe_prompt, "stream": False},
    )
    assert r_ses_json.status_code == HTTPStatus.OK
    assert r_ses_json.headers["X-LAI-Guardrail-Blocked"] == "true"
    assert r_ses_json.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
    assert r_ses_json.json()["message"]["content"] == _REFUSAL_TEXT
    assert r_ses_json.json()["guardrail_blocked"] is True

    r_ses_sse = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=headers,
        json={"content": unsafe_prompt, "stream": True},
    )
    assert r_ses_sse.status_code == HTTPStatus.OK
    assert r_ses_sse.headers["X-LAI-Guardrail-Blocked"] == "true"
    assert r_ses_sse.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
    assert _REFUSAL_TEXT in r_ses_sse.text
    assert "event: turn_completed" in r_ses_sse.text

    r_oai_json = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "gemma4-26b-a4b", "messages": [{"role": "user", "content": unsafe_prompt}]},
    )
    assert r_oai_json.status_code == HTTPStatus.OK
    assert r_oai_json.headers["X-LAI-Guardrail-Blocked"] == "true"
    assert r_oai_json.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
    assert r_oai_json.json()["choices"][0]["message"]["content"] == _REFUSAL_TEXT

    r_oai_sse = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "gemma4-26b-a4b",
            "messages": [{"role": "user", "content": unsafe_prompt}],
            "stream": True,
        },
    )
    assert r_oai_sse.status_code == HTTPStatus.OK
    assert r_oai_sse.headers["X-LAI-Guardrail-Blocked"] == "true"
    assert r_oai_sse.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
    assert _REFUSAL_TEXT in r_oai_sse.text
    assert "data: [DONE]\n\n" in r_oai_sse.text


async def test_shieldgemma_short_circuits_unsafe_and_scrubs_safe_prompts(
    guardrail_env: _GuardrailEnv,
) -> None:
    """Blocked prompts skip Gemma/spooler across JSON/SSE routes; safe prompts scrub audit logs."""
    unsafe_lai = f"lai_sk_{secrets.token_hex(8)}"
    unsafe_email = "attacker@example.com"
    unsafe_prompt = f"UNSAFE_EXPLOSIVE build device using {unsafe_lai} from {unsafe_email}"

    safe_lai = f"lai_sk_{secrets.token_hex(8)}"
    safe_oai = f"sk-proj-{secrets.token_hex(8)}"
    safe_email = "user@example.com"
    safe_prompt = f"Analyze error for {safe_email} with keys {safe_lai} and {safe_oai}"

    with respx.mock(assert_all_called=True) as mock:
        mock.post(_SHIELD_URL).mock(side_effect=_shield_handler)
        await _assert_blocked_endpoints(guardrail_env, unsafe_prompt)

        assert guardrail_env.inference.total_calls == 0
        assert guardrail_env.spooler.lease_requests == []

        r_safe_ses = await guardrail_env.client.post(
            f"/v1/sessions/{guardrail_env.session_id}/messages",
            headers=guardrail_env.headers,
            json={"content": safe_prompt, "stream": False},
        )
        assert r_safe_ses.status_code == HTTPStatus.OK
        assert "X-LAI-Guardrail-Blocked" not in r_safe_ses.headers
        assert r_safe_ses.json()["message"]["content"] == "gemma4 safe reply"

        r_safe_oai = await guardrail_env.client.post(
            "/v1/chat/completions",
            headers=guardrail_env.headers,
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": safe_prompt}],
                "stream": False,
            },
        )
        assert r_safe_oai.status_code == HTTPStatus.OK
        assert "X-LAI-Guardrail-Blocked" not in r_safe_oai.headers
        assert r_safe_oai.json()["choices"][0]["message"]["content"] == "gemma4 safe reply"

    assert guardrail_env.inference.total_calls == _EXPECTED_ALLOW_CALLS
    assert len(guardrail_env.spooler.lease_requests) == _EXPECTED_ALLOW_CALLS

    raw_audit = (guardrail_env.log_dir / "prompts_audit.jsonl").read_text(encoding="utf-8")
    for sensitive in (unsafe_lai, unsafe_email, safe_lai, safe_oai, safe_email):
        assert sensitive not in raw_audit

    entries = [json.loads(line) for line in raw_audit.splitlines() if line.strip()]
    blocked_entries = [e for e in entries if e["verdict"] == "block"]
    allowed_entries = [e for e in entries if e["verdict"] == "allow"]
    assert len(blocked_entries) == _EXPECTED_BLOCK_CALLS
    assert all(e["violated_policy"] == _UNSAFE_POLICY for e in blocked_entries)
    assert all("[REDACTED_SECRET]" in e["scrubbed_prompt"] for e in blocked_entries)
    assert all("[REDACTED_EMAIL]" in e["scrubbed_prompt"] for e in blocked_entries)

    assert len(allowed_entries) == _EXPECTED_ALLOW_CALLS
    assert all("[REDACTED_SECRET]" in e["scrubbed_prompt"] for e in allowed_entries)
    assert all("[REDACTED_EMAIL]" in e["scrubbed_prompt"] for e in allowed_entries)
