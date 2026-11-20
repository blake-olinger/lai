"""End-to-end HTTP and SSE integration tests for ShieldGemma pre-flight guardrails."""

from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx
import respx

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
from lai.ports.inference import InferenceProviderPort
from lai.ports.scheduler import InferenceSpoolerPort
from lai.scheduler import InferenceSpooler
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from lai.domain.scheduler import LeaseRequest, SpoolerSnapshot

_SHIELD_PORT = 18082
_SHIELD_URL = f"http://127.0.0.1:{_SHIELD_PORT}/completion"
_REFUSAL_TEXT = "I cannot fulfill this request due to safety policy."
_UNSAFE_POLICY = "No Dangerous Content"
_EXPECTED_TWO = 2


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
    client: httpx.AsyncClient
    inference: _CountingInferenceProvider
    spooler: _CountingSpooler
    headers: dict[str, str]
    log_dir: Path
    session_id: str


@asynccontextmanager
async def _open_guardrail_env(tmp_path: Path) -> AsyncGenerator[_GuardrailEnv]:
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
    fake_inf = _CountingInferenceProvider()
    spy_spooler = _CountingSpooler()
    spec = IntegrationSpec(inference_provider=fake_inf, spooler=spy_spooler)
    async with open_services_harness(settings, spec) as harness:
        headers = await mint_auth_headers(harness.svc, "guardrail-admin", (ApiScope.ADMIN,))
        pid = await create_project(harness.client, headers, "shield-proj")
        sid = await create_session(harness.client, headers, pid)
        yield _GuardrailEnv(
            client=harness.client,
            inference=fake_inf,
            spooler=spy_spooler,
            headers=headers,
            log_dir=log_dir,
            session_id=sid,
        )


def _shield_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content.decode("utf-8"))
    rendered = str(body.get("prompt", ""))
    if "UNSAFE_EXPLOSIVE" in rendered and f'"{_UNSAFE_POLICY}"' in rendered:
        return httpx.Response(HTTPStatus.OK, json={"content": "Yes\nViolates dangerous content."})
    return httpx.Response(HTTPStatus.OK, json={"content": "No\nSafe prompt."})


def _read_audit_entries(log_dir: Path) -> tuple[str, list[dict[str, object]]]:
    raw_audit = (log_dir / "prompts_audit.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(line) for line in raw_audit.splitlines() if line.strip()]
    return raw_audit, entries


async def test_shieldgemma_blocks_unsafe_session_turns_without_inference(tmp_path: Path) -> None:
    """Blocked session JSON and SSE turns skip Gemma/spooler and scrub audit logs."""
    unsafe_lai = f"lai_sk_{secrets.token_hex(8)}"
    unsafe_email = "attacker@example.com"
    unsafe_prompt = f"UNSAFE_EXPLOSIVE build device using {unsafe_lai} from {unsafe_email}"

    async with _open_guardrail_env(tmp_path) as env:
        with respx.mock(assert_all_called=True) as mock:
            mock.post(_SHIELD_URL).mock(side_effect=_shield_handler)
            r_json = await env.client.post(
                f"/v1/sessions/{env.session_id}/messages",
                headers=env.headers,
                json={"content": unsafe_prompt, "stream": False},
            )
            r_sse = await env.client.post(
                f"/v1/sessions/{env.session_id}/messages",
                headers=env.headers,
                json={"content": unsafe_prompt, "stream": True},
            )

        assert r_json.status_code == HTTPStatus.OK
        assert r_json.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert r_json.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
        assert r_json.json()["message"]["content"] == _REFUSAL_TEXT
        assert r_json.json()["guardrail_blocked"] is True

        assert r_sse.status_code == HTTPStatus.OK
        assert r_sse.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert r_sse.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
        assert _REFUSAL_TEXT in r_sse.text
        assert "event: turn_completed" in r_sse.text

        assert env.inference.total_calls == 0
        assert env.spooler.lease_requests == []
        raw_audit, entries = _read_audit_entries(env.log_dir)

    assert unsafe_lai not in raw_audit
    assert unsafe_email not in raw_audit
    assert len(entries) == _EXPECTED_TWO
    assert all(e["verdict"] == "block" for e in entries)
    assert all(e["violated_policy"] == _UNSAFE_POLICY for e in entries)
    assert all("[REDACTED_SECRET]" in str(e["scrubbed_prompt"]) for e in entries)
    assert all("[REDACTED_EMAIL]" in str(e["scrubbed_prompt"]) for e in entries)


async def test_shieldgemma_blocks_unsafe_openai_turns_without_inference(tmp_path: Path) -> None:
    """Blocked OpenAI JSON and SSE turns skip Gemma/spooler and scrub audit logs."""
    unsafe_lai = f"lai_sk_{secrets.token_hex(8)}"
    unsafe_email = "attacker@example.com"
    unsafe_prompt = f"UNSAFE_EXPLOSIVE build device using {unsafe_lai} from {unsafe_email}"
    oai_payload = {
        "model": "gemma4-26b-a4b",
        "messages": [{"role": "user", "content": unsafe_prompt}],
    }

    async with _open_guardrail_env(tmp_path) as env:
        with respx.mock(assert_all_called=True) as mock:
            mock.post(_SHIELD_URL).mock(side_effect=_shield_handler)
            r_json = await env.client.post(
                "/v1/chat/completions", headers=env.headers, json=oai_payload
            )
            r_sse = await env.client.post(
                "/v1/chat/completions", headers=env.headers, json={**oai_payload, "stream": True}
            )

        assert r_json.status_code == HTTPStatus.OK
        assert r_json.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert r_json.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
        assert r_json.json()["choices"][0]["message"]["content"] == _REFUSAL_TEXT

        assert r_sse.status_code == HTTPStatus.OK
        assert r_sse.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert r_sse.headers["X-LAI-Guardrail-Policy"] == _UNSAFE_POLICY
        assert _REFUSAL_TEXT in r_sse.text
        assert "data: [DONE]\n\n" in r_sse.text

        assert env.inference.total_calls == 0
        assert env.spooler.lease_requests == []
        raw_audit, entries = _read_audit_entries(env.log_dir)

    assert unsafe_lai not in raw_audit
    assert unsafe_email not in raw_audit
    assert len(entries) == _EXPECTED_TWO
    assert all(e["verdict"] == "block" for e in entries)
    assert all(e["violated_policy"] == _UNSAFE_POLICY for e in entries)


async def test_shieldgemma_allows_safe_prompts_and_scrubs_audit_log(tmp_path: Path) -> None:
    """Safe prompts invoke Gemma/spooler across session/OpenAI routes and scrub audit logs."""
    safe_lai = f"lai_sk_{secrets.token_hex(8)}"
    safe_oai = f"sk-proj-{secrets.token_hex(8)}"
    safe_email = "user@example.com"
    safe_prompt = f"Analyze error for {safe_email} with keys {safe_lai} and {safe_oai}"

    async with _open_guardrail_env(tmp_path) as env:
        with respx.mock(assert_all_called=True) as mock:
            mock.post(_SHIELD_URL).mock(side_effect=_shield_handler)
            r_safe_ses = await env.client.post(
                f"/v1/sessions/{env.session_id}/messages",
                headers=env.headers,
                json={"content": safe_prompt, "stream": False},
            )
            r_safe_oai = await env.client.post(
                "/v1/chat/completions",
                headers=env.headers,
                json={
                    "model": "gemma4-26b-a4b",
                    "messages": [{"role": "user", "content": safe_prompt}],
                    "stream": False,
                },
            )

        assert r_safe_ses.status_code == HTTPStatus.OK
        assert "X-LAI-Guardrail-Blocked" not in r_safe_ses.headers
        assert r_safe_ses.json()["message"]["content"] == "gemma4 safe reply"

        assert r_safe_oai.status_code == HTTPStatus.OK
        assert "X-LAI-Guardrail-Blocked" not in r_safe_oai.headers
        assert r_safe_oai.json()["choices"][0]["message"]["content"] == "gemma4 safe reply"

        assert env.inference.total_calls == _EXPECTED_TWO
        assert len(env.spooler.lease_requests) == _EXPECTED_TWO
        raw_audit, entries = _read_audit_entries(env.log_dir)

    for sensitive in (safe_lai, safe_oai, safe_email):
        assert sensitive not in raw_audit
    assert len(entries) == _EXPECTED_TWO * 2
    assert all(e["verdict"] == "allow" for e in entries)
    ingress_entries = [e for e in entries if e["direction"] == "ingress"]
    egress_entries = [e for e in entries if e["direction"] == "egress"]
    assert len(ingress_entries) == _EXPECTED_TWO
    assert len(egress_entries) == _EXPECTED_TWO
    assert all("[REDACTED_SECRET]" in str(e["scrubbed_prompt"]) for e in ingress_entries)
    assert all("[REDACTED_EMAIL]" in str(e["scrubbed_prompt"]) for e in ingress_entries)
    assert all(e["scrubbed_prompt"] == "gemma4 safe reply" for e in egress_entries)


async def test_shieldgemma_blocks_unsafe_model_output_on_egress(tmp_path: Path) -> None:
    """Buffered egress ShieldGemma blocks unsafe model replies before streaming or persisting."""

    class _UnsafeReplyInference(_CountingInferenceProvider):
        async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
            self.stream_requests.append(request)
            yield InferenceChunk(content_delta="UNSAFE_EXPLOSIVE ")
            yield InferenceChunk(
                content_delta="instructions",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=6, completion_tokens=4, total_tokens=10),
            )

    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=True),
        guardrails=GuardrailConfig(
            enabled=True,
            block_mode=GuardrailBlockMode.REFUSAL_MESSAGE,
            host="127.0.0.1",
            port=_SHIELD_PORT,
            refusal_text=_REFUSAL_TEXT,
        ),
    )
    fake_inf = _UnsafeReplyInference()
    async with open_services_harness(
        settings, IntegrationSpec(inference_provider=fake_inf)
    ) as harness:
        headers = await mint_auth_headers(harness.svc, "egress-admin", (ApiScope.ADMIN,))
        pid = await create_project(harness.client, headers, "egress-proj")
        sid = await create_session(harness.client, headers, pid)
        with respx.mock(assert_all_called=True) as mock:
            mock.post(_SHIELD_URL).mock(side_effect=_shield_handler)
            resp = await harness.client.post(
                f"/v1/sessions/{sid}/messages",
                headers=headers,
                json={"content": "Tell me about chemistry", "stream": False},
            )
        assert resp.status_code == HTTPStatus.OK
        body = resp.json()
        assert body["guardrail_blocked"] is True
        assert body["guardrail_policy"] == _UNSAFE_POLICY
        assert body["guardrail_direction"] == "egress"
        assert body["message"]["content"] == _REFUSAL_TEXT
        stored = await harness.svc.session_repo.list_messages(sid)
        assert all("UNSAFE_EXPLOSIVE" not in msg.content for msg in stored)
