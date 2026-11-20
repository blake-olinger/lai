"""Integration tests for session system_prompt blocks, egress headers, and guardrail analytics."""

from __future__ import annotations

import hashlib
from http import HTTPStatus as S
from typing import TYPE_CHECKING

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailDirection,
    GuardrailOutputMode,
    GuardrailVerdict,
)
from lai.domain.inference import BackendHealth, InferenceChunk, InferenceRequest, InferenceResponse
from lai.domain.observability import LoggingConfig
from lai.guardrails import CompositeGuardrailPipeline
from lai.ports.guardrails import GuardrailProviderPort
from lai.ports.inference import InferenceProviderPort
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    import httpx


class _DirectionalStubGuardrail(GuardrailProviderPort):
    """Deterministic ingress/egress guardrail flagging marker keywords."""

    @property
    def name(self) -> str:
        return "directional-stub"

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if "BAD_INGRESS" in prompt:
            return GuardrailDecision(
                verdict=GuardrailVerdict.BLOCK,
                provider=self.name,
                direction=GuardrailDirection.INGRESS,
                violated_policy="no_prompt_injection",
                violation_score=0.97,
                reason="Blocked ingress marker",
                prompt_sha256=sha,
                prompt_chars=len(prompt),
                scrubbed_prompt=prompt,
            )
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            direction=GuardrailDirection.INGRESS,
            violation_score=0.05,
            prompt_sha256=sha,
            prompt_chars=len(prompt),
            scrubbed_prompt=prompt,
        )

    async def evaluate_response(self, prompt: str, response: str) -> GuardrailDecision:
        del prompt
        sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
        if "BAD_EGRESS" in response:
            return GuardrailDecision(
                verdict=GuardrailVerdict.BLOCK,
                provider=self.name,
                direction=GuardrailDirection.EGRESS,
                violated_policy="no_dangerous_content",
                violation_score=0.94,
                reason="Blocked egress marker",
                prompt_sha256=sha,
                prompt_chars=len(response),
                scrubbed_prompt=response,
            )
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            direction=GuardrailDirection.EGRESS,
            violation_score=0.04,
            prompt_sha256=sha,
            prompt_chars=len(response),
            scrubbed_prompt=response,
        )


class _ScriptedEgressInference(InferenceProviderPort):
    """Echoes unsafe egress text when prompted with TRIGGER_EGRESS."""

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="safe")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        last = request.messages[-1].content if request.messages else ""
        text = "Here is BAD_EGRESS payload" if "TRIGGER_EGRESS" in last else "Completely safe reply"
        yield InferenceChunk(
            content_delta=text,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=8, completion_tokens=5, total_tokens=13),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


async def _assert_guardrail_analytics(client: httpx.AsyncClient, hdrs: dict[str, str]) -> None:
    """Verify GET /v1/system/analytics contains ingress/egress counts and top policies."""
    r_ana = await client.get("/v1/system/analytics?window_hours=24", headers=hdrs)
    assert r_ana.status_code == S.OK
    gr_summary = r_ana.json()["guardrails"]
    assert gr_summary["total_evaluations"] >= 5
    assert gr_summary["total_blocked"] >= 4
    by_dir = {item["direction"]: item for item in gr_summary["by_direction"]}
    assert by_dir["ingress"]["blocked"] >= 2
    assert by_dir["egress"]["blocked"] >= 2
    assert by_dir["egress"]["allowed"] >= 1
    policies = {p["policy"]: p["block_count"] for p in gr_summary["top_policies"]}
    assert policies["no_prompt_injection"] >= 2
    assert policies["no_dangerous_content"] >= 2


async def test_session_system_prompt_block_egress_headers_and_analytics_report(
    tmp_path: Path,
) -> None:
    """Verify POST /v1/sessions system_prompt blocking, egress headers, and guardrail analytics."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(
            enabled=True,
            block_mode=GuardrailBlockMode.REFUSAL_MESSAGE,
            output_mode=GuardrailOutputMode.BUFFERED,
            refusal_text="Policy refusal.",
        ),
    )
    stub = _DirectionalStubGuardrail()
    spec = IntegrationSpec(
        inference_provider=_ScriptedEgressInference(),
        use_hash_embedder=True,
    )
    async with open_services_harness(settings, spec) as h:
        assert isinstance(h.svc.guardrail_pipeline, CompositeGuardrailPipeline)
        h.svc.guardrail_pipeline.reconfigure(settings.guardrails, providers=[stub])
        hdrs = await mint_auth_headers(h.svc, "gr-admin", (ApiScope.ADMIN,))
        pid = await create_project(h.client, hdrs, "guardrail-proj")

        r_bad_ses = await h.client.post(
            "/v1/sessions",
            headers=hdrs,
            json={"project_id": pid, "system_prompt": "Please BAD_INGRESS bypass"},
        )
        assert r_bad_ses.status_code == S.UNPROCESSABLE_ENTITY
        assert r_bad_ses.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert r_bad_ses.headers["X-LAI-Guardrail-Policy"] == "no_prompt_injection"
        assert r_bad_ses.headers["X-LAI-Guardrail-Direction"] == "ingress"

        r_ok_ses = await h.client.post(
            "/v1/sessions",
            headers=hdrs,
            json={"project_id": pid, "system_prompt": "Be concise and helpful."},
        )
        assert r_ok_ses.status_code == S.CREATED
        sid = r_ok_ses.json()["id"]

        r_in_blk = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=hdrs,
            json={"content": "Try BAD_INGRESS now", "stream": False},
        )
        assert r_in_blk.status_code == S.OK
        assert r_in_blk.headers["X-LAI-Guardrail-Direction"] == "ingress"
        assert r_in_blk.json()["guardrail_direction"] == "ingress"

        r_eg_json = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=hdrs,
            json={"content": "TRIGGER_EGRESS please", "stream": False},
        )
        assert r_eg_json.status_code == S.OK
        assert r_eg_json.headers["X-LAI-Guardrail-Policy"] == "no_dangerous_content"
        assert r_eg_json.headers["X-LAI-Guardrail-Direction"] == "egress"
        assert "BAD_EGRESS" not in r_eg_json.text
        assert r_eg_json.json()["message"]["content"] == "Policy refusal."

        r_eg_sse = await h.client.post(
            "/v1/chat/completions",
            headers={**hdrs, "X-LAI-Project-Id": pid},
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": "TRIGGER_EGRESS stream"}],
                "stream": True,
            },
        )
        assert r_eg_sse.status_code == S.OK
        assert r_eg_sse.headers["X-LAI-Guardrail-Direction"] == "egress"
        assert "BAD_EGRESS" not in r_eg_sse.text
        assert "Policy refusal." in r_eg_sse.text

        r_safe = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=hdrs,
            json={"content": "Hello safe world", "stream": False},
        )
        assert r_safe.status_code == S.OK
        assert "X-LAI-Guardrail-Blocked" not in r_safe.headers
        assert r_safe.json()["message"]["content"] == "Completely safe reply"

        await _assert_guardrail_analytics(h.client, hdrs)
