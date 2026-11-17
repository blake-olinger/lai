"""Tests for SSE/JSON keepalives, validation edge cases, runtime wiring, and YAML configs."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import yaml

from lai.api.app import create_app
from lai.config.settings import LaiSettings, load_settings
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import (
    ContentDeltaEvent,
    Message,
    MessageRole,
    OrchestratorEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnInput,
    TurnOutput,
)
from lai.ports.orchestrator import OrchestratorPort
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

HTTP_OK = 200
HTTP_CREATED = 201
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_UNPROCESSABLE = 422
DEFAULT_API_PORT = 8400
DEFAULT_LLAMA_PORT = 8081
DEFAULT_SHIELD_PORT = 8082
REPO_ROOT = Path(__file__).resolve().parents[2]


class _SlowOrchestrator(OrchestratorPort):
    """Orchestrator stub with configurable delay for keepalive streaming tests."""

    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    def _output(self, turn_input: TurnInput) -> TurnOutput:
        msg = Message(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=MessageRole.ASSISTANT,
            content="slow reply",
        )
        return TurnOutput(
            session_id=turn_input.session_id,
            message=msg,
            context_mode_used=turn_input.context_mode,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def run_turn(self, turn_input: TurnInput, principal: AuthPrincipal) -> TurnOutput:
        del principal
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        return self._output(turn_input)

    async def run_turn_stream(
        self, turn_input: TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[OrchestratorEvent]:
        del principal
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        out = self._output(turn_input)
        yield ContentDeltaEvent(delta=out.message.content)
        yield TurnCompletedEvent(output=out)


@pytest.fixture(autouse=True)
def _cleanup_root_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_sse_json_keepalives_and_validation_envelopes(tmp_path: Path) -> None:
    """Verify SSE : keepalive comments, RFC 8259 JSON whitespace keepalives, and 404/405/422."""
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        keepalive_interval_seconds=0.04,
        json_keepalive_threshold_seconds=0.04,
    )
    orch = _SlowOrchestrator(delay_seconds=0.14)
    app = create_app(settings, orchestrator=orch)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        created = generate_api_key("keepalive-admin", (ApiScope.ADMIN,))
        await app.state.services.api_key_repo.create(created.record)
        headers = {"Authorization": f"Bearer {created.plaintext_key}"}

        pid = (await client.post("/v1/projects", headers=headers, json={"name": "slow"})).json()[
            "id"
        ]
        sid = (await client.post("/v1/sessions", headers=headers, json={"project_id": pid})).json()[
            "id"
        ]

        # 1. Slow JSON session turn emits RFC 8259 whitespace keepalive before JSON
        r_json = await client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "slow json", "stream": False},
        )
        assert r_json.status_code == HTTP_OK
        assert r_json.text.startswith("\n")
        assert json.loads(r_json.text)["message"]["content"] == "slow reply"

        # 2. Slow SSE session turn emits keepalive events
        r_ses_sse = await client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "slow sse", "stream": True},
        )
        assert "event: keepalive" in r_ses_sse.text
        assert "event: turn_completed" in r_ses_sse.text

        # 3. Slow OpenAI SSE turn emits ': keepalive' comments
        r_oai_sse = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": "slow oai sse"}],
                "stream": True,
            },
        )
        assert ": keepalive\n\n" in r_oai_sse.text
        assert "data: [DONE]\n\n" in r_oai_sse.text

        # 4. Slow OpenAI JSON turn uses whitespace keepalive stream
        r_oai_json = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": "slow oai json"}],
                "stream": False,
            },
        )
        assert r_oai_json.text.startswith("\n")
        assert json.loads(r_oai_json.text)["choices"][0]["message"]["content"] == "slow reply"

        # 5. Edge validation and HTTP error envelopes
        orch.delay_seconds = 0.0
        assert (
            await client.post(
                "/v1/files/stream",
                headers={**headers, "X-LAI-Filename": "a.log"},
                content=b"x",
            )
        ).status_code == HTTP_UNPROCESSABLE
        assert (
            await client.post(
                "/v1/chat/completions",
                headers={**headers, "X-LAI-Context-Mode": "bogus"},
                json={"model": "gemma4-26b-a4b", "messages": [{"role": "user", "content": "hi"}]},
            )
        ).status_code == HTTP_UNPROCESSABLE
        assert (await client.get("/v1/projects/prj_nope", headers=headers)).status_code == (
            HTTP_NOT_FOUND
        )
        assert (await client.delete("/v1/health")).status_code == HTTP_METHOD_NOT_ALLOWED


async def test_runtime_with_oidc_and_cloudflare_edge_verification(tmp_path: Path) -> None:
    """Verify runtime.build_app_services wires OIDC and Cloudflare edge guard cleanly."""
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        auth_providers=("api_key", "oidc"),
        oidc_enabled=True,
        oidc_audience="test-aud",
        cf_access_enabled=True,
        cf_access_client_id="cf-id",
        cf_access_client_secret="cf-secret",  # noqa: S106
    )
    app = create_app(settings, orchestrator=_SlowOrchestrator())

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        created = generate_api_key("runtime-test", (ApiScope.ADMIN,))
        await app.state.services.api_key_repo.create(created.record)
        res = await client.post(
            "/v1/projects",
            headers={
                "Authorization": f"Bearer {created.plaintext_key}",
                "CF-Access-Client-Id": "cf-id",
                "CF-Access-Client-Secret": "cf-secret",
            },
            json={"name": "runtime-proj"},
        )
        assert res.status_code == HTTP_CREATED


def test_example_configs_and_github_workflows_are_valid_yaml() -> None:
    """Validate config.example.yaml, cloudflared template, and all GitHub workflow YAMLs."""
    loaded = load_settings(REPO_ROOT / "config" / "config.example.yaml")
    assert loaded.port == DEFAULT_API_PORT
    assert loaded.llama_port == DEFAULT_LLAMA_PORT
    assert loaded.guardrails.port == DEFAULT_SHIELD_PORT

    tunnel_tpl = REPO_ROOT / "deploy" / "cloudflared" / "config.template.yml"
    parsed_tunnel = yaml.safe_load(tunnel_tpl.read_text(encoding="utf-8"))
    assert isinstance(parsed_tunnel, dict)
    assert "ingress" in parsed_tunnel

    for wf in (
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "security.yml",
        REPO_ROOT / ".github" / "workflows" / "release.yml",
        REPO_ROOT / ".github" / "dependabot.yml",
    ):
        parsed = yaml.safe_load(wf.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"{wf} must parse to a mapping"
