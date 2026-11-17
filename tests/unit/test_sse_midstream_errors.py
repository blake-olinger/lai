"""Tests verifying mid-stream LaiError framing in orchestrator and OpenAI SSE responses."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx
import pytest

from lai.api._streaming import guard_openai_sse_frames, guard_orchestrator_sse_frames
from lai.api.app import create_app
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.chat import (
    Message,
    MessageRole,
    OrchestratorEvent,
    ProgressEvent,
    TokenUsage,
    TurnInput,
    TurnOutput,
)
from lai.domain.errors import InferenceBackendError
from lai.ports.orchestrator import OrchestratorPort
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator
    from pathlib import Path

HTTP_OK = 200


class _MidStreamFailingOrchestrator(OrchestratorPort):
    """Orchestrator stub that emits progress then fails mid-stream after headers flush."""

    async def run_turn(self, turn_input: TurnInput, principal: AuthPrincipal) -> TurnOutput:
        del principal
        msg = Message(
            session_id=turn_input.session_id,
            project_id=turn_input.project_id,
            role=MessageRole.ASSISTANT,
            content="unused",
        )
        return TurnOutput(
            session_id=turn_input.session_id,
            message=msg,
            context_mode_used=turn_input.context_mode,
            usage=TokenUsage(),
        )

    async def run_turn_stream(
        self, turn_input: TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[OrchestratorEvent]:
        del turn_input, principal
        yield ProgressEvent(stage="inference", message="starting")
        msg = "boom"
        raise InferenceBackendError(msg, details={"backend": "ollama"})


@pytest.fixture(autouse=True)
def _cleanup_root_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_guard_sse_frame_helpers_directly() -> None:
    """Verify helper generators serialize mid-stream LaiError cleanly after normal frames."""

    async def _orch_frames() -> AsyncIterator[str]:
        yield 'event: progress\ndata: {"stage":"start"}\n\n'
        msg = "boom"
        raise InferenceBackendError(msg, details={"unit": "1"})

    orch_out = [f async for f in guard_orchestrator_sse_frames(_orch_frames())]
    assert len(orch_out) == 2  # noqa: PLR2004
    assert orch_out[1].startswith("event: error\ndata: ")
    err_payload = json.loads(orch_out[1].split("data: ", 1)[1].strip())
    assert err_payload["error"]["code"] == "inference_backend_error"
    assert err_payload["error"]["message"] == "boom"

    async def _openai_frames() -> AsyncIterator[str]:
        yield ": keepalive\n\n"
        msg = "boom"
        raise InferenceBackendError(msg)

    oai_out = [f async for f in guard_openai_sse_frames(_openai_frames())]
    assert len(oai_out) == 2  # noqa: PLR2004
    assert oai_out[1].endswith("data: [DONE]\n\n")
    oai_err = json.loads(oai_out[1].splitlines()[0].removeprefix("data: "))
    assert oai_err == {"error": {"message": "boom", "type": "inference_backend_error"}}


async def test_asgi_routes_serialize_midstream_lai_errors(tmp_path: Path) -> None:
    """Verify session and OpenAI SSE routes emit structured error frames without RuntimeError."""
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        keepalive_interval_seconds=0.5,
    )
    app = create_app(settings, orchestrator=_MidStreamFailingOrchestrator())

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        created = generate_api_key("sse-err-admin", (ApiScope.ADMIN,))
        await app.state.services.api_key_repo.create(created.record)
        headers = {"Authorization": f"Bearer {created.plaintext_key}"}

        pid = (await client.post("/v1/projects", headers=headers, json={"name": "err"})).json()[
            "id"
        ]
        sid = (await client.post("/v1/sessions", headers=headers, json={"project_id": pid})).json()[
            "id"
        ]

        async with client.stream(
            "POST",
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "hi", "stream": True},
        ) as ses_resp:
            assert ses_resp.status_code == HTTP_OK
            ses_body = (await ses_resp.aread()).decode("utf-8")
        assert "event: progress" in ses_body
        assert "event: error" in ses_body
        assert '"code":"inference_backend_error"' in ses_body

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "gemma4-26b-a4b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as oai_resp:
            assert oai_resp.status_code == HTTP_OK
            oai_body = (await oai_resp.aread()).decode("utf-8")
        assert '"type": "inference_backend_error"' in oai_body
        assert "data: [DONE]" in oai_body
