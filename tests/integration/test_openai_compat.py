"""End-to-end OpenAI /v1/models and /v1/chat/completions RETRIEVAL & MAP_REDUCE tests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx
import pytest

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    ModelProfile,
)
from lai.domain.observability import LoggingConfig
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.security import generate_api_key
from lai.tools import build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator
    from pathlib import Path

_MIN_MAP_CALLS = 2


class _OpenAiTestInferenceProvider(InferenceProviderPort):
    """Fake inference provider verifying RETRIEVAL vs MAP_REDUCE prompt inputs."""

    def __init__(self) -> None:
        self.map_requests: list[InferenceRequest] = []
        self.stream_requests: list[InferenceRequest] = []

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        self.map_requests.append(request)
        return InferenceResponse(
            content="Window error summary: TimeoutError on primary-db.",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        )

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.stream_requests.append(request)
        joined = "\n".join(msg.content for msg in request.messages)
        if "Window error summary:" in joined:
            reply = "Collated map-reduce summary: TimeoutError occurred across app.log."
        else:
            reply = "Retrieved context confirms TimeoutError in app.log."
        yield InferenceChunk(thinking_delta="Inspecting context.")
        yield InferenceChunk(content_delta=reply)
        yield InferenceChunk(
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@dataclass(slots=True)
class _OpenAiEnv:
    client: httpx.AsyncClient
    inf: _OpenAiTestInferenceProvider
    headers: dict[str, str]
    project_id: str
    log_file_id: str
    other_file_id: str


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.fixture
async def oai_env(tmp_path: Path) -> AsyncIterator[_OpenAiEnv]:
    data_dir = tmp_path / "data"
    custom_profile = ModelProfile(
        name="custom-local-test",
        hf_repo="example/custom",
        hf_filename="custom.gguf",
    )
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        direct_file_budget_tokens=256,
        map_window_tokens=256,
        map_window_overlap_tokens=16,
        custom_profiles={"custom-local-test": custom_profile},
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    svc = await build_app_services(settings)
    emb = DeterministicHashEmbedder()
    vec = LanceDbVectorStore(data_dir / "lancedb", dimensions=emb.dimensions)
    svc.embedder = emb
    svc.vector_store = vec
    svc.ingestion_pipeline = LocalIngestionPipeline(
        blob_store=svc.blob_store,
        file_repo=svc.file_repo,
        embedder=emb,
        vector_store=vec,
    )
    svc.tool_registry = build_default_tool_registry(
        svc.file_repo,
        svc.blob_store,
        embedder=emb,
        vector_store=vec,
    )
    inf = _OpenAiTestInferenceProvider()
    svc.inference_provider = inf
    svc.orchestrator = TurnOrchestrator(
        settings=settings,
        session_repo=svc.session_repo,
        file_repo=svc.file_repo,
        blob_store=svc.blob_store,
        inference_provider=inf,
        spooler=svc.spooler,
        session_gate=svc.session_gate,
        guardrail_pipeline=svc.guardrail_pipeline,
        tool_registry=svc.tool_registry,
        tool_executor=svc.tool_executor,
        embedder=emb,
        vector_store=vec,
    )

    api_key = generate_api_key("oai-user", (ApiScope.ADMIN,))
    await svc.api_key_repo.create(api_key.record)
    headers = {"Authorization": f"Bearer {api_key.plaintext_key}"}

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            pid = (
                await client.post("/v1/projects", headers=headers, json={"name": "oai-proj"})
            ).json()["id"]

            log_lines = [
                f"2026-11-17T10:00:{i % 60:02d}Z ERROR db_timeout connection failed step={i}"
                for i in range(90)
            ]
            log_rec = (
                await client.post(
                    "/v1/files?wait=true",
                    headers=headers,
                    data={"project_id": pid},
                    files={"file": ("app.log", "\n".join(log_lines).encode("utf-8"), "text/plain")},
                )
            ).json()
            other_rec = (
                await client.post(
                    "/v1/files?wait=true",
                    headers=headers,
                    data={"project_id": pid},
                    files={
                        "file": (
                            "unrelated.md",
                            b"# SecretUnrelatedMarker\nUnattached notes.\n",
                            "text/markdown",
                        )
                    },
                )
            ).json()

            yield _OpenAiEnv(
                client=client,
                inf=inf,
                headers=headers,
                project_id=pid,
                log_file_id=log_rec["id"],
                other_file_id=other_rec["id"],
            )
    finally:
        await close_app_services(svc)


async def test_openai_models_and_transparent_retrieval_and_map_reduce(
    oai_env: _OpenAiEnv,
) -> None:
    """Verify /v1/models and /v1/chat/completions JSON + SSE across RETRIEVAL and MAP_REDUCE."""
    client = oai_env.client
    inf = oai_env.inf

    # 1. /v1/models lists built-in + custom profile
    r_models = await client.get("/v1/models", headers=oai_env.headers)
    assert r_models.status_code == HTTPStatus.OK
    model_ids = [item["id"] for item in r_models.json()["data"]]
    assert "gemma4-26b-a4b" in model_ids
    assert "custom-local-test" in model_ids

    scoped_headers = {
        **oai_env.headers,
        "X-LAI-Project-Id": oai_env.project_id,
        "X-LAI-File-Ids": oai_env.log_file_id,
    }

    # 2. Specific query auto-routes to RETRIEVAL scoped strictly to X-LAI-File-Ids
    r_ret = await client.post(
        "/v1/chat/completions",
        headers=scoped_headers,
        json={
            "model": "gemma4-26b-a4b",
            "messages": [{"role": "user", "content": "Where did db_timeout occur in app.log?"}],
            "stream": False,
        },
    )
    assert r_ret.status_code == HTTPStatus.OK
    ret_body = r_ret.json()
    assert ret_body["object"] == "chat.completion"
    assert "Retrieved context confirms TimeoutError" in ret_body["choices"][0]["message"]["content"]
    assert ret_body["choices"][0]["message"]["reasoning_content"] == "Inspecting context."
    assert ret_body["usage"]["total_tokens"] > 0
    assert len(inf.map_requests) == 0
    prompt_text = "\n".join(m.content for m in inf.stream_requests[-1].messages)
    assert "app.log" in prompt_text
    assert "SecretUnrelatedMarker" not in prompt_text

    # 3. Holistic query auto-routes to MAP_REDUCE (JSON stream=False)
    r_mr_json = await client.post(
        "/v1/chat/completions",
        headers=scoped_headers,
        json={
            "model": "gemma4-26b-a4b",
            "messages": [{"role": "user", "content": "Summarize all errors across the entire log"}],
            "stream": False,
        },
    )
    assert r_mr_json.status_code == HTTPStatus.OK
    mr_body = r_mr_json.json()
    assert "Collated map-reduce summary" in mr_body["choices"][0]["message"]["content"]
    assert len(inf.map_requests) >= _MIN_MAP_CALLS
    assert mr_body["usage"]["total_tokens"] > ret_body["usage"]["total_tokens"]

    # 4. Holistic query with stream=True emits OpenAI SSE chunks, usage, and [DONE]
    prev_map_count = len(inf.map_requests)
    r_mr_sse = await client.post(
        "/v1/chat/completions",
        headers=scoped_headers,
        json={
            "model": "gemma4-26b-a4b",
            "messages": [{"role": "user", "content": "Give a full summary of all errors"}],
            "stream": True,
        },
    )
    assert r_mr_sse.status_code == HTTPStatus.OK
    assert len(inf.map_requests) - prev_map_count >= _MIN_MAP_CALLS
    assert ": keepalive\n\n" in r_mr_sse.text
    assert "data: [DONE]\n\n" in r_mr_sse.text

    chunk_frames = [
        json.loads(line.removeprefix("data: "))
        for line in r_mr_sse.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert any(
        frame["object"] == "chat.completion.chunk"
        and "Collated map-reduce summary" in (frame["choices"][0]["delta"].get("content") or "")
        for frame in chunk_frames
    )
    assert any(
        frame["choices"][0]["finish_reason"] == "stop" and frame["usage"] is not None
        for frame in chunk_frames
    )

    # 5. body.file_ids, body.context_mode, and max_completion_tokens override X-LAI-* headers
    prev_stream_count = len(inf.stream_requests)
    r_override = await client.post(
        "/v1/chat/completions",
        headers={
            **oai_env.headers,
            "X-LAI-Project-Id": oai_env.project_id,
            "X-LAI-File-Ids": oai_env.log_file_id,
            "X-LAI-Context-Mode": "map_reduce",
        },
        json={
            "model": "gemma4-26b-a4b",
            "messages": [{"role": "user", "content": "Read unrelated.md directly"}],
            "file_ids": [oai_env.other_file_id],
            "context_mode": "direct",
            "max_completion_tokens": 77,
            "stream": False,
        },
    )
    assert r_override.status_code == HTTPStatus.OK
    assert len(inf.stream_requests) == prev_stream_count + 1
    override_req = inf.stream_requests[-1]
    assert override_req.max_tokens == 77  # noqa: PLR2004
    override_text = "\n".join(m.content for m in override_req.messages)
    assert "SecretUnrelatedMarker" in override_text
    assert "db_timeout" not in override_text
