"""Integration tests for sessions/files CRUD routes and FastAPI 422 validation handling."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from lai.api.app import create_app
from lai.api.errors import handle_request_validation_error
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.errors import NotFoundError
from lai.domain.files import FileStatus
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import BackendHealth, InferenceChunk, InferenceRequest, InferenceResponse
from lai.domain.observability import LoggingConfig
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.security import generate_api_key
from lai.tools import build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator

    from lai.api.deps import AppServices

_EXPECTED_THREE = 3
_EXPECTED_TWO = 2


class _EchoInferenceProvider(InferenceProviderPort):
    """Inference backend recording requests and echoing prompt summaries deterministically."""

    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(
            content=f"echo:{request.messages[-1].content}",
            usage=TokenUsage(prompt_tokens=6, completion_tokens=3, total_tokens=9),
        )

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.requests.append(request)
        yield InferenceChunk(content_delta=f"reply:{request.messages[-1].content}")
        yield InferenceChunk(
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@dataclass(slots=True)
class _Env:
    svc: AppServices
    client: httpx.AsyncClient
    inf: _EchoInferenceProvider
    embedder: DeterministicHashEmbedder
    vector_store: LanceDbVectorStore
    admin_hdrs: dict[str, str]
    write_only_hdrs: dict[str, str]
    project_id: str


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[_Env]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    svc = await build_app_services(settings)
    emb = DeterministicHashEmbedder()
    vec = LanceDbVectorStore(data_dir / "lancedb", dimensions=emb.dimensions)
    inf = _EchoInferenceProvider()
    svc.embedder, svc.vector_store, svc.inference_provider = emb, vec, inf
    svc.ingestion_pipeline = LocalIngestionPipeline(
        blob_store=svc.blob_store, file_repo=svc.file_repo, embedder=emb, vector_store=vec
    )
    svc.tool_registry = build_default_tool_registry(
        svc.file_repo, svc.blob_store, embedder=emb, vector_store=vec
    )
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
    admin = generate_api_key("routes-admin", (ApiScope.ADMIN,))
    writer = generate_api_key(
        "routes-no-chat-file-read", (ApiScope.PROJECTS_READ, ApiScope.TOOLS_SAFE)
    )
    await svc.api_key_repo.create(admin.record)
    await svc.api_key_repo.create(writer.record)
    admin_hdrs = {"Authorization": f"Bearer {admin.plaintext_key}"}
    write_only_hdrs = {"Authorization": f"Bearer {writer.plaintext_key}"}

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            pid = (
                await client.post("/v1/projects", headers=admin_hdrs, json={"name": "routes-p"})
            ).json()["id"]
            yield _Env(svc, client, inf, emb, vec, admin_hdrs, write_only_hdrs, pid)
    finally:
        await close_app_services(svc)


async def test_sessions_crud_pagination_scopes_and_negative_paths(env: _Env) -> None:
    """Verify full /v1/sessions CRUD, pagination, cascade delete, scope 403, and 404 paths."""
    cli, hdrs, pid = env.client, env.admin_hdrs, env.project_id

    r_create_404 = await cli.post("/v1/sessions", headers=hdrs, json={"project_id": "prj_missing"})
    assert r_create_404.status_code == HTTPStatus.NOT_FOUND
    assert r_create_404.json()["error"]["code"] == "not_found"

    sids: list[str] = []
    for idx in range(_EXPECTED_THREE):
        payload: dict[str, str] = {
            "project_id": pid,
            "title": f"S{idx}",
            "system_prompt": "Concise.",
        }
        if idx == 1:
            payload["model_profile"] = "gemma4-26b-a4b"
        res = await cli.post("/v1/sessions", headers=hdrs, json=payload)
        assert res.status_code == HTTPStatus.CREATED
        sids.append(str(res.json()["id"]))

    r_403 = await cli.get(f"/v1/sessions?project_id={pid}", headers=env.write_only_hdrs)
    assert r_403.status_code == HTTPStatus.FORBIDDEN
    r_404 = await cli.get("/v1/sessions?project_id=prj_missing", headers=hdrs)
    assert r_404.status_code == HTTPStatus.NOT_FOUND

    page1 = (await cli.get(f"/v1/sessions?project_id={pid}&limit=2&offset=0", headers=hdrs)).json()
    page2 = (await cli.get(f"/v1/sessions?project_id={pid}&limit=2&offset=2", headers=hdrs)).json()
    assert page1["count"] == _EXPECTED_TWO
    assert page2["count"] == 1
    assert {str(item["id"]) for item in page1["items"] + page2["items"]} == set(sids)

    r_post_404 = await cli.post(
        "/v1/sessions/ses_missing/messages", headers=hdrs, json={"content": "hi"}
    )
    assert r_post_404.status_code == HTTPStatus.NOT_FOUND
    assert (
        await cli.get("/v1/sessions/ses_missing", headers=hdrs)
    ).status_code == HTTPStatus.NOT_FOUND

    target_sid = sids[0]
    r_turn = await cli.post(
        f"/v1/sessions/{target_sid}/messages", headers=hdrs, json={"content": "Check status"}
    )
    assert r_turn.status_code == HTTPStatus.OK
    assert r_turn.json()["message"]["content"] == "reply:Check status"
    assert len(env.inf.requests) == 1

    r_det = await cli.get(f"/v1/sessions/{target_sid}", headers=hdrs)
    assert r_det.status_code == HTTPStatus.OK
    det = r_det.json()
    assert det["session"]["id"] == target_sid
    assert det["session"]["system_prompt"] == "Concise."
    assert [(m["role"], m["content"]) for m in det["messages"]] == [
        ("user", "Check status"),
        ("assistant", "reply:Check status"),
    ]

    r_del = await cli.delete(f"/v1/sessions/{target_sid}", headers=hdrs)
    assert r_del.status_code == HTTPStatus.OK
    assert r_del.json() == {"id": target_sid, "deleted": True}
    assert await env.svc.session_repo.list_messages(target_sid) == []
    r_gone = await cli.get(f"/v1/sessions/{target_sid}", headers=hdrs)
    assert r_gone.status_code == HTTPStatus.NOT_FOUND
    r_del2 = await cli.delete(f"/v1/sessions/{target_sid}", headers=hdrs)
    assert r_del2.status_code == HTTPStatus.NOT_FOUND


async def test_files_async_stream_pagination_chunks_delete_and_negative_paths(env: _Env) -> None:
    """Verify /v1/files listing, wait=false stream ingestion, chunks, delete, and 403/404/422."""
    cli, hdrs, pid = env.client, env.admin_hdrs, env.project_id

    r_403 = await cli.get(f"/v1/files?project_id={pid}", headers=env.write_only_hdrs)
    assert r_403.status_code == HTTPStatus.FORBIDDEN
    for method, url, extra in (
        ("GET", "/v1/files?project_id=prj_missing", {}),
        ("POST", "/v1/files/stream?project_id=prj_missing&wait=false", {"X-LAI-Filename": "m.py"}),
        ("GET", "/v1/files/fil_missing", {}),
        ("DELETE", "/v1/files/fil_missing", {}),
        ("GET", "/v1/files/fil_missing/chunks", {}),
    ):
        r_nf = await cli.request(method, url, headers={**hdrs, **extra})
        assert r_nf.status_code == HTTPStatus.NOT_FOUND

    r_blank_hdr = await cli.post(
        f"/v1/files/stream?project_id={pid}&wait=false",
        headers={**hdrs, "X-LAI-Filename": "   "},
        content=b"x=1\n",
    )
    assert r_blank_hdr.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert r_blank_hdr.json()["error"]["details"]["field"] == "X-LAI-Filename"

    for bad_name in ("", "   "):
        r_bad_mp = await cli.post(
            "/v1/files",
            headers=hdrs,
            data={"project_id": pid},
            files={"file": (bad_name, b"hello", "text/plain")},
        )
        assert r_bad_mp.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
        assert r_bad_mp.json()["error"]["code"] == "validation_failed"

    r_async = await cli.post(
        f"/v1/files/stream?project_id={pid}&wait=false",
        headers={**hdrs, "X-LAI-Filename": "worker.py", "Content-Type": "text/x-python"},
        content=b"def compute_checksum() -> int:\n    return 42\n",
    )
    assert r_async.status_code == HTTPStatus.ACCEPTED
    rec1 = r_async.json()
    assert rec1["status"] == FileStatus.PENDING.value
    fid1 = str(rec1["id"])

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and rec1["status"] != FileStatus.READY.value:
        await asyncio.sleep(0.02)
        rec1 = (await cli.get(f"/v1/files/{fid1}", headers=hdrs)).json()
    assert rec1["status"] == FileStatus.READY.value
    assert (await cli.get(f"/v1/files/{fid1}/chunks", headers=hdrs)).json()["count"] >= 1
    assert await env.vector_store.search_vectors(pid, await env.embedder.embed_query("checksum"))

    r_up2 = await cli.post(
        "/v1/files?wait=true",
        headers=hdrs,
        data={"project_id": pid},
        files={"file": ("notes.log", b"2026-11-17T00:00:00Z INFO booted\n", "text/plain")},
    )
    assert r_up2.status_code == HTTPStatus.CREATED
    fid2 = str(r_up2.json()["id"])

    p1 = (await cli.get(f"/v1/files?project_id={pid}&limit=1&offset=0", headers=hdrs)).json()
    p2 = (await cli.get(f"/v1/files?project_id={pid}&limit=1&offset=1", headers=hdrs)).json()
    assert (p1["count"], p2["count"]) == (1, 1)
    assert {str(p1["items"][0]["id"]), str(p2["items"][0]["id"])} == {fid1, fid2}

    r_bad_rng = await cli.get(f"/v1/files/{fid1}/chunks?start_line=10&end_line=5", headers=hdrs)
    assert r_bad_rng.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert r_bad_rng.json()["error"]["details"]["field"] == "end_line"

    r_del1 = await cli.delete(f"/v1/files/{fid1}", headers=hdrs)
    assert r_del1.json() == {"id": fid1, "deleted": True}
    assert (await cli.get(f"/v1/files/{fid1}", headers=hdrs)).status_code == HTTPStatus.NOT_FOUND
    with pytest.raises(NotFoundError):
        await env.svc.blob_store.read_bytes(Path(rec1["blob_path"]))
    vec_after = await env.vector_store.search_vectors(
        pid, await env.embedder.embed_query("checksum")
    )
    assert {h.chunk.file_id for h in vec_after} == {fid2}
    env.svc.vector_store = None
    assert (await cli.delete(f"/v1/files/{fid2}", headers=hdrs)).json()["deleted"] is True


def _assert_422(resp: httpx.Response, req_id: str, expected_loc: str) -> None:
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert resp.headers["X-Request-Id"] == req_id
    err = resp.json()["error"]
    assert (err["code"], err["request_id"], err["details"]["first_location"]) == (
        "validation_failed",
        req_id,
        expected_loc,
    )
    assert int(err["details"]["error_count"]) >= 1


async def test_fastapi_request_validation_error_envelopes_and_request_id_parity(env: _Env) -> None:
    """Verify handle_request_validation_error across JSON body, query, multipart, and edge cases."""
    cli, hdrs = env.client, env.admin_hdrs
    r1 = await cli.post("/v1/projects", headers={**hdrs, "X-Request-Id": "r_b1"}, json={"name": ""})
    r2 = await cli.post("/v1/projects", headers={**hdrs, "X-Request-Id": "r_b2"}, json={})
    r3 = await cli.get(
        f"/v1/sessions?project_id={env.project_id}&limit=0",
        headers={**hdrs, "X-Request-Id": "r_q"},
    )
    r4 = await cli.post(
        "/v1/files",
        headers={**hdrs, "X-Request-Id": "r_mp"},
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    _assert_422(r1, "r_b1", "body.name")
    _assert_422(r2, "r_b2", "body.name")
    _assert_422(r3, "r_q", "query.limit")
    _assert_422(r4, "r_mp", "body.project_id")

    raw_req = Request(
        {"type": "http", "method": "POST", "path": "/x", "headers": [], "state": {"request_id": ""}}
    )
    r_empty = await handle_request_validation_error(raw_req, RequestValidationError([]))
    assert r_empty.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "X-Request-Id" not in r_empty.headers
    assert json.loads(bytes(r_empty.body))["error"]["details"] == {"error_count": "0"}

    r_no_loc = await handle_request_validation_error(
        raw_req, RequestValidationError([{"type": "value_error", "loc": (), "msg": "bad"}])
    )
    assert json.loads(bytes(r_no_loc.body))["error"]["details"] == {"error_count": "1"}
