"""Unit tests for FastAPI health, system, projects, files, sessions, and OpenAI routes."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from http import HTTPStatus as S
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain import chat as dc
from lai.domain import errors as errs
from lai.domain import tools as dt
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.files import FileStatus
from lai.domain.observability import LoggingConfig
from lai.inference import GpuTelemetry
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.ports.tools import ToolPluginPort
from lai.security import generate_api_key
from tests._helpers import StubOrchestratorBase, bearer_headers

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from lai.api.deps import AppServices

EXPECTED_CHUNK_COUNT = 1
_CHAT_JSON: dict[str, object] = {
    "model": "gemma4-26b-a4b",
    "messages": [{"role": "user", "content": "hi"}],
}


class GatedReadTool(ToolPluginPort):
    """Guarded read tool plugin used to exercise tool approval routes."""

    @property
    def spec(self) -> dt.ToolSpec:
        return dt.ToolSpec(
            name="read_note",
            description="Guarded read note tool",
            permission_tier=dt.PermissionTier.READ,
            requires_approval_by_default=True,
        )

    async def execute(
        self, arguments: dict[str, Any], context: dt.ToolExecutionContext
    ) -> dt.ToolResult:
        del arguments, context
        return dt.ToolResult(tool_name="read_note", success=True, output="saved")


class FakeOrch(StubOrchestratorBase):
    """Configurable OrchestratorPort double created upfront per test scenario."""

    def __init__(
        self,
        *,
        next_error: Exception | None = None,
        blocked: bool = False,
        policy: str | None = None,
    ) -> None:
        self.calls: list[dc.TurnInput] = []
        self.next_error = next_error
        self.blocked = blocked
        self.policy = policy

    def _prepare(self, inp: dc.TurnInput) -> dc.TurnOutput:
        self.calls.append(inp)
        if self.next_error is not None:
            raise self.next_error
        msg = dc.Message(
            session_id=inp.session_id,
            project_id=inp.project_id,
            role=dc.MessageRole.ASSISTANT,
            content="refused" if self.blocked else "ok reply",
        )
        return dc.TurnOutput(
            session_id=inp.session_id,
            message=msg,
            context_mode_used=inp.context_mode,
            usage=dc.TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            guardrail_blocked=self.blocked,
            guardrail_policy=self.policy,
        )

    async def run_turn(self, turn_input: dc.TurnInput, principal: AuthPrincipal) -> dc.TurnOutput:
        del principal
        return self._prepare(turn_input)

    async def run_turn_stream(
        self, turn_input: dc.TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[dc.OrchestratorEvent]:
        del principal
        out = self._prepare(turn_input)
        yield dc.ContentDeltaEvent(delta=out.message.content)
        yield dc.TurnCompletedEvent(output=out)


@dataclass(slots=True)
class RoutesEnv:
    """Isolated unit-route ASGI environment with prebuilt tokens and services."""

    svc: AppServices
    orch: FakeOrch
    client: httpx.AsyncClient
    toks: dict[str, str]


@asynccontextmanager
async def open_routes_env(
    tmp_path: Path,
    *,
    orch: FakeOrch | None = None,
    rate_limit_rpm: int = 120,
    raise_app_exceptions: bool = True,
) -> AsyncGenerator[RoutesEnv]:
    """Build isolated unit-route AppServices + ASGI client upfront without mid-test mutation."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        rate_limit_rpm=rate_limit_rpm,
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=True),
    )
    resolved_orch = orch or FakeOrch()
    base_svc = await build_app_services(settings, orchestrator=resolved_orch)
    emb = DeterministicHashEmbedder()
    vec = LanceDbVectorStore(data_dir / "lancedb", dimensions=emb.dimensions)
    svc = replace(
        base_svc,
        embedder=emb,
        vector_store=vec,
        ingestion_pipeline=LocalIngestionPipeline(
            blob_store=base_svc.blob_store,
            file_repo=base_svc.file_repo,
            embedder=emb,
            vector_store=vec,
        ),
        gpu_telemetry_fn=lambda: GpuTelemetry(
            gpu_name="RTX 3090",
            memory_used_mb=1024,
            memory_total_mb=24576,
            utilization_percent=5,
        ),
    )
    svc.tool_registry.register(GatedReadTool())

    tokens: dict[str, str] = {}
    writer_scopes = (
        ApiScope.PROJECTS_WRITE,
        ApiScope.FILES_WRITE,
        ApiScope.CHAT_WRITE,
        ApiScope.TOOLS_SAFE,
    )
    for role, scopes in (
        ("admin", (ApiScope.ADMIN,)),
        ("reader", (ApiScope.PROJECTS_READ, ApiScope.CHAT_READ)),
        ("writer", writer_scopes),
    ):
        created = generate_api_key(role, scopes)
        await svc.api_key_repo.create(created.record)
        tokens[role] = created.plaintext_key

    app = create_app(settings, services=svc)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield RoutesEnv(svc=svc, orch=resolved_orch, client=client, toks=tokens)
    finally:
        await close_app_services(svc)


async def _mk_proj(env: RoutesEnv, hdrs: dict[str, str], name: str) -> str:
    r = await env.client.post("/v1/projects", headers=hdrs, json={"name": name})
    return str(r.json()["id"])


async def _mk_ses(env: RoutesEnv, hdrs: dict[str, str]) -> str:
    pid = await _mk_proj(env, hdrs, "c")
    r = await env.client.post("/v1/sessions", headers=hdrs, json={"project_id": pid})
    return str(r.json()["id"])


async def test_health_status_and_alerts_authorization(tmp_path: Path) -> None:
    """Verify /v1/health request ID and ADMIN scope on /v1/system/status and /v1/system/alerts."""
    async with open_routes_env(tmp_path) as env:
        r_hp = await env.client.get("/v1/health", headers={"X-Request-Id": "req_1"})
        assert (r_hp.status_code, r_hp.headers["X-Request-Id"]) == (S.OK, "req_1")
        assert (await env.client.get("/v1/system/status")).status_code == S.UNAUTHORIZED
        r_forbid = await env.client.get(
            "/v1/system/status", headers=bearer_headers(env.toks["reader"])
        )
        assert r_forbid.status_code == S.FORBIDDEN

        with respx.mock(assert_all_called=True) as mock:
            mock.get("http://127.0.0.1:8081/health").respond(S.OK, json={"status": "ok"})
            mock.get("http://127.0.0.1:8082/health").respond(S.OK, json={"status": "ok"})
            r_st = await env.client.get(
                "/v1/system/status", headers=bearer_headers(env.toks["admin"])
            )
        assert r_st.status_code == S.OK
        assert r_st.json()["gpu"]["gpu_name"] == "RTX 3090"
        assert r_st.json()["guardrail_health"]["healthy"] is True

        r_al = await env.client.get("/v1/system/alerts", headers=bearer_headers(env.toks["reader"]))
        r_ok = await env.client.get("/v1/system/alerts", headers=bearer_headers(env.toks["admin"]))
        assert (r_al.status_code, r_ok.json()["count"]) == (S.FORBIDDEN, 0)


async def test_projects_crud_and_not_found_envelopes(tmp_path: Path) -> None:
    """Verify project creation, listing, retrieval, and 404 envelopes."""
    async with open_routes_env(tmp_path) as env:
        hdrs = bearer_headers(env.toks["writer"])
        pid = await _mk_proj(env, hdrs, "logs")
        assert (await env.client.get("/v1/projects", headers=hdrs)).json()["count"] == 1
        assert (await env.client.get(f"/v1/projects/{pid}", headers=hdrs)).json()["id"] == pid
        r_get_404 = await env.client.get("/v1/projects/prj_nope", headers=hdrs)
        r_del_404 = await env.client.delete("/v1/projects/prj_nope", headers=hdrs)
        assert (r_get_404.status_code, r_del_404.status_code) == (S.NOT_FOUND, S.NOT_FOUND)


async def test_files_multipart_stream_async_poll_and_cascade_delete(tmp_path: Path) -> None:
    """Verify multipart/stream file upload, ?wait=false polling, and blob cascade delete."""
    async with open_routes_env(tmp_path) as env:
        hdrs = bearer_headers(env.toks["writer"])
        pid = await _mk_proj(env, hdrs, "logs")

        r_up = await env.client.post(
            "/v1/files?wait=true",
            headers=hdrs,
            data={"project_id": pid},
            files={"file": ("app.log", b"2026-01-01T00:00:00Z ERROR Boom\n", "text/plain")},
        )
        f_rec = r_up.json()
        fid = f_rec["id"]
        chk = (await env.client.get(f"/v1/files/{fid}/chunks", headers=hdrs)).json()
        assert (r_up.status_code, f_rec["status"]) == (S.CREATED, FileStatus.READY.value)
        assert chk["count"] >= EXPECTED_CHUNK_COUNT

        st_h = {**hdrs, "X-LAI-Filename": "worker.py", "Content-Type": "text/x-python"}
        r_str = await env.client.post(
            f"/v1/files/stream?project_id={pid}&wait=true",
            headers=st_h,
            content=b"def run() -> int:\n    return 42\n",
        )
        assert r_str.status_code == S.CREATED

        r_async = await env.client.post(
            "/v1/files?wait=false",
            headers=hdrs,
            data={"project_id": pid},
            files={"file": ("notes.md", b"# Summary\nOk.\n", "text/markdown")},
        )
        assert r_async.status_code == S.ACCEPTED
        deadline, polled = time.monotonic() + 3.0, r_async.json()
        while time.monotonic() < deadline and polled["status"] != FileStatus.READY.value:
            await asyncio.sleep(0.02)
            polled = (await env.client.get(f"/v1/files/{polled['id']}", headers=hdrs)).json()
        assert polled["status"] == FileStatus.READY.value

        assert (await env.client.delete(f"/v1/files/{fid}", headers=hdrs)).status_code == S.OK
        with pytest.raises(errs.NotFoundError):
            await env.svc.blob_store.read_bytes(Path(f_rec["blob_path"]))
        assert (await env.client.delete(f"/v1/projects/{pid}", headers=hdrs)).status_code == S.OK
        with pytest.raises(errs.NotFoundError):
            await env.svc.blob_store.read_bytes(Path(r_str.json()["blob_path"]))


async def test_session_message_returns_reply(tmp_path: Path) -> None:
    """Verify non-streaming session turn returns orchestrator reply."""
    async with open_routes_env(tmp_path) as env:
        hdrs = bearer_headers(env.toks["writer"])
        sid = await _mk_ses(env, hdrs)
        r_msg = await env.client.post(
            f"/v1/sessions/{sid}/messages", headers=hdrs, json={"content": "hi"}
        )
        assert r_msg.json()["message"]["content"] == "ok reply"


async def test_session_message_guardrail_blocked_sse_sets_headers(tmp_path: Path) -> None:
    """Verify blocked session SSE stream sets X-LAI-Guardrail-* headers and turn_completed."""
    orch = FakeOrch(blocked=True, policy="no_prompt_injection")
    async with open_routes_env(tmp_path, orch=orch) as env:
        hdrs = bearer_headers(env.toks["writer"])
        sid = await _mk_ses(env, hdrs)
        r_sse = await env.client.post(
            f"/v1/sessions/{sid}/messages", headers=hdrs, json={"content": "x", "stream": True}
        )
        assert r_sse.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert r_sse.headers["X-LAI-Guardrail-Policy"] == "no_prompt_injection"
        assert "event: turn_completed" in r_sse.text


async def test_openai_models_and_routing_headers(tmp_path: Path) -> None:
    """Verify /v1/models auth and X-LAI-* routing headers on /v1/chat/completions."""
    async with open_routes_env(tmp_path) as env:
        hdrs = bearer_headers(env.toks["writer"])
        assert (await env.client.get("/v1/models")).status_code == S.UNAUTHORIZED
        models = (await env.client.get("/v1/models", headers=hdrs)).json()["data"]
        assert "gemma4-26b-a4b" in [m["id"] for m in models]

        pid = await _mk_proj(env, hdrs, "oai")
        fids = [
            (
                await env.client.post(
                    "/v1/files?wait=true",
                    headers=hdrs,
                    data={"project_id": pid},
                    files={"file": (fn, b"INFO ok\n", "text/plain")},
                )
            ).json()["id"]
            for fn in ("a.log", "b.log")
        ]

        oai_hdrs = {
            **hdrs,
            "X-LAI-Project-Id": pid,
            "X-LAI-File-Ids": f"{fids[0]}, {fids[1]}",
            "X-LAI-Context-Mode": "retrieval",
        }
        r_c = await env.client.post("/v1/chat/completions", headers=oai_hdrs, json=_CHAT_JSON)
        assert r_c.status_code == S.OK
        assert env.orch.calls[-1].attached_file_ids == tuple(fids)
        assert env.orch.calls[-1].context_mode is dc.ContextMode.RETRIEVAL


async def test_openai_stream_guardrail_blocked_sse_sets_headers(tmp_path: Path) -> None:
    """Verify blocked OpenAI SSE stream emits guardrail headers, chunks, and [DONE]."""
    orch = FakeOrch(blocked=True, policy="no_hate_speech")
    async with open_routes_env(tmp_path, orch=orch) as env:
        hdrs = bearer_headers(env.toks["writer"])
        r_oai = await env.client.post(
            "/v1/chat/completions", headers=hdrs, json={**_CHAT_JSON, "stream": True}
        )
        assert r_oai.headers["X-LAI-Guardrail-Blocked"] == "true"
        assert "chat.completion.chunk" in r_oai.text
        assert "data: [DONE]\n\n" in r_oai.text
