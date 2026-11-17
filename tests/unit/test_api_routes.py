"""Unit tests for FastAPI routes, error envelopes, alerts, OpenAI compat, and CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from http import HTTPStatus as S
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings
from lai.domain import chat as dc
from lai.domain import errors as errs
from lai.domain import tools as dt
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.files import FileStatus
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.inference import GpuTelemetry
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.ports.orchestrator import OrchestratorPort
from lai.ports.tools import ToolPluginPort
from lai.security import TokenBucketRateLimiter, generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from lai.api.deps import AppServices

HTTP_CLIENT_DISCONNECTED = 499
EXPECTED_MIN_TOOLS = 3
EXPECTED_CHUNK_COUNT = 1
_CHAT_URL = "/v1/chat/completions"
_CHAT_JSON: dict[str, object] = {
    "model": "gemma4-26b-a4b",
    "messages": [{"role": "user", "content": "hi"}],
}


class _WriteTool(ToolPluginPort):
    @property
    def spec(self) -> dt.ToolSpec:
        return dt.ToolSpec(
            name="write_note",
            description="Guarded note tool",
            permission_tier=dt.PermissionTier.WRITE,
            requires_approval_by_default=True,
        )

    async def execute(
        self, arguments: dict[str, Any], context: dt.ToolExecutionContext
    ) -> dt.ToolResult:
        del arguments, context
        return dt.ToolResult(tool_name="write_note", success=True, output="saved")


class _FakeOrch(OrchestratorPort):
    def __init__(self) -> None:
        self.calls: list[dc.TurnInput] = []
        self.next_error: Exception | None = None
        self.blocked = False
        self.policy: str | None = None

    def _prepare(self, inp: dc.TurnInput) -> dc.TurnOutput:
        self.calls.append(inp)
        if self.next_error is not None:
            err, self.next_error = self.next_error, None
            raise err
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


def _auth(token: str, **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", **extra}


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@dataclass(slots=True)
class _Env:
    svc: AppServices
    orch: _FakeOrch
    client: httpx.AsyncClient
    toks: dict[str, str]


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[_Env]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=True),
    )
    orch = _FakeOrch()
    svc = await build_app_services(settings, orchestrator=orch)
    emb = DeterministicHashEmbedder()
    vec = LanceDbVectorStore(data_dir / "lancedb", dimensions=emb.dimensions)
    svc.embedder, svc.vector_store = emb, vec
    svc.ingestion_pipeline = LocalIngestionPipeline(
        blob_store=svc.blob_store, file_repo=svc.file_repo, embedder=emb, vector_store=vec
    )
    svc.gpu_telemetry_fn = lambda: GpuTelemetry(
        gpu_name="RTX 3090", memory_used_mb=1024, memory_total_mb=24576, utilization_percent=5
    )
    svc.tool_registry.register(_WriteTool())

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
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield _Env(svc=svc, orch=orch, client=client, toks=tokens)
    finally:
        await close_app_services(svc)


async def test_health_status_alerts_and_files_lifecycle(env: _Env) -> None:
    """Verify /v1/health, auth/scope checks, /v1/system/*, projects, and file lifecycle."""
    svc, client, toks = env.svc, env.client, env.toks
    r_hp = await client.get("/v1/health", headers={"X-Request-Id": "req_1"})
    assert r_hp.status_code == S.OK
    assert r_hp.headers["X-Request-Id"] == "req_1"
    assert (await client.get("/v1/system/status")).status_code == S.UNAUTHORIZED

    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://127.0.0.1:8081/health").respond(S.OK, json={"status": "ok"})
        mock.get("http://127.0.0.1:8082/health").respond(S.OK, json={"status": "ok"})
        r_st = await client.get("/v1/system/status", headers=_auth(toks["reader"]))
    assert r_st.status_code == S.OK
    assert r_st.json()["gpu"]["gpu_name"] == "RTX 3090"
    assert r_st.json()["guardrail_health"]["healthy"] is True
    r_al_403 = await client.get("/v1/system/alerts", headers=_auth(toks["reader"]))
    assert r_al_403.status_code == S.FORBIDDEN
    assert (await client.get("/v1/system/alerts", headers=_auth(toks["admin"]))).json()[
        "count"
    ] == 0

    hdrs = _auth(toks["writer"])
    pid = (await client.post("/v1/projects", headers=hdrs, json={"name": "logs"})).json()["id"]
    assert (await client.get("/v1/projects", headers=hdrs)).json()["count"] == 1

    r_up = await client.post(
        "/v1/files?wait=true",
        headers=hdrs,
        data={"project_id": pid},
        files={"file": ("app.log", b"2026-01-01T00:00:00Z ERROR Boom\n", "text/plain")},
    )
    assert r_up.status_code == S.CREATED
    f_rec = r_up.json()
    fid = f_rec["id"]
    assert f_rec["status"] == FileStatus.READY.value
    chk = (await client.get(f"/v1/files/{fid}/chunks", headers=hdrs)).json()
    assert chk["count"] >= EXPECTED_CHUNK_COUNT

    r_str = await client.post(
        f"/v1/files/stream?project_id={pid}&wait=true",
        headers={**hdrs, "X-LAI-Filename": "worker.py", "Content-Type": "text/x-python"},
        content=b"def run() -> int:\n    return 42\n",
    )
    assert r_str.status_code == S.CREATED

    r_async = await client.post(
        "/v1/files?wait=false",
        headers=hdrs,
        data={"project_id": pid},
        files={"file": ("notes.md", b"# Summary\nOk.\n", "text/markdown")},
    )
    assert r_async.status_code == S.ACCEPTED
    deadline, polled = time.monotonic() + 3.0, r_async.json()
    while time.monotonic() < deadline and polled["status"] != FileStatus.READY.value:
        await asyncio.sleep(0.02)
        polled = (await client.get(f"/v1/files/{polled['id']}", headers=hdrs)).json()
    assert polled["status"] == FileStatus.READY.value

    r_del_f = await client.delete(f"/v1/files/{fid}", headers=hdrs)
    assert r_del_f.status_code == S.OK
    with pytest.raises(errs.NotFoundError):
        await svc.blob_store.read_bytes(Path(f_rec["blob_path"]))
    r_del_p = await client.delete(f"/v1/projects/{pid}", headers=hdrs)
    assert r_del_p.status_code == S.OK
    with pytest.raises(errs.NotFoundError):
        await svc.blob_store.read_bytes(Path(r_str.json()["blob_path"]))


async def test_sessions_and_openai_compat(env: _Env) -> None:
    """Verify sessions, OpenAI JSON/SSE, X-LAI-* routing headers, and guardrail headers."""
    orch, client, toks = env.orch, env.client, env.toks
    hdrs = _auth(toks["writer"])
    pid = (await client.post("/v1/projects", headers=hdrs, json={"name": "c"})).json()["id"]
    sid = (await client.post("/v1/sessions", headers=hdrs, json={"project_id": pid})).json()["id"]

    r_msg = await client.post(f"/v1/sessions/{sid}/messages", headers=hdrs, json={"content": "hi"})
    assert r_msg.json()["message"]["content"] == "ok reply"

    orch.blocked, orch.policy = True, "no_prompt_injection"
    r_sse = await client.post(
        f"/v1/sessions/{sid}/messages", headers=hdrs, json={"content": "x", "stream": True}
    )
    assert r_sse.headers["X-LAI-Guardrail-Blocked"] == "true"
    assert r_sse.headers["X-LAI-Guardrail-Policy"] == "no_prompt_injection"
    assert "event: turn_completed" in r_sse.text

    orch.blocked, orch.policy = False, None
    models = (await client.get("/v1/models", headers=hdrs)).json()["data"]
    assert "gemma4-26b-a4b" in [m["id"] for m in models]
    oai_hdrs = {
        **hdrs,
        "X-LAI-Project-Id": "prj_1",
        "X-LAI-File-Ids": "fil_a, fil_b",
        "X-LAI-Context-Mode": "retrieval",
    }
    r_c = await client.post("/v1/chat/completions", headers=oai_hdrs, json=_CHAT_JSON)
    assert r_c.status_code == S.OK
    assert orch.calls[-1].attached_file_ids == ("fil_a", "fil_b")
    assert orch.calls[-1].context_mode is dc.ContextMode.RETRIEVAL

    orch.blocked, orch.policy = True, "no_hate_speech"
    r_oai = await client.post(
        "/v1/chat/completions", headers=hdrs, json={**_CHAT_JSON, "stream": True}
    )
    assert r_oai.headers["X-LAI-Guardrail-Blocked"] == "true"
    assert "chat.completion.chunk" in r_oai.text
    assert "data: [DONE]\n\n" in r_oai.text


async def test_tools_errors_500_alert_and_429(env: _Env) -> None:
    """Verify tool approvals, error envelopes, 500 FATAL_EXCEPTION alert, and 429 Retry-After."""
    svc, orch, client, toks = env.svc, env.orch, env.client, env.toks
    hdrs = _auth(toks["writer"])
    assert (await client.get("/v1/tools", headers=hdrs)).json()["count"] >= EXPECTED_MIN_TOOLS
    pid = (await client.post("/v1/projects", headers=hdrs, json={"name": "tp"})).json()["id"]
    sid = (await client.post("/v1/sessions", headers=hdrs, json={"project_id": pid})).json()["id"]
    ctx = dt.ToolExecutionContext(
        project_id=pid,
        session_id=sid,
        principal_id="usr_t",
        principal_scopes=(ApiScope.TOOLS_EXEC,),
    )
    with pytest.raises(errs.ToolApprovalRequiredError) as exc_info:
        await svc.tool_executor.invoke("write_note", {"note": "hi"}, ctx)
    apr_url = f"/v1/tool-approvals/{exc_info.value.approval_id}/resolve"
    assert (
        await client.post(apr_url, headers=hdrs, json={"approved": True})
    ).status_code == S.FORBIDDEN
    r_a = await client.post(apr_url, headers=_auth(toks["admin"]), json={"approved": True})
    assert r_a.json()["status"] == dt.ApprovalStatus.EXECUTED.value

    for err, expected in (
        (errs.ToolApprovalRequiredError("ok", approval_id="a", tool_name="s"), S.CONFLICT),
        (errs.PayloadTooLargeError("big"), S.REQUEST_ENTITY_TOO_LARGE),
        (errs.GuardrailBlockedError("no", violated_policy="p"), S.UNPROCESSABLE_CONTENT),
        (errs.ClientDisconnectedError("bye"), HTTP_CLIENT_DISCONNECTED),
        (errs.InferenceBackendError("bad"), S.BAD_GATEWAY),
        (errs.QueueFullError("full"), S.SERVICE_UNAVAILABLE),
    ):
        orch.next_error = err
        res = await client.post("/v1/chat/completions", headers=hdrs, json=_CHAT_JSON)
        assert res.status_code == expected
        assert res.json()["error"]["request_id"]

    orch.next_error = RuntimeError("secret crash")
    r5 = await client.post(_CHAT_URL, headers={**hdrs, "X-Request-Id": "r5"}, json=_CHAT_JSON)
    assert r5.status_code == S.INTERNAL_SERVER_ERROR
    assert "secret crash" not in json.dumps(r5.json())
    items = (await client.get("/v1/system/alerts", headers=_auth(toks["admin"]))).json()["items"]
    cat = AlertCategory.FATAL_EXCEPTION.value
    assert any(a["category"] == cat and a["request_id"] == "r5" for a in items)

    svc.rate_limiter = TokenBucketRateLimiter(requests_per_minute=1)
    assert (await client.get("/v1/projects", headers=hdrs)).status_code == S.OK
    r_429 = await client.get("/v1/projects", headers=hdrs)
    assert r_429.status_code == S.TOO_MANY_REQUESTS
    assert int(r_429.headers["Retry-After"]) >= 1


def test_cli_runner_smoke(tmp_path: Path) -> None:
    """Smoke-test Typer CLI init, keys create/list/revoke, and tunnel render-config."""
    runner, cfg, out = CliRunner(), tmp_path / "cli" / "config.yaml", tmp_path / "cf.yml"
    r1 = runner.invoke(cli_app, ["init", "-c", str(cfg), "--data-dir", str(tmp_path / "d")])
    assert r1.exit_code == 0
    r2 = runner.invoke(cli_app, ["keys", "create", "-c", str(cfg), "-n", "k", "--admin"])
    assert "lai_sk_" in r2.output
    kids = runner.invoke(cli_app, ["keys", "list", "-c", str(cfg)]).output.splitlines()
    r3 = runner.invoke(cli_app, ["keys", "revoke", kids[0].split("\t")[0], "-c", str(cfg)])
    assert r3.exit_code == 0
    tun_cmd = ["tunnel", "render-config", "-c", str(cfg), "-H", "ai.t", "-o", str(out)]
    assert runner.invoke(cli_app, tun_cmd).exit_code == 0
    assert "ai.t" in out.read_text(encoding="utf-8")
