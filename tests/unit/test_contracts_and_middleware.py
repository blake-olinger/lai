"""Unit tests for Wave 3 contracts: scopes, settings wrapping, middleware, lifespan, and RRF."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Literal, cast

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import lai.domain as domain_pkg
from lai.api.app import create_app
from lai.api.middleware import RequestContextMiddleware
from lai.api.runtime import build_app_services, close_app_services
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings, load_settings, load_yaml_mapping
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal, expand_scopes
from lai.domain.errors import AuthorizationError, ValidationFailedError
from lai.domain.files import FileChunk, FileKind, FileRecord, FileStatus, ScoredChunk
from lai.domain.rrf import DEFAULT_RRF_K, DEFAULT_RRF_LIMIT, reciprocal_rank_fusion
from lai.domain.tools import ToolExecutionContext
from lai.ingestion.rrf import reciprocal_rank_fusion as ingestion_rrf
from lai.security import scopes as security_scopes
from lai.tools.plugins.search_files import SearchFilesPlugin

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from starlette.types import Message, Receive, Scope, Send

    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.storage import EmbedderPort, VectorStorePort

_REQ_ID_RE = re.compile(r"^req_[0-9a-f]{32}$")
_HTTP_OK = 200
_HTTP_FORBIDDEN = 403
_MAX_REQ_ID_LEN = 128
_OVER_LIMIT_INPUTS = 22
_EXPECTED_TWO_HITS = 2
_DEFAULT_K1_SCORE = 1.0 / (DEFAULT_RRF_K + 1)
_DUP_SCORE = 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 2)


@pytest.fixture(autouse=True)
def _cleanup_root_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _principal(*scopes: ApiScope) -> AuthPrincipal:
    return AuthPrincipal(principal_id="usr_contract", auth_method=AuthMethod.API_KEY, scopes=scopes)


def _hit(
    cid: str,
    file_id: str = "fil_1",
    project_id: str = "prj_1",
    start_line: int = 1,
    source: Literal["vector", "fts", "hybrid", "direct"] = "fts",
) -> ScoredChunk:
    chunk = FileChunk(
        id=cid,
        file_id=file_id,
        project_id=project_id,
        chunk_index=0,
        start_line=start_line,
        end_line=start_line + 4,
        token_count=10,
        text=f"text {cid}",
    )
    return ScoredChunk(chunk=chunk, score=1.0, source=source)


async def _mw_call(
    raw_id: bytes | None,
    preset: list[tuple[bytes, bytes]] | None = None,
    *,
    scope_type: str = "http",
    state_val: object | None = None,
) -> tuple[Scope, dict[bytes, bytes], list[Scope]]:
    sent: list[Message] = []
    called: list[Scope] = []

    async def _app(s: Scope, _r: Receive, send: Send) -> None:
        called.append(s)
        if s["type"] == "http":
            hdrs = list(preset or [])
            await send({"type": "http.response.start", "status": _HTTP_OK, "headers": hdrs})
            await send({"type": "http.response.body", "body": b"ok"})

    async def _recv() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(msg: Message) -> None:
        sent.append(msg)

    hdrs: list[tuple[bytes, bytes]] = [(b"x-other", b"1")]
    if raw_id is not None:
        hdrs.append((b"x-request-id", raw_id))
    scope: Scope = {"type": scope_type, "method": "GET", "path": "/v1/health", "headers": hdrs}
    if state_val is not None:
        scope["state"] = state_val
    await RequestContextMiddleware(_app)(scope, _recv, _send)
    return (
        scope,
        dict(cast("list[tuple[bytes, bytes]]", sent[0]["headers"])) if sent else {},
        called,
    )


def test_domain_and_security_scopes_are_unified() -> None:
    """AuthPrincipal.has_scope honors full scope hierarchy identically to security.scopes."""
    assert domain_pkg.expand_scopes is expand_scopes
    assert security_scopes.expand_scopes is expand_scopes

    for granted, implied, denied in (
        (ApiScope.PROJECTS_WRITE, ApiScope.PROJECTS_READ, ApiScope.FILES_READ),
        (ApiScope.FILES_WRITE, ApiScope.FILES_READ, ApiScope.CHAT_READ),
        (ApiScope.CHAT_WRITE, ApiScope.CHAT_READ, ApiScope.TOOLS_SAFE),
        (ApiScope.TOOLS_EXEC, ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE),
    ):
        principal = _principal(granted, granted)
        assert principal.has_scope(granted)
        assert principal.has_scope(implied)
        assert not principal.has_scope(denied)
        for scope in ApiScope:
            assert principal.has_scope(scope) is security_scopes.has_effective_scope(
                principal, scope
            )
        security_scopes.require_scope(principal, implied)
        with pytest.raises(AuthorizationError) as exc_info:
            security_scopes.require_scope(principal, denied)
        assert exc_info.value.status_code == _HTTP_FORBIDDEN

    reader = _principal(ApiScope.FILES_READ, ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE)
    for forbidden in (ApiScope.FILES_WRITE, ApiScope.TOOLS_EXEC, ApiScope.ADMIN):
        assert not reader.has_scope(forbidden)
    assert all(_principal(ApiScope.ADMIN).has_scope(s) for s in ApiScope)


def test_settings_wrap_yaml_and_pydantic_errors_into_validation_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_yaml_mapping and load_settings wrap YAML/Pydantic errors while LaiSettings stays raw."""
    malformed = tmp_path / "broken.yaml"
    malformed.write_text("port: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValidationFailedError) as yaml_err:
        load_yaml_mapping(malformed)
    assert yaml_err.value.details == {"path": str(malformed)}

    runner = CliRunner()
    for name, body in (
        ("broken.yaml", "port: [unclosed\n"),
        ("bad_port.yaml", "port: 70000\n"),
        ("extra.yaml", "unknown_setting: true\n"),
    ):
        cfg = tmp_path / name
        cfg.write_text(body, encoding="utf-8")
        with pytest.raises(ValidationFailedError) as cfg_err:
            load_settings(cfg)
        assert cfg_err.value.details["path"] == str(cfg)
        cli_res = runner.invoke(cli_app, ["serve", "--config", str(cfg)])
        assert cli_res.exit_code == 1
        assert "Configuration error (validation_failed):" in cli_res.output

    with pytest.raises(ValidationFailedError) as override_err:
        load_settings(port=0)
    assert "path" not in override_err.value.details

    monkeypatch.setenv("LAI_PORT", "not_an_int")
    with pytest.raises(ValidationFailedError):
        load_settings()
    monkeypatch.delenv("LAI_PORT")

    with pytest.raises(ValidationError):
        LaiSettings(port=0)
    with pytest.raises(ValidationError):
        LaiSettings.model_validate({"port": "x"})


async def test_middleware_request_id_sanitization_and_asgi_paths() -> None:
    """Middleware validates strict ASCII IDs, passes non-HTTP scopes, and keeps preset headers."""
    valid = ((b"req_abc-1.2:3", "req_abc-1.2:3"), (b"  pad  ", "pad"), (b"a" * 128, "a" * 128))
    for raw, expected in valid:
        scope, resp_hdrs, _ = await _mw_call(raw)
        assert scope["state"]["request_id"] == expected
        assert resp_hdrs[b"x-request-id"] == expected.encode("ascii")

    bad_a = (None, b"", b"   ", b"req_1\r\nX-Inject: 1", b"req_1\r\n", b"\nreq_1", b"req\x00id")
    bad_b = (b"req\x1b[31m", b"req_\xffx", b"caf\xc3\xa9", b"a b", b"a/b", b"x" * 129)
    for bad in (*bad_a, *bad_b):
        scope, resp_hdrs, _ = await _mw_call(bad)
        assigned = cast("str", scope["state"]["request_id"])
        assert _REQ_ID_RE.fullmatch(assigned) is not None
        if bad is not None:
            assert assigned != bad.decode("ascii", errors="ignore").strip()
        assert resp_hdrs[b"x-request-id"] == assigned.encode("ascii")

    ws_scope, ws_hdrs, called = await _mw_call(None, scope_type="websocket")
    assert (called, ws_hdrs) == ([ws_scope], {})
    assert "state" not in ws_scope

    scope, resp_hdrs, _ = await _mw_call(b"caller", preset=[(b"X-Request-Id", b"downstream_id")])
    assert scope["state"]["request_id"] == "caller"
    assert resp_hdrs[b"X-Request-Id"] == b"downstream_id"
    nondict_scope, _, _ = await _mw_call(b"ok_id", state_val="not-a-dict")
    assert nondict_scope["state"] == "not-a-dict"


async def test_create_app_lifespan_builds_closes_and_cancels_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_app lifespan builds/closes default services and cancels prebuilt tasks."""
    monkeypatch.setenv("LAI_DATA_DIR", str(tmp_path / "env_data"))
    monkeypatch.setenv("LAI_MODELS_DIR", str(tmp_path / "env_models"))
    monkeypatch.setenv("LAI_BIN_DIR", str(tmp_path / "env_bin"))
    app_default = create_app()
    async with (
        app_default.router.lifespan_context(app_default),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_default), base_url="http://testserver"
        ) as client,
    ):
        resp = await client.get("/v1/health", headers={"X-Request-Id": "bad\x00id"})
        assert resp.status_code == _HTTP_OK
        assert _REQ_ID_RE.fullmatch(resp.headers["X-Request-Id"]) is not None

    svc_dir = tmp_path / "svc"
    prebuilt = await build_app_services(
        LaiSettings(data_dir=svc_dir / "d", models_dir=svc_dir / "m", bin_dir=svc_dir / "b")
    )
    bg_task = asyncio.create_task(asyncio.sleep(60))
    prebuilt.background_tasks.add(bg_task)
    app_prebuilt = create_app(services=prebuilt)
    try:
        async with app_prebuilt.router.lifespan_context(app_prebuilt):
            assert app_prebuilt.state.services is prebuilt
        assert bg_task.cancelled() is True
        assert not prebuilt.background_tasks
    finally:
        await close_app_services(prebuilt)


def test_domain_and_ingestion_rrf_provenance_tiebreaking_and_defaults() -> None:
    """Domain RRF tracks provenance, validation, dedup, and ties; ingestion keeps 'hybrid'."""
    mc = ScoredChunk.model_construct
    fts_hits = [
        _hit("chk_b", "fil_1", start_line=5, source="fts"),
        _hit("chk_b", "fil_1", start_line=5, source="fts"),
        _hit("chk_a", "fil_2", start_line=10, source="fts"),
        _hit("chk_d", "fil_3", start_line=1, source="direct"),
        mc(chunk=_hit("chk_ef", "fil_4").chunk, score=1.0, source=cast("Literal['fts']", "")),
    ]
    vec_hits = [
        _hit("chk_c", "fil_1", start_line=1, source="vector"),
        _hit("chk_b", "fil_1", start_line=5, source="fts"),
        mc(chunk=_hit("chk_ev", "fil_5").chunk, score=1.0, source=cast("Literal['vector']", "")),
    ]
    fused = reciprocal_rank_fusion(fts_hits, vec_hits)
    assert [(h.chunk.id, h.source) for h in fused] == [
        ("chk_b", "hybrid"),
        ("chk_c", "vector"),
        ("chk_a", "fts"),
        ("chk_d", "direct"),
        ("chk_ev", "vector"),
        ("chk_ef", "fts"),
    ]
    assert fused[0].score == pytest.approx(_DUP_SCORE)

    many = [_hit(f"chk_{i:02d}") for i in range(_OVER_LIMIT_INPUTS)]
    for fn in (reciprocal_rank_fusion, ingestion_rrf):
        out = fn(many, [])
        assert len(out) == DEFAULT_RRF_LIMIT
        assert out[0].score == pytest.approx(_DEFAULT_K1_SCORE)
    assert reciprocal_rank_fusion(many, [], limit=0) == []
    with pytest.raises(ValidationFailedError):
        reciprocal_rank_fusion(many, [], k=0)

    tie_a = _hit("chk_z", "fil_1", start_line=20, source="hybrid")
    tie_b = _hit("chk_a", "fil_1", start_line=10, source="hybrid")
    assert [h.chunk.id for h in reciprocal_rank_fusion([tie_a], [tie_b])] == ["chk_a", "chk_z"]


async def test_search_files_plugin_uses_unified_rrf_without_premature_eviction() -> None:
    """SearchFilesPlugin delegates to domain RRF and keeps valid hits past out-of-scope ones."""
    f1 = FileRecord(
        id="fil_1",
        project_id="prj_1",
        filename="app.log",
        size_bytes=10,
        sha256="a" * 64,
        blob_path="b/app.log",
        kind=FileKind.LOG,
        status=FileStatus.READY,
    )
    bad = [_hit(f"chk_b{i}", "fil_2", "prj_2", i) for i in (1, 2)]
    good = [_hit(f"chk_ok{i}", "fil_1", "prj_1", i, "vector") for i in (1, 2)]
    exp_vec = ("prj_1", [1.0, 0.0], ("fil_1",), _EXPECTED_TWO_HITS)

    class _DepsStub:
        async def search_fts(
            self, pid: str, q: str, *, file_ids: Sequence[str] | None = None, limit: int = 20
        ) -> list[ScoredChunk]:
            assert (pid, q, file_ids, limit) == ("prj_1", "err", ("fil_1",), _EXPECTED_TWO_HITS)
            return bad

        async def get_file(self, file_id: str) -> FileRecord | None:
            return f1 if file_id == "fil_1" else None

        async def embed_query(self, query: str) -> list[float]:
            assert query == "err"
            return [1.0, 0.0]

        async def search_vectors(
            self,
            pid: str,
            qv: Sequence[float],
            *,
            file_ids: Sequence[str] | None = None,
            limit: int = 20,
        ) -> list[ScoredChunk]:
            assert (pid, qv, file_ids, limit) == exp_vec
            return good

    deps = _DepsStub()
    plugin = SearchFilesPlugin(
        cast("FileRepositoryPort", deps),
        embedder=cast("EmbedderPort", deps),
        vector_store=cast("VectorStorePort", deps),
    )
    ctx = ToolExecutionContext(
        project_id="prj_1", session_id="ses_1", principal_id="usr_1", allowed_file_ids=("fil_1",)
    )
    res = await plugin.execute({"query": "err", "limit": _EXPECTED_TWO_HITS}, ctx)
    data = cast("dict[str, object]", json.loads(res.output))
    assert (res.success, res.truncated, data["count"]) == (True, False, _EXPECTED_TWO_HITS)
    results = cast("list[dict[str, object]]", data["results"])
    assert [r["chunk_id"] for r in results] == ["chk_ok1", "chk_ok2"]
    assert [r["source"] for r in results] == ["vector", "vector"]
