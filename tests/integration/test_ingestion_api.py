"""Integration tests for multipart/streaming file ingestion, log summaries, and RRF search."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.files import FileKind, FileStatus
from lai.domain.observability import LoggingConfig
from lai.ingestion import (
    DeterministicHashEmbedder,
    LanceDbVectorStore,
    LocalIngestionPipeline,
    reciprocal_rank_fusion,
)
from lai.security import generate_api_key
from lai.tools import build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

_FP_HEX_LEN = 16
_MIN_LOG_LINES = 2200
_MIN_TRACEBACK_CHUNKS = 3
_EXPECTED_TIMEOUT_ERRORS = 4
_EXPECTED_VALUE_ERRORS = 3
_EXPECTED_WARNINGS = 11


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _build_multi_thousand_line_log() -> bytes:
    """Build a >2,200-line multi-exception log with Python tracebacks and timestamps."""
    lines: list[str] = []
    for idx in range(2200):
        sec = idx % 60
        minute = (idx // 60) % 60
        hour = idx // 3600
        ts = f"2026-11-17T{hour:02d}:{minute:02d}:{sec:02d}Z"
        if idx in (200, 700, 1200, 1800):
            lines.extend(
                [
                    f"{ts} ERROR Database pool checkout failed on worker-{idx}",
                    "Traceback (most recent call last):",
                    '  File "src/db_pool.py", line 42, in acquire_pool_connection',
                    "    return await pool.checkout(timeout=5.0)",
                    "TimeoutError: connection pool exhausted for primary-db",
                ]
            )
        elif idx in (450, 950, 1550):
            lines.extend(
                [
                    f"{ts} ERROR Auth token validation failed on request-{idx}",
                    "Traceback (most recent call last):",
                    '  File "src/auth_service.py", line 28, in verify_bearer_token',
                    "    claims = decode_token(raw_token)",
                    "ValueError: invalid token signature in header",
                ]
            )
        elif idx % 215 == 0:
            lines.append(f"{ts} WARN High memory watermark reached at tick={idx}")
        else:
            lines.append(f"{ts} INFO Worker heartbeat healthy tick={idx}")
    return ("\n".join(lines) + "\n").encode("utf-8")


async def _upload_project_files(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    pid: str,
    other_pid: str,
) -> dict[str, object]:
    db_code = (
        b"async def acquire_pool_connection(pool, timeout: float = 5.0):\n"
        b"    # Checkout connection from primary-db pool\n"
        b"    return await pool.checkout(timeout=timeout)\n"
    )
    auth_code = (
        b"def verify_bearer_token(raw_token: str) -> dict[str, str]:\n"
        b"    if not raw_token:\n"
        b"        raise ValueError('invalid token signature in header')\n"
        b"    return {'sub': 'user'}\n"
    )

    r_auth = await client.post(
        "/v1/files?wait=true",
        headers=headers,
        data={"project_id": pid},
        files={"file": ("auth_service.py", auth_code, "text/x-python")},
    )
    assert r_auth.status_code == HTTPStatus.CREATED
    auth_rec = r_auth.json()
    assert auth_rec["kind"] == FileKind.CODE.value
    assert auth_rec["status"] == FileStatus.READY.value
    assert "verify_bearer_token" in auth_rec["code_summary"]["symbols"]

    r_db = await client.post(
        f"/v1/files/stream?project_id={pid}&wait=true",
        headers={
            **headers,
            "X-LAI-Filename": "db_pool.py",
            "Content-Type": "text/x-python",
        },
        content=db_code,
    )
    assert r_db.status_code == HTTPStatus.CREATED
    db_rec = r_db.json()
    assert db_rec["kind"] == FileKind.CODE.value
    assert "acquire_pool_connection" in db_rec["code_summary"]["symbols"]

    r_log = await client.post(
        "/v1/files/stream?wait=true",
        headers={
            **headers,
            "X-LAI-Project-Id": pid,
            "X-LAI-Filename": "prod_cluster.log",
            "Content-Type": "text/plain",
        },
        content=_build_multi_thousand_line_log(),
    )
    assert r_log.status_code == HTTPStatus.CREATED

    await client.post(
        "/v1/files?wait=true",
        headers=headers,
        data={"project_id": other_pid},
        files={
            "file": (
                "other.log",
                b"2026-11-17T00:00:00Z ERROR TimeoutError: isolated project\n",
                "text/plain",
            )
        },
    )
    return dict(r_log.json())


async def _assert_log_summary_and_chunks(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    log_rec: dict[str, object],
) -> str:
    log_id = str(log_rec["id"])
    assert log_rec["kind"] == FileKind.LOG.value
    assert log_rec["status"] == FileStatus.READY.value

    summary = cast("dict[str, Any]", log_rec["log_summary"])
    assert isinstance(summary, dict)
    assert summary["total_lines"] >= _MIN_LOG_LINES
    assert summary["error_count"] == _EXPECTED_TIMEOUT_ERRORS + _EXPECTED_VALUE_ERRORS
    assert summary["warning_count"] == _EXPECTED_WARNINGS
    assert summary["first_timestamp"] is not None
    assert summary["last_timestamp"] is not None

    top_items = cast("list[dict[str, Any]]", summary["top_exceptions"])
    by_exc_type = {str(item["exception_type"]): item for item in top_items}
    assert by_exc_type["TimeoutError"]["count"] == _EXPECTED_TIMEOUT_ERRORS
    assert len(str(by_exc_type["TimeoutError"]["fingerprint"])) == _FP_HEX_LEN
    assert by_exc_type["ValueError"]["count"] == _EXPECTED_VALUE_ERRORS
    assert len(str(by_exc_type["ValueError"]["fingerprint"])) == _FP_HEX_LEN

    chunks = (await client.get(f"/v1/files/{log_id}/chunks", headers=headers)).json()["items"]
    tb_chunks = [c for c in chunks if "Traceback (most recent call last):" in c["text"]]
    assert len(tb_chunks) >= _MIN_TRACEBACK_CHUNKS
    for tb_chunk in tb_chunks:
        assert (
            "TimeoutError: connection pool exhausted for primary-db" in tb_chunk["text"]
            or "ValueError: invalid token signature in header" in tb_chunk["text"]
        )

    first_tb = tb_chunks[0]
    filtered = (
        await client.get(
            f"/v1/files/{log_id}/chunks",
            headers=headers,
            params={"start_line": first_tb["start_line"], "end_line": first_tb["end_line"]},
        )
    ).json()["items"]
    assert any(c["id"] == first_tb["id"] for c in filtered)
    return log_id


async def test_multipart_and_streaming_ingestion_log_summary_and_hybrid_rrf(
    tmp_path: Path,
) -> None:
    """Upload code + >2k-line log via HTTP and verify trace preservation, summaries, and RRF."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
    )
    svc = await build_app_services(settings)
    embedder = DeterministicHashEmbedder()
    vector_store = LanceDbVectorStore(data_dir / "lancedb", dimensions=embedder.dimensions)
    svc.embedder = embedder
    svc.vector_store = vector_store
    svc.ingestion_pipeline = LocalIngestionPipeline(
        blob_store=svc.blob_store,
        file_repo=svc.file_repo,
        embedder=embedder,
        vector_store=vector_store,
    )
    svc.tool_registry = build_default_tool_registry(
        svc.file_repo, svc.blob_store, embedder=embedder, vector_store=vector_store
    )

    api_key = generate_api_key(
        "ingest-writer",
        (ApiScope.PROJECTS_WRITE, ApiScope.FILES_WRITE, ApiScope.CHAT_WRITE),
    )
    await svc.api_key_repo.create(api_key.record)
    headers = {"Authorization": f"Bearer {api_key.plaintext_key}"}

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            pid = (
                await client.post("/v1/projects", headers=headers, json={"name": "ingest-p1"})
            ).json()["id"]
            other_pid = (
                await client.post("/v1/projects", headers=headers, json={"name": "ingest-p2"})
            ).json()["id"]
            log_rec = await _upload_project_files(client, headers, pid, other_pid)
            log_id = await _assert_log_summary_and_chunks(client, headers, log_rec)

        query = "TimeoutError connection pool exhausted primary-db"
        fts_hits = await svc.file_repo.search_fts(pid, query, limit=20)
        vec_hits = await vector_store.search_vectors(
            pid, await embedder.embed_query(query), limit=20
        )
        assert fts_hits
        assert vec_hits
        assert all(hit.chunk.project_id == pid for hit in fts_hits)
        assert all(hit.chunk.project_id == pid for hit in vec_hits)

        fused = reciprocal_rank_fusion(fts_hits, vec_hits, limit=10)
        assert fused
        assert log_id in {hit.chunk.file_id for hit in fused}
        assert any("connection pool exhausted" in hit.chunk.text for hit in fused)
    finally:
        await close_app_services(svc)
