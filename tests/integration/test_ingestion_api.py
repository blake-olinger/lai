"""Integration tests for multipart/streaming file ingestion, log summaries, and RRF search."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.files import FileKind, FileStatus
from lai.domain.observability import LoggingConfig
from lai.ingestion import reciprocal_rank_fusion
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from pathlib import Path

    import httpx

_FP_HEX_LEN = 16
_MIN_LOG_LINES = 2200
_MIN_TRACEBACK_CHUNKS = 3
_EXPECTED_TIMEOUT_ERRORS = 4
_EXPECTED_VALUE_ERRORS = 3
_EXPECTED_WARNINGS = 11
_INGEST_SCOPES = (ApiScope.PROJECTS_WRITE, ApiScope.FILES_WRITE, ApiScope.CHAT_WRITE)


def _ingest_settings(tmp_path: Path) -> LaiSettings:
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
    )


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


async def _upload_large_log(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    pid: str,
) -> dict[str, object]:
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
    return dict(r_log.json())


async def test_code_upload_multipart_and_streaming_extracts_symbols(tmp_path: Path) -> None:
    """Multipart and streaming Python uploads classify FileKind.CODE and extract symbols."""
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
    spec = IntegrationSpec(use_hash_embedder=True)
    async with open_services_harness(_ingest_settings(tmp_path), spec) as harness:
        headers = await mint_auth_headers(harness.svc, "ingest-code", _INGEST_SCOPES)
        pid = await create_project(harness.client, headers, "ingest-code-p1")

        r_auth = await harness.client.post(
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

        r_db = await harness.client.post(
            f"/v1/files/stream?project_id={pid}&wait=true",
            headers={**headers, "X-LAI-Filename": "db_pool.py", "Content-Type": "text/x-python"},
            content=db_code,
        )
        assert r_db.status_code == HTTPStatus.CREATED
        db_rec = r_db.json()
        assert db_rec["kind"] == FileKind.CODE.value
        assert "acquire_pool_connection" in db_rec["code_summary"]["symbols"]


async def test_large_log_streaming_ingestion_preserves_tracebacks_and_summaries(
    tmp_path: Path,
) -> None:
    """Streaming >2k-line log preserves whole tracebacks, summary counts, and line filters."""
    spec = IntegrationSpec(use_hash_embedder=True)
    async with open_services_harness(_ingest_settings(tmp_path), spec) as harness:
        headers = await mint_auth_headers(harness.svc, "ingest-log", _INGEST_SCOPES)
        pid = await create_project(harness.client, headers, "ingest-log-p1")
        log_rec = await _upload_large_log(harness.client, headers, pid)
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

        chunks = (await harness.client.get(f"/v1/files/{log_id}/chunks", headers=headers)).json()[
            "items"
        ]
        tb_chunks = [c for c in chunks if "Traceback (most recent call last):" in c["text"]]
        assert len(tb_chunks) >= _MIN_TRACEBACK_CHUNKS
        for tb_chunk in tb_chunks:
            assert (
                "TimeoutError: connection pool exhausted for primary-db" in tb_chunk["text"]
                or "ValueError: invalid token signature in header" in tb_chunk["text"]
            )

        first_tb = tb_chunks[0]
        filtered = (
            await harness.client.get(
                f"/v1/files/{log_id}/chunks",
                headers=headers,
                params={"start_line": first_tb["start_line"], "end_line": first_tb["end_line"]},
            )
        ).json()["items"]
        assert any(c["id"] == first_tb["id"] for c in filtered)


async def test_hybrid_fts_vector_search_and_rrf_respect_project_isolation(
    tmp_path: Path,
) -> None:
    """FTS5, LanceDB vectors, and RRF fuse matches strictly within the target project."""
    spec = IntegrationSpec(use_hash_embedder=True)
    async with open_services_harness(_ingest_settings(tmp_path), spec) as harness:
        headers = await mint_auth_headers(harness.svc, "ingest-rrf", _INGEST_SCOPES)
        pid = await create_project(harness.client, headers, "ingest-p1")
        other_pid = await create_project(harness.client, headers, "ingest-p2")

        log_rec = await _upload_large_log(harness.client, headers, pid)
        log_id = str(log_rec["id"])
        await harness.client.post(
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

        assert harness.svc.embedder is not None
        assert harness.svc.vector_store is not None
        query = "TimeoutError connection pool exhausted primary-db"
        fts_hits = await harness.svc.file_repo.search_fts(pid, query, limit=20)
        vec_hits = await harness.svc.vector_store.search_vectors(
            pid, await harness.svc.embedder.embed_query(query), limit=20
        )
        assert fts_hits
        assert vec_hits
        assert all(hit.chunk.project_id == pid for hit in fts_hits)
        assert all(hit.chunk.project_id == pid for hit in vec_hits)

        fused = reciprocal_rank_fusion(fts_hits, vec_hits, limit=10)
        assert fused
        assert log_id in {hit.chunk.file_id for hit in fused}
        assert any("connection pool exhausted" in hit.chunk.text for hit in fused)
