"""Integration tests for LocalIngestionPipeline, LanceDbVectorStore, RRF, and alerting."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pytest

from lai.db import SqliteDatabase, SqliteFileRepository, SqliteProjectRepository
from lai.domain.errors import NotFoundError, ValidationFailedError
from lai.domain.files import FileChunk, FileKind, FileStatus, ScoredChunk
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.domain.projects import Project
from lai.ingestion.blob_store import LocalBlobStore
from lai.ingestion.embedder import DeterministicHashEmbedder
from lai.ingestion.pipeline import LocalIngestionPipeline
from lai.ingestion.rrf import reciprocal_rank_fusion
from lai.ingestion.vector_store import LanceDbVectorStore
from lai.observability.logging import configure_observability
from lai.ports.storage import EmbedderPort, VectorStorePort

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator, Sequence
    from pathlib import Path

EXPECTED_SEMAPHORE_CAP = 4
CONCURRENT_UPLOADS = 8
TEST_DIMENSIONS = 64
_STREAM_CHUNK_BYTES = 64


class _TrackingEmbedder(EmbedderPort):
    """Embedder wrapper tracking peak concurrent calls to verify semaphore gating."""

    def __init__(self, inner: EmbedderPort, delay_seconds: float = 0.02) -> None:
        self._inner = inner
        self._delay = delay_seconds
        self.active = 0
        self.max_active = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self._delay)
            return await self._inner.embed_texts(texts)
        finally:
            self.active -= 1

    async def embed_query(self, query: str) -> list[float]:
        return await self._inner.embed_query(query)


class _BrokenVectorStore(VectorStorePort):
    """VectorStorePort raising storage errors on upsert to verify alerting and FAILED status."""

    def __init__(self) -> None:
        self.upsert_calls: list[tuple[str, int, int]] = []
        self.search_calls: list[tuple[str, int, tuple[str, ...] | None, int]] = []
        self.delete_calls: list[tuple[str, str]] = []

    async def upsert_chunks(
        self,
        project_id: str,
        chunks: Sequence[FileChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.upsert_calls.append((project_id, len(chunks), len(vectors)))
        msg = "lancedb disk write error"
        raise OSError(msg)

    async def search_vectors(
        self,
        project_id: str,
        query_vector: Sequence[float],
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        fids = tuple(file_ids) if file_ids is not None else None
        self.search_calls.append((project_id, len(query_vector), fids, limit))
        return []

    async def delete_file_vectors(self, project_id: str, file_id: str) -> None:
        self.delete_calls.append((project_id, file_id))


async def _stream_bytes(payload: bytes) -> AsyncIterator[bytes]:
    yield payload[:_STREAM_CHUNK_BYTES]
    if len(payload) > _STREAM_CHUNK_BYTES:
        yield payload[_STREAM_CHUNK_BYTES:]


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_end_to_end_ingestion_hybrid_search_and_isolation(tmp_path: Path) -> None:
    """Pipeline ingests log/code/markdown into SQLite + LanceDB with strict project isolation."""
    blob_store = LocalBlobStore(tmp_path / "blobs")
    embedder = DeterministicHashEmbedder(dimensions=TEST_DIMENSIONS)
    vector_store = LanceDbVectorStore(tmp_path / "lancedb", dimensions=TEST_DIMENSIONS)
    async with SqliteDatabase(tmp_path / "lai.db") as db:
        projects = SqliteProjectRepository(db)
        await projects.create(Project(id="prj_alpha", name="Alpha"))
        await projects.create(Project(id="prj_beta", name="Beta"))
        file_repo = SqliteFileRepository(db)
        pipeline = LocalIngestionPipeline(
            blob_store=blob_store,
            file_repo=file_repo,
            embedder=embedder,
            vector_store=vector_store,
        )

        log_payload = (
            b"2026-11-17T12:00:00Z ERROR Database connection pool exhausted\n"
            b"Traceback (most recent call last):\n"
            b'  File "db.py", line 14, in connect\n'
            b"    raise TimeoutError('pool timeout')\n"
            b"TimeoutError: pool timeout\n"
        )
        code_payload = (
            b"import asyncio\n\n"
            b"async def retry_connect(attempts: int) -> None:\n"
            b"    await asyncio.sleep(attempts)\n"
        )
        other_proj_payload = b"2026-11-17T12:05:00Z ERROR Secret project database timeout"

        log_rec = await pipeline.ingest_upload(
            "prj_alpha", "app.log", "text/plain", _stream_bytes(log_payload)
        )
        code_rec = await pipeline.ingest_upload(
            "prj_alpha", "client.py", "text/x-python", _stream_bytes(code_payload)
        )
        md_rec = await pipeline.ingest_upload(
            "prj_alpha", "RUNBOOK.md", "text/markdown", _stream_bytes(b"# Runbook\n\nCheck pool.")
        )
        beta_rec = await pipeline.ingest_upload(
            "prj_beta", "beta.log", "text/plain", _stream_bytes(other_proj_payload)
        )

        assert md_rec.status is FileStatus.READY
        assert md_rec.kind is FileKind.MARKDOWN
        assert log_rec.status is FileStatus.READY
        assert log_rec.kind is FileKind.LOG
        assert log_rec.log_summary is not None
        assert log_rec.log_summary.error_count == 1
        assert code_rec.status is FileStatus.READY
        assert code_rec.kind is FileKind.CODE
        assert code_rec.code_summary is not None
        assert code_rec.code_summary.symbols == ("retry_connect",)

        query = "database pool timeout"
        q_vec = await embedder.embed_query(query)
        fts_hits = await file_repo.search_fts("prj_alpha", query)
        vec_hits = await vector_store.search_vectors("prj_alpha", q_vec)
        assert vec_hits
        assert all(hit.chunk.project_id == "prj_alpha" for hit in vec_hits)

        fused = reciprocal_rank_fusion(fts_hits, vec_hits, limit=5)
        assert fused
        assert fused[0].chunk.file_id == log_rec.id

        # Verify SQL-injection project_id cannot leak prj_alpha or prj_beta vectors
        assert await vector_store.search_vectors("prj_' OR '1'='1", q_vec) == []
        assert await vector_store.search_vectors("prj_alpha", q_vec, file_ids=[]) == []

        # Verify file_ids filtering and deletion
        only_code = await vector_store.search_vectors("prj_alpha", q_vec, file_ids=[code_rec.id])
        assert all(hit.chunk.file_id == code_rec.id for hit in only_code)
        await vector_store.delete_file_vectors("prj_beta", beta_rec.id)
        assert await vector_store.search_vectors("prj_beta", q_vec) == []
        with pytest.raises(ValidationFailedError):
            await vector_store.search_vectors("prj_alpha", [0.1, 0.2])


async def test_reindex_concurrency_gate_and_storage_failure_alert(tmp_path: Path) -> None:
    """Pipeline gates CPU indexing via Semaphore(4), reindexes cleanly, and alerts on failure."""
    log_dir = tmp_path / "logs"
    manager = configure_observability(
        LoggingConfig(log_dir=log_dir, level="INFO", json_console=False, enable_file_logs=True)
    )
    try:
        blob_store = LocalBlobStore(tmp_path / "blobs")
        inner_embedder = DeterministicHashEmbedder(dimensions=TEST_DIMENSIONS)
        tracking_embedder = _TrackingEmbedder(inner_embedder, delay_seconds=0.02)
        vector_store = LanceDbVectorStore(tmp_path / "lancedb", dimensions=TEST_DIMENSIONS)
        async with SqliteDatabase(tmp_path / "lai.db") as db:
            projects = SqliteProjectRepository(db)
            await projects.create(Project(id="prj_conc", name="Concurrent"))
            await projects.create(Project(id="prj_fail", name="Fail"))
            file_repo = SqliteFileRepository(db)
            pipeline = LocalIngestionPipeline(
                blob_store=blob_store,
                file_repo=file_repo,
                embedder=tracking_embedder,
                vector_store=vector_store,
            )

            # 8 concurrent uploads must not exceed Semaphore(4)
            tasks = [
                pipeline.ingest_upload(
                    "prj_conc",
                    f"log_{i}.log",
                    "text/plain",
                    _stream_bytes(f"2026-11-17T10:00:0{i}Z INFO worker {i} ready\n".encode()),
                )
                for i in range(CONCURRENT_UPLOADS)
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == CONCURRENT_UPLOADS
            assert 1 <= tracking_embedder.max_active <= EXPECTED_SEMAPHORE_CAP

            # Reindex existing file and check missing file error
            reindexed = await pipeline.reindex_file(results[0].id)
            assert reindexed.status is FileStatus.READY
            with pytest.raises(NotFoundError):
                await pipeline.reindex_file("fil_missing")

            # Trigger indexing failure -> FAILED status + AlertCategory.STORAGE_FAILURE
            broken_store = _BrokenVectorStore()
            broken_pipeline = LocalIngestionPipeline(
                blob_store=blob_store,
                file_repo=file_repo,
                embedder=inner_embedder,
                vector_store=broken_store,
            )
            with pytest.raises(OSError, match="lancedb disk write error"):
                await broken_pipeline.ingest_upload(
                    "prj_fail",
                    "broken.log",
                    "text/plain",
                    _stream_bytes(b"2026-11-17T12:00:00Z ERROR disk error"),
                )
            assert len(broken_store.upsert_calls) == 1

            failed_files = await file_repo.list_files("prj_fail")
            assert len(failed_files) == 1
            assert failed_files[0].status is FileStatus.FAILED
            assert failed_files[0].error_message == "lancedb disk write error"

            alerts = manager.recent_alerts(limit=10)
            assert any(
                alert.category is AlertCategory.STORAGE_FAILURE and alert.component == "ingestion"
                for alert in alerts
            )
    finally:
        manager.close()
        for handler in list(logging.getLogger().handlers):
            handler.flush()
            handler.close()

    entries = [
        json.loads(line)
        for line in (log_dir / "lai.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ingested_logs = [e for e in entries if e.get("event") == "file_ingested"]
    assert ingested_logs
    assert "size_bytes" in ingested_logs[0]
    assert "chunk_count" in ingested_logs[0]
    assert "duration_ms" in ingested_logs[0]
