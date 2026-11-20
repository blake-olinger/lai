"""End-to-end async ingestion pipeline gated by a dedicated CPU semaphore."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.base import utc_now
from lai.domain.errors import LaiError, NotFoundError, ValidationFailedError
from lai.domain.files import (
    CodeFileSummary,
    FileChunk,
    FileKind,
    FileRecord,
    FileStatus,
    LogFileSummary,
)
from lai.domain.ids import new_file_id
from lai.domain.observability import AlertCategory
from lai.ingestion.chunkers.code import chunk_code_file
from lai.ingestion.chunkers.logs import chunk_log_file
from lai.ingestion.chunkers.prose import chunk_prose_file
from lai.ingestion.detector import detect_file_kind, estimate_tokens
from lai.observability.logging import get_component_logger
from lai.ports.storage import (
    BlobStorePort,
    EmbedderPort,
    IngestionPipelinePort,
    VectorStorePort,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from lai.ports.repositories import FileRepositoryPort

DEFAULT_MAX_CONCURRENT_INDEXING = 4
DEFAULT_TARGET_CHUNK_TOKENS = 600
_SAMPLE_CHARS_LEN = 8192
_MIN_CHUNK_TOKENS = 32


def _chunk_by_kind(
    record: FileRecord,
    kind: FileKind,
    text: str,
    target_tokens: int,
) -> tuple[list[FileChunk], LogFileSummary | None, CodeFileSummary | None]:
    """Dispatch chunking and deterministic summary extraction based on FileKind."""
    if kind is FileKind.LOG:
        chunks, log_summary = chunk_log_file(
            file_id=record.id,
            project_id=record.project_id,
            text=text,
            target_tokens=target_tokens,
        )
        return chunks, log_summary, None
    if kind is FileKind.CODE:
        chunks, code_summary = chunk_code_file(
            file_id=record.id,
            project_id=record.project_id,
            filename=record.filename,
            text=text,
            target_tokens=target_tokens,
        )
        return chunks, None, code_summary
    chunks = chunk_prose_file(
        file_id=record.id,
        project_id=record.project_id,
        kind=kind,
        text=text,
        target_tokens=target_tokens,
    )
    return chunks, None, None


def _decode_and_chunk(
    record: FileRecord,
    raw_bytes: bytes,
    target_chunk_tokens: int,
) -> tuple[FileKind, int, list[FileChunk], LogFileSummary | None, CodeFileSummary | None]:
    """Decode uploaded bytes, detect kind, count tokens, and chunk off the event loop."""
    text = raw_bytes.decode("utf-8", errors="replace")
    del raw_bytes
    kind = detect_file_kind(record.filename, record.content_type, text[:_SAMPLE_CHARS_LEN])
    token_count = estimate_tokens(text)
    chunks, log_summary, code_summary = _chunk_by_kind(record, kind, text, target_chunk_tokens)
    return kind, token_count, chunks, log_summary, code_summary


class LocalIngestionPipeline(IngestionPipelinePort):
    """Streaming upload, classification, chunking, CPU embedding, and indexing pipeline."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        blob_store: BlobStorePort,
        file_repo: FileRepositoryPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_INDEXING,
        target_chunk_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
    ) -> None:
        if max_concurrent_jobs < 1:
            msg = f"max_concurrent_jobs must be >= 1, got {max_concurrent_jobs}"
            raise ValidationFailedError(
                msg,
                details={"max_concurrent_jobs": str(max_concurrent_jobs)},
            )
        if target_chunk_tokens < _MIN_CHUNK_TOKENS:
            msg = f"target_chunk_tokens must be >= {_MIN_CHUNK_TOKENS}, got {target_chunk_tokens}"
            raise ValidationFailedError(
                msg,
                details={"target_chunk_tokens": str(target_chunk_tokens)},
            )
        self._blob_store = blob_store
        self._file_repo = file_repo
        self._embedder = embedder
        self._vector_store = vector_store
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._max_concurrent_jobs = max_concurrent_jobs
        self._target_chunk_tokens = target_chunk_tokens
        self._logger = get_component_logger("ingestion")

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Return the CPU indexing concurrency semaphore."""
        return self._semaphore

    async def ingest_upload(
        self,
        project_id: str,
        filename: str,
        content_type: str,
        stream: AsyncIterator[bytes],
    ) -> FileRecord:
        """Stream upload bytes to blob storage, persist metadata, and index file chunks."""
        file_id = new_file_id()
        try:
            blob_path, size_bytes, sha256 = await self._blob_store.write_stream(
                project_id,
                file_id,
                filename,
                stream,
            )
        except OSError:
            self._logger.exception(
                "blob_write_failed",
                project_id=project_id,
                file_id=file_id,
                filename=filename,
                alert_category=AlertCategory.STORAGE_FAILURE.value,
            )
            raise

        initial_kind = detect_file_kind(filename, content_type, "")
        record = FileRecord(
            id=file_id,
            project_id=project_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            blob_path=str(blob_path),
            kind=initial_kind,
            status=FileStatus.PROCESSING,
        )
        try:
            persisted = await self._file_repo.upsert_file(record)
        except Exception:
            self._logger.exception(
                "file_metadata_persist_failed",
                project_id=project_id,
                file_id=file_id,
                filename=filename,
                size_bytes=size_bytes,
                alert_category=AlertCategory.STORAGE_FAILURE.value,
            )
            raise

        return await self._run_gated_index(persisted)

    async def reindex_file(self, file_id: str) -> FileRecord:
        """Re-parse, re-chunk, and re-index an existing uploaded project file."""
        existing = await self._file_repo.get_file(file_id)
        if existing is None:
            msg = f"File not found: {file_id}"
            raise NotFoundError(msg, details={"file_id": file_id})

        processing = existing.model_copy(
            update={
                "status": FileStatus.PROCESSING,
                "error_message": None,
                "updated_at": utc_now(),
            }
        )
        persisted = await self._file_repo.upsert_file(processing)
        return await self._run_gated_index(persisted)

    async def _run_gated_index(self, record: FileRecord) -> FileRecord:
        """Execute CPU parsing, chunking, embedding, and indexing under the semaphore."""
        async with self._semaphore:
            started = time.perf_counter()
            try:
                ready = await self._index_record(record)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._logger.exception(
                    "file_indexing_failed",
                    project_id=record.project_id,
                    file_id=record.id,
                    filename=record.filename,
                    size_bytes=record.size_bytes,
                    duration_ms=round(elapsed_ms, 2),
                    alert_category=AlertCategory.STORAGE_FAILURE.value,
                )
                await self._mark_failed(record, str(exc))
                raise

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._logger.info(
                "file_ingested",
                project_id=ready.project_id,
                file_id=ready.id,
                filename=ready.filename,
                kind=ready.kind.value,
                size_bytes=ready.size_bytes,
                token_count=ready.token_count,
                chunk_count=ready.chunk_count,
                duration_ms=round(elapsed_ms, 2),
            )
            return ready

    async def _index_record(self, record: FileRecord) -> FileRecord:
        """Read blob bytes, classify, chunk, embed, and update SQLite + LanceDB indexes."""
        raw_bytes = await self._blob_store.read_bytes(Path(record.blob_path))
        kind, token_count, chunks, log_summary, code_summary = await asyncio.to_thread(
            _decode_and_chunk,
            record,
            raw_bytes,
            self._target_chunk_tokens,
        )
        del raw_bytes

        vectors = await self._embedder.embed_texts([c.text for c in chunks]) if chunks else []
        await self._file_repo.replace_chunks(record.id, chunks)
        await self._vector_store.delete_file_vectors(record.project_id, record.id)
        if chunks:
            await self._vector_store.upsert_chunks(record.project_id, chunks, vectors)

        ready = record.model_copy(
            update={
                "kind": kind,
                "status": FileStatus.READY,
                "token_count": token_count,
                "chunk_count": len(chunks),
                "log_summary": log_summary,
                "code_summary": code_summary,
                "error_message": None,
                "updated_at": utc_now(),
            }
        )
        return await self._file_repo.upsert_file(ready)

    async def _mark_failed(self, record: FileRecord, error_message: str) -> None:
        """Best-effort update of FileRecord to FAILED when indexing raises."""
        failed = record.model_copy(
            update={
                "status": FileStatus.FAILED,
                "error_message": error_message[:500],
                "updated_at": utc_now(),
            }
        )
        try:
            await self._file_repo.upsert_file(failed)
        except (OSError, LaiError, RuntimeError, ValueError):
            self._logger.warning(
                "file_mark_failed_skipped",
                project_id=record.project_id,
                file_id=record.id,
            )
