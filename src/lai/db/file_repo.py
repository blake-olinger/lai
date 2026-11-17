"""SQLite repository for uploaded files, line chunks, and safe FTS5 search."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from lai.db._codecs import clamp_limit, clamp_offset, format_dt, parse_dt
from lai.domain.errors import ValidationFailedError
from lai.domain.files import (
    CodeFileSummary,
    FileChunk,
    FileKind,
    FileRecord,
    FileStatus,
    LogFileSummary,
    LogSeverity,
    ScoredChunk,
)
from lai.ports.repositories import FileRepositoryPort

if TYPE_CHECKING:
    from collections.abc import Sequence

    import aiosqlite

    from lai.db.connection import SqliteDatabase

MAX_FTS_QUERY_TERMS = 32

_FILE_COLUMNS = (
    "id, project_id, filename, content_type, size_bytes, sha256, blob_path, "
    "kind, status, token_count, chunk_count, log_summary_json, code_summary_json, "
    "error_message, created_at, updated_at"
)
_CHUNK_COLUMNS = (
    "id, file_id, project_id, chunk_index, start_line, end_line, token_count, "
    "text, symbol_name, severity, metadata_json"
)


def sanitize_fts5_query(raw_query: str) -> str:
    """Convert arbitrary user text into a syntax-safe FTS5 MATCH expression."""
    cleaned = raw_query.replace("\x00", " ").strip()
    if not cleaned:
        return ""

    terms: list[str] = []
    seen: set[str] = set()
    for token in cleaned.split():
        if not any(ch.isalnum() for ch in token):
            continue
        if token in seen:
            continue
        seen.add(token)
        escaped = token.replace('"', '""')
        terms.append(f'"{escaped}"')
        if len(terms) >= MAX_FTS_QUERY_TERMS:
            break
    return " OR ".join(terms)


def _row_to_file(row: aiosqlite.Row) -> FileRecord:
    """Map a files SQLite row to a FileRecord domain model."""
    raw_log = cast("str | None", row["log_summary_json"])
    raw_code = cast("str | None", row["code_summary_json"])
    return FileRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        filename=str(row["filename"]),
        content_type=str(row["content_type"]),
        size_bytes=int(row["size_bytes"]),
        sha256=str(row["sha256"]),
        blob_path=str(row["blob_path"]),
        kind=FileKind(str(row["kind"])),
        status=FileStatus(str(row["status"])),
        token_count=int(row["token_count"]),
        chunk_count=int(row["chunk_count"]),
        log_summary=LogFileSummary.model_validate_json(raw_log) if raw_log is not None else None,
        code_summary=(
            CodeFileSummary.model_validate_json(raw_code) if raw_code is not None else None
        ),
        error_message=cast("str | None", row["error_message"]),
        created_at=parse_dt(str(row["created_at"])),
        updated_at=parse_dt(str(row["updated_at"])),
    )


def _row_to_chunk(row: aiosqlite.Row) -> FileChunk:
    """Map a file_chunks SQLite row to a FileChunk domain model."""
    raw_severity = cast("str | None", row["severity"])
    raw_meta = cast("dict[str, str]", json.loads(str(row["metadata_json"])))
    return FileChunk(
        id=str(row["id"]),
        file_id=str(row["file_id"]),
        project_id=str(row["project_id"]),
        chunk_index=int(row["chunk_index"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        token_count=int(row["token_count"]),
        text=str(row["text"]),
        symbol_name=cast("str | None", row["symbol_name"]),
        severity=LogSeverity(raw_severity) if raw_severity is not None else None,
        metadata=raw_meta,
    )


class SqliteFileRepository(FileRepositoryPort):
    """SQLite persistence and FTS5 search adapter implementing FileRepositoryPort."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def upsert_file(self, record: FileRecord) -> FileRecord:
        """Create or update a file metadata record."""
        log_json = record.log_summary.model_dump_json() if record.log_summary is not None else None
        code_json = (
            record.code_summary.model_dump_json() if record.code_summary is not None else None
        )
        await self._db.execute_write(
            """
            INSERT INTO files (
                id, project_id, filename, content_type, size_bytes, sha256,
                blob_path, kind, status, token_count, chunk_count,
                log_summary_json, code_summary_json, error_message,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename = excluded.filename,
                content_type = excluded.content_type,
                size_bytes = excluded.size_bytes,
                sha256 = excluded.sha256,
                blob_path = excluded.blob_path,
                kind = excluded.kind,
                status = excluded.status,
                token_count = excluded.token_count,
                chunk_count = excluded.chunk_count,
                log_summary_json = excluded.log_summary_json,
                code_summary_json = excluded.code_summary_json,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at;
            """,
            (
                record.id,
                record.project_id,
                record.filename,
                record.content_type,
                record.size_bytes,
                record.sha256,
                record.blob_path,
                record.kind.value,
                record.status.value,
                record.token_count,
                record.chunk_count,
                log_json,
                code_json,
                record.error_message,
                format_dt(record.created_at),
                format_dt(record.updated_at),
            ),
        )
        return record

    async def get_file(self, file_id: str) -> FileRecord | None:
        """Fetch a file metadata record by ID."""
        row = await self._db.fetch_one(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE id = ?;",  # noqa: S608
            (file_id,),
        )
        return _row_to_file(row) if row is not None else None

    async def list_files(
        self,
        project_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[FileRecord]:
        """List files belonging to a project."""
        rows = await self._db.fetch_all(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE project_id = ? "  # noqa: S608
            "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?;",
            (project_id, clamp_limit(limit), clamp_offset(offset)),
        )
        return [_row_to_file(row) for row in rows]

    async def replace_chunks(self, file_id: str, chunks: Sequence[FileChunk]) -> None:
        """Atomically replace all indexed chunks and FTS5 entries for a file."""
        for chunk in chunks:
            if chunk.file_id != file_id:
                msg = f"Chunk {chunk.id!r} belongs to {chunk.file_id!r}, expected {file_id!r}."
                raise ValidationFailedError(msg)

        chunk_rows = [
            (
                chunk.id,
                chunk.file_id,
                chunk.project_id,
                chunk.chunk_index,
                chunk.start_line,
                chunk.end_line,
                chunk.token_count,
                chunk.text,
                chunk.symbol_name,
                chunk.severity.value if chunk.severity is not None else None,
                json.dumps(chunk.metadata),
            )
            for chunk in chunks
        ]
        fts_rows = [
            (
                chunk.text,
                chunk.symbol_name,
                chunk.id,
                chunk.file_id,
                chunk.project_id,
                chunk.severity.value if chunk.severity is not None else None,
            )
            for chunk in chunks
        ]

        async with self._db.write_transaction() as conn:
            await conn.execute("DELETE FROM chunks_fts WHERE file_id = ?;", (file_id,))
            await conn.execute("DELETE FROM file_chunks WHERE file_id = ?;", (file_id,))
            if chunk_rows:
                await conn.executemany(
                    """
                    INSERT INTO file_chunks (
                        id, file_id, project_id, chunk_index, start_line, end_line,
                        token_count, text, symbol_name, severity, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    chunk_rows,
                )
                await conn.executemany(
                    """
                    INSERT INTO chunks_fts (
                        content, symbol_name, chunk_id, file_id, project_id, severity
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    fts_rows,
                )

    async def list_chunks(
        self,
        file_id: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[FileChunk]:
        """Return ordered chunks for a file, optionally overlapping [start_line, end_line]."""
        rows = await self._db.fetch_all(
            f"SELECT {_CHUNK_COLUMNS} FROM file_chunks WHERE file_id = ? "  # noqa: S608
            "AND (? IS NULL OR end_line >= ?) "
            "AND (? IS NULL OR start_line <= ?) "
            "ORDER BY chunk_index ASC, start_line ASC;",
            (file_id, start_line, start_line, end_line, end_line),
        )
        return [_row_to_chunk(row) for row in rows]

    async def search_fts(
        self,
        project_id: str,
        query: str,
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        """Run BM25 full-text search across chunks scoped to a project."""
        safe_limit = clamp_limit(limit)
        if safe_limit == 0:
            return []
        if file_ids is not None and len(file_ids) == 0:
            return []

        fts_query = sanitize_fts5_query(query)
        if not fts_query:
            return []

        file_filter_json = json.dumps(list(file_ids)) if file_ids is not None else None
        rows = await self._db.fetch_all(
            """
            SELECT
                fc.id,
                fc.file_id,
                fc.project_id,
                fc.chunk_index,
                fc.start_line,
                fc.end_line,
                fc.token_count,
                fc.text,
                fc.symbol_name,
                fc.severity,
                fc.metadata_json,
                -bm25(chunks_fts) AS relevance_score
            FROM chunks_fts
            JOIN file_chunks AS fc ON fc.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
              AND chunks_fts.project_id = ?
              AND fc.project_id = ?
              AND (? IS NULL OR fc.file_id IN (SELECT value FROM json_each(?)))
            ORDER BY bm25(chunks_fts) ASC, fc.chunk_index ASC
            LIMIT ?;
            """,
            (fts_query, project_id, project_id, file_filter_json, file_filter_json, safe_limit),
        )
        return [
            ScoredChunk(
                chunk=_row_to_chunk(row),
                score=float(row["relevance_score"]),
                source="fts",
            )
            for row in rows
        ]

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file record, its chunks, and its FTS5 entries."""
        async with self._db.write_transaction() as conn:
            await conn.execute("DELETE FROM chunks_fts WHERE file_id = ?;", (file_id,))
            cursor = await conn.execute("DELETE FROM files WHERE id = ?;", (file_id,))
            return cursor.rowcount > 0
