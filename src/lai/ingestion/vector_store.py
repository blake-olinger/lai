"""LanceDB-backed project-scoped vector similarity store for file chunks."""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import lancedb
import lancedb.pydantic as ldb_pydantic
from lancedb.pydantic import LanceModel
from pydantic import create_model

from lai.domain.errors import ValidationFailedError
from lai.domain.files import FileChunk, LogSeverity, ScoredChunk
from lai.ports.storage import VectorStorePort

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEFAULT_VECTOR_TABLE = "file_chunks"
DEFAULT_VECTOR_DIMENSIONS = 384
_SAFE_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _TableListingLike(Protocol):
    @property
    def tables(self) -> list[str]: ...


class _VectorQueryLike(Protocol):
    def distance_type(self, distance_type: str) -> _VectorQueryLike: ...

    def where(self, where: str, prefilter: bool = ...) -> _VectorQueryLike: ...

    def limit(self, limit: int | None) -> _VectorQueryLike: ...

    def to_list(self) -> list[dict[str, object]]: ...


class _MergeInsertLike(Protocol):
    def when_matched_update_all(self) -> _MergeInsertLike: ...

    def when_not_matched_insert_all(self) -> _MergeInsertLike: ...

    def execute(self, data: object) -> object: ...


class _LanceTableLike(Protocol):
    def merge_insert(self, on: str) -> _MergeInsertLike: ...

    def search(self, query: Sequence[float]) -> _VectorQueryLike: ...

    def delete(self, where: str) -> object: ...


class _LanceDbLike(Protocol):
    def list_tables(self) -> _TableListingLike: ...

    def open_table(self, name: str) -> _LanceTableLike: ...

    def create_table(
        self,
        name: str,
        *,
        schema: object,
        exist_ok: bool = ...,
    ) -> _LanceTableLike: ...


def _sql_quote(value: str, field_name: str) -> str:
    """Safely SQL-quote a non-empty identifier literal for LanceDB DataFusion filters."""
    if not value or "\x00" in value:
        msg = f"Invalid {field_name} for vector filter: {value!r}"
        raise ValidationFailedError(msg, details={"field": field_name})
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _build_where_clause(project_id: str, file_ids: Sequence[str] | None) -> str:
    """Construct a safe SQL filter predicate scoped to project_id and optional file_ids."""
    clause = f"project_id = {_sql_quote(project_id, 'project_id')}"
    if file_ids is not None:
        quoted = ", ".join(_sql_quote(fid, "file_id") for fid in file_ids)
        clause = f"{clause} AND file_id IN ({quoted})"
    return clause


def _build_lancedb_schema(dimensions: int) -> type[LanceModel]:
    """Build a LanceModel schema dynamically for the configured vector dimensionality."""
    vector_factory = cast(
        "Callable[[int], object]",
        ldb_pydantic.Vector,
    )
    vector_type: Any = vector_factory(dimensions)
    return create_model(
        "FileChunkVectorRow",
        __base__=LanceModel,
        id=(str, ...),
        file_id=(str, ...),
        project_id=(str, ...),
        chunk_index=(int, ...),
        start_line=(int, ...),
        end_line=(int, ...),
        token_count=(int, ...),
        text=(str, ...),
        symbol_name=(str | None, None),
        severity=(str | None, None),
        metadata_json=(str, "{}"),
        vector=(vector_type, ...),
    )


def _row_to_scored_chunk(row: dict[str, object]) -> ScoredChunk:
    """Convert a LanceDB query result row back into a domain ScoredChunk."""
    raw_meta = row.get("metadata_json")
    metadata: dict[str, str] = {}
    if isinstance(raw_meta, str) and raw_meta:
        parsed_meta: object = json.loads(raw_meta)
        if isinstance(parsed_meta, dict):
            raw_dict = cast("dict[object, object]", parsed_meta)
            metadata = {str(k): str(v) for k, v in raw_dict.items()}

    raw_sev = row.get("severity")
    severity = LogSeverity(raw_sev) if isinstance(raw_sev, str) and raw_sev else None
    raw_symbol = row.get("symbol_name")
    symbol_name = raw_symbol if isinstance(raw_symbol, str) and raw_symbol else None

    chunk = FileChunk(
        id=str(row["id"]),
        file_id=str(row["file_id"]),
        project_id=str(row["project_id"]),
        chunk_index=int(cast("int", row["chunk_index"])),
        start_line=int(cast("int", row["start_line"])),
        end_line=int(cast("int", row["end_line"])),
        token_count=int(cast("int", row["token_count"])),
        text=str(row["text"]),
        symbol_name=symbol_name,
        severity=severity,
        metadata=metadata,
    )

    raw_dist = row.get("_distance")
    distance = (
        float(raw_dist)
        if isinstance(raw_dist, int | float) and not math.isnan(float(raw_dist))
        else 1.0
    )
    score = max(0.0, 1.0 - distance)
    return ScoredChunk(chunk=chunk, score=score, source="vector")


class LanceDbVectorStore(VectorStorePort):
    """Project-partitioned LanceDB vector index for FileChunk embeddings."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        dimensions: int = DEFAULT_VECTOR_DIMENSIONS,
        table_name: str = DEFAULT_VECTOR_TABLE,
    ) -> None:
        if dimensions < 1:
            msg = f"dimensions must be >= 1, got {dimensions}"
            raise ValidationFailedError(msg, details={"dimensions": str(dimensions)})
        if not _SAFE_TABLE_NAME_RE.match(table_name):
            msg = f"Invalid LanceDB table name: {table_name!r}"
            raise ValidationFailedError(msg, details={"table_name": table_name})

        self._db_path = Path(db_path).resolve()
        self._dimensions = dimensions
        self._table_name = table_name
        self._schema = _build_lancedb_schema(dimensions)
        self._lock = threading.Lock()
        self._db: _LanceDbLike | None = None

    def _connect_sync(self) -> _LanceDbLike:
        """Connect lazily to the on-disk LanceDB directory."""
        if self._db is None:
            self._db_path.mkdir(parents=True, exist_ok=True)
            self._db = cast("_LanceDbLike", lancedb.connect(self._db_path))
        return self._db

    def _get_or_create_table_sync(self) -> _LanceTableLike:
        """Open or create the vector chunks table."""
        db = self._connect_sync()
        if self._table_name in db.list_tables().tables:
            return db.open_table(self._table_name)
        return db.create_table(self._table_name, schema=self._schema, exist_ok=True)

    def _open_existing_table_sync(self) -> _LanceTableLike | None:
        """Open the vector chunks table only if it already exists."""
        db = self._connect_sync()
        if self._table_name not in db.list_tables().tables:
            return None
        return db.open_table(self._table_name)

    def _upsert_sync(self, rows: list[dict[str, object]]) -> None:
        """Idempotently upsert chunk vector rows under the store lock."""
        with self._lock:
            table = self._get_or_create_table_sync()
            (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )

    def _search_sync(
        self,
        query_vector: list[float],
        where_clause: str,
        limit: int,
    ) -> list[ScoredChunk]:
        """Run a prefiltered cosine similarity search against LanceDB."""
        with self._lock:
            table = self._open_existing_table_sync()
            if table is None:
                return []
            raw_rows = (
                table.search(query_vector)
                .distance_type("cosine")
                .where(where_clause, prefilter=True)
                .limit(limit)
                .to_list()
            )
        return [_row_to_scored_chunk(row) for row in raw_rows]

    def _delete_sync(self, where_clause: str) -> None:
        """Delete matching rows from the vector table if it exists."""
        with self._lock:
            table = self._open_existing_table_sync()
            if table is None:
                return
            table.delete(where_clause)

    async def upsert_chunks(
        self,
        project_id: str,
        chunks: Sequence[FileChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update chunk embeddings scoped to project_id."""
        _sql_quote(project_id, "project_id")
        if len(chunks) != len(vectors):
            msg = "chunks and vectors must have identical length."
            raise ValidationFailedError(
                msg,
                details={"chunks": str(len(chunks)), "vectors": str(len(vectors))},
            )
        if not chunks:
            return

        rows: list[dict[str, object]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.project_id != project_id:
                msg = (
                    f"Chunk {chunk.id} belongs to project {chunk.project_id!r}, "
                    f"expected {project_id!r}"
                )
                raise ValidationFailedError(msg, details={"chunk_id": chunk.id})
            if len(vector) != self._dimensions:
                msg = (
                    f"Vector dimension mismatch for chunk {chunk.id}: "
                    f"expected {self._dimensions}, got {len(vector)}"
                )
                raise ValidationFailedError(msg, details={"chunk_id": chunk.id})
            rows.append(
                {
                    "id": chunk.id,
                    "file_id": chunk.file_id,
                    "project_id": chunk.project_id,
                    "chunk_index": chunk.chunk_index,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                    "symbol_name": chunk.symbol_name,
                    "severity": chunk.severity.value if chunk.severity is not None else None,
                    "metadata_json": json.dumps(chunk.metadata, sort_keys=True),
                    "vector": [float(v) for v in vector],
                }
            )

        await asyncio.to_thread(self._upsert_sync, rows)

    async def search_vectors(
        self,
        project_id: str,
        query_vector: Sequence[float],
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        """Return top-k semantically similar chunks filtered by project_id and file_ids."""
        if len(query_vector) != self._dimensions:
            msg = (
                f"Query vector dimension mismatch: expected {self._dimensions}, "
                f"got {len(query_vector)}"
            )
            raise ValidationFailedError(msg, details={"dimensions": str(len(query_vector))})
        where_clause = _build_where_clause(project_id, file_ids)
        if limit <= 0 or (file_ids is not None and len(file_ids) == 0):
            return []
        return await asyncio.to_thread(
            self._search_sync,
            [float(v) for v in query_vector],
            where_clause,
            limit,
        )

    async def delete_file_vectors(self, project_id: str, file_id: str) -> None:
        """Delete all vector rows for file_id within project_id."""
        where_clause = (
            f"project_id = {_sql_quote(project_id, 'project_id')} "
            f"AND file_id = {_sql_quote(file_id, 'file_id')}"
        )
        await asyncio.to_thread(self._delete_sync, where_clause)
