"""Blob storage, embedding, vector store, and ingestion protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from lai.domain.files import FileChunk, FileRecord, ScoredChunk


@runtime_checkable
class BlobStorePort(Protocol):
    """Streaming filesystem blob store for uploaded project artifacts."""

    async def write_stream(
        self,
        project_id: str,
        file_id: str,
        filename: str,
        chunks: AsyncIterator[bytes],
    ) -> tuple[Path, int, str]:
        """Persist streamed bytes and return (blob_path, size_bytes, sha256)."""
        ...

    async def read_text_range(
        self,
        blob_path: Path,
        start_line: int = 1,
        end_line: int | None = None,
        *,
        max_chars: int | None = None,
    ) -> str:
        """Read UTF-8 text lines from blob_path within [start_line, end_line]."""
        ...

    async def read_bytes(self, blob_path: Path) -> bytes:
        """Read the raw bytes of a persisted blob."""
        ...

    async def delete_blob(self, blob_path: Path) -> bool:
        """Delete a persisted blob if it exists."""
        ...


@runtime_checkable
class EmbedderPort(Protocol):
    """Local text embedding generator contract."""

    @property
    def model_name(self) -> str:
        """Return the configured embedding model identifier."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the embedding vector dimensionality."""
        ...

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense vector embeddings for document texts."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Generate a dense vector embedding for a search query."""
        ...


@runtime_checkable
class VectorStorePort(Protocol):
    """Project-scoped vector similarity index contract."""

    async def upsert_chunks(
        self,
        project_id: str,
        chunks: Sequence[FileChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update chunk embeddings within a project partition."""
        ...

    async def search_vectors(
        self,
        project_id: str,
        query_vector: Sequence[float],
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        """Return top-k semantically similar chunks scoped to a project."""
        ...

    async def delete_file_vectors(self, project_id: str, file_id: str) -> None:
        """Remove all vector entries associated with file_id."""
        ...


@runtime_checkable
class IngestionPipelinePort(Protocol):
    """End-to-end file upload, parsing, chunking, and indexing contract."""

    async def ingest_upload(
        self,
        project_id: str,
        filename: str,
        content_type: str,
        stream: AsyncIterator[bytes],
    ) -> FileRecord:
        """Stream, parse, chunk, and index an uploaded file."""
        ...

    async def reindex_file(self, file_id: str) -> FileRecord:
        """Re-parse and re-index an already stored file."""
        ...
