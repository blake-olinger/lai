"""File upload, chunking, parsing, embedding, and vector storage layer."""

from __future__ import annotations

from lai.ingestion.blob_store import DEFAULT_MAX_UPLOAD_BYTES, LocalBlobStore
from lai.ingestion.chunkers import (
    DEFAULT_CODE_CHUNK_TOKENS,
    DEFAULT_LOG_CHUNK_TOKENS,
    DEFAULT_PROSE_CHUNK_TOKENS,
    chunk_code_file,
    chunk_log_file,
    chunk_prose_file,
)
from lai.ingestion.detector import detect_file_kind, detect_language, estimate_tokens
from lai.ingestion.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DeterministicHashEmbedder,
    FastEmbedCpuAdapter,
)
from lai.ingestion.pipeline import (
    DEFAULT_MAX_CONCURRENT_INDEXING,
    DEFAULT_TARGET_CHUNK_TOKENS,
    LocalIngestionPipeline,
)
from lai.ingestion.rrf import DEFAULT_RRF_K, DEFAULT_RRF_LIMIT, reciprocal_rank_fusion
from lai.ingestion.vector_store import (
    DEFAULT_VECTOR_DIMENSIONS,
    DEFAULT_VECTOR_TABLE,
    LanceDbVectorStore,
)

__all__ = [
    "DEFAULT_CODE_CHUNK_TOKENS",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LOG_CHUNK_TOKENS",
    "DEFAULT_MAX_CONCURRENT_INDEXING",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DEFAULT_PROSE_CHUNK_TOKENS",
    "DEFAULT_RRF_K",
    "DEFAULT_RRF_LIMIT",
    "DEFAULT_TARGET_CHUNK_TOKENS",
    "DEFAULT_VECTOR_DIMENSIONS",
    "DEFAULT_VECTOR_TABLE",
    "DeterministicHashEmbedder",
    "FastEmbedCpuAdapter",
    "LanceDbVectorStore",
    "LocalBlobStore",
    "LocalIngestionPipeline",
    "chunk_code_file",
    "chunk_log_file",
    "chunk_prose_file",
    "detect_file_kind",
    "detect_language",
    "estimate_tokens",
    "reciprocal_rank_fusion",
]
