"""Local CPU-only FastEmbed adapter and deterministic offline hash embedder."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from typing import TYPE_CHECKING, Protocol, cast

from fastembed import TextEmbedding

from lai.domain.errors import ValidationFailedError
from lai.ports.storage import EmbedderPort

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSIONS = 384
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_NGRAM_SIZE = 3


class _TextEmbeddingBackend(Protocol):
    """Minimal typing protocol for fastembed.TextEmbedding CPU backend."""

    def embed(
        self,
        documents: Sequence[str],
        batch_size: int = 256,
    ) -> Iterable[Iterable[float]]:
        """Generate vector embeddings for document texts."""
        ...

    def query_embed(self, query: str) -> Iterable[Iterable[float]]:
        """Generate vector embedding for a single query."""
        ...


class FastEmbedCpuAdapter(EmbedderPort):
    """CPU-only fastembed TextEmbedding adapter using 0 MB GPU VRAM."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        cache_dir: Path | str | None = None,
        threads: int | None = None,
        batch_size: int = 64,
        local_files_only: bool = False,
        backend_factory: Callable[[], _TextEmbeddingBackend] | None = None,
    ) -> None:
        if dimensions < 1:
            msg = f"dimensions must be >= 1, got {dimensions}"
            raise ValidationFailedError(msg, details={"dimensions": str(dimensions)})
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValidationFailedError(msg, details={"batch_size": str(batch_size)})

        self._model_name = model_name
        self._dimensions = dimensions
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._threads = threads
        self._batch_size = batch_size
        self._local_files_only = local_files_only
        self._backend_factory = backend_factory
        self._lock = threading.Lock()
        self._backend: _TextEmbeddingBackend | None = None

    @property
    def model_name(self) -> str:
        """Return the configured FastEmbed model identifier."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return the configured embedding vector dimensionality."""
        return self._dimensions

    def _get_backend(self) -> _TextEmbeddingBackend:
        """Lazily initialize the CPU-pinned FastEmbed model once under a thread lock."""
        with self._lock:
            if self._backend is None:
                if self._backend_factory is not None:
                    self._backend = self._backend_factory()
                elif self._local_files_only:
                    instance = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=self._cache_dir,
                        threads=self._threads,
                        providers=["CPUExecutionProvider"],
                        local_files_only=True,
                    )
                    self._backend = cast("_TextEmbeddingBackend", instance)
                else:
                    instance = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=self._cache_dir,
                        threads=self._threads,
                        providers=["CPUExecutionProvider"],
                    )
                    self._backend = cast("_TextEmbeddingBackend", instance)
            return self._backend

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Run CPU embedding generation synchronously inside a worker thread."""
        backend = self._get_backend()
        raw_vectors = backend.embed(texts, batch_size=self._batch_size)
        return [[float(val) for val in vec] for vec in raw_vectors]

    def _embed_query_sync(self, query: str) -> list[float]:
        """Run CPU query embedding generation synchronously inside a worker thread."""
        backend = self._get_backend()
        for vec in backend.query_embed(query):
            return [float(val) for val in vec]
        msg = "FastEmbed query_embed returned no vectors."
        raise RuntimeError(msg)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense CPU embeddings for document chunks via asyncio.to_thread."""
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_texts_sync, list(texts))

    async def embed_query(self, query: str) -> list[float]:
        """Generate a dense CPU embedding for a search query via asyncio.to_thread."""
        return await asyncio.to_thread(self._embed_query_sync, query)


def _hash_feature(feature: str, dimensions: int) -> tuple[int, float]:
    """Map a text feature deterministically to (bucket_index, signed_weight)."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    raw_int = int.from_bytes(digest, "little", signed=False)
    bucket = (raw_int >> 1) % dimensions
    sign = 1.0 if (raw_int & 1) == 0 else -1.0
    return bucket, sign


def _l2_normalize(vector: list[float]) -> list[float]:
    """Normalize vector to unit L2 length."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        out = [0.0] * len(vector)
        out[0] = 1.0
        return out
    return [v / norm for v in vector]


def _embed_deterministic(text: str, dimensions: int) -> list[float]:
    """Compute a deterministic L2-normalized feature-hashed vector for text."""
    vec = [0.0] * dimensions
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)

    for tok in tokens:
        idx, sign = _hash_feature(f"w:{tok}", dimensions)
        vec[idx] += sign * 2.0
        if len(tok) >= _NGRAM_SIZE:
            for i in range(len(tok) - _NGRAM_SIZE + 1):
                g_idx, g_sign = _hash_feature(f"g:{tok[i : i + _NGRAM_SIZE]}", dimensions)
                vec[g_idx] += g_sign

    if not any(vec):
        idx, sign = _hash_feature(f"raw:{text}", dimensions)
        vec[idx] = sign
    return _l2_normalize(vec)


class DeterministicHashEmbedder(EmbedderPort):
    """Instant, zero-download deterministic embedder for offline unit & integration tests."""

    def __init__(
        self,
        *,
        model_name: str = "deterministic-hash-384",
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        if dimensions < 1:
            msg = f"dimensions must be >= 1, got {dimensions}"
            raise ValidationFailedError(msg, details={"dimensions": str(dimensions)})
        self._model_name = model_name
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        """Return the deterministic test embedder identifier."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Deterministically embed a sequence of texts."""
        return [_embed_deterministic(text, self._dimensions) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Deterministically embed a search query."""
        return _embed_deterministic(query, self._dimensions)
