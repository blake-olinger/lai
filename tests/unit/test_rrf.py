"""Unit tests for Reciprocal Rank Fusion (RRF) and embedding adapters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import pytest

from lai.domain.errors import ValidationFailedError
from lai.domain.files import FileChunk, ScoredChunk
from lai.ingestion.embedder import DeterministicHashEmbedder, FastEmbedCpuAdapter
from lai.ingestion.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

TEST_DIM = 64
EXPECTED_RRF_K = 60
EXPECTED_TWO = 2
EXPECTED_THREE = 3


def _make_scored_chunk(
    chunk_id: str,
    score: float,
    source: Literal["fts", "vector"] = "fts",
) -> ScoredChunk:
    chunk = FileChunk(
        id=chunk_id,
        file_id="fil_1",
        project_id="prj_1",
        chunk_index=0,
        start_line=1,
        end_line=5,
        token_count=10,
        text=f"chunk {chunk_id}",
    )
    return ScoredChunk(chunk=chunk, score=score, source=source)


def test_reciprocal_rank_fusion_scoring_and_deduplication() -> None:
    """RRF combines reciprocal ranks across lists and deduplicates intra-list repeats."""
    c1 = _make_scored_chunk("chk_1", 10.0, "fts")
    c2 = _make_scored_chunk("chk_2", 9.0, "fts")
    c3 = _make_scored_chunk("chk_3", 8.0, "fts")

    v2 = _make_scored_chunk("chk_2", 0.99, "vector")
    v4 = _make_scored_chunk("chk_4", 0.95, "vector")
    v1 = _make_scored_chunk("chk_1", 0.90, "vector")

    # Duplicate c1 inside fts_hits should only count at rank 1
    fused = reciprocal_rank_fusion([c1, c1, c2, c3], [v2, v4, v1], k=EXPECTED_RRF_K, limit=10)
    assert [hit.chunk.id for hit in fused] == ["chk_2", "chk_1", "chk_4", "chk_3"]
    assert all(hit.source == "hybrid" for hit in fused)

    expected_c2 = 1.0 / (EXPECTED_RRF_K + 2) + 1.0 / (EXPECTED_RRF_K + 1)
    expected_c1 = 1.0 / (EXPECTED_RRF_K + 1) + 1.0 / (EXPECTED_RRF_K + 3)
    assert math.isclose(fused[0].score, expected_c2, rel_tol=1e-9)
    assert math.isclose(fused[1].score, expected_c1, rel_tol=1e-9)


def test_reciprocal_rank_fusion_limits_and_validation() -> None:
    """RRF respects limit, handles empty inputs, and rejects invalid k < 1."""
    hits = [_make_scored_chunk(f"chk_{i}", float(10 - i)) for i in range(5)]
    assert len(reciprocal_rank_fusion(hits, [], limit=EXPECTED_TWO)) == EXPECTED_TWO
    assert reciprocal_rank_fusion(hits, [], limit=0) == []
    assert reciprocal_rank_fusion([], []) == []

    with pytest.raises(ValidationFailedError):
        reciprocal_rank_fusion(hits, [], k=0)


class _FakeTextEmbeddingBackend:
    """In-memory fake TextEmbedding backend verifying CPU-only adapter delegation."""

    def __init__(self) -> None:
        self.embedded_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed(
        self,
        documents: Sequence[str],
        batch_size: int = 256,
    ) -> Iterable[Iterable[float]]:
        del batch_size
        self.embedded_batches.append(list(documents))
        return [[0.25, 0.75, 0.0, 0.5] for _ in documents]

    def query_embed(self, query: str) -> Iterable[Iterable[float]]:
        self.queries.append(query)
        return [[0.5, 0.5, 0.5, 0.5]]


async def test_deterministic_and_fastembed_cpu_adapters() -> None:
    """DeterministicHashEmbedder and FastEmbedCpuAdapter implement EmbedderPort cleanly."""
    hash_embedder = DeterministicHashEmbedder(dimensions=TEST_DIM)
    assert hash_embedder.dimensions == TEST_DIM
    vecs = await hash_embedder.embed_texts(
        ["database connection timeout", "css margin padding", ""]
    )
    assert len(vecs) == EXPECTED_THREE
    for vec in vecs:
        assert len(vec) == TEST_DIM
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)

    query_vec = await hash_embedder.embed_query("database connection timeout")
    sim_relevant = sum(a * b for a, b in zip(vecs[0], query_vec, strict=True))
    sim_unrelated = sum(a * b for a, b in zip(vecs[1], query_vec, strict=True))
    assert sim_relevant > sim_unrelated

    fake_backend = _FakeTextEmbeddingBackend()
    cpu_adapter = FastEmbedCpuAdapter(
        dimensions=4,
        backend_factory=lambda: fake_backend,
    )
    assert await cpu_adapter.embed_texts([]) == []
    assert fake_backend.embedded_batches == []

    doc_vecs = await cpu_adapter.embed_texts(["alpha", "beta"])
    q_vec = await cpu_adapter.embed_query("alpha")
    assert doc_vecs == [[0.25, 0.75, 0.0, 0.5], [0.25, 0.75, 0.0, 0.5]]
    assert q_vec == [0.5, 0.5, 0.5, 0.5]
    assert fake_backend.embedded_batches == [["alpha", "beta"]]
    assert fake_backend.queries == ["alpha"]
