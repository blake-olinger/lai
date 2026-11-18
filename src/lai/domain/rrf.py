"""Unified Reciprocal Rank Fusion (RRF) domain algorithm for ranked chunk retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lai.domain.errors import ValidationFailedError
from lai.domain.files import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import FileChunk

DEFAULT_RRF_K = 60
DEFAULT_RRF_LIMIT = 20
_SourceTag = Literal["vector", "fts", "hybrid", "direct"]


def reciprocal_rank_fusion(
    fts_hits: Sequence[ScoredChunk],
    vector_hits: Sequence[ScoredChunk],
    k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_RRF_LIMIT,
    *,
    preserve_source: bool = True,
) -> list[ScoredChunk]:
    """Fuse ranked FTS and vector ScoredChunks via Reciprocal Rank Fusion."""
    if k < 1:
        msg = f"RRF constant k must be >= 1, got {k}"
        raise ValidationFailedError(msg, details={"k": str(k)})
    if limit <= 0:
        return []

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks_by_id: dict[str, FileChunk] = {}
    seen_lists: dict[str, set[_SourceTag]] = {}
    first_source: dict[str, _SourceTag] = {}
    source_batches: tuple[tuple[_SourceTag, Sequence[ScoredChunk]], ...] = (
        ("fts", fts_hits),
        ("vector", vector_hits),
    )

    for default_source, source_hits in source_batches:
        seen_in_source: set[str] = set()
        rank = 0
        for hit in source_hits:
            chunk_id = hit.chunk.id
            if chunk_id in seen_in_source:
                continue
            seen_in_source.add(chunk_id)
            rank += 1
            chunks_by_id.setdefault(chunk_id, hit.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            prev_best = best_rank.get(chunk_id)
            best_rank[chunk_id] = rank if prev_best is None else min(prev_best, rank)
            seen_lists.setdefault(chunk_id, set()).add(default_source)
            first_source.setdefault(chunk_id, hit.source or default_source)

    ordered_ids = sorted(
        scores,
        key=lambda cid: (
            -scores[cid],
            best_rank[cid],
            chunks_by_id[cid].file_id,
            chunks_by_id[cid].start_line,
            cid,
        ),
    )[:limit]

    return [
        ScoredChunk(
            chunk=chunks_by_id[cid],
            score=scores[cid],
            source=(
                "hybrid" if (not preserve_source or len(seen_lists[cid]) > 1) else first_source[cid]
            ),
        )
        for cid in ordered_ids
    ]
