"""Reciprocal Rank Fusion (RRF) combining FTS5 lexical and vector semantic hits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.errors import ValidationFailedError
from lai.domain.files import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import FileChunk

DEFAULT_RRF_K = 60
DEFAULT_RRF_LIMIT = 20


def reciprocal_rank_fusion(
    fts_hits: Sequence[ScoredChunk],
    vector_hits: Sequence[ScoredChunk],
    k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_RRF_LIMIT,
) -> list[ScoredChunk]:
    """Fuse ranked FTS and vector ScoredChunks using Reciprocal Rank Fusion."""
    if k < 1:
        msg = f"RRF constant k must be >= 1, got {k}"
        raise ValidationFailedError(msg, details={"k": str(k)})
    if limit <= 0:
        return []

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks_by_id: dict[str, FileChunk] = {}

    for source_hits in (fts_hits, vector_hits):
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
            source="hybrid",
        )
        for cid in ordered_ids
    ]
