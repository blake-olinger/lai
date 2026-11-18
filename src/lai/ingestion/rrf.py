"""Compatibility re-export of domain Reciprocal Rank Fusion for ingestion callers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.rrf import (
    DEFAULT_RRF_K,
    DEFAULT_RRF_LIMIT,
)
from lai.domain.rrf import (
    reciprocal_rank_fusion as _domain_reciprocal_rank_fusion,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import ScoredChunk

__all__ = ["DEFAULT_RRF_K", "DEFAULT_RRF_LIMIT", "reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    fts_hits: Sequence[ScoredChunk],
    vector_hits: Sequence[ScoredChunk],
    k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_RRF_LIMIT,
) -> list[ScoredChunk]:
    """Fuse ranked FTS and vector ScoredChunks via domain RRF with 'hybrid' source tags."""
    return _domain_reciprocal_rank_fusion(
        fts_hits,
        vector_hits,
        k=k,
        limit=limit,
        preserve_source=False,
    )
