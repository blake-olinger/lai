"""Phase 1 search_files tool plugin combining FTS and optional vector search."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

from lai.domain.errors import ValidationFailedError
from lai.domain.rrf import reciprocal_rank_fusion
from lai.domain.tools import PermissionTier, ToolResult, ToolSpec
from lai.ports.tools import ToolPluginPort
from lai.tools.plugins._scoping import (
    optional_int,
    require_non_empty_str,
    validate_allowed_file_id,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import FileRecord, ScoredChunk
    from lai.domain.tools import ToolExecutionContext
    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.storage import EmbedderPort, VectorStorePort

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50


def _parse_file_ids(arguments: dict[str, Any]) -> tuple[str, ...]:
    """Validate optional file_ids list argument."""
    raw: object = arguments.get("file_ids")
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        msg = "Argument 'file_ids' must be a list of file IDs."
        raise ValidationFailedError(msg, details={"field": "file_ids"})
    items = cast("Sequence[object]", raw)
    parsed: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            msg = "Every entry in 'file_ids' must be a non-empty string."
            raise ValidationFailedError(msg, details={"field": "file_ids"})
        parsed.append(item.strip())
    return tuple(dict.fromkeys(parsed))


class SearchFilesPlugin(ToolPluginPort):
    """Read-only project- and file-scoped search over indexed file chunks."""

    def __init__(
        self,
        file_repo: FileRepositoryPort,
        *,
        embedder: EmbedderPort | None = None,
        vector_store: VectorStorePort | None = None,
        max_limit: int = _MAX_LIMIT,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._file_repo = file_repo
        self._embedder = embedder
        self._vector_store = vector_store
        self._max_limit = max(max_limit, 1)
        self._spec = ToolSpec(
            name="search_files",
            description=(
                "Search indexed chunks across project files using keyword FTS and "
                "optional semantic similarity, strictly scoped to the active project."
            ),
            permission_tier=PermissionTier.READ,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "file_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": self._max_limit},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            requires_approval_by_default=False,
            timeout_seconds=timeout_seconds,
        )

    @property
    def spec(self) -> ToolSpec:
        """Return the declarative specification for search_files."""
        return self._spec

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute project- and session-scoped full-text / hybrid search."""
        query = require_non_empty_str(arguments, "query")
        requested_file_ids = _parse_file_ids(arguments)
        requested_limit = optional_int(arguments, "limit", _DEFAULT_LIMIT)
        if requested_limit is None or requested_limit < 1:
            msg = "Argument 'limit' must be >= 1."
            raise ValidationFailedError(msg, details={"field": "limit"})
        limit = min(requested_limit, self._max_limit)

        for fid in requested_file_ids:
            validate_allowed_file_id(fid, context)

        scoped_file_ids: tuple[str, ...] | None = (
            requested_file_ids
            if requested_file_ids
            else (context.allowed_file_ids if context.allowed_file_ids else None)
        )

        ranked = await self._run_search(
            project_id=context.project_id,
            query=query,
            file_ids=scoped_file_ids,
            limit=limit,
        )
        filtered = await self._enforce_isolation_and_resolve_files(
            ranked=ranked,
            context=context,
            requested_file_ids=frozenset(requested_file_ids),
        )
        truncated = len(filtered) > limit
        top_items = filtered[:limit]

        payload = {
            "query": query,
            "count": len(top_items),
            "results": [
                {
                    "chunk_id": item.chunk.id,
                    "file_id": item.chunk.file_id,
                    "filename": file_rec.filename if file_rec is not None else None,
                    "start_line": item.chunk.start_line,
                    "end_line": item.chunk.end_line,
                    "score": item.score,
                    "source": item.source,
                    "symbol_name": item.chunk.symbol_name,
                    "severity": (
                        item.chunk.severity.value if item.chunk.severity is not None else None
                    ),
                    "text": item.chunk.text,
                }
                for item, file_rec in top_items
            ],
        }
        return ToolResult(
            tool_name=self._spec.name,
            success=True,
            output=json.dumps(payload),
            truncated=truncated,
        )

    async def _run_search(
        self,
        *,
        project_id: str,
        query: str,
        file_ids: tuple[str, ...] | None,
        limit: int,
    ) -> list[ScoredChunk]:
        """Run FTS search and fuse with vector search when embedder+vector_store exist."""
        if self._embedder is not None and self._vector_store is not None:
            fts_task = self._file_repo.search_fts(
                project_id,
                query,
                file_ids=file_ids,
                limit=limit,
            )
            query_vector, fts_results = await asyncio.gather(
                self._embedder.embed_query(query),
                fts_task,
            )
            vec_results = await self._vector_store.search_vectors(
                project_id,
                query_vector,
                file_ids=file_ids,
                limit=limit,
            )
            return reciprocal_rank_fusion(
                fts_results,
                vec_results,
                limit=max(limit, len(fts_results) + len(vec_results)),
            )

        fts_only = await self._file_repo.search_fts(
            project_id,
            query,
            file_ids=file_ids,
            limit=limit,
        )
        return sorted(fts_only, key=lambda item: item.score, reverse=True)

    async def _enforce_isolation_and_resolve_files(
        self,
        *,
        ranked: Sequence[ScoredChunk],
        context: ToolExecutionContext,
        requested_file_ids: frozenset[str],
    ) -> list[tuple[ScoredChunk, FileRecord | None]]:
        """Drop any out-of-scope chunks and look up file metadata once per file_id."""
        allowed_set = frozenset(context.allowed_file_ids)
        file_cache: dict[str, FileRecord | None] = {}
        accepted: list[tuple[ScoredChunk, FileRecord | None]] = []

        for item in ranked:
            chunk = item.chunk
            if chunk.project_id != context.project_id:
                continue
            if allowed_set and chunk.file_id not in allowed_set:
                continue
            if requested_file_ids and chunk.file_id not in requested_file_ids:
                continue

            if chunk.file_id not in file_cache:
                file_cache[chunk.file_id] = await self._file_repo.get_file(chunk.file_id)
            file_rec = file_cache[chunk.file_id]
            if file_rec is not None and file_rec.project_id != context.project_id:
                continue
            accepted.append((item, file_rec))
        return accepted
