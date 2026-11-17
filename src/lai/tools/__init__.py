"""Guarded tool registry, policy engine, executor, and Phase 1 read plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.tools.executor import GuardedToolExecutor
from lai.tools.plugins import (
    DEFAULT_MAX_LINES_PER_CALL,
    GetFileOverviewPlugin,
    ReadFileLinesPlugin,
    SearchFilesPlugin,
)
from lai.tools.policy import DefaultToolPolicy
from lai.tools.registry import InMemoryToolRegistry

if TYPE_CHECKING:
    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.storage import BlobStorePort, EmbedderPort, VectorStorePort


def build_default_tool_registry(
    file_repo: FileRepositoryPort,
    blob_store: BlobStorePort,
    *,
    embedder: EmbedderPort | None = None,
    vector_store: VectorStorePort | None = None,
) -> InMemoryToolRegistry:
    """Create an InMemoryToolRegistry pre-populated with Phase 1 read tools."""
    return InMemoryToolRegistry(
        plugins=(
            SearchFilesPlugin(file_repo, embedder=embedder, vector_store=vector_store),
            ReadFileLinesPlugin(file_repo, blob_store),
            GetFileOverviewPlugin(file_repo),
        )
    )


__all__ = [
    "DEFAULT_MAX_LINES_PER_CALL",
    "DefaultToolPolicy",
    "GetFileOverviewPlugin",
    "GuardedToolExecutor",
    "InMemoryToolRegistry",
    "ReadFileLinesPlugin",
    "SearchFilesPlugin",
    "build_default_tool_registry",
]
