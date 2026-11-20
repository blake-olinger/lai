"""Deterministic context-mode routing across Direct, Retrieval, and Map-Reduce."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.chat import ContextMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import FileRecord

_MULTI_FILE_HOLISTIC_CUES: tuple[str, ...] = (
    "all errors",
    "every file",
    "audit all",
    "across the log",
    "count all",
    "full summary",
)

_HOLISTIC_CUES: tuple[str, ...] = (
    *_MULTI_FILE_HOLISTIC_CUES,
    "summarize",
    "entire",
    "whole",
    "timeline",
    "overview",
)


def classify_context_mode(  # noqa: PLR0913
    requested_mode: ContextMode,
    user_message: str,
    files: Sequence[FileRecord],
    *,
    direct_budget_tokens: int,
    max_map_reduce_tokens: int | None = None,
    explicit_files: bool = True,
) -> ContextMode:
    """Select Direct, Retrieval, or Map-Reduce mode for the active turn."""
    if requested_mode is not ContextMode.AUTO:
        return requested_mode

    if not files:
        return ContextMode.DIRECT

    total_tokens = sum(max(0, record.token_count) for record in files)
    if total_tokens <= direct_budget_tokens:
        return ContextMode.DIRECT

    if max_map_reduce_tokens is not None and total_tokens > max_map_reduce_tokens:
        return ContextMode.RETRIEVAL

    normalized_query = " ".join(user_message.lower().split())
    active_cues = (
        _MULTI_FILE_HOLISTIC_CUES if (not explicit_files and len(files) > 1) else _HOLISTIC_CUES
    )
    if any(cue in normalized_query for cue in active_cues):
        return ContextMode.MAP_REDUCE

    return ContextMode.RETRIEVAL
