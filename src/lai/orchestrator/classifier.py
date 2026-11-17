"""Deterministic context-mode routing across Direct, Retrieval, and Map-Reduce."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.chat import ContextMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import FileRecord

_HOLISTIC_CUES: tuple[str, ...] = (
    "summarize",
    "entire",
    "whole",
    "all errors",
    "every file",
    "timeline",
    "overview",
    "audit all",
    "across the log",
    "count all",
    "full summary",
)


def classify_context_mode(
    requested_mode: ContextMode,
    user_message: str,
    files: Sequence[FileRecord],
    *,
    direct_budget_tokens: int,
) -> ContextMode:
    """Select Direct, Retrieval, or Map-Reduce mode for the active turn."""
    if requested_mode is not ContextMode.AUTO:
        return requested_mode

    if not files:
        return ContextMode.DIRECT

    total_tokens = sum(max(0, record.token_count) for record in files)
    if total_tokens <= direct_budget_tokens:
        return ContextMode.DIRECT

    normalized_query = " ".join(user_message.lower().split())
    if any(cue in normalized_query for cue in _HOLISTIC_CUES):
        return ContextMode.MAP_REDUCE

    return ContextMode.RETRIEVAL
