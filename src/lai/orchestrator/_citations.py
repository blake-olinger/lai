"""Line-accurate citation matching and extraction helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.chat import Citation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.files import FileRecord, ScoredChunk

_CITATION_RE = re.compile(r"\[([^\]\r\n:]+):L(\d+)(?:-L?(\d+))?\]")
_MAX_FALLBACK_CITATIONS = 8


def _match_file_by_label(label: str, files: Sequence[FileRecord]) -> FileRecord | None:
    """Match a citation filename label against target project FileRecords."""
    clean = label.strip()
    for record in files:
        if record.filename == clean:
            return record
    basename_matches = [record for record in files if Path(record.filename).name == clean]
    if len(basename_matches) == 1:
        return basename_matches[0]
    lowered = clean.lower()
    ci_matches = [
        record
        for record in files
        if record.filename.lower() == lowered or Path(record.filename).name.lower() == lowered
    ]
    return ci_matches[0] if len(ci_matches) == 1 else None


def _find_matching_chunk_id(
    file_id: str,
    start_line: int,
    end_line: int,
    chunks: Sequence[ScoredChunk],
) -> str | None:
    """Return the first retrieved chunk_id overlapping [start_line, end_line] for file_id."""
    for hit in chunks:
        chunk = hit.chunk
        if (
            chunk.file_id == file_id
            and chunk.start_line <= end_line
            and chunk.end_line >= start_line
        ):
            return chunk.id
    return None


def extract_citations(
    text: str,
    files: Sequence[FileRecord],
    *,
    retrieved_chunks: Sequence[ScoredChunk] = (),
    fallback_citations: Sequence[Citation] = (),
) -> tuple[Citation, ...]:
    """Extract deduplicated line-accurate Citations from model output or provenance fallbacks."""
    if not files:
        return ()

    citations: list[Citation] = []
    seen: set[tuple[str, int | None, int | None]] = set()

    for match in _CITATION_RE.finditer(text):
        label, raw_start, raw_end = match.group(1), match.group(2), match.group(3)
        record = _match_file_by_label(label, files)
        if record is None:
            continue
        start_line = max(1, int(raw_start))
        end_line = max(start_line, int(raw_end)) if raw_end is not None else start_line
        key = (record.id, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        chunk_id = _find_matching_chunk_id(record.id, start_line, end_line, retrieved_chunks)
        citations.append(
            Citation(
                file_id=record.id,
                filename=record.filename,
                chunk_id=chunk_id,
                start_line=start_line,
                end_line=end_line,
            )
        )

    if citations:
        return tuple(citations)

    deduped_fallback: list[Citation] = []
    for item in fallback_citations:
        fb_key = (item.file_id, item.start_line, item.end_line)
        if fb_key in seen:
            continue
        seen.add(fb_key)
        deduped_fallback.append(item)
        if len(deduped_fallback) >= _MAX_FALLBACK_CITATIONS:
            break
    return tuple(deduped_fallback)
