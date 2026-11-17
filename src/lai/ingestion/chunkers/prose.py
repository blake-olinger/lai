"""Heading, paragraph, and JSON-aware prose chunker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.files import FileChunk, FileKind
from lai.ingestion.detector import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_PROSE_CHUNK_TOKENS = 600
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class _ProseBlock:
    """Contiguous paragraph, fenced code block, or heading block with optional heading."""

    start_line: int
    end_line: int
    heading: str | None = None
    is_heading: bool = False


def _split_long_single_line(line_text: str, target_tokens: int) -> list[str]:
    """Split a single long line into <= target_tokens segments along separators."""
    if estimate_tokens(line_text) <= target_tokens:
        return [line_text]
    max_chars = max(64, target_tokens * _CHARS_PER_TOKEN)
    pieces: list[str] = []
    start = 0
    n = len(line_text)

    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            split_at = max(
                line_text.rfind(" ", start + 1, end),
                line_text.rfind("},", start + 1, end) + 2,
                line_text.rfind(",", start + 1, end) + 1,
            )
            if split_at > start:
                end = split_at
        piece = line_text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces or [line_text]


@dataclass(slots=True)
class _MdScanState:
    """Mutable cursor state for Markdown heading and fence extraction."""

    blocks: list[_ProseBlock]
    current_heading: str | None = None
    cur_start: int | None = None
    fence_marker: str = ""

    def flush(self, end_line: int) -> None:
        """Close the currently open paragraph/code block at end_line."""
        if self.cur_start is not None and self.cur_start <= end_line:
            self.blocks.append(
                _ProseBlock(
                    start_line=self.cur_start,
                    end_line=end_line,
                    heading=self.current_heading,
                )
            )
            self.cur_start = None


def _handle_fence_line(state: _MdScanState, line_no: int, marker: str) -> None:
    """Toggle fenced code block state without splitting headings inside fences."""
    if not state.fence_marker:
        if state.cur_start is None:
            state.cur_start = line_no
        state.fence_marker = marker
    elif marker == state.fence_marker:
        state.fence_marker = ""


def _extract_markdown_blocks(raw_lines: Sequence[str]) -> list[_ProseBlock]:
    """Split Markdown into heading, paragraph, and atomic fenced-code blocks."""
    state = _MdScanState(blocks=[])
    for idx, line in enumerate(raw_lines):
        line_no = idx + 1
        if (fence_match := _FENCE_RE.match(line)) is not None:
            _handle_fence_line(state, line_no, fence_match.group(1))
            continue
        if state.fence_marker:
            continue
        if not line.strip():
            state.flush(line_no - 1)
            continue
        if (heading_match := _HEADING_RE.match(line)) is not None:
            state.flush(line_no - 1)
            state.current_heading = heading_match.group(2).strip()
            state.blocks.append(
                _ProseBlock(
                    start_line=line_no,
                    end_line=line_no,
                    heading=state.current_heading,
                    is_heading=True,
                )
            )
            continue
        if state.cur_start is None:
            state.cur_start = line_no

    state.flush(len(raw_lines))
    return state.blocks


def _extract_paragraph_blocks(raw_lines: Sequence[str]) -> list[_ProseBlock]:
    """Split plain text or JSON lines into blank-line-separated blocks."""
    blocks: list[_ProseBlock] = []
    cur_start: int | None = None

    for idx, line in enumerate(raw_lines):
        line_no = idx + 1
        if not line.strip():
            if cur_start is not None:
                blocks.append(_ProseBlock(start_line=cur_start, end_line=line_no - 1))
                cur_start = None
            continue
        if cur_start is None:
            cur_start = line_no

    if cur_start is not None:
        blocks.append(_ProseBlock(start_line=cur_start, end_line=len(raw_lines)))
    return blocks


def _split_oversized_block(
    *,
    raw_lines: Sequence[str],
    block: _ProseBlock,
    target_tokens: int,
) -> list[tuple[int, int, str, str | None]]:
    """Split an oversized block cleanly by line or long-line window."""
    if block.start_line == block.end_line:
        segments = _split_long_single_line(raw_lines[block.start_line - 1], target_tokens)
        return [(block.start_line, block.end_line, seg, block.heading) for seg in segments]

    items: list[tuple[int, int, str, str | None]] = []
    cur_start = block.start_line
    for line_no in range(block.start_line, block.end_line + 1):
        candidate = "\n".join(raw_lines[cur_start - 1 : line_no])
        if line_no > cur_start and estimate_tokens(candidate) > target_tokens:
            piece = "\n".join(raw_lines[cur_start - 1 : line_no - 1])
            items.append((cur_start, line_no - 1, piece, block.heading))
            cur_start = line_no

    tail = "\n".join(raw_lines[cur_start - 1 : block.end_line])
    if estimate_tokens(tail) > target_tokens and cur_start == block.end_line:
        items.extend(
            (cur_start, block.end_line, seg, block.heading)
            for seg in _split_long_single_line(tail, target_tokens)
        )
    else:
        items.append((cur_start, block.end_line, tail, block.heading))
    return items


def _pack_prose_blocks(
    *,
    raw_lines: Sequence[str],
    blocks: list[_ProseBlock],
    target_tokens: int,
    split_on_heading: bool,
) -> list[tuple[int, int, str, str | None]]:
    """Pack contiguous prose blocks up to target_tokens while respecting headings."""
    packed: list[tuple[int, int, str, str | None]] = []
    pending: tuple[int, int, str | None] | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            s_line, e_line, heading = pending
            packed.append((s_line, e_line, "\n".join(raw_lines[s_line - 1 : e_line]), heading))
            pending = None

    for block in blocks:
        block_text = "\n".join(raw_lines[block.start_line - 1 : block.end_line])
        if estimate_tokens(block_text) > target_tokens:
            flush_pending()
            packed.extend(
                _split_oversized_block(
                    raw_lines=raw_lines,
                    block=block,
                    target_tokens=target_tokens,
                )
            )
            continue

        if pending is None:
            pending = (block.start_line, block.end_line, block.heading)
            continue

        cur_start, _cur_end, cur_heading = pending
        if split_on_heading and block.is_heading and block.heading != cur_heading:
            flush_pending()
            pending = (block.start_line, block.end_line, block.heading)
            continue

        merged = "\n".join(raw_lines[cur_start - 1 : block.end_line])
        if estimate_tokens(merged) <= target_tokens:
            pending = (cur_start, block.end_line, cur_heading or block.heading)
        else:
            flush_pending()
            pending = (block.start_line, block.end_line, block.heading)

    flush_pending()
    return packed


def chunk_prose_file(
    *,
    file_id: str,
    project_id: str,
    kind: FileKind,
    text: str,
    target_tokens: int = DEFAULT_PROSE_CHUNK_TOKENS,
) -> list[FileChunk]:
    """Split markdown, JSON, or plain text into ~target_tokens structural chunks."""
    if not text or not text.strip():
        return []

    raw_lines = text.splitlines()
    if kind is FileKind.MARKDOWN:
        blocks = _extract_markdown_blocks(raw_lines)
        split_on_heading = True
    else:
        blocks = _extract_paragraph_blocks(raw_lines)
        split_on_heading = False

    packed = _pack_prose_blocks(
        raw_lines=raw_lines,
        blocks=blocks,
        target_tokens=max(32, target_tokens),
        split_on_heading=split_on_heading,
    )

    chunks: list[FileChunk] = []
    for idx, (start_line, end_line, chunk_text, heading) in enumerate(packed):
        metadata: dict[str, str] = {"kind": kind.value}
        if heading is not None:
            metadata["heading"] = heading
        chunks.append(
            FileChunk(
                file_id=file_id,
                project_id=project_id,
                chunk_index=idx,
                start_line=start_line,
                end_line=end_line,
                token_count=estimate_tokens(chunk_text),
                text=chunk_text,
                symbol_name=heading,
                metadata=metadata,
            )
        )
    return chunks
