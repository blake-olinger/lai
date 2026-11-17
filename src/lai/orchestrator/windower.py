"""Line-preserving overlapping text windower for Map-Reduce analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.errors import ValidationFailedError
from lai.ingestion.detector import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_WINDOW_TOKENS = 9000
DEFAULT_OVERLAP_TOKENS = 300
_HEADER_OVERHEAD_RESERVE = 24


class TextWindow(StrictModel):
    """Single line-bounded overlapping window extracted from a project file."""

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    window_index: int = Field(ge=0)
    total_windows: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    token_count: int = Field(ge=0)
    text: str


def _format_window_text(
    filename: str,
    lines: Sequence[str],
    start_idx: int,
    end_idx: int,
) -> str:
    """Render an inclusive 0-based line range with its [filename:Lstart-Lend] header."""
    start_line = start_idx + 1
    end_line = end_idx + 1
    header = f"[{filename}:L{start_line}-L{end_line}]"
    body = "\n".join(lines[start_idx : end_idx + 1])
    return f"{header}\n{body}"


def _pick_window_end(
    *,
    filename: str,
    lines: Sequence[str],
    line_tokens: Sequence[int],
    start_idx: int,
    window_tokens: int,
) -> int:
    """Greedily pick the largest inclusive end_idx fitting within window_tokens."""
    n_lines = len(lines)
    header_sample = f"[{filename}:L{start_idx + 1}-L{n_lines}]"
    header_cost = max(1, estimate_tokens(header_sample) + 1)
    reserve = max(header_cost, min(_HEADER_OVERHEAD_RESERVE, window_tokens // 4))
    body_budget = max(1, window_tokens - reserve)

    acc_tokens = 0
    end_idx = start_idx
    for idx in range(start_idx, n_lines):
        cost = line_tokens[idx]
        if idx > start_idx and acc_tokens + cost > body_budget:
            break
        acc_tokens += cost
        end_idx = idx

    while end_idx > start_idx:
        candidate = _format_window_text(filename, lines, start_idx, end_idx)
        if estimate_tokens(candidate) <= window_tokens:
            break
        end_idx -= 1

    return end_idx


def _compute_next_start(
    *,
    line_tokens: Sequence[int],
    start_idx: int,
    end_idx: int,
    overlap_tokens: int,
) -> int:
    """Compute the next window start index keeping <= overlap_tokens of trailing lines."""
    if overlap_tokens <= 0 or end_idx <= start_idx:
        return end_idx + 1

    acc = 0
    overlap_start = end_idx + 1
    for back_idx in range(end_idx, start_idx, -1):
        cost = line_tokens[back_idx]
        if acc + cost > overlap_tokens:
            break
        acc += cost
        overlap_start = back_idx

    return max(start_idx + 1, overlap_start)


def slice_text_into_windows(
    file_id: str,
    filename: str,
    text: str,
    *,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextWindow]:
    """Slice file text on clean line boundaries with overlap and [filename:Lstart-Lend] headers."""
    if not file_id.strip():
        msg = "file_id must not be empty."
        raise ValidationFailedError(msg, details={"field": "file_id"})
    if not filename.strip():
        msg = "filename must not be empty."
        raise ValidationFailedError(msg, details={"field": "filename"})
    if window_tokens < 1:
        msg = f"window_tokens must be >= 1, got {window_tokens}."
        raise ValidationFailedError(msg, details={"window_tokens": str(window_tokens)})
    if overlap_tokens < 0:
        msg = f"overlap_tokens must be >= 0, got {overlap_tokens}."
        raise ValidationFailedError(msg, details={"overlap_tokens": str(overlap_tokens)})

    if not text:
        return []
    lines = text.splitlines()
    if not lines:
        return []

    effective_overlap = (
        overlap_tokens if overlap_tokens < window_tokens else max(0, window_tokens // 3)
    )
    line_tokens = [max(1, estimate_tokens(line) + 1) for line in lines]

    spans: list[tuple[int, int, str, int]] = []
    start_idx = 0
    n_lines = len(lines)
    while start_idx < n_lines:
        end_idx = _pick_window_end(
            filename=filename,
            lines=lines,
            line_tokens=line_tokens,
            start_idx=start_idx,
            window_tokens=window_tokens,
        )
        rendered = _format_window_text(filename, lines, start_idx, end_idx)
        spans.append((start_idx + 1, end_idx + 1, rendered, estimate_tokens(rendered)))
        if end_idx >= n_lines - 1:
            break
        start_idx = _compute_next_start(
            line_tokens=line_tokens,
            start_idx=start_idx,
            end_idx=end_idx,
            overlap_tokens=effective_overlap,
        )

    total_windows = len(spans)
    return [
        TextWindow(
            file_id=file_id,
            filename=filename,
            window_index=idx,
            total_windows=total_windows,
            start_line=start_line,
            end_line=end_line,
            token_count=token_count,
            text=rendered,
        )
        for idx, (start_line, end_line, rendered, token_count) in enumerate(spans)
    ]
