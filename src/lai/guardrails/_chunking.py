"""Conservative token estimation and overlapping sliding-window chunking for ShieldGemma."""

from __future__ import annotations

import re

_TOKEN_UNIT_RE = re.compile(r"[A-Za-z0-9]+|[ \t\r]+|\n|.")
_MIN_SLOT_TEXT_BUDGET = 256
_DEFAULT_PROMPT_OVERHEAD_TOKENS = 512
_SLOT_UTILIZATION_RATIO = 0.70
_MAX_OVERLAP_TOKENS = 512
_MIN_OVERLAP_DIVISOR = 6
_CLIP_MARKER = "\n...[truncated for egress context]...\n"
_MIN_SEGMENT_TOKENS = 64


def estimate_conservative_tokens(text: str) -> int:
    """Return a conservative upper-bound token estimate without external tokenizers."""
    if not text:
        return 0
    total = 0
    for unit in _TOKEN_UNIT_RE.findall(text):
        first = unit[0]
        if first.isascii() and first.isalnum():
            total += (len(unit) + 2) // 3
        elif first in (" ", "\t", "\r"):
            total += max(1, (len(unit) + 3) // 4)
        else:
            total += len(unit.encode("utf-8"))
    return total


def compute_slot_text_budget(ctx_size: int, parallel_slots: int) -> int:
    """Return safe conservative text token budget per ShieldGemma llama-server slot."""
    per_slot = ctx_size // max(1, parallel_slots)
    raw = int(per_slot * _SLOT_UTILIZATION_RATIO) - _DEFAULT_PROMPT_OVERHEAD_TOKENS
    return max(_MIN_SLOT_TEXT_BUDGET, raw)


def _split_long_piece_by_chars(piece: str, max_piece_tokens: int) -> list[str]:
    """Break a single unspaced/giant piece into token-bounded character slices."""
    chunks: list[str] = []
    start = 0
    length = len(piece)
    step = max(1, max_piece_tokens)
    while start < length:
        end = min(length, start + step)
        while end > start + 1 and estimate_conservative_tokens(piece[start:end]) > max_piece_tokens:
            end = start + max(1, (end - start) // 2)
        while (
            end < length
            and estimate_conservative_tokens(piece[start : min(length, end + step)])
            <= max_piece_tokens
        ):
            end = min(length, end + step)
        chunks.append(piece[start:end])
        start = end
    return chunks


def _split_long_line(line: str, target_piece_tokens: int) -> list[str]:
    """Split an oversized line on whitespace first, falling back to character slicing."""
    parts = re.findall(r"\S+\s*|\s+", line)
    segments: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for part in parts:
        part_tokens = estimate_conservative_tokens(part)
        if part_tokens > target_piece_tokens:
            if buf:
                segments.append("".join(buf))
                buf = []
                buf_tokens = 0
            segments.extend(_split_long_piece_by_chars(part, target_piece_tokens))
            continue
        if buf and buf_tokens + part_tokens > target_piece_tokens:
            segments.append("".join(buf))
            buf = [part]
            buf_tokens = part_tokens
        else:
            buf.append(part)
            buf_tokens += part_tokens
    if buf:
        segments.append("".join(buf))
    return segments


def _atomize_text(text: str, max_tokens: int) -> list[tuple[str, int]]:
    """Split text into atomic (segment, token_estimate) pairs bounded for overlap packing."""
    piece_limit = max(_MIN_SEGMENT_TOKENS, max_tokens // 4)
    atoms: list[tuple[str, int]] = []
    for line in text.splitlines(keepends=True):
        line_tokens = estimate_conservative_tokens(line)
        if line_tokens <= piece_limit:
            atoms.append((line, line_tokens))
        else:
            atoms.extend(
                (seg, estimate_conservative_tokens(seg))
                for seg in _split_long_line(line, piece_limit)
            )
    return atoms


def _carry_overlap_tail(
    window: list[tuple[str, int]],
    overlap_tokens: int,
) -> tuple[list[tuple[str, int]], int]:
    """Keep trailing segments from the previous window within overlap_tokens."""
    if overlap_tokens <= 0 or len(window) <= 1:
        return [], 0
    carried: list[tuple[str, int]] = []
    carried_tokens = 0
    # Leave at least the first segment out so window progress is strictly monotonic.
    for seg, seg_tokens in reversed(window[1:]):
        if carried and carried_tokens + seg_tokens > overlap_tokens:
            break
        if seg_tokens > overlap_tokens and not carried:
            break
        carried.append((seg, seg_tokens))
        carried_tokens += seg_tokens
    carried.reverse()
    return carried, carried_tokens


def split_overlapping_windows(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split oversized text into overlapping windows bounded by max_tokens."""
    bounded_max = max(_MIN_SEGMENT_TOKENS, max_tokens)
    if not text or estimate_conservative_tokens(text) <= bounded_max:
        return [text]

    effective_overlap = (
        min(_MAX_OVERLAP_TOKENS, bounded_max // _MIN_OVERLAP_DIVISOR)
        if overlap_tokens is None
        else max(0, min(overlap_tokens, bounded_max // 2))
    )
    atoms = _atomize_text(text, bounded_max)
    windows: list[str] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0

    for seg, seg_tokens in atoms:
        if current and current_tokens + seg_tokens > bounded_max:
            windows.append("".join(piece for piece, _ in current))
            current, current_tokens = _carry_overlap_tail(current, effective_overlap)
            while current and current_tokens + seg_tokens > bounded_max:
                _, dropped_tokens = current.pop(0)
                current_tokens -= dropped_tokens
        current.append((seg, seg_tokens))
        current_tokens += seg_tokens

    if current:
        windows.append("".join(piece for piece, _ in current))
    return windows


def clip_prompt_context(prompt: str, *, max_tokens: int) -> str:
    """Return a head+tail excerpt of prompt fitting within max_tokens for egress framing."""
    bounded_max = max(_MIN_SEGMENT_TOKENS, max_tokens)
    if estimate_conservative_tokens(prompt) <= bounded_max:
        return prompt
    marker_tokens = estimate_conservative_tokens(_CLIP_MARKER)
    half_budget = max(16, (bounded_max - marker_tokens) // 2)

    head_end = min(len(prompt), half_budget * 3)
    while head_end > 1 and estimate_conservative_tokens(prompt[:head_end]) > half_budget:
        head_end = max(1, (head_end * 3) // 4)

    tail_len = min(len(prompt) - head_end, half_budget * 3)
    while tail_len > 1 and estimate_conservative_tokens(prompt[-tail_len:]) > half_budget:
        tail_len = max(1, (tail_len * 3) // 4)

    tail = prompt[-tail_len:] if tail_len > 0 else ""
    return f"{prompt[:head_end]}{_CLIP_MARKER}{tail}"


def bisect_text(text: str, *, min_chars: int = 256) -> tuple[str, str] | None:
    """Split text into two overlapping halves for HTTP 400 context-overflow recovery."""
    length = len(text)
    if length <= min_chars:
        return None
    mid = length // 2
    overlap = min(128, mid // 4)
    left_end = mid + overlap
    right_start = max(0, mid - overlap)
    if left_end >= length or right_start <= 0:
        return text[:mid], text[mid:]
    return text[:left_end], text[right_start:]
