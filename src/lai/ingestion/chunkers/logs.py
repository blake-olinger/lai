"""Structure-aware log chunker with stack-trace coalescing and deterministic summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.files import FileChunk, LogFileSummary, LogSeverity
from lai.ingestion.chunkers._log_parser import (
    EXCEPTION_LINE_RE,
    POWERSHELL_AT_RE,
    POWERSHELL_FQID_RE,
    POWERSHELL_PLUS_RE,
    PYTHON_CHAIN_RE,
    PYTHON_TRACEBACK_START_RE,
    STACK_FRAME_RE,
    LogBlock,
    ParsedLogLine,
    max_severity,
    parse_log_line,
)
from lai.ingestion.chunkers._log_summary import build_log_summary
from lai.ingestion.detector import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_LOG_CHUNK_TOKENS = 600
_MAX_CHAIN_HEADER_LINES = 2


def _next_nonempty_line(lines: Sequence[str], start_idx: int) -> str | None:
    """Peek forward for the next non-blank raw line starting at start_idx."""
    for idx in range(start_idx, len(lines)):
        if lines[idx].strip():
            return lines[idx]
    return None


def _is_stack_continuation(raw: str) -> bool:
    """Return True if raw line is unambiguously a stack-frame or chain continuation."""
    return bool(
        STACK_FRAME_RE.match(raw)
        or PYTHON_CHAIN_RE.match(raw)
        or POWERSHELL_AT_RE.match(raw)
        or POWERSHELL_PLUS_RE.match(raw)
    )


def _is_indented_continuation(raw: str) -> bool:
    """Return True if raw line is indented non-empty continuation text."""
    return bool(raw.strip()) and raw[0] in {" ", "\t"}


def _is_ps_error_header(block: LogBlock, parsed: ParsedLogLine) -> bool:
    """Return True if a PowerShell At/+/CategoryInfo line should attach to block."""
    if not (POWERSHELL_AT_RE.match(parsed.raw) or POWERSHELL_PLUS_RE.match(parsed.raw)):
        return False
    if block.in_powershell_err or block.severity in {LogSeverity.ERROR, LogSeverity.CRITICAL}:
        return True
    return len(block.lines) == 1 and not block.has_timestamp


def _is_traceback_header_attach(block: LogBlock, parsed: ParsedLogLine) -> bool:
    """Return True if a Traceback start or exception header should attach to current block."""
    if PYTHON_TRACEBACK_START_RE.match(parsed.raw):
        return (
            block.in_python_tb
            or block.severity in {LogSeverity.ERROR, LogSeverity.CRITICAL}
            or (
                len(block.lines) <= _MAX_CHAIN_HEADER_LINES
                and PYTHON_CHAIN_RE.match(block.lines[-1]) is not None
            )
        )
    if parsed.timestamp is None and EXCEPTION_LINE_RE.match(parsed.raw.strip()):
        return block.in_python_tb or block.severity in {LogSeverity.ERROR, LogSeverity.CRITICAL}
    return False


def _should_attach(block: LogBlock, parsed: ParsedLogLine) -> bool:
    """Determine whether parsed line belongs to the currently open atomic log block."""
    raw = parsed.raw
    if _is_stack_continuation(raw):
        return True
    if _is_ps_error_header(block, parsed):
        return True
    if _is_traceback_header_attach(block, parsed):
        return True
    if parsed.timestamp is None and parsed.severity is None and _is_indented_continuation(raw):
        return True
    return bool(
        block.in_python_tb
        and parsed.timestamp is None
        and EXCEPTION_LINE_RE.match(raw.strip()) is not None
    )


def _update_block_state(block: LogBlock, parsed: ParsedLogLine) -> None:
    """Append a parsed line to block and update traceback/severity state."""
    raw = parsed.raw
    block.end_line = parsed.line_no
    block.lines.append(raw)
    block.severity = max_severity(block.severity, parsed.severity)
    if parsed.timestamp is not None:
        if block.first_timestamp is None:
            block.first_timestamp = parsed.timestamp
        block.last_timestamp = parsed.timestamp

    if PYTHON_TRACEBACK_START_RE.match(raw) or PYTHON_CHAIN_RE.match(raw):
        block.in_python_tb = True
        block.in_stack_trace = True
    elif STACK_FRAME_RE.match(raw):
        block.in_stack_trace = True
    elif (
        POWERSHELL_AT_RE.match(raw)
        or POWERSHELL_PLUS_RE.match(raw)
        or POWERSHELL_FQID_RE.match(raw)
    ):
        block.in_powershell_err = True
        block.in_stack_trace = True
    elif block.in_python_tb and EXCEPTION_LINE_RE.match(raw.strip()):
        block.in_python_tb = False
        block.in_stack_trace = True


def _new_block(parsed: ParsedLogLine) -> LogBlock:
    """Create a fresh atomic LogBlock starting at parsed line."""
    block = LogBlock(
        start_line=parsed.line_no,
        end_line=parsed.line_no,
        has_timestamp=parsed.timestamp is not None,
    )
    _update_block_state(block, parsed)
    if EXCEPTION_LINE_RE.match(parsed.raw.strip()):
        block.in_stack_trace = True
    return block


def _finalize_block(block: LogBlock) -> LogBlock:
    """Ensure stack-trace or PowerShell-error blocks carry at least ERROR severity."""
    if block.severity is None and (block.in_stack_trace or block.in_powershell_err):
        block.severity = LogSeverity.ERROR
    return block


def _coalesce_blocks(raw_lines: Sequence[str]) -> list[LogBlock]:
    """Group raw log lines into atomic LogBlocks without splitting stack traces."""
    blocks: list[LogBlock] = []
    current: LogBlock | None = None

    for idx, raw in enumerate(raw_lines):
        line_no = idx + 1
        if not raw.strip():
            if current is not None and current.in_stack_trace:
                nxt = _next_nonempty_line(raw_lines, idx + 1)
                if nxt is not None and (
                    _is_stack_continuation(nxt) or PYTHON_TRACEBACK_START_RE.match(nxt)
                ):
                    current.end_line = line_no
                    current.lines.append(raw)
                    continue
            if current is not None:
                blocks.append(_finalize_block(current))
                current = None
            continue

        parsed = parse_log_line(line_no, raw)
        if current is not None and _should_attach(current, parsed):
            _update_block_state(current, parsed)
            continue

        if current is not None:
            blocks.append(_finalize_block(current))
        current = _new_block(parsed)

    if current is not None:
        blocks.append(_finalize_block(current))
    return blocks


def _build_chunk(
    *,
    file_id: str,
    project_id: str,
    chunk_index: int,
    raw_lines: Sequence[str],
    blocks: list[LogBlock],
) -> FileChunk:
    """Materialize a FileChunk from one or more contiguous atomic LogBlocks."""
    start_line = blocks[0].start_line
    end_line = blocks[-1].end_line
    text = "\n".join(raw_lines[start_line - 1 : end_line])

    chunk_sev: LogSeverity | None = None
    first_ts: str | None = None
    last_ts: str | None = None
    for block in blocks:
        chunk_sev = max_severity(chunk_sev, block.severity)
        if block.first_timestamp is not None:
            if first_ts is None:
                first_ts = block.first_timestamp
            last_ts = block.last_timestamp or block.first_timestamp

    metadata: dict[str, str] = {}
    if first_ts is not None:
        metadata["first_timestamp"] = first_ts
    if last_ts is not None:
        metadata["last_timestamp"] = last_ts

    return FileChunk(
        file_id=file_id,
        project_id=project_id,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        token_count=estimate_tokens(text),
        text=text,
        severity=chunk_sev,
        metadata=metadata,
    )


def _pack_blocks_into_chunks(
    *,
    file_id: str,
    project_id: str,
    raw_lines: Sequence[str],
    blocks: list[LogBlock],
    target_tokens: int,
) -> list[FileChunk]:
    """Pack atomic LogBlocks into ~target_tokens chunks without splitting any block."""
    chunks: list[FileChunk] = []
    pending: list[LogBlock] = []

    for block in blocks:
        if not pending:
            pending.append(block)
            continue

        candidate_text = "\n".join(raw_lines[pending[0].start_line - 1 : block.end_line])
        if estimate_tokens(candidate_text) <= target_tokens:
            pending.append(block)
            continue

        chunks.append(
            _build_chunk(
                file_id=file_id,
                project_id=project_id,
                chunk_index=len(chunks),
                raw_lines=raw_lines,
                blocks=pending,
            )
        )
        pending = [block]

    if pending:
        chunks.append(
            _build_chunk(
                file_id=file_id,
                project_id=project_id,
                chunk_index=len(chunks),
                raw_lines=raw_lines,
                blocks=pending,
            )
        )
    return chunks


def chunk_log_file(
    *,
    file_id: str,
    project_id: str,
    text: str,
    target_tokens: int = DEFAULT_LOG_CHUNK_TOKENS,
) -> tuple[list[FileChunk], LogFileSummary]:
    """Parse, stack-trace-coalesce, chunk, and summarize a log file deterministically."""
    if not text:
        return [], LogFileSummary(total_lines=0)
    raw_lines = text.splitlines()
    blocks = _coalesce_blocks(raw_lines)
    chunks = _pack_blocks_into_chunks(
        file_id=file_id,
        project_id=project_id,
        raw_lines=raw_lines,
        blocks=blocks,
        target_tokens=max(1, target_tokens),
    )
    summary = build_log_summary(len(raw_lines), blocks)
    return chunks, summary
