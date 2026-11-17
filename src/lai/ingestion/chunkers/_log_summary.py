"""Deterministic LogFileSummary and exception fingerprint aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.files import ExceptionFingerprint, LogFileSummary, LogSeverity
from lai.ingestion.chunkers._log_parser import extract_exception_signature

if TYPE_CHECKING:
    from lai.ingestion.chunkers._log_parser import LogBlock

_TOP_EXCEPTIONS_LIMIT = 20


@dataclass(slots=True)
class _ExceptionAcc:
    """Mutable accumulator for deduplicated exception fingerprints."""

    fingerprint: str
    exception_type: str
    message_sample: str
    count: int
    first_line: int
    last_line: int


def build_log_summary(total_lines: int, blocks: list[LogBlock]) -> LogFileSummary:
    """Build deterministic LogFileSummary with top 20 deduplicated ExceptionFingerprints."""
    severity_counts: dict[LogSeverity, int] = {}
    first_ts: str | None = None
    last_ts: str | None = None
    exceptions: dict[str, _ExceptionAcc] = {}

    for block in blocks:
        if block.severity is not None:
            severity_counts[block.severity] = severity_counts.get(block.severity, 0) + 1
        if block.first_timestamp is not None:
            if first_ts is None:
                first_ts = block.first_timestamp
            last_ts = block.last_timestamp or block.first_timestamp

        sig = extract_exception_signature(block.lines, block.severity)
        if sig is None:
            continue
        fp_id, exc_type, sample = sig
        existing = exceptions.get(fp_id)
        if existing is None:
            exceptions[fp_id] = _ExceptionAcc(
                fingerprint=fp_id,
                exception_type=exc_type,
                message_sample=sample,
                count=1,
                first_line=block.start_line,
                last_line=block.end_line,
            )
        else:
            existing.count += 1
            existing.last_line = max(existing.last_line, block.end_line)

    ranked = sorted(
        exceptions.values(),
        key=lambda item: (-item.count, item.first_line, item.fingerprint),
    )[:_TOP_EXCEPTIONS_LIMIT]
    top_exceptions = tuple(
        ExceptionFingerprint(
            fingerprint=item.fingerprint,
            exception_type=item.exception_type,
            message_sample=item.message_sample,
            count=item.count,
            first_line=item.first_line,
            last_line=item.last_line,
        )
        for item in ranked
    )

    error_count = severity_counts.get(LogSeverity.ERROR, 0) + severity_counts.get(
        LogSeverity.CRITICAL, 0
    )
    warning_count = severity_counts.get(LogSeverity.WARNING, 0)
    return LogFileSummary(
        total_lines=total_lines,
        severity_counts=severity_counts,
        error_count=error_count,
        warning_count=warning_count,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        top_exceptions=top_exceptions,
    )
