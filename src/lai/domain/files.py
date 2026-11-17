"""Uploaded file, chunking, log/code summary, and retrieval domain models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_chunk_id, new_file_id


class FileKind(StrEnum):
    """Logical classification of an ingested project file."""

    LOG = "log"
    CODE = "code"
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    OTHER = "other"


class FileStatus(StrEnum):
    """Ingestion lifecycle status for an uploaded file."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class LogSeverity(StrEnum):
    """Normalized log entry severity levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExceptionFingerprint(StrictModel):
    """Aggregated signature for repeated exceptions or error patterns in a log."""

    fingerprint: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)
    message_sample: str
    count: int = Field(ge=1)
    first_line: int = Field(ge=1)
    last_line: int = Field(ge=1)


class LogFileSummary(StrictModel):
    """Deterministic structural summary extracted from a log file."""

    total_lines: int = Field(ge=0)
    severity_counts: dict[LogSeverity, int] = Field(default_factory=dict)
    error_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    top_exceptions: tuple[ExceptionFingerprint, ...] = ()


class CodeFileSummary(StrictModel):
    """Deterministic structural summary extracted from a source code file."""

    language: str = Field(min_length=1)
    total_lines: int = Field(ge=0)
    symbols: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()


class FileRecord(StrictModel):
    """Persisted metadata for an uploaded file within a project."""

    id: str = Field(default_factory=new_file_id)
    project_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    content_type: str = "application/octet-stream"
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    blob_path: str = Field(min_length=1)
    kind: FileKind
    status: FileStatus = FileStatus.PENDING
    token_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    log_summary: LogFileSummary | None = None
    code_summary: CodeFileSummary | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class FileChunk(StrictModel):
    """Indexed line-ranged chunk extracted from a project file."""

    id: str = Field(default_factory=new_chunk_id)
    file_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    token_count: int = Field(ge=0)
    text: str
    symbol_name: str | None = None
    severity: LogSeverity | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(StrictModel):
    """Retrieved file chunk annotated with relevance score and source."""

    chunk: FileChunk
    score: float
    source: Literal["vector", "fts", "hybrid", "direct"] = "hybrid"
