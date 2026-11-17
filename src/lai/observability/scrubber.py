"""Deterministic PII/secret prompt scrubber and 24h rotating prompt audit logger."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from lai.observability.rotation import DailyRotatingStreamWriter

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from lai.domain.guardrails import PromptAuditEntry

_ELLIPSIS = "..."

_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
_SECRET_TOKEN_RE = re.compile(
    r"\b(?:"
    r"lai_sk_[A-Za-z0-9_-]+"
    r"|sk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{8,}"
    r"|ghp_[A-Za-z0-9]{8,}"
    r"|github_pat_[A-Za-z0-9_]{8,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r")\b"
)
_ASSIGNMENT_RE = re.compile(
    r"""(?i)\b(password|passwd|secret|token|api_key)\b(\s*[:=]\s*)(?:"[^"]*"|'[^']*'|\S+)"""
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PII_RE = re.compile(r"(?:\b\d{3}-\d{2}-\d{4}\b|(?<!\d)(?:\d[ -]?){12,18}\d(?!\d))")


def fingerprint_prompt(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 encoded prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate_preserving_tags(text: str, max_chars: int) -> str:
    """Truncate text to max_chars without splitting redaction placeholder tags."""
    if len(text) <= max_chars:
        return text
    ellipsis_len = len(_ELLIPSIS)
    if max_chars <= ellipsis_len:
        return text[: max(max_chars, 0)]
    cutoff = max_chars - ellipsis_len
    open_idx = text.rfind("[REDACTED_", 0, cutoff)
    if open_idx != -1:
        close_idx = text.find("]", open_idx)
        if close_idx == -1 or close_idx >= cutoff:
            cutoff = open_idx
    return f"{text[:cutoff]}{_ELLIPSIS}"


def scrub_sensitive_text(text: str, *, max_chars: int = 2000) -> str:
    """Redact secrets, tokens, emails, and PII digits before bounded logging."""
    redacted = _BEARER_RE.sub("Bearer [REDACTED_SECRET]", text)
    redacted = _SECRET_TOKEN_RE.sub("[REDACTED_SECRET]", redacted)
    redacted = _ASSIGNMENT_RE.sub(r"\1\2[REDACTED_SECRET]", redacted)
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _PII_RE.sub("[REDACTED_PII]", redacted)
    return _truncate_preserving_tags(redacted, max_chars)


class PromptAuditLogger:
    """24-hour rotating JSONL writer for scrubbed guardrail prompt audits."""

    def __init__(
        self,
        log_dir: Path,
        *,
        retention_days: int = 30,
        compress_rotated: bool = True,
    ) -> None:
        self._writer = DailyRotatingStreamWriter(
            log_dir / "prompts_audit.jsonl",
            retention_days=retention_days,
            compress=compress_rotated,
            logger_name="lai.observability.prompt_audit",
        )

    def log_entry(self, entry: PromptAuditEntry) -> None:
        """Persist a single PromptAuditEntry line to the daily rotating audit log."""
        self._writer.write_line(entry.model_dump_json())

    def close(self) -> None:
        """Close the underlying rotating file writer."""
        self._writer.close()

    def __enter__(self) -> PromptAuditLogger:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the audit logger when leaving context."""
        del exc_type, exc, tb
        self.close()
