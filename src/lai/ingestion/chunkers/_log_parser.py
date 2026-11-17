"""Log line parsing, timestamp/severity extraction, and exception fingerprinting helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import cast

from lai.domain.files import LogSeverity

SEVERITY_RANK: dict[LogSeverity, int] = {
    LogSeverity.TRACE: 0,
    LogSeverity.DEBUG: 1,
    LogSeverity.INFO: 2,
    LogSeverity.WARNING: 3,
    LogSeverity.ERROR: 4,
    LogSeverity.CRITICAL: 5,
}

_TIMESTAMP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?P<ts>\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    ),
    re.compile(r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"),
    re.compile(r"^\[(?P<ts>\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]"),
)

_SEVERITY_WORD_RE = re.compile(
    r"(?:^|[\s\[|(:=,])"
    r"(?P<lvl>CRITICAL|FATAL|CRIT|EMERGENCY|ALERT|PANIC|ERROR|ERR|SEVERE|"
    r"WARNING|WARN|WRN|INFO|INF|NOTICE|DEBUG|DBG|TRACE|TRC|VERBOSE)"
    r"(?=$|[\s\]|):=,])",
    re.IGNORECASE,
)

_SEVERITY_MAP: dict[str, LogSeverity] = {
    "CRITICAL": LogSeverity.CRITICAL,
    "FATAL": LogSeverity.CRITICAL,
    "CRIT": LogSeverity.CRITICAL,
    "EMERGENCY": LogSeverity.CRITICAL,
    "ALERT": LogSeverity.CRITICAL,
    "PANIC": LogSeverity.CRITICAL,
    "ERROR": LogSeverity.ERROR,
    "ERR": LogSeverity.ERROR,
    "SEVERE": LogSeverity.ERROR,
    "WARNING": LogSeverity.WARNING,
    "WARN": LogSeverity.WARNING,
    "WRN": LogSeverity.WARNING,
    "INFO": LogSeverity.INFO,
    "INF": LogSeverity.INFO,
    "NOTICE": LogSeverity.INFO,
    "DEBUG": LogSeverity.DEBUG,
    "DBG": LogSeverity.DEBUG,
    "TRACE": LogSeverity.TRACE,
    "TRC": LogSeverity.TRACE,
    "VERBOSE": LogSeverity.TRACE,
}

PYTHON_TRACEBACK_START_RE = re.compile(r"^\s*Traceback \(most recent call last\):\s*$")
PYTHON_CHAIN_RE = re.compile(
    r"^\s*(?:During handling of the above exception, another exception occurred:|"
    r"The above exception was the direct cause of the following exception:)\s*$"
)
STACK_FRAME_RE = re.compile(
    r"^(?:\s+at\s+\S|\s+File \".+\", line \d+|\s+\.\.\.\s+\d+\s+more\b|"
    r"\s+---\s+End of inner exception stack trace\s+---|"
    r"\s+Suppressed:\s+|\s*Caused by:\s+)"
)
CAUSED_BY_RE = re.compile(r"^\s*Caused by:\s+(?P<exc>.+)$")
POWERSHELL_AT_RE = re.compile(r"^At\s+.+?:\d+\s+char:\d+\s*$")
POWERSHELL_PLUS_RE = re.compile(r"^\s*\+\s+(?:CategoryInfo|FullyQualifiedErrorId|~+|\S)")
POWERSHELL_CATEGORY_RE = re.compile(
    r"^\s*\+\s*CategoryInfo\s*:\s*(?P<info>.+?)(?:,\s*(?P<exc>[A-Za-z_][\w.]*)\s*)?$"
)
POWERSHELL_FQID_RE = re.compile(r"^\s*\+\s*FullyQualifiedErrorId\s*:\s*(?P<fqid>.+)$")

EXCEPTION_LINE_RE = re.compile(
    r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit|Panic|Fault))\b"
    r"(?:\s*:\s*(?P<msg>.*))?$"
)
EMBEDDED_EXCEPTION_RE = re.compile(
    r"\b(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception))\b(?:\s*:\s*(?P<msg>[^\r\n]+))?"
)

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.I)
_ISO_TS_RE = re.compile(
    r"\b\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_NUM_RE = re.compile(r"\b\d+\b")
_WS_RE = re.compile(r"\s+")

_MAX_SAMPLE_CHARS = 200


@dataclass(slots=True)
class ParsedLogLine:
    """Normalized structural metadata parsed from a single log line."""

    line_no: int
    raw: str
    timestamp: str | None
    severity: LogSeverity | None


@dataclass(slots=True)
class LogBlock:
    """Atomic log event block that must never be split across chunk boundaries."""

    start_line: int
    end_line: int
    lines: list[str] = field(default_factory=list)
    severity: LogSeverity | None = None
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    in_python_tb: bool = False
    in_stack_trace: bool = False
    in_powershell_err: bool = False
    has_timestamp: bool = False


def parse_json_log_line(line: str) -> tuple[str | None, LogSeverity | None] | None:
    """Extract timestamp and severity from structured JSON log lines when present."""
    stripped = line.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    obj = cast("dict[str, object]", parsed)
    ts_val = obj.get("timestamp") or obj.get("ts") or obj.get("time") or obj.get("@timestamp")
    lvl_val = obj.get("level") or obj.get("severity") or obj.get("lvl")
    ts = str(ts_val) if isinstance(ts_val, str) and ts_val else None
    sev = _SEVERITY_MAP.get(str(lvl_val).upper()) if isinstance(lvl_val, str) else None
    return ts, sev


def parse_log_line(line_no: int, raw: str) -> ParsedLogLine:
    """Extract timestamp and normalized LogSeverity from a single raw log line."""
    json_meta = parse_json_log_line(raw)
    if json_meta is not None:
        return ParsedLogLine(
            line_no=line_no,
            raw=raw,
            timestamp=json_meta[0],
            severity=json_meta[1],
        )

    timestamp: str | None = None
    remainder = raw
    for pattern in _TIMESTAMP_PATTERNS:
        match = pattern.match(raw)
        if match is not None:
            timestamp = match.group("ts")
            remainder = raw[match.end() :]
            break

    severity: LogSeverity | None = None
    sev_match = _SEVERITY_WORD_RE.search(remainder)
    if sev_match is not None:
        severity = _SEVERITY_MAP.get(sev_match.group("lvl").upper())

    return ParsedLogLine(
        line_no=line_no,
        raw=raw,
        timestamp=timestamp,
        severity=severity,
    )


def max_severity(a: LogSeverity | None, b: LogSeverity | None) -> LogSeverity | None:
    """Return the higher of two optional LogSeverity values."""
    if a is None:
        return b
    if b is None:
        return a
    return a if SEVERITY_RANK[a] >= SEVERITY_RANK[b] else b


def normalize_message(text: str) -> str:
    """Normalize variable tokens (IDs, timestamps, IPs, numbers) for fingerprinting."""
    cleaned = _UUID_RE.sub("<UUID>", text)
    cleaned = _HEX_RE.sub("<HEX>", cleaned)
    cleaned = _ISO_TS_RE.sub("<TS>", cleaned)
    cleaned = _IP_RE.sub("<IP>", cleaned)
    cleaned = _NUM_RE.sub("<N>", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def _strip_log_prefix(raw: str) -> str:
    """Remove leading timestamp and severity token to isolate the error message body."""
    text = raw.strip()
    for pattern in _TIMESTAMP_PATTERNS:
        match = pattern.match(text)
        if match is not None:
            text = text[match.end() :].lstrip(" -|:")
            break
    sev_match = _SEVERITY_WORD_RE.match(text)
    if sev_match is not None:
        text = text[sev_match.end() :].lstrip(" -|:")
    return text


def _extract_from_lines(lines: list[str]) -> tuple[str, str] | None:
    """Scan lines in reverse/priority order for explicit Python/Java/PowerShell exceptions."""
    for line in reversed(lines):
        stripped = line.strip()
        if (exc_match := EXCEPTION_LINE_RE.match(stripped)) is not None:
            exc_type = exc_match.group("exc")
            msg = (exc_match.group("msg") or "").strip()
            return exc_type, msg
        if (cat_match := POWERSHELL_CATEGORY_RE.match(line)) is not None:
            exc_name = cat_match.group("exc") or "PowerShellError"
            return exc_name, cat_match.group("info").strip()

    for line in lines:
        if (caused := CAUSED_BY_RE.match(line)) is not None:
            body = caused.group("exc").strip()
            if (exc_match := EXCEPTION_LINE_RE.match(body)) is not None:
                return exc_match.group("exc"), (exc_match.group("msg") or "").strip()
            return "CausedByException", body
        if (emb := EMBEDDED_EXCEPTION_RE.search(line)) is not None:
            return emb.group("exc"), (emb.group("msg") or "").strip()
    return None


def extract_exception_signature(
    lines: list[str],
    severity: LogSeverity | None,
) -> tuple[str, str, str] | None:
    """Extract (fingerprint, exception_type, message_sample) for an error/traceback block."""
    extracted = _extract_from_lines(lines)
    if extracted is not None:
        exc_type, raw_msg = extracted
    elif severity in {LogSeverity.ERROR, LogSeverity.CRITICAL}:
        exc_type = "CriticalError" if severity is LogSeverity.CRITICAL else "LogError"
        raw_msg = _strip_log_prefix(lines[0])
    else:
        return None

    sample = (raw_msg or exc_type)[:_MAX_SAMPLE_CHARS]
    normalized = normalize_message(raw_msg)
    digest = hashlib.sha256(f"{exc_type}:{normalized}".encode()).hexdigest()[:16]
    return digest, exc_type, sample
