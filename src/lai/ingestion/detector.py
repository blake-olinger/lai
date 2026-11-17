"""File type classification, language detection, and fast token estimation."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import tiktoken

from lai.domain.files import FileKind

_CHARS_PER_TOKEN_FALLBACK = 4
_SAMPLE_LINE_LIMIT = 40
_JSON_SAMPLE_MAX_CHARS = 65_536
_MIN_LOG_MATCHES = 2

_CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".tf": "hcl",
    ".hcl": "hcl",
}

_SPECIAL_CODE_FILENAMES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "jenkinsfile": "groovy",
}

_LOG_EXTENSIONS = frozenset({".log", ".out", ".err", ".trace", ".syslog"})
_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown", ".mdx"})
_JSON_EXTENSIONS = frozenset({".json", ".jsonl", ".ndjson", ".geojson"})
_TEXT_EXTENSIONS = frozenset({".txt", ".text", ".csv", ".tsv", ".rst", ".env", ".conf"})

_LOG_LINE_HINT_RE = re.compile(
    r"^(?:\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}|"
    r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}|"
    r"\[(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\])",
    re.IGNORECASE,
)
_LOG_TRACEBACK_HINT_RE = re.compile(
    r"(?:Traceback \(most recent call last\):|\bCaused by:|\+\s*CategoryInfo\b)",
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


@lru_cache(maxsize=1)
def _load_cl100k_encoding() -> tiktoken.Encoding | None:
    """Load tiktoken cl100k_base encoding, returning None if offline/unavailable."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except (OSError, ValueError, RuntimeError):
        return None


def clear_token_encoder_cache() -> None:
    """Clear cached tiktoken encoder handle (primarily for offline fallback tests)."""
    _load_cl100k_encoding.cache_clear()


def _heuristic_token_count(text: str) -> int:
    """Estimate token count offline using ~4 chars per token."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN_FALLBACK - 1) // _CHARS_PER_TOKEN_FALLBACK)


def estimate_tokens(text: str, *, allow_tiktoken: bool = True) -> int:
    """Estimate tokens using tiktoken cl100k_base with graceful offline fallback."""
    if not text:
        return 0
    if allow_tiktoken:
        encoding = _load_cl100k_encoding()
        if encoding is not None:
            return len(encoding.encode(text, disallowed_special=()))
    return _heuristic_token_count(text)


def detect_language(filename: str) -> str:
    """Infer source programming language identifier from filename."""
    path = Path(filename)
    lower_name = path.name.lower()
    if lower_name in _SPECIAL_CODE_FILENAMES:
        return _SPECIAL_CODE_FILENAMES[lower_name]
    ext = path.suffix.lower()
    if ext in _CODE_EXTENSIONS:
        return _CODE_EXTENSIONS[ext]
    if ext in _MARKDOWN_EXTENSIONS:
        return "markdown"
    if ext in _JSON_EXTENSIONS:
        return "json"
    return "unknown"


def _looks_like_log(sample_text: str) -> bool:
    """Heuristically check whether sample text resembles log entries or stack traces."""
    if not sample_text:
        return False
    if _LOG_TRACEBACK_HINT_RE.search(sample_text):
        return True
    matches = 0
    non_empty = 0
    for line in sample_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        non_empty += 1
        if _LOG_LINE_HINT_RE.search(stripped):
            matches += 1
            if matches >= _MIN_LOG_MATCHES:
                return True
        if non_empty >= _SAMPLE_LINE_LIMIT:
            break
    return False


def _looks_like_json(sample_text: str) -> bool:
    """Heuristically check whether sample text is valid JSON or JSONL."""
    stripped = sample_text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return False
    if len(stripped) <= _JSON_SAMPLE_MAX_CHARS:
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return True
    first_line = stripped.splitlines()[0].strip()
    if first_line.startswith("{"):
        try:
            parsed: object = json.loads(first_line)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict)
    return False


_MIME_KIND_MAP: dict[str, FileKind] = {
    "application/json": FileKind.JSON,
    "text/json": FileKind.JSON,
    "text/markdown": FileKind.MARKDOWN,
    "text/x-markdown": FileKind.MARKDOWN,
    "text/x-log": FileKind.LOG,
    "application/x-log": FileKind.LOG,
    "text/x-python": FileKind.CODE,
    "application/x-python-code": FileKind.CODE,
    "application/javascript": FileKind.CODE,
    "text/javascript": FileKind.CODE,
    "application/typescript": FileKind.CODE,
    "text/x-rust": FileKind.CODE,
    "text/x-go": FileKind.CODE,
}

_EXTENSION_KIND_MAP: dict[str, FileKind] = {
    **dict.fromkeys(_LOG_EXTENSIONS, FileKind.LOG),
    **dict.fromkeys(_MARKDOWN_EXTENSIONS, FileKind.MARKDOWN),
    **dict.fromkeys(_JSON_EXTENSIONS, FileKind.JSON),
    **dict.fromkeys(_CODE_EXTENSIONS, FileKind.CODE),
}


def _classify_by_content_type(content_type: str, sample_text: str) -> FileKind | None:
    """Classify FileKind from MIME content_type header when extension is unknown."""
    mime = content_type.split(";", maxsplit=1)[0].strip().lower()
    if not mime:
        return None
    if mime in _MIME_KIND_MAP:
        return _MIME_KIND_MAP[mime]
    if mime.endswith("+json"):
        return FileKind.JSON
    if mime == "text/plain":
        return FileKind.LOG if _looks_like_log(sample_text) else FileKind.TEXT
    return None


def _classify_by_sample(sample_text: str) -> FileKind:
    """Classify FileKind using content heuristics when extension and MIME are inconclusive."""
    if _looks_like_json(sample_text):
        return FileKind.JSON
    if _looks_like_log(sample_text):
        return FileKind.LOG
    if _MARKDOWN_HEADING_RE.search(sample_text):
        return FileKind.MARKDOWN
    return FileKind.OTHER


def detect_file_kind(
    filename: str,
    content_type: str = "",
    sample_text: str = "",
) -> FileKind:
    """Classify an uploaded project file into a logical FileKind."""
    path = Path(filename)
    lower_name = path.name.lower()
    ext = path.suffix.lower()

    if lower_name in _SPECIAL_CODE_FILENAMES:
        return FileKind.CODE
    if (by_ext := _EXTENSION_KIND_MAP.get(ext)) is not None:
        return by_ext
    if ext in _TEXT_EXTENSIONS:
        return FileKind.LOG if _looks_like_log(sample_text) else FileKind.TEXT
    if (by_mime := _classify_by_content_type(content_type, sample_text)) is not None:
        return by_mime
    return _classify_by_sample(sample_text)
