"""Specialized structural file chunkers for logs, source code, and prose."""

from __future__ import annotations

from lai.ingestion.chunkers.code import DEFAULT_CODE_CHUNK_TOKENS, chunk_code_file
from lai.ingestion.chunkers.logs import DEFAULT_LOG_CHUNK_TOKENS, chunk_log_file
from lai.ingestion.chunkers.prose import DEFAULT_PROSE_CHUNK_TOKENS, chunk_prose_file

__all__ = [
    "DEFAULT_CODE_CHUNK_TOKENS",
    "DEFAULT_LOG_CHUNK_TOKENS",
    "DEFAULT_PROSE_CHUNK_TOKENS",
    "chunk_code_file",
    "chunk_log_file",
    "chunk_prose_file",
]
