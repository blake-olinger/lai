"""Unit tests for LocalBlobStore streaming safety and FileKind/token detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tiktoken

from lai.domain.errors import NotFoundError, PayloadTooLargeError, ValidationFailedError
from lai.domain.files import FileKind
from lai.ingestion.blob_store import LocalBlobStore
from lai.ingestion.detector import (
    clear_token_encoder_cache,
    detect_file_kind,
    detect_language,
    estimate_tokens,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

EXPECTED_TWO_TOKENS = 2
EXPECTED_THREE_TOKENS = 3
LARGE_BLOB_BYTES = 70_000


async def _iter_chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def test_local_blob_store_streams_hashes_and_reads_ranges(tmp_path: Path) -> None:
    """LocalBlobStore writes under blobs_dir/<project>/<sha[:2]>/<file_id> and reads lines."""
    blobs_dir = tmp_path / "blobs"
    store = LocalBlobStore(blobs_dir, max_upload_bytes=1024)
    content = b"line one\r\nline two\r\nline three\nline four\n"
    expected_sha = hashlib.sha256(content).hexdigest()

    blob_path, size_bytes, sha256 = await store.write_stream(
        "prj_alpha",
        "fil_001",
        "app.log",
        _iter_chunks(content[:10], b"", content[10:]),
    )
    assert size_bytes == len(content)
    assert sha256 == expected_sha
    assert blob_path == blobs_dir.resolve() / "prj_alpha" / expected_sha[:2] / "fil_001"
    assert await store.read_bytes(blob_path) == content
    assert (
        await store.read_text_range(blob_path, start_line=2, end_line=3) == "line two\nline three"
    )
    assert await store.read_text_range(blob_path, start_line=10) == ""

    assert await store.delete_blob(blob_path) is True
    assert await store.delete_blob(blob_path) is False

    # Exercise multi-chunk > 64 KiB streaming flush and line-range validation
    big_store = LocalBlobStore(blobs_dir, max_upload_bytes=100_000)
    big_payload = b"a" * LARGE_BLOB_BYTES
    big_path, big_size, _ = await big_store.write_stream(
        "prj_alpha", "fil_big_ok", "big.txt", _iter_chunks(big_payload)
    )
    assert big_size == LARGE_BLOB_BYTES
    assert len(await big_store.read_bytes(big_path)) == LARGE_BLOB_BYTES
    with pytest.raises(ValidationFailedError):
        await big_store.read_text_range(big_path, start_line=0)
    with pytest.raises(ValidationFailedError):
        await big_store.read_text_range(big_path, start_line=3, end_line=2)


async def test_local_blob_store_rejects_nul_oversize_and_traversal(tmp_path: Path) -> None:
    """LocalBlobStore blocks NUL binary payloads, size overruns, and path traversal."""
    blobs_dir = tmp_path / "blobs"
    store = LocalBlobStore(blobs_dir, max_upload_bytes=16)

    with pytest.raises(ValidationFailedError):
        await store.write_stream(
            "prj_1",
            "fil_nul",
            "binary.dat",
            _iter_chunks(b"abc\x00def"),
        )

    with pytest.raises(PayloadTooLargeError):
        await store.write_stream(
            "prj_1",
            "fil_big",
            "big.log",
            _iter_chunks(b"0123456789", b"0123456789"),
        )

    # Confirm aborted streams left no lingering .tmp artifacts
    tmp_dir = blobs_dir / "prj_1" / ".tmp"
    assert not tmp_dir.exists() or list(tmp_dir.iterdir()) == []

    with pytest.raises(ValidationFailedError):
        await store.write_stream("../escape", "fil_1", "a.txt", _iter_chunks(b"ok"))

    outside = tmp_path / "secret.txt"
    outside.write_text("do not read", encoding="utf-8")
    with pytest.raises(ValidationFailedError):
        await store.read_bytes(outside)
    with pytest.raises(ValidationFailedError):
        await store.read_text_range(Path("../secret.txt"))
    with pytest.raises(NotFoundError):
        await store.read_bytes(blobs_dir / "prj_1" / "ab" / "missing")


def test_detector_file_kind_and_offline_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """detector classifies all FileKinds and falls back gracefully when tiktoken is offline."""
    assert detect_file_kind("service.log") is FileKind.LOG
    assert detect_file_kind("worker.py") is FileKind.CODE
    assert detect_file_kind("Dockerfile") is FileKind.CODE
    assert detect_file_kind("README.md") is FileKind.MARKDOWN
    assert detect_file_kind("config.json") is FileKind.JSON
    assert detect_file_kind("notes.txt", sample_text="plain shopping list") is FileKind.TEXT
    assert (
        detect_file_kind(
            "crash.txt",
            sample_text="2026-11-17T10:00:00Z ERROR failed\nTraceback (most recent call last):\n",
        )
        is FileKind.LOG
    )
    assert (
        detect_file_kind("unknown.bin", content_type="application/octet-stream") is FileKind.OTHER
    )
    assert detect_file_kind("payload", sample_text='{"ok": true}') is FileKind.JSON
    assert detect_file_kind("events", sample_text='{"a": 1}\n{"a": 2}\n') is FileKind.JSON
    assert detect_file_kind("doc", content_type="text/markdown") is FileKind.MARKDOWN
    assert detect_file_kind("api", content_type="application/vnd.custom+json") is FileKind.JSON
    assert detect_file_kind("script", content_type="text/x-python") is FileKind.CODE
    assert detect_file_kind("plain", content_type="text/plain", sample_text="hi") is FileKind.TEXT
    assert detect_file_kind("notes", sample_text="# Section One\n\nBody") is FileKind.MARKDOWN
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("README.md") == "markdown"
    assert detect_language("data.json") == "json"
    assert detect_language("file.unknown") == "unknown"

    # Special tokens inside logs must never crash tiktoken
    special_count = estimate_tokens("hello <|endoftext|> world")
    assert special_count >= 1
    assert estimate_tokens("") == 0

    # Simulate offline environment where tiktoken cannot fetch encoding
    def _raise_offline(_name: str) -> tiktoken.Encoding:
        msg = "offline"
        raise OSError(msg)

    monkeypatch.setattr(tiktoken, "get_encoding", _raise_offline)
    assert estimate_tokens("hello world", allow_tiktoken=False) == EXPECTED_THREE_TOKENS

    clear_token_encoder_cache()
    try:
        assert estimate_tokens("12345678") == EXPECTED_TWO_TOKENS
    finally:
        clear_token_encoder_cache()
