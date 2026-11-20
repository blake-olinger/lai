"""Streaming content-addressed local filesystem blob store."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO

from lai.domain.errors import NotFoundError, PayloadTooLargeError, ValidationFailedError
from lai.ports.storage import BlobStorePort

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

DEFAULT_MAX_UPLOAD_BYTES = 262_144_000
_FLUSH_BUFFER_BYTES = 65_536
_FORBIDDEN_SEGMENT_CHARS = ("/", "\\", "\x00")
_POSIX_DIR_MODE = 0o700
_POSIX_FILE_MODE = 0o600


def _chmod_posix(path: Path, mode: int) -> None:
    """Apply POSIX file or directory permissions on a best-effort basis."""
    if os.name == "posix":
        with suppress(OSError):
            path.chmod(mode)


def _validate_segment(value: str, field_name: str) -> str:
    """Validate that a path segment cannot traverse directories or collide with temp dirs."""
    if (
        not value
        or value in {".", ".."}
        or value.startswith(".")
        or any(ch in value for ch in _FORBIDDEN_SEGMENT_CHARS)
    ):
        msg = f"Invalid {field_name} for blob storage: {value!r}"
        raise ValidationFailedError(msg, details={"field": field_name})
    return value


def _validate_filename(filename: str) -> str:
    """Validate uploaded filename metadata without trusting it as a filesystem path."""
    if not filename.strip() or "\x00" in filename:
        msg = f"Invalid uploaded filename: {filename!r}"
        raise ValidationFailedError(msg, details={"field": "filename"})
    return filename


def _open_temp_file(tmp_path: Path, blobs_root: Path) -> BinaryIO:
    """Create parent directories with private POSIX permissions and open staging file."""
    blobs_root.mkdir(parents=True, exist_ok=True)
    _chmod_posix(blobs_root, _POSIX_DIR_MODE)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_posix(tmp_path.parent, _POSIX_DIR_MODE)
    handle = tmp_path.open("wb")
    _chmod_posix(tmp_path, _POSIX_FILE_MODE)
    return handle


def _write_and_clear(handle: BinaryIO, payload: bytes) -> None:
    """Write buffered bytes to the staging file."""
    handle.write(payload)


def _finalize_temp_file(handle: BinaryIO, tmp_path: Path, dest_path: Path, trailing: bytes) -> None:
    """Flush trailing bytes, close staging file, and atomically move into content path."""
    if trailing:
        handle.write(trailing)
    handle.flush()
    handle.close()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_posix(dest_path.parent, _POSIX_DIR_MODE)
    tmp_path.replace(dest_path)
    _chmod_posix(dest_path, _POSIX_FILE_MODE)


def _cleanup_temp_file(handle: BinaryIO | None, tmp_path: Path) -> None:
    """Close and delete a staging file if an upload fails or aborts."""
    if handle is not None and not handle.closed:
        handle.close()
    tmp_path.unlink(missing_ok=True)


def _read_lines_range_sync(
    resolved_path: Path,
    start_line: int,
    end_line: int | None,
) -> str:
    """Read inclusive 1-based UTF-8 text lines from disk with bounded memory."""
    if not resolved_path.is_file():
        msg = f"Blob not found: {resolved_path}"
        raise NotFoundError(msg, details={"blob_path": str(resolved_path)})

    selected: list[str] = []
    with resolved_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            selected.append(raw_line.rstrip("\r\n"))
    return "\n".join(selected)


def _read_bytes_sync(resolved_path: Path) -> bytes:
    """Read raw bytes of an existing blob file."""
    if not resolved_path.is_file():
        msg = f"Blob not found: {resolved_path}"
        raise NotFoundError(msg, details={"blob_path": str(resolved_path)})
    return resolved_path.read_bytes()


def _validate_stream_chunk(
    chunk: bytes,
    new_total_bytes: int,
    max_upload_bytes: int,
    filename: str,
) -> None:
    """Reject NUL-containing binary payloads and uploads exceeding max_upload_bytes."""
    if b"\x00" in chunk:
        msg = "Binary payloads containing NUL bytes are not supported."
        raise ValidationFailedError(msg, details={"filename": filename})
    if new_total_bytes > max_upload_bytes:
        msg = f"Upload exceeds maximum allowed size of {max_upload_bytes} bytes."
        raise PayloadTooLargeError(
            msg,
            details={"max_upload_bytes": str(max_upload_bytes)},
        )


def _delete_blob_sync(resolved_path: Path) -> bool:
    """Delete an existing blob file and return whether it existed."""
    if not resolved_path.is_file():
        return False
    resolved_path.unlink(missing_ok=True)
    return True


class LocalBlobStore(BlobStorePort):
    """HTTP-upload-only local filesystem blob store keyed by project, SHA prefix, and file_id."""

    def __init__(
        self,
        blobs_dir: Path,
        *,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        if max_upload_bytes < 1:
            msg = "max_upload_bytes must be >= 1."
            raise ValidationFailedError(msg, details={"max_upload_bytes": str(max_upload_bytes)})
        self._blobs_dir = blobs_dir.resolve()
        self._max_upload_bytes = max_upload_bytes

    @property
    def blobs_dir(self) -> Path:
        """Return the resolved root directory for stored blobs."""
        return self._blobs_dir

    @property
    def max_upload_bytes(self) -> int:
        """Return the maximum permitted upload payload size in bytes."""
        return self._max_upload_bytes

    def set_max_upload_bytes(self, max_upload_bytes: int) -> None:
        """Update maximum upload payload size in bytes in place."""
        if max_upload_bytes < 1:
            msg = "max_upload_bytes must be >= 1."
            raise ValidationFailedError(msg, details={"max_upload_bytes": str(max_upload_bytes)})
        self._max_upload_bytes = max_upload_bytes

    def _resolve_blob_path(self, blob_path: Path) -> Path:
        """Resolve blob_path and verify it stays strictly within blobs_dir."""
        candidate = (
            blob_path.resolve()
            if blob_path.is_absolute()
            else (self._blobs_dir / blob_path).resolve()
        )
        if candidate == self._blobs_dir or not candidate.is_relative_to(self._blobs_dir):
            msg = f"Blob path escapes configured blobs directory: {blob_path}"
            raise ValidationFailedError(msg, details={"blob_path": str(blob_path)})
        return candidate

    async def write_stream(
        self,
        project_id: str,
        file_id: str,
        filename: str,
        chunks: AsyncIterator[bytes],
    ) -> tuple[Path, int, str]:
        """Stream bytes to disk, compute SHA-256, reject NUL binaries and oversized uploads."""
        safe_project = _validate_segment(project_id, "project_id")
        safe_file = _validate_segment(file_id, "file_id")
        _validate_filename(filename)

        tmp_path = self._blobs_dir / safe_project / ".tmp" / f"{safe_file}.{uuid.uuid4().hex}.part"
        hasher = hashlib.sha256()
        total_bytes = 0
        pending = bytearray()
        handle: BinaryIO | None = None

        try:
            handle = await asyncio.to_thread(_open_temp_file, tmp_path, self._blobs_dir)
            async for chunk in chunks:
                if not chunk:
                    continue
                total_bytes += len(chunk)
                _validate_stream_chunk(chunk, total_bytes, self._max_upload_bytes, filename)
                hasher.update(chunk)
                pending.extend(chunk)
                if len(pending) >= _FLUSH_BUFFER_BYTES:
                    await asyncio.to_thread(_write_and_clear, handle, bytes(pending))
                    pending.clear()

            sha256_hex = hasher.hexdigest()
            dest_path = self._blobs_dir / safe_project / sha256_hex[:2] / safe_file
            await asyncio.to_thread(
                _finalize_temp_file, handle, tmp_path, dest_path, bytes(pending)
            )
        except BaseException:
            await asyncio.to_thread(_cleanup_temp_file, handle, tmp_path)
            raise
        else:
            return dest_path, total_bytes, sha256_hex

    async def read_text_range(
        self,
        blob_path: Path,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        """Read UTF-8 text lines from blob_path within inclusive [start_line, end_line]."""
        if start_line < 1:
            msg = f"start_line must be >= 1, got {start_line}"
            raise ValidationFailedError(msg, details={"start_line": str(start_line)})
        if end_line is not None and end_line < start_line:
            msg = f"end_line ({end_line}) must be >= start_line ({start_line})"
            raise ValidationFailedError(msg, details={"end_line": str(end_line)})
        resolved = self._resolve_blob_path(blob_path)
        return await asyncio.to_thread(_read_lines_range_sync, resolved, start_line, end_line)

    async def read_bytes(self, blob_path: Path) -> bytes:
        """Read the raw bytes of a persisted blob."""
        resolved = self._resolve_blob_path(blob_path)
        return await asyncio.to_thread(_read_bytes_sync, resolved)

    async def delete_blob(self, blob_path: Path) -> bool:
        """Delete a persisted blob if it exists."""
        resolved = self._resolve_blob_path(blob_path)
        return await asyncio.to_thread(_delete_blob_sync, resolved)
