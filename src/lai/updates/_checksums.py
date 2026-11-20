"""Shared SHA-256 verification, SemVer parsing, and staging helpers for self-updates."""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    import structlog

__all__ = [
    "CHUNK_BYTES",
    "REQUIREMENTS_LOCK_FILENAME",
    "SHA256_RE",
    "WHEEL_RE",
    "compute_sha256",
    "make_upgrade_error",
    "parse_semver_tuple",
    "parse_sha256sums",
    "prepare_dest_path",
    "updates_logger",
    "verify_file_sha256",
    "verify_wheel_sha256",
]

CHUNK_BYTES = 65536
REQUIREMENTS_LOCK_FILENAME = "requirements-lock.txt"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)(?:[.\-+A-Za-z0-9]*)?$")
WHEEL_RE = re.compile(r"^lai-(\d+\.\d+\.\d+[A-Za-z0-9._+]*)-.+\.whl$")


def updates_logger() -> structlog.stdlib.BoundLogger:
    """Return the structured component logger for the updates subsystem."""
    return get_component_logger("updates")


def make_upgrade_error(code: str, msg: str, status: int = 500) -> LaiError:
    """Construct a typed LaiError for the updates subsystem."""
    return LaiError(msg, code=code, status_code=status)


def parse_semver_tuple(version: str) -> tuple[int, int, int]:
    """Parse a SemVer string (with optional leading 'v' or suffix) into (major, minor, patch)."""
    match = _SEMVER_RE.fullmatch(version.strip())
    if match is None:
        msg = f"Invalid SemVer version: {version!r}"
        raise ValueError(msg)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse sha256sum-formatted lines into a {filename: lowercase_hex_digest} mapping."""
    checksums: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:  # noqa: PLR2004
            raise make_upgrade_error(
                "upgrade_invalid_checksums", f"Malformed SHA256SUMS line: {raw_line!r}"
            )
        digest = parts[0].strip().lower()
        filename = Path(parts[1].strip().lstrip("*")).name
        if not SHA256_RE.fullmatch(digest) or not filename:
            raise make_upgrade_error(
                "upgrade_invalid_checksums", f"Invalid SHA256SUMS entry: {raw_line!r}"
            )
        checksums[filename] = digest
    return checksums


def compute_sha256(path: Path) -> str:
    """Compute the lowercase SHA-256 hex digest of a file using streaming reads."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_sha256(file_path: Path, expected_sha256: str) -> None:
    """Verify file_path matches expected_sha256 using constant-time comparison."""
    expected = expected_sha256.strip().lower()
    actual = compute_sha256(file_path)
    if not SHA256_RE.fullmatch(expected) or not hmac.compare_digest(actual, expected):
        updates_logger().error(
            "upgrade_wheel_checksum_mismatch",
            wheel_path=str(file_path),
            expected_sha256=expected,
            actual_sha256=actual,
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        msg = f"SHA-256 mismatch for {file_path.name}: expected {expected}, got {actual}."
        raise make_upgrade_error("upgrade_checksum_mismatch", msg)


def verify_wheel_sha256(wheel_path: Path, expected_sha256: str) -> None:
    """Verify wheel_path matches expected_sha256 using constant-time comparison."""
    verify_file_sha256(wheel_path, expected_sha256)


def prepare_dest_path(dest_dir: Path, filename: str) -> Path:
    """Validate filename has no directory traversal and prepare destination path."""
    if Path(filename).name != filename:
        raise make_upgrade_error("upgrade_invalid_wheel", f"Unsafe filename: {filename!r}", 400)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / filename
