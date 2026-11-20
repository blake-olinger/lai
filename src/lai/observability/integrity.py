"""Streaming SHA-256 integrity verification helpers shared across Layer 3 subsystems."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import TYPE_CHECKING

from lai.domain.errors import InferenceBackendError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from pathlib import Path

_SHA256_CHUNK_BYTES = 1024 * 1024
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

__all__ = [
    "compute_file_sha256",
    "verify_file_sha256",
]


def compute_file_sha256(path: Path) -> str:
    """Compute the lowercase SHA-256 hex digest of a file using streaming 1 MiB reads."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA256_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_file_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    """Verify file SHA-256 case-insensitively with constant-time comparison.

    Logs a ``SECURITY_ALERT`` and raises ``InferenceBackendError`` on mismatch.
    """
    expected = expected_sha256.strip().lower()
    if _SHA256_HEX_RE.fullmatch(expected) is None:
        get_component_logger("observability").error(
            "model_integrity_violation",
            label=label,
            path=str(path),
            expected_sha256=expected,
            reason="invalid_expected_sha256",
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        msg = f"Invalid expected SHA-256 digest for {label}: must be 64 hex characters."
        raise InferenceBackendError(msg, details={"label": label, "path": str(path)})

    if not path.is_file():
        msg = f"File not found for SHA-256 verification ({label}): {path}"
        raise InferenceBackendError(msg, details={"label": label, "path": str(path)})

    try:
        actual = compute_file_sha256(path)
    except OSError as exc:
        msg = f"Failed reading {label} for SHA-256 verification ({path}): {exc}"
        raise InferenceBackendError(msg, details={"label": label, "path": str(path)}) from exc

    if not hmac.compare_digest(actual, expected):
        get_component_logger("observability").error(
            "model_integrity_violation",
            label=label,
            path=str(path),
            expected_sha256=expected,
            actual_sha256=actual,
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        msg = (
            f"SHA-256 integrity mismatch for {label} ({path.name}): "
            f"expected {expected}, got {actual}"
        )
        raise InferenceBackendError(
            msg,
            details={
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
            },
        )
    return actual
