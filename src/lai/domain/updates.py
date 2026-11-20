"""Domain models and configuration for self-update and rollback orchestration."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator

from lai.domain.base import StrictModel

__all__ = [
    "ReleaseArtifact",
    "UpdateConfig",
    "UpgradeCheckResult",
    "UpgradeResult",
    "UpgradeStatus",
]

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class UpdateConfig(StrictModel):
    """Configuration governing release discovery, slot layout, health checks, and rollback."""

    enabled: bool = False
    github_repo: str = Field(default="blake-olinger/lai", min_length=3)
    releases_dir: Path = Path("/opt/lai/releases")
    current_symlink: Path = Path("/opt/lai/current")
    backups_dir: Path = Path("/var/lib/lai/backups")
    service_name: str = Field(default="lai.service", min_length=1)
    health_url: str = Field(default="http://127.0.0.1:8400/v1/health", min_length=1)
    health_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    health_poll_interval_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    keep_releases: int = Field(default=3, ge=1, le=20)
    auto_rollback: bool = True
    allow_prerelease: bool = False

    @field_validator("github_repo")
    @classmethod
    def _validate_github_repo(cls, value: str) -> str:
        """Ensure github_repo follows strict owner/repo slug format."""
        cleaned = value.strip()
        if not _GITHUB_REPO_RE.fullmatch(cleaned):
            msg = f"Invalid GitHub repository slug {value!r}; expected 'owner/repo'."
            raise ValueError(msg)
        return cleaned


class ReleaseArtifact(StrictModel):
    """Verified release wheel metadata retrieved from a remote or local source."""

    version: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    wheel_filename: str = Field(min_length=1)
    sha256: str
    download_url: str | None = None
    local_path: Path | None = None

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        """Normalize and enforce a 64-character lowercase hexadecimal SHA-256 digest."""
        normalized = value.strip().lower()
        if not _SHA256_HEX_RE.fullmatch(normalized):
            msg = f"Invalid SHA-256 digest {value!r}; expected 64 hexadecimal characters."
            raise ValueError(msg)
        return normalized


class UpgradeCheckResult(StrictModel):
    """Outcome of checking a release source for a newer version."""

    current_version: str = Field(min_length=1)
    latest_version: str = Field(min_length=1)
    update_available: bool
    artifact: ReleaseArtifact | None = None


class UpgradeStatus(StrEnum):
    """Terminal state reported after an upgrade or rollback attempt."""

    UP_TO_DATE = "up_to_date"
    DRY_RUN = "dry_run"
    UPGRADED = "upgraded"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    SKIPPED_DISABLED = "skipped_disabled"


class UpgradeResult(StrictModel):
    """Structured result describing an upgrade or rollback execution."""

    status: UpgradeStatus
    from_version: str = Field(min_length=1)
    to_version: str | None = None
    slot_path: Path | None = None
    backup_path: Path | None = None
    rolled_back: bool = False
    error: str | None = None
