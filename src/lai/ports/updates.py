"""Layer-5 hexagonal port contracts for self-update, slots, service control, and DB backups."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from lai.domain.updates import ReleaseArtifact

__all__ = [
    "DatabaseBackupPort",
    "HealthProbePort",
    "ReleaseSourcePort",
    "ServiceControllerPort",
    "SlotInstallerPort",
]


@runtime_checkable
class ReleaseSourcePort(Protocol):
    """Contract for discovering and fetching integrity-verified release wheels."""

    async def get_latest(self, *, allow_prerelease: bool = False) -> ReleaseArtifact:
        """Return metadata for the newest eligible release artifact."""
        ...

    async def get_version(self, version: str) -> ReleaseArtifact:
        """Return metadata for a specific target release version."""
        ...

    async def download_wheel(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path:
        """Download or stage the wheel into dest_dir and verify its SHA-256 digest."""
        ...

    async def download_requirements(
        self,
        artifact: ReleaseArtifact,
        dest_dir: Path,
    ) -> Path | None:
        """Download or stage verified locked requirements into dest_dir when present."""
        ...


@runtime_checkable
class SlotInstallerPort(Protocol):
    """Contract for installing, activating, tracking, and pruning release slots."""

    def current_slot(self) -> Path | None:
        """Return the currently active release slot path, if the symlink exists."""
        ...

    def previous_slot(self) -> Path | None:
        """Return the previously active release slot path eligible for rollback."""
        ...

    async def install_slot(
        self,
        version: str,
        wheel_path: Path,
        *,
        requirements_path: Path | None = None,
    ) -> Path:
        """Provision an isolated release slot virtualenv from wheel_path and locked requirements."""
        ...

    def activate_slot(self, slot_path: Path) -> Path | None:
        """Atomically point the active symlink to slot_path and return the previous slot."""
        ...

    def prune_old_slots(self, keep: int, *, active: Path | None = None) -> list[Path]:
        """Remove oldest release slots beyond keep while preserving active and previous slots."""
        ...


@runtime_checkable
class ServiceControllerPort(Protocol):
    """Contract for stopping and restarting the supervised LAI service."""

    async def stop(self, service_name: str) -> None:
        """Stop the named system service, raising on non-zero exit."""
        ...

    async def restart(self, service_name: str) -> None:
        """Restart the named system service, raising on non-zero exit."""
        ...


@runtime_checkable
class HealthProbePort(Protocol):
    """Contract for polling the HTTP health endpoint after a service restart."""

    async def wait_healthy(
        self,
        url: str,
        *,
        expected_version: str | None,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> bool:
        """Poll url until healthy (and matching expected_version if set) or timeout expires."""
        ...


@runtime_checkable
class DatabaseBackupPort(Protocol):
    """Contract for WAL-safe online SQLite backups and integrity-verified restores."""

    async def create_backup(self, db_path: Path, backups_dir: Path, label: str) -> Path:
        """Create an atomic consistent backup of db_path inside backups_dir."""
        ...

    async def restore_backup(self, backup_path: Path, db_path: Path) -> None:
        """Verify backup_path integrity and atomically restore it over db_path."""
        ...
