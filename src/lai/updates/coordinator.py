"""Hexagonal upgrade and rollback coordinator orchestrating release updates safely."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory
from lai.domain.updates import (
    ReleaseArtifact,
    UpdateConfig,
    UpgradeCheckResult,
    UpgradeResult,
    UpgradeStatus,
)
from lai.updates._checksums import parse_semver_tuple, updates_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from lai.ports.updates import (
        DatabaseBackupPort,
        HealthProbePort,
        ReleaseSourcePort,
        ServiceControllerPort,
        SlotInstallerPort,
    )

__all__ = ["UpgradeCoordinator"]

_SECURITY_CODES = frozenset({"upgrade_checksum_mismatch", "upgrade_checksum_missing"})
_APP_ERR = AlertCategory.APPLICATION_ERROR
_DB_ERR = AlertCategory.STORAGE_FAILURE
_SEC_ERR = AlertCategory.SECURITY_ALERT
_TIMEOUT_ERR = AlertCategory.TIMEOUT


def _alert(
    event: str,
    cat: AlertCategory,
    error: str,
    exc: Exception | None = None,
    version: str | None = None,
) -> str:
    """Emit a structured error log tagged with AlertCategory and return error."""
    updates_logger().error(
        event,
        error=error,
        version=version,
        exception_type=type(exc).__name__ if exc is not None else None,
        alert_category=cat.value,
    )
    return error


def _res(  # noqa: PLR0913
    status: UpgradeStatus,
    from_ver: str,
    *,
    to_ver: str | None = None,
    slot: Path | None = None,
    backup: Path | None = None,
    rolled_back: bool = False,
    error: str | None = None,
) -> UpgradeResult:
    return UpgradeResult(
        status=status,
        from_version=from_ver,
        to_version=to_ver,
        slot_path=slot,
        backup_path=backup,
        rolled_back=rolled_back,
        error=error,
    )


class UpgradeCoordinator:
    """Orchestrates version checks, verified downloads, DB backups, activation, and rollback."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        config: UpdateConfig,
        source: ReleaseSourcePort,
        slots: SlotInstallerPort,
        service: ServiceControllerPort,
        health_probe: HealthProbePort,
        backup: DatabaseBackupPort | None = None,
    ) -> None:
        self._config, self._source, self._slots = config, source, slots
        self._service, self._health_probe, self._backup = service, health_probe, backup

    async def check(self, current_version: str) -> UpgradeCheckResult:
        """Query the configured release source and determine if an upgrade is available."""
        artifact = await self._source.get_latest(allow_prerelease=self._config.allow_prerelease)
        newer = parse_semver_tuple(artifact.version) > parse_semver_tuple(current_version)
        return UpgradeCheckResult(
            current_version=current_version,
            latest_version=artifact.version,
            update_available=newer,
            artifact=artifact,
        )

    async def _resolve_artifact(self, target_version: str | None) -> ReleaseArtifact:
        if target_version is not None:
            return await self._source.get_version(target_version)
        return await self._source.get_latest(allow_prerelease=self._config.allow_prerelease)

    async def _restart_and_verify(self, ver: str | None) -> tuple[bool, str | None]:
        try:
            await self._service.restart(self._config.service_name)
        except (LaiError, OSError) as exc:
            msg = f"Service restart failed: {exc}"
            return False, _alert("upgrade_restart_failed", _APP_ERR, msg, exc, ver)
        try:
            healthy = await self._health_probe.wait_healthy(
                self._config.health_url,
                expected_version=ver,
                timeout_seconds=self._config.health_timeout_seconds,
                poll_interval_seconds=self._config.health_poll_interval_seconds,
            )
        except (LaiError, OSError) as exc:
            msg = f"Health probe error: {exc}"
            return False, _alert("upgrade_health_error", _APP_ERR, msg, exc, ver)
        if not healthy:
            msg = f"Health probe failed for {ver or 'service'} at {self._config.health_url}."
            return False, _alert("upgrade_health_timeout", _TIMEOUT_ERR, msg, version=ver)
        return True, None

    async def _execute_rollback(
        self,
        *,
        slot: Path,
        ver: str | None,
        db_path: Path | None,
        backup_path: Path | None,
    ) -> tuple[bool, str | None]:
        if backup_path is not None and db_path is not None and self._backup is not None:
            try:
                await self._service.stop(self._config.service_name)
            except (LaiError, OSError) as exc:
                msg = f"Rollback service stop failed; refusing live DB overwrite: {exc}"
                return False, _alert("upgrade_rollback_stop_failed", _APP_ERR, msg, exc, ver)
            try:
                await self._backup.restore_backup(backup_path, db_path)
            except (LaiError, OSError) as exc:
                msg = f"Rollback DB restore failed: {exc}"
                return False, _alert("upgrade_rollback_db_failed", _DB_ERR, msg, exc, ver)
        try:
            self._slots.activate_slot(slot)
        except (LaiError, OSError) as exc:
            msg = f"Rollback symlink restore failed: {exc}"
            return False, _alert("upgrade_rollback_activate_failed", _APP_ERR, msg, exc, ver)
        return await self._restart_and_verify(ver)

    async def _prepare_slot(
        self, artifact: ReleaseArtifact, db_path: Path | None
    ) -> tuple[Path | None, Path | None, str | None]:
        backup_path: Path | None = None
        ver = artifact.version
        with tempfile.TemporaryDirectory(prefix="lai-upgrade-") as tmp_dir:
            stage_dir = Path(tmp_dir)
            try:
                wheel_path = await self._source.download_wheel(artifact, stage_dir)
                req_path = await self._source.download_requirements(artifact, stage_dir)
            except (LaiError, OSError) as exc:
                cat = (
                    _SEC_ERR
                    if isinstance(exc, LaiError) and exc.code in _SECURITY_CODES
                    else _APP_ERR
                )
                return None, None, _alert("upgrade_download_failed", cat, str(exc), exc, ver)
            if db_path is not None and self._backup is not None:
                try:
                    backup_path = await self._backup.create_backup(
                        db_path, self._config.backups_dir, f"pre-v{ver}"
                    )
                except (LaiError, OSError) as exc:
                    return None, None, _alert("upgrade_backup_failed", _DB_ERR, str(exc), exc, ver)
            try:
                slot_path = await self._slots.install_slot(
                    ver, wheel_path, requirements_path=req_path
                )
            except (LaiError, OSError) as exc:
                err = _alert("upgrade_install_failed", _APP_ERR, str(exc), exc, ver)
                return None, backup_path, err
        return slot_path, backup_path, None

    async def _activate_and_verify(
        self,
        *,
        current_ver: str,
        target_ver: str,
        slot_path: Path,
        backup_path: Path | None,
        db_path: Path | None,
    ) -> UpgradeResult:
        try:
            prev_slot = self._slots.activate_slot(slot_path)
        except (LaiError, OSError) as exc:
            err = _alert("upgrade_activate_failed", _APP_ERR, str(exc), exc, target_ver)
            return _res(
                UpgradeStatus.FAILED,
                current_ver,
                to_ver=target_ver,
                slot=slot_path,
                backup=backup_path,
                error=err,
            )

        healthy, reason = await self._restart_and_verify(target_ver)
        if healthy:
            self._slots.prune_old_slots(self._config.keep_releases, active=slot_path)
            updates_logger().info(
                "upgrade_completed", from_version=current_ver, to_version=target_ver
            )
            return _res(
                UpgradeStatus.UPGRADED,
                current_ver,
                to_ver=target_ver,
                slot=slot_path,
                backup=backup_path,
            )

        failure = reason or "Post-activation verification failed."
        if not self._config.auto_rollback or prev_slot is None:
            return _res(
                UpgradeStatus.FAILED,
                current_ver,
                to_ver=target_ver,
                slot=slot_path,
                backup=backup_path,
                error=failure,
            )
        rb_ok, rb_err = await self._execute_rollback(
            slot=prev_slot, ver=current_ver, db_path=db_path, backup_path=backup_path
        )
        if rb_ok:
            return _res(
                UpgradeStatus.ROLLED_BACK,
                current_ver,
                to_ver=target_ver,
                slot=prev_slot,
                backup=backup_path,
                rolled_back=True,
                error=failure,
            )
        return _res(
            UpgradeStatus.FAILED,
            current_ver,
            to_ver=target_ver,
            slot=slot_path,
            backup=backup_path,
            error=f"{failure}; automatic rollback failed: {rb_err}",
        )

    async def _run_pre_activate_hook(
        self,
        hook: Callable[[Path], Awaitable[Path | None]] | None,
        slot_path: Path,
        ver: str,
    ) -> str | None:
        if hook is None:
            return None
        try:
            await hook(slot_path)
        except (LaiError, OSError) as exc:
            return _alert("upgrade_pre_activate_failed", _APP_ERR, str(exc), exc, ver)
        return None

    async def apply(  # noqa: PLR0913
        self,
        current_version: str,
        *,
        target_version: str | None = None,
        dry_run: bool = False,
        db_path: Path | None = None,
        respect_enabled: bool = False,
        force: bool = False,
        pre_activate_hook: Callable[[Path], Awaitable[Path | None]] | None = None,
    ) -> UpgradeResult:
        """Execute a full verified upgrade with automatic rollback on post-activation failure."""
        if respect_enabled and not self._config.enabled:
            return _res(UpgradeStatus.SKIPPED_DISABLED, current_version)
        artifact = await self._resolve_artifact(target_version)
        ver = artifact.version
        if not force and parse_semver_tuple(ver) <= parse_semver_tuple(current_version):
            return _res(UpgradeStatus.UP_TO_DATE, current_version, to_ver=ver)
        if dry_run:
            return _res(UpgradeStatus.DRY_RUN, current_version, to_ver=ver)
        slot_path, backup_path, prep_err = await self._prepare_slot(artifact, db_path)
        if prep_err is not None or slot_path is None:
            return _res(
                UpgradeStatus.FAILED,
                current_version,
                to_ver=ver,
                backup=backup_path,
                error=prep_err or "Upgrade preparation failed.",
            )
        hook_err = await self._run_pre_activate_hook(pre_activate_hook, slot_path, ver)
        if hook_err is not None:
            return _res(
                UpgradeStatus.FAILED,
                current_version,
                to_ver=ver,
                slot=slot_path,
                backup=backup_path,
                error=hook_err,
            )
        return await self._activate_and_verify(
            current_ver=current_version,
            target_ver=ver,
            slot_path=slot_path,
            backup_path=backup_path,
            db_path=db_path,
        )

    async def rollback(
        self,
        current_version: str,
        *,
        db_path: Path | None = None,
        backup_path: Path | None = None,
    ) -> UpgradeResult:
        """Manually switch back to the previous release slot and verify service health."""
        target_slot = self._slots.previous_slot()
        if target_slot is None:
            err = "No previous release slot available."
            msg = _alert("upgrade_rollback_missing", _APP_ERR, err, version=current_version)
            return _res(UpgradeStatus.FAILED, current_version, error=msg)
        target_ver = target_slot.name.lstrip("vV") or None
        rb_ok, rb_err = await self._execute_rollback(
            slot=target_slot, ver=target_ver, db_path=db_path, backup_path=backup_path
        )
        status = UpgradeStatus.ROLLED_BACK if rb_ok else UpgradeStatus.FAILED
        return _res(
            status,
            current_version,
            to_ver=target_ver,
            slot=target_slot,
            backup=backup_path,
            rolled_back=rb_ok,
            error=None if rb_ok else (rb_err or "Manual rollback failed."),
        )
