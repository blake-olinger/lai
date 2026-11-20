"""Release slot provisioning, atomic symlink switching, and old-slot pruning."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.errors import LaiError
from lai.ports.updates import SlotInstallerPort
from lai.updates._service_control import (
    CommandRunner,
    HttpHealthProbe,
    SystemdServiceController,
    default_command_runner,
)
from lai.updates.sources import parse_semver_tuple

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CommandRunner",
    "HttpHealthProbe",
    "ReleaseSlotManager",
    "SystemdServiceController",
]

_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - POSIX hosts use bin/python
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _resolve_uv_executable(configured: str) -> str:
    if configured != "uv":
        return configured
    if shutil.which("uv") is not None:
        return "uv"
    posix_uv = Path("/usr/local/bin/uv")
    if os.name == "posix" and posix_uv.is_file():  # pragma: no cover - Linux systemd PATH fallback
        return str(posix_uv)
    return configured


def _replace_symlink(staging: Path, target: Path) -> None:
    try:
        staging.replace(target)
    except PermissionError:  # pragma: no cover - Windows directory-symlink replacement fallback
        if os.name == "posix" or not target.is_symlink():
            raise
        target.unlink()
        staging.replace(target)


def _slot_sort_key(slot: Path) -> tuple[int, tuple[int, int, int], str]:
    mtime_ns = slot.stat().st_mtime_ns
    try:
        semver = parse_semver_tuple(slot.name)
    except ValueError:
        semver = (0, 0, 0)
    return (mtime_ns, semver, slot.name)


class ReleaseSlotManager(SlotInstallerPort):
    """Manage versioned release slots, uv virtualenvs, atomic symlinks, and pruning."""

    def __init__(  # noqa: PLR0913
        self,
        releases_dir: Path,
        current_symlink: Path,
        *,
        uv_executable: str = "uv",
        python_version: str = "3.13",
        require_hashes: bool = True,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._releases_dir = releases_dir
        self._current_symlink = current_symlink
        self._uv_executable = uv_executable
        self._python_version = python_version
        self._require_hashes = require_hashes
        self._runner: CommandRunner = command_runner or default_command_runner
        self._previous_slot: Path | None = None
        self._previous_marker = releases_dir / ".previous"

    def current_slot(self) -> Path | None:
        """Return the resolved active slot directory pointed to by current_symlink."""
        if not self._current_symlink.is_symlink():
            return None
        resolved = self._current_symlink.resolve()
        return resolved if resolved.is_dir() else None

    def previous_slot(self) -> Path | None:
        """Return the previous release slot from memory, marker file, or slot history."""
        current = self.current_slot()
        current_resolved = current.resolve() if current is not None else None
        if (
            self._previous_slot is not None
            and self._previous_slot.is_dir()
            and self._previous_slot.resolve() != current_resolved
        ):
            return self._previous_slot.resolve()
        if self._previous_marker.is_file():
            raw = self._previous_marker.read_text(encoding="utf-8").strip()
            if raw:
                candidate = Path(raw)
                if candidate.is_dir() and candidate.resolve() != current_resolved:
                    return candidate.resolve()
        for slot in self._list_slots_newest_first():
            if slot.resolve() != current_resolved:
                return slot.resolve()
        return None

    def _list_slots_newest_first(self) -> list[Path]:
        if not self._releases_dir.is_dir():
            return []
        slots = [
            entry
            for entry in self._releases_dir.iterdir()
            if entry.is_dir() and not entry.is_symlink() and entry.name.startswith("v")
        ]
        return sorted(slots, key=_slot_sort_key, reverse=True)

    def _prepare_slot_dir(self, version: str, wheel_path: Path) -> Path:
        clean_version = version.strip().lstrip("vV")
        if not _SAFE_VERSION_RE.fullmatch(clean_version) or ".." in clean_version:
            msg = f"Unsafe release version for slot directory: {version!r}"
            raise LaiError(msg, code="upgrade_invalid_version", status_code=400)
        if not wheel_path.is_file():
            msg = f"Wheel file does not exist: {wheel_path}"
            raise LaiError(msg, code="upgrade_invalid_wheel", status_code=404)
        self._releases_dir.mkdir(parents=True, exist_ok=True)
        slot_path = (self._releases_dir / f"v{clean_version}").resolve()
        current = self.current_slot()
        if current is not None and current.resolve() == slot_path:
            msg = f"Cannot overwrite currently active release slot: {slot_path}"
            raise LaiError(msg, code="upgrade_slot_active", status_code=409)
        if slot_path.exists():
            shutil.rmtree(slot_path)
        slot_path.mkdir(parents=True, exist_ok=True)
        return slot_path

    async def _run_checked(self, args: Sequence[str], *, step: str, slot_path: Path) -> None:
        try:
            returncode, stdout, stderr = await self._runner(args)
        except OSError as exc:
            shutil.rmtree(slot_path, ignore_errors=True)
            msg = f"Failed to execute {step} for {slot_path.name}: {exc}"
            raise LaiError(msg, code="upgrade_slot_install_failed", status_code=500) from exc
        if returncode != 0:
            shutil.rmtree(slot_path, ignore_errors=True)
            detail = stderr or stdout or f"exit code {returncode}"
            msg = f"Slot {step} failed for {slot_path.name}: {detail}"
            raise LaiError(msg, code="upgrade_slot_install_failed", status_code=500)

    def _validate_requirements_path(self, requirements_path: Path | None) -> None:
        if self._require_hashes and requirements_path is None:
            msg = "Locked requirements-lock.txt is required when require_hashes=True."
            raise LaiError(msg, code="upgrade_requirements_missing", status_code=400)
        if requirements_path is not None and not requirements_path.is_file():
            msg = f"Requirements lockfile does not exist: {requirements_path}"
            raise LaiError(msg, code="upgrade_requirements_missing", status_code=404)

    async def install_slot(
        self,
        version: str,
        wheel_path: Path,
        *,
        requirements_path: Path | None = None,
    ) -> Path:
        """Create releases_dir/v<version>/.venv and install hashed deps + wheel via uv."""
        self._validate_requirements_path(requirements_path)
        slot_path = await asyncio.to_thread(self._prepare_slot_dir, version, wheel_path)
        uv = _resolve_uv_executable(self._uv_executable)
        venv_dir = slot_path / ".venv"
        venv_python = _venv_python_path(venv_dir)
        await self._run_checked(
            [uv, "venv", "--python", self._python_version, str(venv_dir)],
            step="uv venv",
            slot_path=slot_path,
        )
        if requirements_path is not None:
            dep_cmd = [uv, "pip", "install", "--python", str(venv_python)]
            if self._require_hashes:
                dep_cmd.append("--require-hashes")
            dep_cmd.extend(["-r", str(requirements_path)])
            await self._run_checked(
                dep_cmd, step="uv pip install requirements", slot_path=slot_path
            )
            await self._run_checked(
                [uv, "pip", "install", "--python", str(venv_python), "--no-deps", str(wheel_path)],
                step="uv pip install wheel",
                slot_path=slot_path,
            )
        else:
            await self._run_checked(
                [uv, "pip", "install", "--python", str(venv_python), str(wheel_path)],
                step="uv pip install",
                slot_path=slot_path,
            )
        return slot_path

    def activate_slot(self, slot_path: Path) -> Path | None:
        """Atomically flip current_symlink to slot_path via a staging symlink and replace."""
        target = slot_path.resolve()
        if not target.is_dir():
            msg = f"Release slot directory does not exist: {slot_path}"
            raise LaiError(msg, code="upgrade_slot_missing", status_code=404)
        previous = self.current_slot()
        self._current_symlink.parent.mkdir(parents=True, exist_ok=True)
        staging = self._current_symlink.parent / (
            f".{self._current_symlink.name}.staging-{uuid.uuid4().hex[:8]}"
        )
        try:
            staging.symlink_to(target, target_is_directory=True)
            _replace_symlink(staging, self._current_symlink)
        except OSError as exc:
            staging.unlink(missing_ok=True)
            msg = f"Failed to activate release slot {target}: {exc}"
            raise LaiError(msg, code="upgrade_activation_failed", status_code=500) from exc

        if previous is not None and previous.resolve() != target:
            self._previous_slot = previous.resolve()
            self._releases_dir.mkdir(parents=True, exist_ok=True)
            self._previous_marker.write_text(str(self._previous_slot), encoding="utf-8")
        return previous

    def prune_old_slots(self, keep: int, *, active: Path | None = None) -> list[Path]:
        """Delete oldest slots beyond keep while always preserving active and previous slots."""
        if keep < 1:
            msg = f"keep must be >= 1, got {keep}"
            raise ValueError(msg)
        candidates = self._list_slots_newest_first()
        preserved = {
            path.resolve()
            for path in (active, self.current_slot(), self.previous_slot())
            if path is not None
        }
        retained = {slot.resolve() for slot in candidates[:keep]} | preserved
        removed: list[Path] = []
        for slot in candidates:
            if slot.resolve() not in retained:
                shutil.rmtree(slot)
                removed.append(slot)
        return removed
