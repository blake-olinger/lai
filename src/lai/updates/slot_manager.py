"""Release slot provisioning, atomic symlink switching, systemd control, and health probing."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import httpx

from lai.domain.errors import LaiError
from lai.ports.updates import HealthProbePort, ServiceControllerPort, SlotInstallerPort
from lai.updates.sources import parse_semver_tuple

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

__all__ = [
    "CommandRunner",
    "HttpHealthProbe",
    "ReleaseSlotManager",
    "SystemdServiceController",
]

_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")
_HTTP_OK = 200


class CommandRunner(Protocol):
    """Async subprocess execution protocol returning (returncode, stdout, stderr)."""

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Run args without a shell and return (returncode, stdout, stderr)."""
        ...


async def _default_command_runner(args: Sequence[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    returncode = proc.returncode if proc.returncode is not None else 1
    return (
        returncode,
        stdout_bytes.decode("utf-8", errors="replace").strip(),
        stderr_bytes.decode("utf-8", errors="replace").strip(),
    )


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - POSIX hosts use bin/python
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


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

    def __init__(
        self,
        releases_dir: Path,
        current_symlink: Path,
        *,
        uv_executable: str = "uv",
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._releases_dir = releases_dir
        self._current_symlink = current_symlink
        self._uv_executable = uv_executable
        self._runner: CommandRunner = command_runner or _default_command_runner
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

    async def install_slot(self, version: str, wheel_path: Path) -> Path:
        """Create releases_dir/v<version>/.venv and install wheel_path into it via uv."""
        slot_path = await asyncio.to_thread(self._prepare_slot_dir, version, wheel_path)
        venv_dir = slot_path / ".venv"
        venv_python = _venv_python_path(venv_dir)
        await self._run_checked(
            [self._uv_executable, "venv", str(venv_dir)],
            step="uv venv",
            slot_path=slot_path,
        )
        await self._run_checked(
            [self._uv_executable, "pip", "install", "--python", str(venv_python), str(wheel_path)],
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


class SystemdServiceController(ServiceControllerPort):
    """ServiceControllerPort executing `systemctl restart <service_name>`."""

    def __init__(
        self,
        *,
        systemctl_executable: str = "systemctl",
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._systemctl = systemctl_executable
        self._runner: CommandRunner = command_runner or _default_command_runner

    async def restart(self, service_name: str) -> None:
        """Restart service_name via systemctl, raising LaiError on failure."""
        cleaned = service_name.strip()
        if not cleaned:
            msg = "Service name must not be empty."
            raise LaiError(msg, code="upgrade_service_restart_failed", status_code=400)
        try:
            returncode, stdout, stderr = await self._runner([self._systemctl, "restart", cleaned])
        except OSError as exc:
            msg = f"Failed to execute systemctl restart {cleaned}: {exc}"
            raise LaiError(msg, code="upgrade_service_restart_failed", status_code=500) from exc
        if returncode != 0:
            detail = stderr or stdout or f"exit code {returncode}"
            msg = f"systemctl restart {cleaned} failed: {detail}"
            raise LaiError(msg, code="upgrade_service_restart_failed", status_code=500)


class HttpHealthProbe(HealthProbePort):
    """HealthProbePort polling an HTTP JSON health endpoint until healthy or timed out."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        request_timeout_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http_client = http_client
        self._request_timeout_seconds = request_timeout_seconds
        self._clock = clock
        self._sleeper = sleeper

    @asynccontextmanager
    async def _client_ctx(self) -> AsyncGenerator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._request_timeout_seconds)
        ) as client:
            yield client

    async def _probe_once(
        self,
        client: httpx.AsyncClient,
        url: str,
        expected_version: str | None,
    ) -> bool:
        try:
            response = await client.get(url)
        except (httpx.HTTPError, OSError):
            return False
        if response.status_code != _HTTP_OK:
            return False
        try:
            payload: object = response.json()
        except ValueError:
            return False
        if not isinstance(payload, dict):
            return False
        data = cast("dict[str, object]", payload)
        if data.get("status") != "ok":
            return False
        return expected_version is None or data.get("version") == expected_version

    async def wait_healthy(
        self,
        url: str,
        *,
        expected_version: str | None,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> bool:
        """Poll url until status=='ok' (and version matches expected_version) or timeout elapses."""
        deadline = self._clock() + timeout_seconds
        async with self._client_ctx() as client:
            while True:
                if await self._probe_once(client, url, expected_version):
                    return True
                remaining = deadline - self._clock()
                if remaining <= 0.0:
                    return False
                await self._sleeper(min(poll_interval_seconds, remaining))
