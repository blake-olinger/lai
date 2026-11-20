"""Helpers for post-install slot binary refresh and POSIX privilege dropping during upgrades."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.errors import LaiError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.updates import CommandRunner

__all__ = [
    "build_refresh_binaries_command",
    "refresh_slot_binaries",
    "slot_lai_executable",
]


async def _run_command(args: Sequence[str]) -> tuple[int, str, str]:
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


def slot_lai_executable(slot_path: Path) -> Path:
    """Return the path to the `lai` CLI entrypoint inside a release slot's virtualenv."""
    if os.name == "nt":  # pragma: no cover - POSIX hosts use bin/lai
        return slot_path / ".venv" / "Scripts" / "lai.exe"
    return slot_path / ".venv" / "bin" / "lai"


def _resolve_posix_service_user(bin_dir: Path) -> str | None:
    """Resolve the non-root POSIX owner of bin_dir (or 'lai') when running as uid 0."""
    if sys.platform == "win32":
        return None
    if os.geteuid() != 0:  # pragma: no cover - root-only systemd drop
        return None
    import pwd  # noqa: PLC0415 - POSIX-only stdlib module

    if bin_dir.exists():  # pragma: no cover - POSIX root path
        uid = bin_dir.stat().st_uid
        if uid != 0:
            try:
                return str(pwd.getpwuid(uid).pw_name)
            except KeyError:
                pass
    try:  # pragma: no cover - POSIX root path
        return str(pwd.getpwnam("lai").pw_name)
    except KeyError:
        return None


def build_refresh_binaries_command(
    slot_path: Path,
    *,
    config_path: Path | None,
    bin_dir: Path,
) -> list[str]:
    """Construct the argv for running `<new_slot>/.venv/bin/lai models refresh-binaries`."""
    resolved_config = config_path
    if resolved_config is None:
        env_cfg = os.environ.get("LAI_CONFIG_FILE", "").strip()
        if env_cfg:
            resolved_config = Path(env_cfg)
    cmd = [str(slot_lai_executable(slot_path)), "models", "refresh-binaries"]
    if resolved_config is not None:
        cmd.extend(["--config", str(resolved_config)])
    target_user = _resolve_posix_service_user(bin_dir)
    if target_user is not None:
        runuser = shutil.which("runuser") or "/usr/sbin/runuser"
        return [runuser, "-u", target_user, "--", *cmd]
    return cmd


def _parse_binary_path_output(stdout: str, bin_dir: Path) -> Path:
    for line in stdout.splitlines():
        if line.startswith("llama_server:"):
            raw = line.split(":", 1)[1].strip()
            if raw:
                return Path(raw)
    return bin_dir / "llama" / "current" / "llama-server"


async def refresh_slot_binaries(
    slot_path: Path,
    *,
    config_path: Path | None,
    bin_dir: Path,
    command_runner: CommandRunner | None = None,
) -> Path:
    """Invoke the newly installed slot's `lai models refresh-binaries` before activation."""
    cmd: Sequence[str] = build_refresh_binaries_command(
        slot_path,
        config_path=config_path,
        bin_dir=bin_dir,
    )
    runner = command_runner or _run_command
    try:
        returncode, stdout, stderr = await runner(cmd)
    except OSError as exc:
        msg = f"Failed to execute slot binary refresh ({cmd[0]}): {exc}"
        raise LaiError(msg, code="upgrade_binaries_refresh_failed", status_code=500) from exc
    if returncode != 0:
        detail = stderr or stdout or f"exit code {returncode}"
        msg = f"Slot binary refresh failed for {slot_path.name}: {detail}"
        raise LaiError(msg, code="upgrade_binaries_refresh_failed", status_code=500)
    return _parse_binary_path_output(stdout, bin_dir)
