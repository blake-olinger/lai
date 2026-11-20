"""Systemd service control, subprocess command runner, and HTTP health probe."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol, cast

import httpx

from lai.domain.errors import LaiError
from lai.ports.updates import HealthProbePort, ServiceControllerPort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

__all__ = [
    "CommandRunner",
    "HttpHealthProbe",
    "SystemdServiceController",
    "default_command_runner",
]

_HTTP_OK = 200


class CommandRunner(Protocol):
    """Async subprocess execution protocol returning (returncode, stdout, stderr)."""

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Run args without a shell and return (returncode, stdout, stderr)."""
        ...


async def default_command_runner(args: Sequence[str]) -> tuple[int, str, str]:
    """Execute a subprocess without a shell and return (returncode, stdout, stderr)."""
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


class SystemdServiceController(ServiceControllerPort):
    """ServiceControllerPort executing `systemctl stop/restart <service_name>`."""

    def __init__(
        self,
        *,
        systemctl_executable: str = "systemctl",
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._systemctl = systemctl_executable
        self._runner: CommandRunner = command_runner or default_command_runner

    async def _run_action(self, action: str, service_name: str, *, error_code: str) -> None:
        cleaned = service_name.strip()
        if not cleaned:
            msg = "Service name must not be empty."
            raise LaiError(msg, code=error_code, status_code=400)
        try:
            returncode, stdout, stderr = await self._runner([self._systemctl, action, cleaned])
        except OSError as exc:
            msg = f"Failed to execute systemctl {action} {cleaned}: {exc}"
            raise LaiError(msg, code=error_code, status_code=500) from exc
        if returncode != 0:
            detail = stderr or stdout or f"exit code {returncode}"
            msg = f"systemctl {action} {cleaned} failed: {detail}"
            raise LaiError(msg, code=error_code, status_code=500)

    async def restart(self, service_name: str) -> None:
        """Restart service_name via systemctl, raising LaiError on failure."""
        await self._run_action("restart", service_name, error_code="upgrade_service_restart_failed")

    async def stop(self, service_name: str) -> None:
        """Stop service_name via systemctl, raising LaiError on failure."""
        await self._run_action("stop", service_name, error_code="upgrade_service_stop_failed")


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
