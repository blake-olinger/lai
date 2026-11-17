"""Subprocess lifecycle supervisor and GPU telemetry for local llama-server."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

from lai.domain.errors import InferenceBackendError
from lai.domain.inference import BackendHealth, ModelProfile
from lai.domain.observability import AlertCategory
from lai.inference._winjob import (
    GpuTelemetry,
    WindowsJobObject,
    kill_process_group,
    parse_nvidia_smi_output,
    pump_stream,
    terminate_process_group,
)
from lai.inference.flags import build_llama_server_args
from lai.inference.llamacpp_client import LlamaCppProvider
from lai.observability.logging import get_component_logger
from lai.observability.rotation import DailyRotatingStreamWriter
from lai.ports.inference import InferenceSupervisorPort

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    import structlog

_GPU_QUERY_TIMEOUT_SECONDS = 3.0
_THREAD_JOIN_TIMEOUT_SECONDS = 2.0
_FORCE_KILL_WAIT_SECONDS = 5.0
_MIN_POLL_SLEEP_SECONDS = 0.01

__all__ = ["GpuTelemetry", "LlamaCppSupervisor", "query_gpu_telemetry"]


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


def query_gpu_telemetry() -> GpuTelemetry | None:
    """Query GPU memory and utilization via nvidia-smi, returning None if unavailable."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GPU_QUERY_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None
    return parse_nvidia_smi_output(completed.stdout)


class LlamaCppSupervisor(InferenceSupervisorPort):
    """Lifecycle supervisor managing local llama-server subprocess and health polling."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        executable_path: Path,
        models_dir: Path,
        log_dir: Path,
        host: str = "127.0.0.1",
        port: int = 8081,
        startup_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.25,
        shutdown_timeout_seconds: float = 10.0,
        model_path_resolver: Callable[[ModelProfile], Path] | None = None,
        health_probe: Callable[[], Awaitable[BackendHealth]] | None = None,
    ) -> None:
        self._executable_path = executable_path
        self._models_dir = models_dir
        self._log_dir = log_dir
        self._host = host
        self._port = port
        self._startup_timeout_seconds = startup_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._model_path_resolver = model_path_resolver
        self._health_probe = health_probe

        self._lifecycle_lock = asyncio.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._active_profile: ModelProfile | None = None
        self._job_object: WindowsJobObject | None = None
        self._log_writer: DailyRotatingStreamWriter | None = None
        self._output_thread: threading.Thread | None = None
        self._health_client: LlamaCppProvider | None = None

    def active_profile(self) -> ModelProfile | None:
        """Return the currently loaded model profile if the process is running."""
        if self._process is None or self._process.poll() is not None:
            return None
        return self._active_profile

    def _resolve_executable(self) -> str:
        if self._executable_path.exists():
            return str(self._executable_path)
        if len(self._executable_path.parts) == 1:
            resolved = shutil.which(str(self._executable_path))
            if resolved is not None:
                return resolved
        msg = f"llama-server executable not found: {self._executable_path}"
        raise InferenceBackendError(msg, details={"executable_path": str(self._executable_path)})

    def _resolve_model_path(self, profile: ModelProfile) -> Path:
        path = (
            self._model_path_resolver(profile)
            if self._model_path_resolver is not None
            else self._models_dir / profile.hf_filename
        )
        if not path.exists():
            msg = f"GGUF model file not found: {path}"
            raise InferenceBackendError(msg, details={"model_path": str(path)})
        return path

    async def _probe_health(self) -> BackendHealth:
        if self._health_probe is not None:
            return await self._health_probe()
        if self._health_client is None:
            self._health_client = LlamaCppProvider(
                host=self._host,
                port=self._port,
                health_timeout_seconds=2.0,
            )
        return await self._health_client.health()

    def _spawn_process(self, profile: ModelProfile) -> subprocess.Popen[bytes]:
        exe = self._resolve_executable()
        model_path = self._resolve_model_path(profile)
        cmd = [exe, *build_llama_server_args(profile, model_path, self._host, self._port)]
        writer = DailyRotatingStreamWriter(
            self._log_dir / "llama-server.log",
            logger_name="lai.inference.llama_server",
        )
        try:
            proc = self._popen_process(cmd)
        except OSError as exc:
            writer.close()
            _logger().exception("llama_server_spawn_failed", executable=exe, error=str(exc))
            msg = f"Failed to spawn llama-server: {exc}"
            raise InferenceBackendError(msg) from exc

        self._process = proc
        self._log_writer = writer
        if sys.platform == "win32":
            self._attach_windows_job(proc.pid)
        if proc.stdout is not None:
            thread = threading.Thread(
                target=pump_stream,
                args=(proc.stdout, writer),
                daemon=True,
                name="llama-server-log-pump",
            )
            thread.start()
            self._output_thread = thread
        _logger().info("llama_server_spawned", profile=profile.name, pid=proc.pid, port=self._port)
        return proc

    @staticmethod
    def _popen_process(cmd: list[str]) -> subprocess.Popen[bytes]:
        if sys.platform == "win32":
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )

    def _attach_windows_job(self, pid: int) -> None:
        try:
            job = WindowsJobObject()
            if job.active:
                job.assign_pid(pid)
                self._job_object = job
        except OSError as exc:
            _logger().warning("windows_job_attach_failed", pid=pid, error=str(exc))

    async def _cleanup_resources(self) -> None:
        thread = self._output_thread
        self._output_thread = None
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, _THREAD_JOIN_TIMEOUT_SECONDS)
        if self._job_object is not None:
            self._job_object.close()
            self._job_object = None
        if self._log_writer is not None:
            self._log_writer.close()
            self._log_writer = None
        if self._health_client is not None:
            await self._health_client.aclose()
            self._health_client = None
        self._process = None
        self._active_profile = None

    async def _stop_locked(self) -> None:
        proc = self._process
        if proc is not None and proc.poll() is None:
            terminate_process_group(proc)
            try:
                await asyncio.to_thread(proc.wait, self._shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                kill_process_group(proc)
                await asyncio.to_thread(proc.wait, _FORCE_KILL_WAIT_SECONDS)
            _logger().info("llama_server_stopped", pid=proc.pid)
        await self._cleanup_resources()

    async def _await_startup_ready(
        self,
        proc: subprocess.Popen[bytes],
        profile: ModelProfile,
    ) -> BackendHealth:
        deadline = time.monotonic() + self._startup_timeout_seconds
        while True:
            returncode = proc.poll()
            if returncode is not None:
                await self._cleanup_resources()
                _logger().error(
                    "llama_server_startup_crashed",
                    profile=profile.name,
                    pid=proc.pid,
                    returncode=returncode,
                    alert_category=AlertCategory.SUBPROCESS_CRASH.value,
                )
                msg = f"llama-server exited unexpectedly during startup with code {returncode}"
                raise InferenceBackendError(msg, details={"returncode": str(returncode)})

            health = await self._probe_health()
            if health.healthy:
                self._active_profile = profile
                _logger().info("llama_server_ready", profile=profile.name, pid=proc.pid)
                loaded = health.model_loaded or profile.name
                return health.model_copy(update={"model_loaded": loaded})

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep_for = min(
                self._poll_interval_seconds,
                max(_MIN_POLL_SLEEP_SECONDS, remaining),
            )
            await asyncio.sleep(sleep_for)

        _logger().error(
            "llama_server_startup_timeout",
            profile=profile.name,
            pid=proc.pid,
            timeout_seconds=self._startup_timeout_seconds,
            alert_category=AlertCategory.TIMEOUT.value,
        )
        await self._stop_locked()
        msg = f"llama-server startup timed out after {self._startup_timeout_seconds}s"
        raise InferenceBackendError(msg)

    async def ensure_running(self, profile: ModelProfile) -> BackendHealth:
        """Start or verify llama-server for the requested model profile."""
        async with self._lifecycle_lock:
            if self._process is not None:
                returncode = self._process.poll()
                if returncode is not None:
                    await self._record_unexpected_exit(self._process, returncode)
                elif self._active_profile == profile:
                    current = await self._probe_health()
                    if current.healthy:
                        loaded = current.model_loaded or profile.name
                        return current.model_copy(update={"model_loaded": loaded})
                    await self._stop_locked()
                else:
                    await self._stop_locked()

            proc = self._spawn_process(profile)
            try:
                return await self._await_startup_ready(proc, profile)
            except asyncio.CancelledError:
                await self._stop_locked()
                raise

    async def _record_unexpected_exit(
        self,
        proc: subprocess.Popen[bytes],
        returncode: int,
    ) -> None:
        profile_name = self._active_profile.name if self._active_profile else None
        _logger().error(
            "llama_server_crashed",
            profile=profile_name,
            pid=proc.pid,
            returncode=returncode,
            alert_category=AlertCategory.SUBPROCESS_CRASH.value,
        )
        await self._cleanup_resources()

    async def stop(self) -> None:
        """Gracefully terminate the supervised inference process."""
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def health(self) -> BackendHealth:
        """Return process and endpoint health status."""
        proc = self._process
        if proc is None:
            return BackendHealth(healthy=False, status="stopped")
        returncode = proc.poll()
        if returncode is not None:
            await self._record_unexpected_exit(proc, returncode)
            return BackendHealth(
                healthy=False,
                status="crashed",
                error=f"llama-server exited unexpectedly with code {returncode}",
            )
        health = await self._probe_health()
        if health.healthy and health.model_loaded is None and self._active_profile is not None:
            return health.model_copy(update={"model_loaded": self._active_profile.name})
        return health
