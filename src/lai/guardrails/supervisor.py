"""Supervised CPU-only llama-server process lifecycle for ShieldGemma-2B."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lai.domain.errors import InferenceBackendError
from lai.domain.inference import BackendHealth
from lai.domain.observability import AlertCategory
from lai.guardrails._process import (
    kill_process_tree,
    make_pdeathsig_preexec,
    terminate_process_tree,
)
from lai.guardrails.shieldgemma import ShieldGemmaCpuProvider
from lai.observability.integrity import verify_file_sha256
from lai.observability.logging import get_component_logger
from lai.observability.rotation import DailyRotatingStreamWriter
from lai.ports.guardrails import GuardrailSupervisorPort

if TYPE_CHECKING:
    import httpx

    from lai.domain.guardrails import GuardrailConfig

_DEFAULT_STOP_TIMEOUT_SECONDS = 5.0
_DEFAULT_RETENTION_DAYS = 14


def build_shieldgemma_server_args(config: GuardrailConfig, model_path: Path) -> list[str]:
    """Build CPU-only llama-server CLI arguments for ShieldGemma-2B."""
    return [
        "--model",
        str(model_path),
        "--ngl",
        "0",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--threads",
        str(config.cpu_threads),
        "--ctx-size",
        str(config.ctx_size),
        "--parallel",
        str(config.parallel_slots),
        "--no-mmap",
    ]


class ShieldGemmaSupervisor(GuardrailSupervisorPort):
    """Lifecycle manager for the CPU-only ShieldGemma-2B llama-server subprocess."""

    def __init__(  # noqa: PLR0913
        self,
        executable_path: Path,
        model_path: Path,
        *,
        log_dir: Path = Path("data/logs"),
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        stop_timeout_seconds: float = _DEFAULT_STOP_TIMEOUT_SECONDS,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        compress_rotated: bool = True,
        http_client: httpx.AsyncClient | None = None,
        expected_sha256: str | None = None,
        verify_sha256: bool = False,
    ) -> None:
        self._executable_path = executable_path
        self._model_path = model_path
        self._log_dir = log_dir
        self._startup_timeout_seconds = startup_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._retention_days = retention_days
        self._compress_rotated = compress_rotated
        self._http_client = http_client
        self._expected_sha256 = (
            expected_sha256.strip().lower() if expected_sha256 is not None else None
        )
        self._verify_sha256 = verify_sha256

        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._config: GuardrailConfig | None = None
        self._stream_writer: DailyRotatingStreamWriter | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._last_exit_code: int | None = None
        self._log = get_component_logger("guardrails")

    async def ensure_running(self, config: GuardrailConfig) -> BackendHealth:
        """Start or verify the CPU-only ShieldGemma llama-server process."""
        async with self._lock:
            if (
                self._process is not None
                and self._process.returncode is None
                and self._config == config
            ):
                current = await self.health()
                if current.healthy:
                    return current

            if self._process is not None:
                await self._stop_locked()

            self._validate_paths()
            await self._verify_model_integrity(config)
            await self._spawn_locked(config)
            return await self._wait_for_ready_locked(config)

    async def stop(self) -> None:
        """Gracefully terminate the supervised ShieldGemma server process."""
        async with self._lock:
            await self._stop_locked()

    async def health(self) -> BackendHealth:
        """Return process status and /health probe snapshot."""
        if self._process is None:
            if self._last_exit_code is not None:
                return BackendHealth(
                    healthy=False,
                    status="crashed",
                    error=f"Process exited unexpectedly with code {self._last_exit_code}",
                )
            return BackendHealth(healthy=False, status="stopped")

        if self._process.returncode is not None:
            return BackendHealth(
                healthy=False,
                status="crashed",
                error=f"Process exited with code {self._process.returncode}",
            )

        if self._config is None:
            return BackendHealth(healthy=False, status="unconfigured")
        provider = ShieldGemmaCpuProvider(self._config, client=self._http_client)
        return await provider.health()

    def _validate_paths(self) -> None:
        """Ensure executable and GGUF model files exist before spawning."""
        if not self._executable_path.exists():
            msg = f"ShieldGemma executable not found: {self._executable_path}"
            raise InferenceBackendError(msg, details={"path": str(self._executable_path)})
        if not self._model_path.exists():
            msg = f"ShieldGemma GGUF model not found: {self._model_path}"
            raise InferenceBackendError(msg, details={"path": str(self._model_path)})

    async def _verify_model_integrity(self, config: GuardrailConfig) -> None:
        """Verify ShieldGemma GGUF SHA-256 before spawn when integrity verification is enabled."""
        if not self._verify_sha256:
            return
        expected = self._expected_sha256 or config.sha256
        if not expected:
            self._log.error(
                "shieldgemma_model_integrity_missing_digest",
                path=str(self._model_path),
                alert_category=AlertCategory.SECURITY_ALERT.value,
            )
            msg = "ShieldGemma SHA-256 verification enabled but no expected digest is configured."
            raise InferenceBackendError(msg, details={"path": str(self._model_path)})

        await asyncio.to_thread(
            verify_file_sha256,
            self._model_path,
            expected,
            label="shieldgemma",
        )

    async def _spawn_locked(self, config: GuardrailConfig) -> None:
        """Spawn llama-server in a new POSIX session and start rotating log pumping."""
        self._stopping = False
        self._last_exit_code = None
        self._config = config
        writer = DailyRotatingStreamWriter(
            self._log_dir / "shieldgemma-server.log",
            retention_days=self._retention_days,
            compress=self._compress_rotated,
            logger_name="lai.guardrails.supervisor.stream",
        )
        args = build_shieldgemma_server_args(config, self._model_path)
        try:
            if sys.platform != "win32":
                proc = await asyncio.create_subprocess_exec(
                    str(self._executable_path),
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=make_pdeathsig_preexec(),
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    str(self._executable_path),
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
        except Exception:
            writer.close()
            raise
        self._stream_writer = writer
        self._process = proc
        self._monitor_task = asyncio.create_task(self._pump_and_watch(proc))

    async def _pump_and_watch(self, proc: asyncio.subprocess.Process) -> None:
        """Stream child stdout/stderr lines and emit crash alert on unexpected exit."""
        try:
            if proc.stdout is not None:
                while True:
                    raw_line = await proc.stdout.readline()
                    if not raw_line:
                        break
                    text = raw_line.decode("utf-8", errors="replace")
                    if self._stream_writer is not None:
                        self._stream_writer.write_line(text)
            returncode = await proc.wait()
        except asyncio.CancelledError:
            return

        if not self._stopping:
            self._last_exit_code = returncode
            self._log.error(
                "shieldgemma_subprocess_crashed",
                pid=proc.pid,
                returncode=returncode,
                alert_category=AlertCategory.SUBPROCESS_CRASH.value,
            )

    async def _wait_for_ready_locked(self, config: GuardrailConfig) -> BackendHealth:
        """Poll /health until ready, or fail on crash or startup timeout."""
        deadline = time.monotonic() + self._startup_timeout_seconds
        provider = ShieldGemmaCpuProvider(config, client=self._http_client)

        while True:
            proc = self._process
            if proc is None or proc.returncode is not None:
                if self._monitor_task is not None:
                    await self._monitor_task
                exit_code = proc.returncode if proc is not None else self._last_exit_code
                await self._stop_locked(preserve_last_exit_code=True)
                msg = f"ShieldGemma server exited before becoming healthy (code={exit_code})"
                raise InferenceBackendError(msg, details={"returncode": str(exit_code)})

            status = await provider.health()
            if status.healthy:
                self._log.info(
                    "shieldgemma_server_ready",
                    host=config.host,
                    port=config.port,
                    pid=proc.pid,
                )
                return status

            if time.monotonic() >= deadline:
                self._log.error(
                    "shieldgemma_startup_timeout",
                    host=config.host,
                    port=config.port,
                    timeout_seconds=self._startup_timeout_seconds,
                    alert_category=AlertCategory.TIMEOUT.value,
                )
                await self._stop_locked()
                msg = (
                    f"ShieldGemma server failed health check within "
                    f"{self._startup_timeout_seconds}s"
                )
                raise InferenceBackendError(msg, details={"last_status": status.status})

            await asyncio.sleep(self._poll_interval_seconds)

    async def _terminate_and_drain(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate or kill proc if still running and await process exit."""
        if proc.returncode is None:
            terminate_process_tree(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout_seconds)
            except TimeoutError:
                kill_process_tree(proc)
                await proc.wait()
        else:
            await proc.wait()

    async def _stop_locked(self, *, preserve_last_exit_code: bool = False) -> None:
        """Terminate running process, release resources, and close rotating log writer."""
        self._stopping = True
        proc = self._process
        if proc is not None:
            await self._terminate_and_drain(proc)

        if self._monitor_task is not None:
            if not self._monitor_task.done():
                self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        if proc is not None:
            transport = cast("asyncio.BaseTransport | None", getattr(proc, "_transport", None))
            if transport is not None:
                transport.close()
            await asyncio.sleep(0)

        if self._stream_writer is not None:
            self._stream_writer.close()
            self._stream_writer = None

        self._process = None
        if not preserve_last_exit_code:
            self._last_exit_code = None
        self._stopping = False
