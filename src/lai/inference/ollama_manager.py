"""Ollama model setup, Modelfile generation, and process supervisor."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

import httpx

from lai.domain.errors import InferenceBackendError
from lai.domain.inference import BackendHealth, ModelProfile
from lai.domain.observability import AlertCategory
from lai.inference._winjob import (
    WindowsJobObject,
    kill_process_group,
    pump_stream,
    terminate_process_group,
)
from lai.observability.logging import get_component_logger
from lai.observability.rotation import DailyRotatingStreamWriter
from lai.ports.inference import InferenceSupervisorPort

if TYPE_CHECKING:
    from collections.abc import Mapping

    import structlog

    from lai.domain.guardrails import GuardrailConfig

_HTTP_OK = 200
_MIN_POLL_SLEEP = 0.05
_THREAD_JOIN_TIMEOUT = 2.0
_FORCE_KILL_WAIT = 5.0


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


def resolve_ollama_executable() -> Path:
    """Locate the signed Ollama CLI executable on PATH or standard Windows install path."""
    if (on_path := shutil.which("ollama")) is not None:
        return Path(on_path)
    if sys.platform == "win32" and (local_app := os.environ.get("LOCALAPPDATA")):
        candidate = Path(local_app) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            return candidate
    msg = "ollama executable not found on PATH or standard install directory."
    raise InferenceBackendError(msg)


def derive_hf_quant_tag(hf_filename: str) -> str:
    """Extract Ollama HuggingFace quant tag from a GGUF filename."""
    stem = hf_filename.removesuffix(".gguf")
    if (dot_idx := stem.find(".")) != -1:
        return stem[dot_idx + 1 :]
    if "-it-" in stem:
        return stem.split("-it-", maxsplit=1)[1]
    parts = stem.rsplit("-", maxsplit=1)
    return parts[1] if len(parts) == 2 else stem  # noqa: PLR2004


def build_ollama_serve_env(
    base_url: str,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build Ollama env vars tuned for Gemma 4 Blackwell compatibility and dual-model residency."""
    env = dict(base_env) if base_env is not None else dict(os.environ)
    parsed = urlparse(base_url)
    env["OLLAMA_HOST"] = parsed.netloc or "127.0.0.1:11434"
    env["OLLAMA_FLASH_ATTENTION"] = "0"
    env["OLLAMA_KV_CACHE_TYPE"] = "f16"
    env["OLLAMA_MAX_LOADED_MODELS"] = "2"
    env["OLLAMA_NUM_PARALLEL"] = "1"
    env["OLLAMA_GPU_OVERHEAD"] = "1073741824"
    return env


def build_ollama_modelfile(
    base_ref: str,
    *,
    num_ctx: int,
    num_gpu: int | None = None,
) -> str:
    """Render an Ollama Modelfile aliasing a HuggingFace GGUF with tuned parameters."""
    lines = [f"FROM {base_ref}", f"PARAMETER num_ctx {num_ctx}", "PARAMETER use_mmap true"]
    if num_gpu is not None:
        lines.append(f"PARAMETER num_gpu {num_gpu}")
    return "\n".join(lines) + "\n"


def _run_ollama_cli(
    ollama_exe: Path,
    args: list[str],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run an Ollama CLI command synchronously and fail loudly on non-zero exit."""
    env_dict = dict(env) if env else None
    completed = subprocess.run(  # noqa: S603
        [str(ollama_exe), *args], env=env_dict, shell=False, check=False
    )
    if completed.returncode != 0:
        msg = f"ollama {' '.join(args)} failed with exit code {completed.returncode}"
        raise InferenceBackendError(msg, details={"returncode": str(completed.returncode)})


def _create_aliased_model(
    ollama_exe: Path,
    alias: str,
    modelfile_text: str,
    *,
    env: Mapping[str, str],
) -> None:
    """Write a temporary Modelfile and run `ollama create <alias>`."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(modelfile_text)
        tmp_path = Path(tmp.name)
    try:
        _run_ollama_cli(ollama_exe, ["create", alias, "-f", str(tmp_path)], env=env)
    finally:
        tmp_path.unlink(missing_ok=True)


async def setup_ollama_models(
    profile: ModelProfile,
    guardrail: GuardrailConfig,
    *,
    base_url: str = "http://127.0.0.1:11434",
    ollama_exe: Path | None = None,
    skip_guardrail: bool = False,
) -> dict[str, str]:
    """Pull HuggingFace GGUFs into Ollama and create tuned local aliases."""
    exe = ollama_exe or resolve_ollama_executable()
    env = build_ollama_serve_env(base_url)
    p = profile
    specs: list[tuple[str, str, str, str, int, int | None]] = [
        ("primary_model", p.hf_repo, p.hf_filename, p.name, p.ctx_size, None)
    ]
    if guardrail.enabled and not skip_guardrail:
        g = guardrail
        specs.append(("guardrail_model", g.hf_repo, g.hf_filename, g.ollama_model, g.ctx_size, 0))
    resolved: dict[str, str] = {}
    for key, repo, filename, alias, num_ctx, num_gpu in specs:
        src = f"hf.co/{repo}:{derive_hf_quant_tag(filename)}"
        await asyncio.to_thread(_run_ollama_cli, exe, ["pull", src], env=env)
        modelfile = build_ollama_modelfile(src, num_ctx=num_ctx, num_gpu=num_gpu)
        await asyncio.to_thread(_create_aliased_model, exe, alias, modelfile, env=env)
        resolved[key] = f"{alias} <- {src}"
    _logger().info("ollama_models_ready", **resolved)
    return resolved


def _extract_tag_models(response: httpx.Response) -> list[str] | None:
    try:
        payload: object = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_models = cast("dict[object, object]", payload).get("models")
    if not isinstance(raw_models, list):
        return None
    return [
        name
        for entry in cast("list[object]", raw_models)
        if isinstance(entry, dict)
        and isinstance((name := cast("dict[object, object]", entry).get("name")), str)
        and name
    ]


class OllamaSupervisor(InferenceSupervisorPort):
    """Lifecycle supervisor ensuring an Ollama server is running with GPU/KV tuning."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        log_dir: Path,
        executable_path: Path | None = None,
        startup_timeout_seconds: float = 45.0,
        poll_interval_seconds: float = 0.25,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._log_dir = log_dir
        self._executable_path = executable_path
        self._startup_timeout_seconds = startup_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

        self._lock = asyncio.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._owns_process = False
        self._active_profile: ModelProfile | None = None
        self._job_object: WindowsJobObject | None = None
        self._log_writer: DailyRotatingStreamWriter | None = None
        self._output_thread: threading.Thread | None = None

    def active_profile(self) -> ModelProfile | None:
        """Return the active ModelProfile when managed or verified."""
        return self._active_profile

    @staticmethod
    def _unhealthy(status: str, started: float, error: str) -> BackendHealth:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return BackendHealth(healthy=False, status=status, latency_ms=latency_ms, error=error)

    async def health(self, *, expected_model: str | None = None) -> BackendHealth:
        """Probe Ollama /api/tags endpoint and verify model availability."""
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
        except httpx.HTTPError as exc:
            return self._unhealthy("unreachable", started, str(exc))
        if response.status_code != _HTTP_OK:
            return self._unhealthy(f"http_{response.status_code}", started, response.text.strip())
        models = _extract_tag_models(response)
        if models is None:
            return self._unhealthy("invalid_response", started, "Invalid /api/tags JSON payload.")
        target = expected_model or (self._active_profile.name if self._active_profile else None)
        matched = (
            next((m for m in models if m == target or m.startswith(f"{target}:")), None)
            if target is not None
            else (models[0] if models else None)
        )
        if target is not None and matched is None:
            return self._unhealthy("model_missing", started, f"Model {target!r} not in /api/tags.")
        ms = round((time.perf_counter() - started) * 1000.0, 2)
        return BackendHealth(
            healthy=True,
            status="ok",
            model_loaded=matched,
            slots_total=1,
            slots_idle=1,
            latency_ms=ms,
        )

    @staticmethod
    def _popen_ollama_serve(cmd: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
        pipe, out = subprocess.PIPE, subprocess.STDOUT
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            return subprocess.Popen(  # noqa: S603
                cmd, stdout=pipe, stderr=out, bufsize=0, env=env, shell=False, creationflags=flags
            )
        return subprocess.Popen(  # noqa: S603
            cmd, stdout=pipe, stderr=out, bufsize=0, env=env, shell=False, start_new_session=True
        )

    def _spawn_process(self) -> subprocess.Popen[bytes]:
        exe = self._executable_path or resolve_ollama_executable()
        env = build_ollama_serve_env(self._base_url)
        log_path = self._log_dir / "ollama-server.log"
        writer = DailyRotatingStreamWriter(log_path, logger_name="lai.inference.ollama_server")
        proc = self._popen_ollama_serve([str(exe), "serve"], env)
        self._process, self._owns_process, self._log_writer = proc, True, writer
        if sys.platform == "win32":
            job = WindowsJobObject()
            if job.active:
                job.assign_pid(proc.pid)
                self._job_object = job
        if proc.stdout is not None:
            args = (proc.stdout, writer)
            thread = threading.Thread(
                target=pump_stream, args=args, daemon=True, name="ollama-server-log-pump"
            )
            thread.start()
            self._output_thread = thread
        _logger().info("ollama_server_spawned", pid=proc.pid, base_url=self._base_url)
        return proc

    async def _await_startup_ready(
        self, proc: subprocess.Popen[bytes], profile: ModelProfile
    ) -> BackendHealth:
        deadline = time.monotonic() + self._startup_timeout_seconds
        while True:
            rc = proc.poll()
            if rc is not None:
                await self._cleanup_resources()
                cat = AlertCategory.SUBPROCESS_CRASH.value
                _logger().error("ollama_server_crashed", returncode=rc, alert_category=cat)
                msg = f"ollama serve exited unexpectedly with code {rc}"
                raise InferenceBackendError(msg)
            current = await self.health(expected_model=profile.name)
            if current.healthy:
                self._active_profile = profile
                return current.model_copy(update={"model_loaded": profile.name})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self._stop_locked()
                msg = f"ollama serve startup timed out after {self._startup_timeout_seconds}s"
                raise InferenceBackendError(msg)
            await asyncio.sleep(min(self._poll_interval_seconds, max(_MIN_POLL_SLEEP, remaining)))

    async def ensure_running(self, profile: ModelProfile) -> BackendHealth:
        """Ensure an Ollama server is reachable, spawning one if needed."""
        async with self._lock:
            existing = await self.health(expected_model=profile.name)
            if existing.healthy:
                self._active_profile = profile
                return existing.model_copy(update={"model_loaded": profile.name})
            proc = self._spawn_process()
            return await self._await_startup_ready(proc, profile)

    async def _cleanup_resources(self) -> None:
        thread = self._output_thread
        self._output_thread = None
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, _THREAD_JOIN_TIMEOUT)
        if self._job_object is not None:
            self._job_object.close()
        if self._log_writer is not None:
            self._log_writer.close()
        self._job_object = self._log_writer = self._process = self._active_profile = None
        self._owns_process = False

    async def _stop_locked(self) -> None:
        proc = self._process
        if self._owns_process and proc is not None and proc.poll() is None:
            terminate_process_group(proc)
            try:
                await asyncio.to_thread(proc.wait, self._shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                kill_process_group(proc)
                await asyncio.to_thread(proc.wait, _FORCE_KILL_WAIT)
        await self._cleanup_resources()

    async def stop(self) -> None:
        """Stop the supervised Ollama server only if spawned by this instance."""
        async with self._lock:
            await self._stop_locked()
