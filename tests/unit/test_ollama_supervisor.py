"""Unit and branch-coverage tests for OllamaSupervisor lifecycle and CLI helpers."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

from lai.config.profiles import GEMMA4_26B_A4B_PROFILE
from lai.domain import AlertCategory, GuardrailConfig, LoggingConfig
from lai.domain.errors import InferenceBackendError
from lai.inference import OllamaSupervisor, ollama_manager, query_gpu_telemetry, setup_ollama_models
from lai.observability import configure_observability

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
EXPECTED_PORT_ERR_CODE = 2
EXPECTED_CREATE_CMD_LEN = 5
EXPECTED_WAIT_TIMEOUT = 10.0
_CREATE_NEW_PROCESS_GROUP = 0x200


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


class _FakePopenProc:
    """Popen double recording waits, termination, and force-kills."""

    def __init__(
        self,
        *,
        stdout_bytes: bytes = b"ollama listening\n",
        initial_rc: int | None = None,
        timeout_once_on_wait: bool = False,
    ) -> None:
        self.pid = 51515
        self.stdout = io.BytesIO(stdout_bytes)
        self._rc = initial_rc
        self._timeout_once = timeout_once_on_wait
        self.wait_timeouts: list[float | None] = []
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._rc

    def terminate(self) -> None:
        self.terminated = True
        self._rc = 0

    def kill(self) -> None:
        self.killed = True
        self._rc = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._timeout_once:
            self._timeout_once = False
            raise subprocess.TimeoutExpired(cmd="ollama", timeout=timeout or 0.0)
        if self._rc is None:
            self._rc = 0
        return self._rc


def test_resolve_ollama_executable_paths_and_missing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_ollama_executable checks PATH, Windows LOCALAPPDATA, and raises when absent."""

    def _which_ollama(name: str) -> str | None:
        return "/usr/bin/ollama" if name == "ollama" else None

    def _which_none(_name: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", _which_ollama)
    assert ollama_manager.resolve_ollama_executable() == Path("/usr/bin/ollama")

    monkeypatch.setattr(shutil, "which", _which_none)
    win_exe = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    win_exe.parent.mkdir(parents=True)
    win_exe.write_bytes(b"MZ")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert ollama_manager.resolve_ollama_executable() == win_exe

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing"))
    with pytest.raises(InferenceBackendError, match="not found"):
        ollama_manager.resolve_ollama_executable()

    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(InferenceBackendError, match="not found"):
        ollama_manager.resolve_ollama_executable()


async def test_setup_ollama_models_runs_cli_creates_modelfile_and_handles_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setup_ollama_models invokes _run_ollama_cli and _create_aliased_model cleanly."""
    recorded: list[tuple[list[str], dict[str, str] | None, bool]] = []
    seen_modelfile_texts: list[str] = []
    created_tmp_paths: list[Path] = []
    exit_code = 0

    def _fake_run(
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        shell: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        assert check is False
        recorded.append((list(cmd), env, shell))
        if len(cmd) >= EXPECTED_CREATE_CMD_LEN and cmd[1] == "create":
            modelfile_path = Path(cmd[4])
            created_tmp_paths.append(modelfile_path)
            seen_modelfile_texts.append(modelfile_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, returncode=exit_code)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    exe = tmp_path / "ollama"
    guard = GuardrailConfig(backend="ollama", enabled=True)

    resolved = await setup_ollama_models(GEMMA4_26B_A4B_PROFILE, guard, ollama_exe=exe)
    assert list(resolved) == ["primary_model", "guardrail_model"]
    assert all(shell is False and env is not None for _, env, shell in recorded)
    assert [cmd[1] for cmd, _, _ in recorded] == ["pull", "create", "pull", "create"]
    assert "FROM hf.co/" in seen_modelfile_texts[0]
    assert "PARAMETER num_gpu 0" in seen_modelfile_texts[1]
    assert all(not p.exists() for p in created_tmp_paths)

    exit_code = EXPECTED_PORT_ERR_CODE
    with pytest.raises(InferenceBackendError, match="exit code 2") as exc_info:
        await setup_ollama_models(
            GEMMA4_26B_A4B_PROFILE, guard, ollama_exe=exe, skip_guardrail=True
        )
    assert exc_info.value.details == {"returncode": "2"}


async def test_ollama_supervisor_health_json_and_model_matching(tmp_path: Path) -> None:
    """OllamaSupervisor.health validates /api/tags JSON and exact/tag model matching."""
    sup = OllamaSupervisor(base_url="http://127.0.0.1:11434", log_dir=tmp_path)
    tags_url = "http://127.0.0.1:11434/api/tags"

    with respx.mock(assert_all_called=True) as mock:
        bad_models = ["bad-entry", {"name": 123}, {"name": "gemma4-26b-a4b-other:latest"}]
        mock.get(tags_url).mock(
            side_effect=[
                httpx.ConnectError("connection refused"),
                httpx.Response(HTTP_UNAVAILABLE, text="daemon starting"),
                httpx.Response(HTTP_OK, content=b"{invalid-json"),
                httpx.Response(HTTP_OK, json=["not-a-dict"]),
                httpx.Response(HTTP_OK, json={"models": "not-a-list"}),
                httpx.Response(HTTP_OK, json={"models": bad_models}),
                httpx.Response(HTTP_OK, json={"models": [{"name": "gemma4-26b-a4b:latest"}]}),
                httpx.Response(HTTP_OK, json={"models": [{"name": "gemma4-26b-a4b"}]}),
                httpx.Response(HTTP_OK, json={"models": []}),
                httpx.Response(HTTP_OK, json={"models": [{"name": "llama3:8b"}]}),
            ]
        )

        unhealthy_statuses = (
            "unreachable",
            "http_503",
            "invalid_response",
            "invalid_response",
            "invalid_response",
        )
        for expected_status in unhealthy_statuses:
            res = await sup.health()
            assert (res.healthy, res.status) == (False, expected_status)

        missing = await sup.health(expected_model="gemma4-26b-a4b")
        assert (missing.healthy, missing.status) == (False, "model_missing")

        for expected_loaded in ("gemma4-26b-a4b:latest", "gemma4-26b-a4b"):
            ok = await sup.health(expected_model="gemma4-26b-a4b")
            assert (ok.healthy, ok.model_loaded) == (True, expected_loaded)

        for expected_unpinned in (None, "llama3:8b"):
            ok = await sup.health()
            assert (ok.healthy, ok.model_loaded) == (True, expected_unpinned)


async def test_ollama_supervisor_spawn_posix_win32_and_graceful_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover _spawn_process on win32/POSIX, model-wait polling, and graceful stop."""
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"MZ")
    tags_url = "http://127.0.0.1:11434/api/tags"
    log_dir = tmp_path / "logs"

    popen_calls: list[tuple[list[str], Mapping[str, str], dict[str, object]]] = []
    spawned: list[_FakePopenProc] = []

    def _recording_popen(cmd: list[str], **kwargs: object) -> _FakePopenProc:
        env_map = cast("dict[str, str]", kwargs["env"])
        popen_calls.append((cmd, env_map, kwargs))
        proc = _FakePopenProc(stdout_bytes=b"ollama server ready\n")
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)
    sup = OllamaSupervisor(
        base_url="http://127.0.0.1:11434",
        log_dir=log_dir,
        executable_path=exe,
        startup_timeout_seconds=1.0,
        poll_interval_seconds=0.01,
        shutdown_timeout_seconds=EXPECTED_WAIT_TIMEOUT,
    )

    monkeypatch.setattr(sys, "platform", "linux")
    with respx.mock(assert_all_called=True) as mock:
        mock.get(tags_url).mock(
            side_effect=[
                httpx.Response(HTTP_OK, json={"models": [{"name": "other:latest"}]}),
                httpx.Response(HTTP_OK, json={"models": [{"name": "gemma4-26b-a4b:latest"}]}),
            ]
        )
        health_posix = await sup.ensure_running(GEMMA4_26B_A4B_PROFILE)
        assert health_posix.healthy is True
    assert popen_calls[-1][2]["start_new_session"] is True
    assert popen_calls[-1][2]["shell"] is False
    await sup.stop()
    assert spawned[0].terminated is True

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP, raising=False
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(tags_url).mock(
            side_effect=[
                httpx.Response(HTTP_OK, json={"models": [{"name": "other:latest"}]}),
                httpx.Response(HTTP_OK, json={"models": [{"name": "other:latest"}]}),
                httpx.Response(HTTP_OK, json={"models": [{"name": "gemma4-26b-a4b:latest"}]}),
            ]
        )
        health_win = await sup.ensure_running(GEMMA4_26B_A4B_PROFILE)
        assert health_win.healthy is True
        assert health_win.model_loaded == GEMMA4_26B_A4B_PROFILE.name
        assert popen_calls[-1][0] == [str(exe), "serve"]
        assert popen_calls[-1][1]["OLLAMA_HOST"] == "127.0.0.1:11434"
        assert popen_calls[-1][2]["creationflags"] == _CREATE_NEW_PROCESS_GROUP

    await sup.stop()
    assert spawned[-1].terminated is True
    assert spawned[-1].wait_timeouts == [EXPECTED_WAIT_TIMEOUT]
    assert "ollama server ready" in (log_dir / "ollama-server.log").read_text(encoding="utf-8")


async def test_ollama_supervisor_crash_and_timeout_force_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover startup crash alert and startup timeout _stop_locked force-kill fallback."""
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"MZ")
    tags_url = "http://127.0.0.1:11434/api/tags"
    log_dir = tmp_path / "logs"

    sup = OllamaSupervisor(
        base_url="http://127.0.0.1:11434",
        log_dir=log_dir,
        executable_path=exe,
        startup_timeout_seconds=1.0,
        poll_interval_seconds=0.01,
    )
    manager = configure_observability(LoggingConfig(log_dir=log_dir, enable_file_logs=False))
    try:
        crash_proc = _FakePopenProc(stdout_bytes=b"panic\n", initial_rc=1)

        def _spawn_crash(_cmd: list[str], **_kw: object) -> _FakePopenProc:
            return crash_proc

        monkeypatch.setattr(subprocess, "Popen", _spawn_crash)
        with respx.mock(assert_all_called=True) as mock:
            mock.get(tags_url).mock(side_effect=httpx.ConnectError("down"))
            with pytest.raises(InferenceBackendError, match="exited unexpectedly with code 1"):
                await sup.ensure_running(GEMMA4_26B_A4B_PROFILE)
        assert any(a.category is AlertCategory.SUBPROCESS_CRASH for a in manager.recent_alerts())
    finally:
        manager.close()

    timeout_proc = _FakePopenProc(stdout_bytes=b"slow\n", timeout_once_on_wait=True)

    def _spawn_timeout(_cmd: list[str], **_kw: object) -> _FakePopenProc:
        return timeout_proc

    monkeypatch.setattr(subprocess, "Popen", _spawn_timeout)
    sup_timeout = OllamaSupervisor(
        base_url="http://127.0.0.1:11434",
        log_dir=log_dir,
        executable_path=exe,
        startup_timeout_seconds=0.03,
        poll_interval_seconds=0.01,
        shutdown_timeout_seconds=1.0,
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(tags_url).mock(return_value=httpx.Response(HTTP_OK, json={"models": []}))
        with pytest.raises(InferenceBackendError, match="timed out"):
            await sup_timeout.ensure_running(GEMMA4_26B_A4B_PROFILE)
    assert timeout_proc.terminated is True
    assert timeout_proc.killed is True


def test_query_gpu_telemetry_timeout_and_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """query_gpu_telemetry returns None on TimeoutExpired or non-zero nvidia-smi exit."""

    def _which_nvidia(_name: str) -> str | None:
        return "/usr/bin/nvidia-smi"

    monkeypatch.setattr(shutil, "which", _which_nvidia)

    def _timeout_run(cmd: Sequence[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=3.0)

    monkeypatch.setattr(subprocess, "run", _timeout_run)
    assert query_gpu_telemetry() is None

    def _fail_run(cmd: Sequence[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="err")

    monkeypatch.setattr(subprocess, "run", _fail_run)
    assert query_gpu_telemetry() is None
