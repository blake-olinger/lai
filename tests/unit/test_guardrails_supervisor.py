"""Unit tests for ShieldGemmaSupervisor subprocess lifecycle and process isolation."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import sys
from ctypes import wintypes
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

import lai.guardrails.supervisor as supervisor_mod
from lai.domain import AlertCategory, GuardrailConfig, LoggingConfig
from lai.domain.errors import InferenceBackendError
from lai.guardrails import ShieldGemmaSupervisor
from lai.guardrails._process import kill_process_tree, terminate_process_tree
from lai.observability import AlertManager, configure_observability
from lai.observability._process import (
    WindowsJobObject,
    attach_win32_job_object,
    close_win32_handle,
    kill_asyncio_process_tree,
    kill_popen_process_group,
    terminate_asyncio_process_tree,
    terminate_popen_process_group,
)
from lai.ports import GuardrailSupervisorPort

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Generator

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
EXPECTED_CRASH_EXIT = 7
HIGH_64_BIT_JOB_HANDLE = 0x8000_0001_0000_1234
HIGH_64_BIT_PROC_HANDLE = 0x8000_0002_0000_5678
FAKE_PID = 4242
FAKE_PGID = 9001


def _make_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    *,
    startup_timeout: float = 2.0,
    poll_interval: float = 0.05,
) -> ShieldGemmaSupervisor:
    """Create a ShieldGemmaSupervisor running an inline Python child script."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")

    def _args(config: GuardrailConfig, path: Path) -> list[str]:
        del config, path
        return ["-u", "-c", script]

    monkeypatch.setattr(supervisor_mod, "build_shieldgemma_server_args", _args)
    return ShieldGemmaSupervisor(
        Path(sys.executable),
        model_path,
        log_dir=tmp_path,
        startup_timeout_seconds=startup_timeout,
        poll_interval_seconds=poll_interval,
        stop_timeout_seconds=1.0,
    )


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_supervisor_validates_missing_paths(tmp_path: Path) -> None:
    """ShieldGemmaSupervisor raises InferenceBackendError when binary or GGUF is missing."""
    missing_exe = tmp_path / "missing-llama-server.exe"
    model_path = tmp_path / "shieldgemma-2b.Q4_K_M.gguf"
    sup = ShieldGemmaSupervisor(missing_exe, model_path, log_dir=tmp_path)
    assert isinstance(sup, GuardrailSupervisorPort)
    assert (await sup.health()).status == "stopped"
    with pytest.raises(InferenceBackendError, match="executable"):
        await sup.ensure_running(GuardrailConfig())
    exe_path = tmp_path / "llama-server.exe"
    exe_path.write_text("stub", encoding="utf-8")
    sup_missing_model = ShieldGemmaSupervisor(exe_path, model_path, log_dir=tmp_path)
    with pytest.raises(InferenceBackendError, match="model"):
        await sup_missing_model.ensure_running(GuardrailConfig())


async def test_supervisor_streams_logs_polls_health_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ShieldGemmaSupervisor streams output to shieldgemma-server.log and terminates cleanly."""
    sup = _make_supervisor(
        tmp_path,
        monkeypatch,
        "import time; print('shieldgemma-ready-line', flush=True); time.sleep(30)",
        startup_timeout=5.0,
    )
    cfg = GuardrailConfig(host="127.0.0.1", port=18085)
    try:
        with respx.mock(assert_all_called=True) as mock:
            mock.get("http://127.0.0.1:18085/health").mock(
                side_effect=[
                    httpx.Response(HTTP_UNAVAILABLE, json={"error": "loading"}),
                    httpx.Response(HTTP_OK, json={"status": "ok", "slots_idle": 2}),
                    httpx.Response(HTTP_OK, json={"status": "ok", "slots_idle": 2}),
                ],
            )
            assert (await sup.ensure_running(cfg)).healthy is True
            if sys.platform == "win32":
                assert sup.job_handle is not None
                assert sup.job_handle > 0
            assert (await sup.ensure_running(cfg)).healthy is True
            await asyncio.sleep(0.05)
    finally:
        await sup.stop()
    assert sup.job_handle is None
    assert (await sup.health()).status == "stopped"
    assert "shieldgemma-ready-line" in (tmp_path / "shieldgemma-server.log").read_text("utf-8")


async def test_supervisor_emits_crash_alert_on_unexpected_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected subprocess exit streams output and emits SUBPROCESS_CRASH alert."""
    sup = _make_supervisor(
        tmp_path,
        monkeypatch,
        (
            "import sys, time; print('fatal model load error', flush=True); "
            f"time.sleep(0.05); sys.exit({EXPECTED_CRASH_EXIT})"
        ),
    )
    manager = AlertManager(cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    cfg = GuardrailConfig(host="127.0.0.1", port=18086)
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("http://127.0.0.1:18086/health").mock(
                return_value=httpx.Response(HTTP_UNAVAILABLE, json={"error": "loading"}),
            )
            with pytest.raises(InferenceBackendError, match="exited before becoming healthy"):
                await sup.ensure_running(cfg)
        assert (await sup.health()).status == "crashed"
        await manager.flush()
    finally:
        await sup.stop()
        manager.close()
    alerts = [e for e in manager.recent_alerts() if e.category is AlertCategory.SUBPROCESS_CRASH]
    assert len(alerts) == 1
    assert alerts[0].component == "guardrails"
    assert "fatal model load error" in (tmp_path / "shieldgemma-server.log").read_text("utf-8")


async def test_supervisor_emits_timeout_alert_when_health_never_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup timeout kills the hung process and emits AlertCategory.TIMEOUT."""
    sup = _make_supervisor(
        tmp_path, monkeypatch, "import time; time.sleep(30)", startup_timeout=0.15
    )
    manager = AlertManager(cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    cfg = GuardrailConfig(host="127.0.0.1", port=18087)
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.get("http://127.0.0.1:18087/health").mock(
                return_value=httpx.Response(HTTP_UNAVAILABLE, json={"error": "loading"}),
            )
            with pytest.raises(InferenceBackendError, match="failed health check"):
                await sup.ensure_running(cfg)
        await manager.flush()
    finally:
        await sup.stop()
        manager.close()
    assert len([e for e in manager.recent_alerts() if e.category is AlertCategory.TIMEOUT]) == 1
    assert (await sup.health()).status == "stopped"


async def test_supervisor_restarts_on_config_change_and_detects_post_start_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supervisor restarts when GuardrailConfig changes and reports crash after startup."""
    sup = _make_supervisor(
        tmp_path, monkeypatch, "import sys, time; time.sleep(0.15); sys.exit(3)", poll_interval=0.02
    )
    manager = AlertManager(cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    cfg1 = GuardrailConfig(host="127.0.0.1", port=18088, cpu_threads=4)
    cfg2 = GuardrailConfig(host="127.0.0.1", port=18089, cpu_threads=8)
    try:
        with respx.mock(assert_all_called=True) as mock:
            mock.get("http://127.0.0.1:18088/health").mock(
                return_value=httpx.Response(HTTP_OK, json={"status": "ok"}),
            )
            mock.get("http://127.0.0.1:18089/health").mock(
                return_value=httpx.Response(HTTP_OK, json={"status": "ok"}),
            )
            assert (await sup.ensure_running(cfg1)).healthy is True
            assert (await sup.ensure_running(cfg2)).healthy is True
        await asyncio.sleep(0.35)
        crashed = await sup.health()
        assert (crashed.healthy, crashed.status) == (False, "crashed")
        await manager.flush()
    finally:
        await sup.stop()
        manager.close()
    alerts = [e for e in manager.recent_alerts() if e.category is AlertCategory.SUBPROCESS_CRASH]
    assert len(alerts) >= 1


class _FakeApi:
    def __init__(self, impl: Callable[..., object]) -> None:
        self.argtypes: list[object] = []
        self.restype: object = None
        self._impl = impl

    def __call__(self, *args: object) -> object:
        return self._impl(*args)


class _FakeKernel32:
    def __init__(self) -> None:
        self.closed: list[object] = []
        self.assigned: list[tuple[object, object]] = []
        self.set_ok, self.assign_ok = 1, 1
        self.job, self.proc = HIGH_64_BIT_JOB_HANDLE, HIGH_64_BIT_PROC_HANDLE
        self.apis: dict[str, _FakeApi] = {
            "CreateJobObjectW": _FakeApi(lambda *_a: self.job),
            "SetInformationJobObject": _FakeApi(lambda *_a: self.set_ok),
            "OpenProcess": _FakeApi(lambda *_a: self.proc),
            "AssignProcessToJobObject": _FakeApi(self._assign),
            "CloseHandle": _FakeApi(self._close),
        }

    def _assign(self, *args: object) -> int:
        self.assigned.append((args[0], args[1]))
        return self.assign_ok

    def _close(self, *args: object) -> int:
        self.closed.append(args[0])
        return 1

    def __getattr__(self, name: str) -> _FakeApi:
        return self.apis[name]


def test_process_win32_job_64bit_handles_and_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Win32 Job Object helpers preserve 64-bit HANDLEs and handle all error/POSIX paths."""
    dll = _FakeKernel32()

    def _win_dll(_name: str, *, use_last_error: bool = False) -> _FakeKernel32 | None:
        return dll if use_last_error else None

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", _win_dll, raising=False)
    handle = attach_win32_job_object(FAKE_PID)
    assert handle == HIGH_64_BIT_JOB_HANDLE
    for name in ("CreateJobObjectW", "OpenProcess"):
        assert dll.apis[name].restype is wintypes.HANDLE
    for name in ("SetInformationJobObject", "AssignProcessToJobObject", "CloseHandle"):
        assert wintypes.HANDLE in dll.apis[name].argtypes
    assert dll.assigned == [(HIGH_64_BIT_JOB_HANDLE, HIGH_64_BIT_PROC_HANDLE)]
    close_win32_handle(handle)
    close_win32_handle(None)
    close_win32_handle(0)
    assert dll.closed == [HIGH_64_BIT_PROC_HANDLE, HIGH_64_BIT_JOB_HANDLE]
    assert attach_win32_job_object(0) is None
    for attr in ("assign_ok", "proc"):
        setattr(dll, attr, 0)
        assert attach_win32_job_object(FAKE_PID) is None
    for attr, match in (("set_ok", "SetInformationJobObject"), ("job", "CreateJobObjectW")):
        setattr(dll, attr, 0)
        assert attach_win32_job_object(FAKE_PID) is None
        with pytest.raises(OSError, match=match):
            WindowsJobObject()
    monkeypatch.setattr(sys, "platform", "linux")
    posix_job = WindowsJobObject()
    assert not posix_job.active
    assert not posix_job.assign_pid(FAKE_PID)
    posix_job.close()
    assert attach_win32_job_object(FAKE_PID) is None
    close_win32_handle(HIGH_64_BIT_JOB_HANDLE)


def test_process_tree_suppression_and_posix_group_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process termination helpers suppress ProcessLookupError/OSError and handle POSIX killpg."""

    class _FakeProc:
        def __init__(self, pid: int, err: type[OSError]) -> None:
            self.pid, self._err, self.calls = pid, err, 0

        def _fail(self) -> None:
            self.calls += 1
            raise self._err("fail")

        terminate = kill = _fail

    assert terminate_process_tree is terminate_asyncio_process_tree
    assert kill_process_tree is kill_asyncio_process_tree
    monkeypatch.setattr(sys, "platform", "win32")
    for err_cls in (ProcessLookupError, OSError):
        ap = cast("asyncio.subprocess.Process", _FakeProc(FAKE_PID, err_cls))
        pp = cast("subprocess.Popen[bytes]", _FakeProc(FAKE_PID, err_cls))
        terminate_asyncio_process_tree(ap)
        kill_asyncio_process_tree(ap)
        terminate_popen_process_group(pp)
        kill_popen_process_group(pp)

    pg_signals: list[tuple[int, int]] = []

    def _getpgid(_pid: int) -> int:
        return FAKE_PGID

    def _killpg(pg: int, sig: int) -> None:
        pg_signals.append((pg, sig))

    def _deny(_pg: int, _sig: int) -> None:
        raise OSError

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getpgid", _getpgid, raising=False)
    monkeypatch.setattr(os, "killpg", _killpg, raising=False)
    raw = _FakeProc(FAKE_PID, OSError)
    terminate_asyncio_process_tree(cast("asyncio.subprocess.Process", raw))
    kill_popen_process_group(cast("subprocess.Popen[bytes]", raw))
    assert len(pg_signals) == 2  # noqa: PLR2004
    assert raw.calls == 0
    monkeypatch.setattr(os, "killpg", _deny, raising=False)
    kill_asyncio_process_tree(cast("asyncio.subprocess.Process", raw))
    assert raw.calls == 1
