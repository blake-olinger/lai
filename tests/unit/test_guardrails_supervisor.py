"""Unit tests for ShieldGemmaSupervisor subprocess lifecycle and crash/timeout alerts."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

import lai.guardrails.supervisor as supervisor_mod
from lai.domain import (
    AlertCategory,
    GuardrailConfig,
    LoggingConfig,
)
from lai.domain.errors import InferenceBackendError
from lai.guardrails import ShieldGemmaSupervisor
from lai.observability import AlertManager, configure_observability
from lai.ports import GuardrailSupervisorPort

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
EXPECTED_CRASH_EXIT = 7


def _fixed_args(args: list[str]) -> Callable[[GuardrailConfig, Path], list[str]]:
    """Return a typed replacement for build_shieldgemma_server_args in tests."""

    def _builder(config: GuardrailConfig, model_path: Path) -> list[str]:
        del config, model_path
        return list(args)

    return _builder


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ShieldGemmaSupervisor streams output to shieldgemma-server.log and terminates cleanly."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    monkeypatch.setattr(
        supervisor_mod,
        "build_shieldgemma_server_args",
        _fixed_args(
            [
                "-u",
                "-c",
                "import time; print('shieldgemma-ready-line', flush=True); time.sleep(30)",
            ],
        ),
    )

    cfg = GuardrailConfig(host="127.0.0.1", port=18085)
    health_url = "http://127.0.0.1:18085/health"
    sup = ShieldGemmaSupervisor(
        Path(sys.executable),
        model_path,
        log_dir=tmp_path,
        startup_timeout_seconds=5.0,
        poll_interval_seconds=0.05,
        stop_timeout_seconds=1.0,
    )

    try:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(health_url).mock(
                side_effect=[
                    httpx.Response(HTTP_UNAVAILABLE, json={"error": "loading"}),
                    httpx.Response(HTTP_OK, json={"status": "ok", "slots_idle": 2}),
                    httpx.Response(HTTP_OK, json={"status": "ok", "slots_idle": 2}),
                ],
            )
            status = await sup.ensure_running(cfg)
            assert status.healthy is True
            reused = await sup.ensure_running(cfg)
            assert reused.healthy is True
            await asyncio.sleep(0.05)
    finally:
        await sup.stop()

    assert (await sup.health()).status == "stopped"
    server_log = (tmp_path / "shieldgemma-server.log").read_text(encoding="utf-8")
    assert "shieldgemma-ready-line" in server_log


async def test_supervisor_emits_crash_alert_on_unexpected_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected subprocess exit streams output and emits SUBPROCESS_CRASH alert."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    monkeypatch.setattr(
        supervisor_mod,
        "build_shieldgemma_server_args",
        _fixed_args(
            [
                "-u",
                "-c",
                (
                    "import sys, time; print('fatal model load error', flush=True); "
                    f"time.sleep(0.05); sys.exit({EXPECTED_CRASH_EXIT})"
                ),
            ],
        ),
    )

    manager = AlertManager(cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    cfg = GuardrailConfig(host="127.0.0.1", port=18086)
    sup = ShieldGemmaSupervisor(
        Path(sys.executable),
        model_path,
        log_dir=tmp_path,
        startup_timeout_seconds=2.0,
        poll_interval_seconds=0.05,
    )

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

    crash_alerts = [
        e for e in manager.recent_alerts() if e.category is AlertCategory.SUBPROCESS_CRASH
    ]
    assert len(crash_alerts) == 1
    assert crash_alerts[0].component == "guardrails"
    server_log = (tmp_path / "shieldgemma-server.log").read_text(encoding="utf-8")
    assert "fatal model load error" in server_log


async def test_supervisor_emits_timeout_alert_when_health_never_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup timeout kills the hung process and emits AlertCategory.TIMEOUT."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    monkeypatch.setattr(
        supervisor_mod,
        "build_shieldgemma_server_args",
        _fixed_args(["-u", "-c", "import time; time.sleep(30)"]),
    )

    manager = AlertManager(cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    cfg = GuardrailConfig(host="127.0.0.1", port=18087)
    sup = ShieldGemmaSupervisor(
        Path(sys.executable),
        model_path,
        log_dir=tmp_path,
        startup_timeout_seconds=0.15,
        poll_interval_seconds=0.05,
        stop_timeout_seconds=1.0,
    )

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

    timeout_alerts = [e for e in manager.recent_alerts() if e.category is AlertCategory.TIMEOUT]
    assert len(timeout_alerts) == 1
    assert (await sup.health()).status == "stopped"


async def test_supervisor_restarts_on_config_change_and_detects_post_start_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor restarts when GuardrailConfig changes and reports crash after startup."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    monkeypatch.setattr(
        supervisor_mod,
        "build_shieldgemma_server_args",
        _fixed_args(["-u", "-c", "import sys, time; time.sleep(0.15); sys.exit(3)"]),
    )

    manager = AlertManager(cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    cfg1 = GuardrailConfig(host="127.0.0.1", port=18088, cpu_threads=4)
    cfg2 = GuardrailConfig(host="127.0.0.1", port=18089, cpu_threads=8)
    sup = ShieldGemmaSupervisor(
        Path(sys.executable),
        model_path,
        log_dir=tmp_path,
        startup_timeout_seconds=2.0,
        poll_interval_seconds=0.02,
    )

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
        crashed_health = await sup.health()
        assert crashed_health.healthy is False
        assert crashed_health.status == "crashed"
        await manager.flush()
    finally:
        await sup.stop()
        manager.close()

    crash_alerts = [
        e for e in manager.recent_alerts() if e.category is AlertCategory.SUBPROCESS_CRASH
    ]
    assert len(crash_alerts) >= 1
