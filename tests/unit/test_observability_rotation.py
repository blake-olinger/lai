"""Unit tests for daily rotating file handlers, stream writers, and observability contracts."""

from __future__ import annotations

import ctypes
import gzip
import hashlib
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from pydantic import ValidationError

from lai.config.profiles import GEMMA4_26B_A4B_PROFILE
from lai.config.settings import LaiSettings
from lai.domain import AlertCategory, LoggingConfig
from lai.domain.errors import InferenceBackendError
from lai.domain.ids import ALERT_ID_PREFIX, new_alert_id
from lai.domain.inference import ModelIntegrityConfig, ModelProfile
from lai.inference import DEFAULT_LLAMA_CPP_TAG, LLAMA_CPP_LINUX_X64_MANIFEST
from lai.observability import (
    AlertManager,
    DailyRotatingStreamWriter,
    SafeTimedRotatingFileHandler,
    WebhookAlertSink,
    compute_file_sha256,
    configure_observability,
    create_daily_rotating_handler,
    verify_file_sha256,
)
from lai.observability._process import (
    kill_asyncio_process_tree,
    kill_popen_process_group,
    make_pdeathsig_preexec,
    terminate_asyncio_process_tree,
    terminate_popen_process_group,
)
from lai.ports import AlertManagerPort, AlertSinkPort

if TYPE_CHECKING:
    import asyncio
    import subprocess
    from collections.abc import Generator

RETENTION_TWO_DAYS = 2
BACKUP_COUNT_FOUR = 4
DEFAULT_COOLDOWN_SECONDS = 60.0
EXPECTED_LOG_MODE = 0o640
PR_SET_PDEATHSIG = 1
FAKE_PID = 4321
FAKE_PGID = 4300


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


class _NonConformingAlertStub:
    """Minimal stub lacking AlertSinkPort and AlertManagerPort methods."""


def _raise_locked(_self: logging.handlers.TimedRotatingFileHandler) -> None:
    raise PermissionError("locked")


async def test_observability_domain_and_port_contracts(tmp_path: Path) -> None:
    """Alert IDs, LoggingConfig defaults, and concrete Alert adapters validate cleanly."""
    alert_id = new_alert_id()
    assert alert_id.startswith(ALERT_ID_PREFIX)
    assert len(alert_id) > len(ALERT_ID_PREFIX)

    cfg = LoggingConfig()
    assert cfg.rotation_when == "midnight"
    assert cfg.compress_rotated is True
    assert cfg.alert_cooldown_seconds == DEFAULT_COOLDOWN_SECONDS
    assert LaiSettings().logging == cfg

    async with httpx.AsyncClient() as http_client:
        sink = WebhookAlertSink("https://example.com/webhook", client=http_client)
        manager = AlertManager(sinks=(sink,), log_dir=tmp_path / "alerts")
        try:
            assert isinstance(sink, AlertSinkPort)
            assert isinstance(manager, AlertManagerPort)
            assert not isinstance(_NonConformingAlertStub(), AlertSinkPort)
            assert not isinstance(_NonConformingAlertStub(), AlertManagerPort)
        finally:
            manager.close()


def test_daily_rotating_handler_compression_pruning_and_permission_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rotating handler compresses rolled logs to .gz, prunes expired files, and handles locks."""
    log_path = tmp_path / "logs" / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for day in ("2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04"):
        (log_path.parent / f"app.log.{day}.gz").write_bytes(b"old")

    handler = create_daily_rotating_handler(
        log_path,
        level=logging.INFO,
        retention_days=RETENTION_TWO_DAYS,
        compress=True,
    )
    try:
        record = logging.LogRecord(
            "lai.test", logging.INFO, __file__, 1, "hello rotation", (), None
        )
        handler.emit(record)
        handler.flush()
        handler.doRollover()
    finally:
        handler.close()

    gz_files = sorted(log_path.parent.glob("app.log.*.gz"))
    assert len(gz_files) <= RETENTION_TWO_DAYS
    assert b"hello rotation" in gzip.decompress(gz_files[-1].read_bytes())

    uncompressed_path = tmp_path / "logs" / "plain.log"
    plain_handler = create_daily_rotating_handler(
        uncompressed_path,
        level=logging.INFO,
        retention_days=RETENTION_TWO_DAYS,
        compress=False,
    )
    try:
        plain_handler.emit(
            logging.LogRecord("lai.test", logging.INFO, __file__, 1, "plain", (), None)
        )
        monkeypatch.setattr(
            logging.handlers.TimedRotatingFileHandler,
            "doRollover",
            _raise_locked,
        )
        assert isinstance(plain_handler, SafeTimedRotatingFileHandler)
        plain_handler.doRollover()
        plain_handler.emit(
            logging.LogRecord("lai.test", logging.INFO, __file__, 1, "after", (), None)
        )
    finally:
        plain_handler.close()


def test_daily_rotating_stream_writer(tmp_path: Path) -> None:
    """DailyRotatingStreamWriter writes lines safely and handles close idempotently."""
    stream_path = tmp_path / "llama-server.log"
    with DailyRotatingStreamWriter(stream_path, retention_days=BACKUP_COUNT_FOUR) as writer:
        writer.write_line("line one\r\n")
        writer.flush()
        writer.handler.doRollover()
        writer.write_line("line two")
    writer.write_line("ignored after close")
    writer.close()

    assert "line two" in stream_path.read_text(encoding="utf-8")
    assert list(tmp_path.glob("llama-server.log.*.gz"))


def test_rotation_applies_posix_0640_permissions_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX log open and gzip rollover apply 0o640 permissions and ignore chmod errors."""
    chmod_calls: list[tuple[str, int]] = []
    fail_chmod = False

    def _fake_chmod(self: Path, mode: int) -> None:
        chmod_calls.append((self.name, mode))
        if fail_chmod:
            msg = "chmod unsupported"
            raise OSError(msg)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "chmod", _fake_chmod)

    log_path = tmp_path / "secure.log"
    with DailyRotatingStreamWriter(log_path, retention_days=RETENTION_TWO_DAYS) as writer:
        writer.write_line("first")
        writer.handler.doRollover()
        fail_chmod = True
        writer.write_line("second")

    assert ("secure.log", EXPECTED_LOG_MODE) in chmod_calls
    assert any(name.endswith(".gz") and mode == EXPECTED_LOG_MODE for name, mode in chmod_calls)


def test_sha256_integrity_verification_and_security_alert(tmp_path: Path) -> None:
    """verify_file_sha256 validates digests case-insensitively and emits SECURITY_ALERT."""
    payload = b"gguf-weights-block" * 70_000
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert compute_file_sha256(artifact) == expected
    assert verify_file_sha256(artifact, expected.upper(), label="model.gguf") == expected

    manager = configure_observability(
        LoggingConfig(log_dir=tmp_path, enable_file_logs=False, alert_cooldown_seconds=0.0)
    )
    try:
        wrong_digest = "0" * 64
        with pytest.raises(InferenceBackendError, match="SHA-256 integrity mismatch"):
            verify_file_sha256(artifact, wrong_digest, label="model.gguf")

        with pytest.raises(InferenceBackendError, match="Invalid expected SHA-256"):
            verify_file_sha256(artifact, "not-a-sha", label="model.gguf")

        with pytest.raises(InferenceBackendError, match="File not found"):
            verify_file_sha256(tmp_path / "missing.gguf", expected, label="missing")

        alerts = manager.recent_alerts(limit=10)
        assert len(alerts) == 2
        assert all(alert.category is AlertCategory.SECURITY_ALERT for alert in alerts)
    finally:
        manager.close()


class _FakeLibc:
    """Stub libc exposing a trackable prctl call for Linux PR_SET_PDEATHSIG tests."""

    def __init__(self, calls: list[tuple[int, int]], *, fail: bool = False) -> None:
        self._calls = calls
        self._fail = fail

    def prctl(self, option: int, arg: int) -> int:
        self._calls.append((option, arg))
        if self._fail:
            msg = "prctl rejected"
            raise OSError(msg)
        return 0


class _FakeProcessHandle:
    """Process stub compatible with POSIX process-group signaling helpers."""

    def __init__(self, pid: int = FAKE_PID) -> None:
        self.pid = pid
        self.terminated = 0
        self.killed = 0

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


def test_make_pdeathsig_preexec_and_posix_process_group_signaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_pdeathsig_preexec calls libc.prctl(1, sig) on Linux and no-ops elsewhere."""
    monkeypatch.setattr(sys, "platform", "darwin")
    noop = make_pdeathsig_preexec()
    noop()

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(sys, "platform", "linux")

    def _ok_cdll(_name: str | None) -> _FakeLibc:
        return _FakeLibc(calls)

    monkeypatch.setattr(ctypes, "CDLL", _ok_cdll)
    preexec = make_pdeathsig_preexec(signal.SIGTERM)
    preexec()
    assert calls == [(PR_SET_PDEATHSIG, signal.SIGTERM)]

    fail_calls: list[tuple[int, int]] = []

    def _fail_cdll(_name: str | None) -> _FakeLibc:
        return _FakeLibc(fail_calls, fail=True)

    monkeypatch.setattr(ctypes, "CDLL", _fail_cdll)
    make_pdeathsig_preexec(signal.SIGTERM)()
    assert fail_calls == [(PR_SET_PDEATHSIG, signal.SIGTERM)]

    def _missing_cdll(_name: str | None) -> object:
        msg = "no libc"
        raise OSError(msg)

    monkeypatch.setattr(ctypes, "CDLL", _missing_cdll)
    make_pdeathsig_preexec()()

    sent_signals: list[tuple[int, int]] = []

    def _fake_getpgid(_pid: int) -> int:
        return FAKE_PGID

    def _fake_killpg(pgid: int, sig: int) -> None:
        sent_signals.append((pgid, sig))

    monkeypatch.setattr(os, "getpgid", _fake_getpgid, raising=False)
    monkeypatch.setattr(os, "killpg", _fake_killpg, raising=False)
    proc = _FakeProcessHandle()
    terminate_popen_process_group(cast("subprocess.Popen[bytes]", proc))
    kill_popen_process_group(cast("subprocess.Popen[bytes]", proc))
    terminate_asyncio_process_tree(cast("asyncio.subprocess.Process", proc))
    kill_asyncio_process_tree(cast("asyncio.subprocess.Process", proc))
    assert len(sent_signals) == 4
    assert sent_signals[0] == (FAKE_PGID, signal.SIGTERM)

    monkeypatch.setattr(os, "killpg", None, raising=False)
    terminate_popen_process_group(cast("subprocess.Popen[bytes]", proc))
    kill_popen_process_group(cast("subprocess.Popen[bytes]", proc))
    assert proc.terminated == 1
    assert proc.killed == 1


def test_domain_integrity_models_and_pinned_manifest() -> None:
    """ModelProfile, ModelIntegrityConfig, and pinned manifest enforce strict SHA-256 hex."""
    assert GEMMA4_26B_A4B_PROFILE.hf_revision is not None
    assert GEMMA4_26B_A4B_PROFILE.sha256 is not None
    assert len(GEMMA4_26B_A4B_PROFILE.sha256) == 64
    assert set(LLAMA_CPP_LINUX_X64_MANIFEST) == {"12.8", "13.5"}
    for spec in LLAMA_CPP_LINUX_X64_MANIFEST.values():
        assert spec.tag == DEFAULT_LLAMA_CPP_TAG
        assert len(spec.sha256) == 64
        assert len(spec.cudart_sha256) == 64

    cfg = ModelIntegrityConfig(
        approved_model_sha256s=("A" * 64,),
        allowed_hf_repos=("  unsloth/gemma-4-26B-A4B-it-GGUF  ",),
    )
    assert cfg.approved_model_sha256s == ("a" * 64,)
    assert cfg.allowed_hf_repos == ("unsloth/gemma-4-26B-A4B-it-GGUF",)
    for bad_call in (
        lambda: ModelIntegrityConfig(approved_model_sha256s=("short",)),
        lambda: ModelIntegrityConfig(allowed_hf_repos=("   ",)),
        lambda: ModelProfile(name="p", hf_repo="r/m", hf_filename="m.gguf", sha256="zz" * 32),
        lambda: ModelProfile(name="p", hf_repo="r/m", hf_filename="m.gguf", hf_revision="  "),
    ):
        with pytest.raises(ValidationError):
            bad_call()
