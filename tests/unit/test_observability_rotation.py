"""Unit tests for daily rotating file handlers, stream writers, and observability contracts."""

from __future__ import annotations

import gzip
import logging
import logging.handlers
from typing import TYPE_CHECKING

from lai.config.settings import LaiSettings
from lai.domain import LoggingConfig
from lai.domain.ids import ALERT_ID_PREFIX, new_alert_id
from lai.observability import (
    DailyRotatingStreamWriter,
    SafeTimedRotatingFileHandler,
    create_daily_rotating_handler,
)
from lai.ports import AlertManagerPort, AlertSinkPort

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

RETENTION_TWO_DAYS = 2
BACKUP_COUNT_FOUR = 4
DEFAULT_COOLDOWN_SECONDS = 60.0


def _raise_locked(_self: logging.handlers.TimedRotatingFileHandler) -> None:
    raise PermissionError("locked")


def test_observability_domain_and_port_contracts() -> None:
    """Alert IDs, LoggingConfig defaults, and Alert ports validate cleanly."""
    alert_id = new_alert_id()
    assert alert_id.startswith(ALERT_ID_PREFIX)
    assert len(alert_id) > len(ALERT_ID_PREFIX)

    cfg = LoggingConfig()
    assert cfg.rotation_when == "midnight"
    assert cfg.compress_rotated is True
    assert cfg.alert_cooldown_seconds == DEFAULT_COOLDOWN_SECONDS
    assert LaiSettings().logging == cfg

    assert not isinstance(object(), AlertSinkPort)
    assert not isinstance(object(), AlertManagerPort)


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
