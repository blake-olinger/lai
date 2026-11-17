"""Unit tests for structured component logging, alerting, and webhook delivery."""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx
import structlog

from lai.domain import AlertCategory, AlertEvent, AlertSeverity, LoggingConfig
from lai.domain.ids import new_alert_id
from lai.observability import (
    AlertingLogHandler,
    AlertManager,
    WebhookAlertSink,
    bind_log_context,
    configure_observability,
    get_component_logger,
)
from lai.ports import AlertSinkPort

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

EXPECTED_RECENT_TWO = 2
EXPECTED_THREE_ALERTS = 3
DEFAULT_COOLDOWN_SECONDS = 60.0
LARGE_LIMIT = 10
MAX_TRACEBACK_LEN = 2000
LONG_ERROR_LEN = 2500
HTTP_OK = 200
HTTP_ERROR = 500


class _MemorySink(AlertSinkPort):
    """In-memory alert sink recording received events."""

    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


class _FailingSink(AlertSinkPort):
    """Test sink raising an error to verify flush exception isolation."""

    async def send_alert(self, event: AlertEvent) -> None:
        del event
        msg = "sink boom"
        raise RuntimeError(msg)


def _raise_long_runtime_error() -> None:
    raise RuntimeError("x" * LONG_ERROR_LEN)


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def test_configure_observability_writes_jsonl_and_errors_with_context(tmp_path: Path) -> None:
    """Component logger writes correlated JSONL to lai.jsonl and tracebacks to errors.jsonl."""
    log_dir = tmp_path / "logs"
    cfg = LoggingConfig(log_dir=log_dir, level="INFO", json_console=True, enable_file_logs=True)
    manager = configure_observability(cfg)
    try:
        log = get_component_logger("spooler")
        with bind_log_context(
            request_id="req_1",
            job_id="job_1",
            session_id="ses_1",
            project_id="prj_1",
            principal_id="usr_1",
        ):
            log.info("lease_acquired")
            with bind_log_context(job_id="job_nested"):
                log.info("nested_job")
            try:
                _ = 1 // 0
            except ZeroDivisionError:
                log.exception("worker_failed")
    finally:
        manager.close()
        for handler in list(logging.getLogger().handlers):
            handler.flush()
            handler.close()

    all_entries = [
        json.loads(line)
        for line in (log_dir / "lai.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all_entries[0]["component"] == "spooler"
    assert all_entries[0]["request_id"] == "req_1"
    assert all_entries[0]["job_id"] == "job_1"
    assert all_entries[1]["job_id"] == "job_nested"
    assert all_entries[1]["session_id"] == "ses_1"

    error_entries = [
        json.loads(line)
        for line in (log_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(error_entries) == 1
    assert error_entries[0]["event"] == "worker_failed"
    assert "ZeroDivisionError" in error_entries[0]["exception"]


async def test_webhook_alert_sink_success_and_error_swallowing() -> None:
    """WebhookAlertSink posts JSON alert payload and swallows HTTP/transport errors."""
    event = AlertEvent(
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.FATAL_EXCEPTION,
        component="inference",
        message="llama crashed",
    )
    url = "https://hooks.example.test/alerts"
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(url).mock(return_value=httpx.Response(HTTP_OK, json={"ok": True}))
        sink = WebhookAlertSink(url, timeout_seconds=2.0)
        await sink.send_alert(event)
        assert route.called
        sent_request = cast("httpx.Request", route.calls[0].request)
        body = json.loads(sent_request.content.decode("utf-8"))
        assert body["event"]["id"] == event.id
        assert body["text"] == "[CRITICAL] inference (fatal_exception): llama crashed"

    with respx.mock(assert_all_called=True) as mock:
        mock.post(url).mock(return_value=httpx.Response(HTTP_ERROR, text="fail"))
        async with httpx.AsyncClient() as shared_client:
            sink = WebhookAlertSink(url, client=shared_client)
            await sink.send_alert(event)

    with respx.mock(assert_all_called=True) as mock:
        mock.post(url).mock(side_effect=httpx.ConnectError("refused"))
        sink = WebhookAlertSink(url)
        await sink.send_alert(event)


async def test_alert_manager_dedup_ring_buffer_and_flush(tmp_path: Path) -> None:
    """AlertManager enforces cooldown boundaries, persists alerts.jsonl, and flushes sinks."""
    now = 100.0

    def clock() -> float:
        return now

    sink = _MemorySink()
    failing_sink = _FailingSink()
    manager = AlertManager(
        sinks=(sink, failing_sink),
        cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
        log_dir=tmp_path,
        clock=clock,
    )
    try:
        base = AlertEvent(
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TIMEOUT,
            component="spooler",
            message="wait timeout",
            exception_type="TimeoutError",
        )
        manager.emit(base)
        now = 159.9
        manager.emit(base.model_copy(update={"id": new_alert_id(), "message": "suppressed"}))
        now = 160.0
        second = base.model_copy(update={"id": new_alert_id(), "message": "after cooldown"})
        manager.emit(second)
        third = base.model_copy(
            update={"id": new_alert_id(), "component": "db", "message": "other component"}
        )
        manager.emit(third)
        await manager.flush()
    finally:
        manager.close()

    assert manager.recent_alerts(limit=0) == []
    assert manager.recent_alerts(limit=-1) == []
    assert len(manager.recent_alerts(limit=1)) == 1
    recent = manager.recent_alerts(limit=EXPECTED_RECENT_TWO)
    assert [item.id for item in recent] == [third.id, second.id]
    assert len(sink.events) == EXPECTED_THREE_ALERTS

    persisted = [
        json.loads(line)
        for line in (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(persisted) == EXPECTED_THREE_ALERTS


def test_alert_manager_sync_deferral_and_alerting_log_handler(tmp_path: Path) -> None:
    """AlertingLogHandler captures critical, explicit category, and timeout records."""
    sink = _MemorySink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0, log_dir=tmp_path)
    handler = AlertingLogHandler(manager)
    try:
        handler.emit(
            logging.LogRecord(
                "lai.observability.alerts", logging.CRITICAL, __file__, 1, "skip", (), None
            )
        )
        handler.emit(
            logging.LogRecord("lai.db", logging.ERROR, __file__, 1, "plain error", (), None)
        )
        handler.emit(logging.LogRecord("lai.db", logging.ERROR, __file__, 1, 12345, (), None))
        handler.emit(
            logging.LogRecord(
                "lai.inference",
                logging.ERROR,
                __file__,
                1,
                {
                    "event": "process exited",
                    "component": "supervisor",
                    "alert_category": "subprocess_crash",
                    "request_id": "req_9",
                    "job_id": "job_9",
                },
                (),
                None,
            )
        )
        handler.emit(
            logging.LogRecord(
                "lai.inference",
                logging.ERROR,
                __file__,
                1,
                json.dumps({"event": "backend timed out", "alert_category": "unknown_cat"}),
                (),
                None,
            )
        )
        try:
            _raise_long_runtime_error()
        except RuntimeError as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            handler.emit(
                logging.LogRecord(
                    "lai.api", logging.CRITICAL, __file__, 1, "fatal crash", (), exc_info
                )
            )
    finally:
        manager.close()

    alerts = manager.recent_alerts(limit=LARGE_LIMIT)
    assert len(alerts) == EXPECTED_THREE_ALERTS
    assert [a.category for a in alerts] == [
        AlertCategory.FATAL_EXCEPTION,
        AlertCategory.TIMEOUT,
        AlertCategory.SUBPROCESS_CRASH,
    ]
    assert alerts[0].severity is AlertSeverity.CRITICAL
    assert alerts[0].exception_type == "RuntimeError"
    assert alerts[0].traceback_excerpt is not None
    assert len(alerts[0].traceback_excerpt) <= MAX_TRACEBACK_LEN
    assert alerts[2].request_id == "req_9"
    assert alerts[2].job_id == "job_9"
    assert list(tmp_path.glob("alerts.jsonl*")) == [tmp_path / "alerts.jsonl"]

    muted_manager = AlertManager(cooldown_seconds=0.0)
    muted_handler = AlertingLogHandler(muted_manager, alert_on_timeouts=False)
    muted_handler.emit(
        logging.LogRecord(
            "lai.inference",
            logging.ERROR,
            __file__,
            1,
            {"event": "timed out", "alert_category": "timeout"},
            (),
            None,
        )
    )
    assert muted_manager.recent_alerts() == []
    # CRITICAL timeout alerts still fire even when alert_on_timeouts=False
    muted_handler.emit(
        logging.LogRecord(
            "lai.inference",
            logging.CRITICAL,
            __file__,
            1,
            {"event": "critical timeout", "alert_category": "timeout"},
            (),
            None,
        )
    )
    assert [item.category for item in muted_manager.recent_alerts()] == [AlertCategory.TIMEOUT]


async def test_alert_manager_flushes_sync_deferred_deliveries() -> None:
    """Events emitted outside an event loop are delivered when flush() is awaited."""
    sink = _MemorySink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0)
    event = AlertEvent(
        severity=AlertSeverity.ERROR,
        category=AlertCategory.STORAGE_FAILURE,
        component="db",
        message="sqlite busy",
    )
    thread = threading.Thread(target=lambda: manager.emit(event))
    thread.start()
    thread.join()
    await manager.flush()
    assert [item.id for item in sink.events] == [event.id]


def test_pre_init_component_logger_routes_to_files_and_alerts(tmp_path: Path) -> None:
    """Logger created before configure_observability routes to JSONL files and AlertManager."""
    structlog.reset_defaults()
    pre_init_log = get_component_logger("pre_init")
    cfg = LoggingConfig(log_dir=tmp_path, level="INFO", json_console=True, enable_file_logs=True)
    manager = configure_observability(cfg)
    try:
        pre_init_log.error("upstream timed out", alert_category=AlertCategory.TIMEOUT.value)
    finally:
        manager.close()
        for handler in list(logging.getLogger().handlers):
            handler.flush()
            handler.close()

    alerts = manager.recent_alerts()
    assert len(alerts) == 1
    assert alerts[0].category is AlertCategory.TIMEOUT
    assert alerts[0].component == "pre_init"
    for filename in ("lai.jsonl", "errors.jsonl"):
        rows = [
            json.loads(line)
            for line in (tmp_path / filename).read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["component"] == "pre_init"
        assert rows[0]["event"] == "upstream timed out"
