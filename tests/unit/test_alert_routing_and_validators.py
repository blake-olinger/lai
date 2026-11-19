"""Unit tests for error-level alert routing, live alerting reconfigure, and SMTP validators."""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING

import pytest
import respx
from pydantic import ValidationError

from lai.config.settings import LaiSettings
from lai.domain.observability import (
    AlertCategory,
    AlertEvent,
    AlertSeverity,
    LoggingConfig,
    SmtpAlertConfig,
)
from lai.observability import (
    AlertingLogHandler,
    AlertManager,
    WebhookAlertSink,
    configure_observability,
    reconfigure_observability_alerting,
    set_runtime_log_level,
)
from lai.ports import AlertSinkPort

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class _RecordingSink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _cleanup_root_handlers() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_alert_on_errors_reconfigure_and_runtime_log_level(tmp_path: Path) -> None:
    """AlertingLogHandler routes plain ERRORs, respects timeout mute, and hot-swaps live."""
    sink1, sink2 = _RecordingSink(), _RecordingSink()
    manager = AlertManager(sinks=(sink1,), cooldown_seconds=60.0)
    handler = AlertingLogHandler(manager, alert_on_timeouts=False, alert_on_errors=True)

    # Plain ERROR -> APPLICATION_ERROR; WARNING ignored even if directly emitted
    handler.emit(logging.LogRecord("lai.db", logging.WARNING, __file__, 1, "warn only", (), None))
    assert manager.recent_alerts() == []
    handler.emit(logging.LogRecord("lai.db", logging.ERROR, __file__, 1, "plain failure", (), None))
    assert [a.category for a in manager.recent_alerts()] == [AlertCategory.APPLICATION_ERROR]

    # Non-critical timeout stays muted when alert_on_timeouts=False even if alert_on_errors=True
    handler.emit(
        logging.LogRecord("lai.api", logging.ERROR, __file__, 1, "upstream timed out", (), None)
    )
    handler.emit(
        logging.LogRecord(
            "lai.api",
            logging.ERROR,
            __file__,
            1,
            {"event": "op failed", "exception_type": "ReadTimeout"},
            (),
            None,
        )
    )
    assert len(manager.recent_alerts()) == 1

    # CRITICAL timeout still fires as FATAL_EXCEPTION when untagged
    handler.emit(
        logging.LogRecord("lai.api", logging.CRITICAL, __file__, 1, "fatal timeout", (), None)
    )
    assert manager.recent_alerts()[0].category is AlertCategory.FATAL_EXCEPTION

    # Live reconfigure swaps sinks & handler toggles without dropping recent alerts
    manager.reconfigure(sinks=(sink2,), cooldown_seconds=0.0)
    handler.reconfigure(alert_on_timeouts=True, alert_on_errors=False)
    handler.emit(logging.LogRecord("lai.db", logging.ERROR, __file__, 1, "ignored now", (), None))
    assert len(manager.recent_alerts()) == 2
    handler.emit(
        logging.LogRecord("lai.api", logging.ERROR, __file__, 1, "upstream timed out", (), None)
    )
    await manager.flush()
    assert len(sink2.events) == 1
    assert sink2.events[0].category is AlertCategory.TIMEOUT

    cfg = LoggingConfig(log_dir=tmp_path, level="INFO", enable_file_logs=True)
    obs_mgr = configure_observability(cfg)
    try:
        reconfigure_observability_alerting(
            obs_mgr,
            cfg.model_copy(update={"alert_on_errors": False, "alert_cooldown_seconds": 5.0}),
        )
        assert set_runtime_log_level("debug") == "DEBUG"
        assert logging.getLogger().level == logging.DEBUG
        with pytest.raises(ValueError, match="Invalid log level"):
            set_runtime_log_level("VERBOSE")
        event = AlertEvent(
            severity=AlertSeverity.ERROR,
            category=AlertCategory.APPLICATION_ERROR,
            component="api",
            message="strict test",
        )
        with respx.mock(assert_all_called=True) as mock:
            mock.post("https://hooks.example.test/strict").respond(204)
            await WebhookAlertSink("https://hooks.example.test/strict").send_alert_strict(event)
    finally:
        obs_mgr.close()


@pytest.mark.parametrize(
    "bad_logging",
    [
        LoggingConfig(alert_webhook_url="ftp://hooks.example.test/x"),
        LoggingConfig(alert_webhook_url="https://hooks.example.test/ bad"),
        LoggingConfig(alert_webhook_url=" https://hooks.example.test/x"),
        LoggingConfig(alert_webhook_url="https:///missing-host"),
        LoggingConfig(alert_smtp=SmtpAlertConfig(password=secrets.token_hex(6))),
        LoggingConfig(alert_smtp=SmtpAlertConfig(subject_prefix="[LAI]\r\nBcc: evil@test")),
        LoggingConfig(alert_smtp=SmtpAlertConfig(enabled=True)),
        LoggingConfig(
            alert_smtp=SmtpAlertConfig(enabled=True, host="bad host", from_address="a@b.test")
        ),
        LoggingConfig(alert_smtp=SmtpAlertConfig(enabled=True, host="smtp.test")),
        LoggingConfig(
            alert_smtp=SmtpAlertConfig(
                enabled=True,
                host="smtp.test",
                from_address="bad-email",
                to_addresses=("ops@test.local",),
            )
        ),
        LoggingConfig(
            alert_smtp=SmtpAlertConfig(
                enabled=True,
                host="smtp.test",
                from_address="a@b@c.test",
                to_addresses=("ops@test.local",),
            )
        ),
        LoggingConfig(
            alert_smtp=SmtpAlertConfig(
                enabled=True,
                host="smtp.test",
                from_address="alerts@test.local\r\n",
                to_addresses=("ops@test.local",),
            )
        ),
        LoggingConfig(
            alert_smtp=SmtpAlertConfig(
                enabled=True,
                host="smtp.test",
                from_address="alerts@test.local",
                to_addresses=(),
                use_admin_emails_fallback=True,
            )
        ),
        LoggingConfig(
            alert_smtp=SmtpAlertConfig(
                enabled=True,
                host="smtp.test",
                from_address="alerts@test.local",
                to_addresses=("not an email",),
            )
        ),
    ],
)
def test_alerting_settings_fail_closed_validation(bad_logging: LoggingConfig) -> None:
    """Invalid webhook URLs and incomplete/injectable SMTP settings fail closed."""
    with pytest.raises(ValidationError):
        LaiSettings(logging=bad_logging)


def test_smtp_settings_accept_admin_emails_fallback() -> None:
    """Enabled SMTP config validates when recipients resolve from oidc_admin_emails."""
    settings = LaiSettings(
        oidc_admin_emails=("owner@example.test",),
        logging=LoggingConfig(
            alert_smtp=SmtpAlertConfig(
                enabled=True,
                host="smtp.example.test",
                from_address="alerts@example.test",
                use_admin_emails_fallback=True,
            )
        ),
    )
    assert settings.logging.alert_smtp.enabled is True
