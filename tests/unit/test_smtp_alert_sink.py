"""Unit tests for SmtpAlertSink, FilteredAlertSink, build_alert_sinks, and test delivery."""

from __future__ import annotations

import secrets
import smtplib
from typing import TYPE_CHECKING, cast

import pytest
import respx

from lai.domain.observability import (
    AlertCategory,
    AlertEvent,
    AlertSeverity,
    LoggingConfig,
    SmtpAlertConfig,
    SmtpTlsMode,
)
from lai.observability import (
    AlertManager,
    FilteredAlertSink,
    SmtpAlertSink,
    build_alert_email,
    build_alert_sinks,
    deliver_test_alert,
    resolve_smtp_recipients,
)
from lai.ports import AlertSinkPort

if TYPE_CHECKING:
    from email.message import EmailMessage
    from types import TracebackType


class _FakeSmtpClient:
    """In-memory smtplib.SMTP / SMTP_SSL test double recording protocol calls."""

    def __init__(  # noqa: PLR0913
        self,
        host: str,
        port: int,
        *,
        timeout: float = 10.0,
        context: object | None = None,
        fail_on_send: Exception | None = None,
        recorded: list[dict[str, object]],
        mode: str,
    ) -> None:
        self._entry: dict[str, object] = {
            "mode": mode,
            "host": host,
            "port": port,
            "timeout": timeout,
            "has_context": context is not None,
            "ehlo_count": 0,
            "starttls": False,
            "login": None,
            "messages": [],
        }
        self._fail_on_send = fail_on_send
        recorded.append(self._entry)

    def __enter__(self) -> _FakeSmtpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb

    def ehlo(self) -> None:
        count = self._entry["ehlo_count"]
        assert isinstance(count, int)
        self._entry["ehlo_count"] = count + 1

    def starttls(self, *, context: object | None = None) -> None:
        self._entry["starttls"] = True
        self._entry["has_context"] = context is not None

    def login(self, user: str, password: str) -> None:
        self._entry["login"] = (user, password)

    def send_message(self, msg: EmailMessage) -> None:
        if self._fail_on_send is not None:
            raise self._fail_on_send
        messages = cast("list[EmailMessage]", self._entry["messages"])
        messages.append(msg)


class _RecordingSink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


def _sample_event(
    *,
    severity: AlertSeverity = AlertSeverity.ERROR,
    category: AlertCategory = AlertCategory.APPLICATION_ERROR,
    message: str = "boom\r\ninjected",
) -> AlertEvent:
    return AlertEvent(
        severity=severity,
        category=category,
        component="inference",
        message=message,
        request_id="req_1",
        job_id="job_1",
        session_id="ses_1",
        project_id="prj_1",
        exception_type="RuntimeError",
        details={"token": "Bearer lai_sk_secret_token_value"},
        traceback_excerpt="Traceback: Bearer lai_sk_another_secret",
    )


def test_resolve_recipients_and_email_scrubbing() -> None:
    """Recipients prefer explicit to_addresses, fall back to admin emails, and scrub secrets."""
    cfg = SmtpAlertConfig(
        enabled=True,
        host="smtp.example.test",
        from_address="alerts@example.test",
        to_addresses=("ops@example.test", "ops@example.test"),
    )
    assert resolve_smtp_recipients(cfg, ("admin@example.test",)) == ("ops@example.test",)

    fallback_cfg = SmtpAlertConfig(
        enabled=True,
        host="smtp.example.test",
        from_address="alerts@example.test",
        use_admin_emails_fallback=True,
    )
    assert resolve_smtp_recipients(fallback_cfg, ("admin@example.test", " ")) == (
        "admin@example.test",
    )
    no_fallback = fallback_cfg.model_copy(update={"use_admin_emails_fallback": False})
    assert resolve_smtp_recipients(no_fallback, ("admin@example.test",)) == ()

    msg = build_alert_email(_sample_event(), cfg, ("ops@example.test",))
    assert "\r" not in str(msg["Subject"])
    assert "\n" not in str(msg["Subject"])
    assert str(msg["Subject"]) == "[LAI Alert] [ERROR] inference: boom injected"
    body = msg.get_content()
    assert "lai_sk_secret_token_value" not in body
    assert "lai_sk_another_secret" not in body
    assert "req_1" in body
    assert "RuntimeError" in body


async def test_smtp_alert_sink_tls_modes_auth_and_error_handling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SmtpAlertSink supports STARTTLS, SSL, NONE, auth, strict errors, and safe swallowing."""
    recorded: list[dict[str, object]] = []
    fail_box: list[Exception | None] = [None]

    def _make_smtp(host: str, port: int, timeout: float = 10.0) -> _FakeSmtpClient:
        return _FakeSmtpClient(
            host, port, timeout=timeout, fail_on_send=fail_box[0], recorded=recorded, mode="smtp"
        )

    def _make_ssl(
        host: str, port: int, timeout: float = 10.0, context: object | None = None
    ) -> _FakeSmtpClient:
        return _FakeSmtpClient(
            host,
            port,
            timeout=timeout,
            context=context,
            fail_on_send=fail_box[0],
            recorded=recorded,
            mode="ssl",
        )

    monkeypatch.setattr(smtplib, "SMTP", _make_smtp)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _make_ssl)

    smtp_cred = secrets.token_hex(8)
    base = SmtpAlertConfig(
        enabled=True,
        host="smtp.example.test",
        port=587,
        tls_mode=SmtpTlsMode.STARTTLS,
        username="mailer",
        password=smtp_cred,
        from_address="alerts@example.test",
        to_addresses=("ops@example.test",),
    )
    event = _sample_event()
    await SmtpAlertSink(base).send_alert(event)
    assert recorded[-1]["starttls"] is True
    assert recorded[-1]["ehlo_count"] == 2
    assert recorded[-1]["login"] == ("mailer", smtp_cred)

    ssl_cfg = base.model_copy(
        update={"tls_mode": SmtpTlsMode.SSL, "port": 465, "username": None, "password": None}
    )
    await SmtpAlertSink(ssl_cfg).send_alert(event)
    assert recorded[-1]["mode"] == "ssl"
    assert recorded[-1]["has_context"] is True
    assert recorded[-1]["login"] is None

    none_cfg = base.model_copy(update={"tls_mode": SmtpTlsMode.NONE, "port": 25})
    await SmtpAlertSink(none_cfg).send_alert(event)
    assert recorded[-1]["starttls"] is False

    fail_box[0] = smtplib.SMTPAuthenticationError(535, b"bad auth")
    with pytest.raises(smtplib.SMTPAuthenticationError):
        await SmtpAlertSink(base).send_alert_strict(event)
    await SmtpAlertSink(base).send_alert(event)
    assert "SmtpAlertSink delivery failed" in capsys.readouterr().err

    with pytest.raises(OSError, match="host"):
        await SmtpAlertSink(base.model_copy(update={"host": ""})).send_alert_strict(event)
    with pytest.raises(OSError, match="recipients"):
        await SmtpAlertSink(
            base.model_copy(update={"to_addresses": (), "use_admin_emails_fallback": False})
        ).send_alert_strict(event)


async def test_filtered_sink_build_sinks_and_test_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FilteredAlertSink enforces severity/category rules and deliver_test_alert reports status."""
    inner = _RecordingSink()
    crit_only = FilteredAlertSink(
        inner,
        name="custom",
        min_severity=AlertSeverity.CRITICAL,
        categories=(AlertCategory.TIMEOUT,),
    )
    await crit_only.send_alert(_sample_event(severity=AlertSeverity.ERROR))
    await crit_only.send_alert(
        _sample_event(severity=AlertSeverity.CRITICAL, category=AlertCategory.APPLICATION_ERROR)
    )
    assert inner.events == []
    await crit_only.send_alert(
        _sample_event(severity=AlertSeverity.CRITICAL, category=AlertCategory.TIMEOUT)
    )
    assert len(inner.events) == 1

    recorded: list[dict[str, object]] = []

    def _smtp_factory(host: str, port: int, timeout: float = 10.0) -> _FakeSmtpClient:
        return _FakeSmtpClient(host, port, timeout=timeout, recorded=recorded, mode="smtp")

    monkeypatch.setattr(smtplib, "SMTP", _smtp_factory)
    cfg = LoggingConfig(
        alert_webhook_url="https://hooks.example.test/test",
        alert_webhook_enabled=True,
        alert_webhook_min_severity=AlertSeverity.CRITICAL,
        alert_smtp=SmtpAlertConfig(
            enabled=True,
            host="smtp.example.test",
            from_address="alerts@example.test",
            to_addresses=("ops@example.test",),
            min_severity=AlertSeverity.CRITICAL,
        ),
    )
    sinks = build_alert_sinks(cfg)
    assert len(sinks) == 2
    manager = AlertManager(sinks=sinks, cooldown_seconds=0.0)

    err_event = _sample_event(severity=AlertSeverity.ERROR)
    skipped = await deliver_test_alert(manager, err_event, bypass_filters=False)
    assert all(item.skipped and not item.delivered for item in skipped)

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://hooks.example.test/test").respond(500, text="boom")
        outcomes = await deliver_test_alert(manager, err_event, bypass_filters=True)
    by_name = {item.sink_name: item for item in outcomes}
    assert by_name["smtp"].delivered is True
    assert by_name["webhook"].delivered is False
    assert by_name["webhook"].error is not None

    disabled_sinks = build_alert_sinks(cfg.model_copy(update={"alert_webhook_enabled": False}))
    assert len(disabled_sinks) == 1
    assert isinstance(disabled_sinks[0], FilteredAlertSink)
    assert disabled_sinks[0].name == "smtp"
