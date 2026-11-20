"""Outbound stdlib SMTP alert sink, severity/category filtering, and test delivery."""

from __future__ import annotations

import asyncio
import ipaddress
import smtplib
import ssl
import sys
from email.message import EmailMessage
from typing import TYPE_CHECKING

import httpx

from lai.domain.observability import (
    AlertCategory,
    AlertEvent,
    AlertSeverity,
    AlertSinkDeliveryResult,
    LoggingConfig,
    SmtpAlertConfig,
    SmtpTlsMode,
)
from lai.observability.alerts import AlertManager, WebhookAlertSink
from lai.observability.scrubber import scrub_sensitive_text
from lai.ports.observability import AlertSinkPort

if TYPE_CHECKING:
    from collections.abc import Sequence

_SEVERITY_RANK: dict[AlertSeverity, int] = {
    AlertSeverity.ERROR: 1,
    AlertSeverity.CRITICAL: 2,
}
_MAX_EMAIL_FIELD_CHARS = 500
_MAX_EMAIL_TRACEBACK_CHARS = 2000


def is_loopback_host(host: str) -> bool:
    """Return True only for localhost or explicit IPv4/IPv6 loopback addresses."""
    cleaned = host.strip().lower().strip("[]")
    if not cleaned:
        return False
    if cleaned == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def _sanitize_header_value(value: str) -> str:
    """Strip CR/LF characters from email header values to prevent header injection."""
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def resolve_smtp_recipients(
    config: SmtpAlertConfig,
    fallback_admin_emails: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve effective SMTP recipients using explicit to_addresses or admin email fallback."""
    explicit = [item.strip() for item in config.to_addresses if item.strip()]
    if explicit:
        return tuple(dict.fromkeys(explicit))
    if config.use_admin_emails_fallback:
        fallback = [item.strip() for item in fallback_admin_emails if item.strip()]
        return tuple(dict.fromkeys(fallback))
    return ()


def build_alert_email(
    event: AlertEvent,
    config: SmtpAlertConfig,
    recipients: Sequence[str],
) -> EmailMessage:
    """Construct a plain-text EmailMessage with scrubbed diagnostic details for an alert."""
    prefix = _sanitize_header_value(config.subject_prefix)
    subject_msg = _sanitize_header_value(event.message)[:120]
    subject = f"{prefix} [{event.severity.value.upper()}] {event.component}: {subject_msg}"

    lines = [
        f"Alert ID:   {event.id}",
        f"Timestamp:  {event.created_at.isoformat()}",
        f"Severity:   {event.severity.value.upper()}",
        f"Category:   {event.category.value}",
        f"Component:  {event.component}",
        f"Message:    {scrub_sensitive_text(event.message, max_chars=_MAX_EMAIL_FIELD_CHARS)}",
    ]
    for label, value in (
        ("Request ID", event.request_id),
        ("Job ID", event.job_id),
        ("Session ID", event.session_id),
        ("Project ID", event.project_id),
        ("Exception", event.exception_type),
    ):
        if value:
            lines.append(f"{label + ':':<12}{value}")

    if event.details:
        lines.append("")
        lines.append("Details:")
        for key, raw_val in sorted(event.details.items()):
            cleaned = scrub_sensitive_text(raw_val, max_chars=_MAX_EMAIL_FIELD_CHARS)
            lines.append(f"  {key}: {cleaned}")

    if event.traceback_excerpt:
        scrubbed_tb = scrub_sensitive_text(
            event.traceback_excerpt,
            max_chars=_MAX_EMAIL_TRACEBACK_CHARS,
        )
        lines.append("")
        lines.append("Traceback excerpt:")
        lines.append(scrubbed_tb)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = _sanitize_header_value(config.from_address or "")
    msg["To"] = ", ".join(_sanitize_header_value(addr) for addr in recipients)
    msg.set_content("\n".join(lines) + "\n")
    return msg


class SmtpAlertSink(AlertSinkPort):
    """Asynchronous stdlib SMTP alert sink supporting STARTTLS, SSL, and plaintext relays."""

    def __init__(
        self,
        config: SmtpAlertConfig,
        *,
        recipients: Sequence[str] = (),
    ) -> None:
        self._config = config
        self._recipients = tuple(recipients) if recipients else resolve_smtp_recipients(config)

    @property
    def recipients(self) -> tuple[str, ...]:
        """Return effective email recipients targeted by this SMTP sink."""
        return self._recipients

    def _deliver_sync(self, message: EmailMessage) -> None:
        host = (self._config.host or "").strip()
        if not host:
            msg = "SMTP alert host is not configured."
            raise OSError(msg)
        if not self._recipients:
            msg = "SMTP alert recipients list is empty."
            raise OSError(msg)
        timeout = self._config.timeout_seconds
        tls_mode = self._config.tls_mode
        password = (self._config.password or "").strip()
        if password and tls_mode is SmtpTlsMode.NONE and not is_loopback_host(host):
            msg = "Refusing plaintext SMTP authentication to non-loopback host."
            raise OSError(msg)

        if tls_mode is SmtpTlsMode.SSL:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, self._config.port, timeout=timeout, context=context) as cli:
                self._authenticate_and_send(cli, message)
            return

        with smtplib.SMTP(host, self._config.port, timeout=timeout) as cli:
            if tls_mode is SmtpTlsMode.STARTTLS:
                cli.ehlo()
                cli.starttls(context=ssl.create_default_context())
                cli.ehlo()
            self._authenticate_and_send(cli, message)

    def _authenticate_and_send(self, client: smtplib.SMTP, message: EmailMessage) -> None:
        username = (self._config.username or "").strip()
        password = (self._config.password or "").strip()
        if username and password:
            if self._config.tls_mode is SmtpTlsMode.NONE and not is_loopback_host(
                self._config.host or ""
            ):
                msg = "Refusing plaintext SMTP authentication to non-loopback host."
                raise OSError(msg)
            client.login(username, password)
        client.send_message(message)

    async def send_alert_strict(self, event: AlertEvent) -> None:
        """Send an alert email via SMTP, propagating SMTP/network errors to the caller."""
        email_message = build_alert_email(event, self._config, self._recipients)
        await asyncio.to_thread(self._deliver_sync, email_message)

    async def send_alert(self, event: AlertEvent) -> None:
        """Send an alert email via SMTP without raising on network or authentication errors."""
        try:
            await self.send_alert_strict(event)
        except (smtplib.SMTPException, OSError) as exc:
            sys.stderr.write(f"SmtpAlertSink delivery failed: {exc}\n")


class FilteredAlertSink(AlertSinkPort):
    """AlertSinkPort decorator applying per-sink severity and category routing filters."""

    def __init__(
        self,
        inner: AlertSinkPort,
        *,
        name: str,
        min_severity: AlertSeverity = AlertSeverity.ERROR,
        categories: Sequence[AlertCategory] = (),
    ) -> None:
        self._inner = inner
        self._name = name
        self._min_severity = min_severity
        self._categories = frozenset(categories)

    @property
    def name(self) -> str:
        """Return human-readable sink identifier (e.g. 'smtp' or 'webhook')."""
        return self._name

    @property
    def inner(self) -> AlertSinkPort:
        """Return the underlying wrapped AlertSinkPort."""
        return self._inner

    def matches(self, event: AlertEvent) -> bool:
        """Return True when event satisfies this sink's severity and category filters."""
        if _SEVERITY_RANK[event.severity] < _SEVERITY_RANK[self._min_severity]:
            return False
        return not self._categories or event.category in self._categories

    async def send_alert(self, event: AlertEvent) -> None:
        """Forward event to the wrapped sink only when routing filters match."""
        if not self.matches(event):
            return
        await self._inner.send_alert(event)


def build_alert_sinks(
    config: LoggingConfig,
    *,
    fallback_admin_emails: Sequence[str] = (),
) -> tuple[AlertSinkPort, ...]:
    """Build all enabled, filter-wrapped alert sinks from LoggingConfig."""
    sinks: list[AlertSinkPort] = []
    if config.alert_webhook_enabled and config.alert_webhook_url:
        sinks.append(
            FilteredAlertSink(
                WebhookAlertSink(config.alert_webhook_url),
                name="webhook",
                min_severity=config.alert_webhook_min_severity,
                categories=config.alert_webhook_categories,
            )
        )
    if config.alert_smtp.enabled:
        recipients = resolve_smtp_recipients(config.alert_smtp, fallback_admin_emails)
        if config.alert_smtp.host and config.alert_smtp.from_address and recipients:
            sinks.append(
                FilteredAlertSink(
                    SmtpAlertSink(config.alert_smtp, recipients=recipients),
                    name="smtp",
                    min_severity=config.alert_smtp.min_severity,
                    categories=config.alert_smtp.categories,
                )
            )
    return tuple(sinks)


async def _deliver_strict_to_sink(
    sink: AlertSinkPort,
    event: AlertEvent,
    *,
    bypass_filters: bool,
) -> AlertSinkDeliveryResult:
    """Deliver a synthetic test alert to one sink and capture structured outcome."""
    sink_name = type(sink).__name__
    target: AlertSinkPort = sink
    if isinstance(sink, FilteredAlertSink):
        sink_name = sink.name
        if not bypass_filters and not sink.matches(event):
            return AlertSinkDeliveryResult(sink_name=sink_name, delivered=False, skipped=True)
        target = sink.inner

    try:
        if isinstance(target, (SmtpAlertSink, WebhookAlertSink)):
            await target.send_alert_strict(event)
        else:
            await target.send_alert(event)
    except (smtplib.SMTPException, httpx.HTTPError, OSError, RuntimeError) as exc:
        prefix = "SMTP delivery failed" if isinstance(target, SmtpAlertSink) else "Delivery failed"
        return AlertSinkDeliveryResult(
            sink_name=sink_name,
            delivered=False,
            error=f"{prefix} ({type(exc).__name__})",
        )
    return AlertSinkDeliveryResult(sink_name=sink_name, delivered=True)


async def deliver_test_alert(
    alert_manager: AlertManager,
    event: AlertEvent,
    *,
    bypass_filters: bool = True,
) -> list[AlertSinkDeliveryResult]:
    """Synchronously test delivery across all active sinks on alert_manager."""
    results: list[AlertSinkDeliveryResult] = []
    for sink in alert_manager.sinks:
        outcome = await _deliver_strict_to_sink(sink, event, bypass_filters=bypass_filters)
        results.append(outcome)
    return results
