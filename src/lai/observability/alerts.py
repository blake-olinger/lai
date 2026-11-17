"""Automatic fatal/timeout alerting, cooldown deduplication, and webhook delivery."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import httpx

from lai.domain.observability import AlertCategory, AlertEvent, AlertSeverity
from lai.observability.rotation import DailyRotatingStreamWriter
from lai.ports.observability import AlertManagerPort, AlertSinkPort

if TYPE_CHECKING:
    from pathlib import Path

_MAX_TRACEBACK_CHARS = 2000


class WebhookAlertSink(AlertSinkPort):
    """HTTP webhook alert sink posting structured alert JSON payloads."""

    def __init__(
        self,
        webhook_url: str,
        *,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def send_alert(self, event: AlertEvent) -> None:
        """Post an alert payload to the configured webhook without raising on failure."""
        summary = (
            f"[{event.severity.value.upper()}] {event.component} "
            f"({event.category.value}): {event.message}"
        )
        payload: dict[str, object] = {
            "text": summary,
            "content": summary,
            "event": event.model_dump(mode="json"),
        }
        try:
            if self._client is not None:
                response = await self._client.post(self._webhook_url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.post(self._webhook_url, json=payload)
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            sys.stderr.write(f"WebhookAlertSink delivery failed: {exc}\n")


class AlertManager(AlertManagerPort):
    """Deduplicating alert manager with ring buffer, JSONL persistence, and async sinks."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        sinks: Sequence[AlertSinkPort] = (),
        cooldown_seconds: float = 60.0,
        max_recent: int = 200,
        log_dir: Path | None = None,
        retention_days: int = 30,
        compress_rotated: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sinks = tuple(sinks)
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[AlertEvent] = deque(maxlen=max_recent)
        self._last_emitted: dict[tuple[str, AlertCategory, str | None], float] = {}
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._deferred: deque[AlertEvent] = deque(maxlen=max_recent)
        self._writer = (
            DailyRotatingStreamWriter(
                log_dir / "alerts.jsonl",
                retention_days=retention_days,
                compress=compress_rotated,
                logger_name="lai.observability.alerts",
            )
            if log_dir is not None
            else None
        )

    def emit(self, event: AlertEvent) -> None:
        """Record an alert event unless suppressed by cooldown, and dispatch to sinks."""
        dedup_key = (event.component, event.category, event.exception_type)
        now = self._clock()
        with self._lock:
            prev = self._last_emitted.get(dedup_key)
            if prev is not None and (now - prev) < self._cooldown_seconds:
                return
            self._last_emitted[dedup_key] = now
            self._recent.append(event)
            if self._writer is not None:
                self._writer.write_line(event.model_dump_json())

        if not self._sinks:
            return
        self._schedule_delivery(event)

    def _schedule_delivery(self, event: AlertEvent) -> None:
        """Dispatch event to all sinks on the running event loop or defer until flush."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with self._lock:
                self._deferred.append(event)
            return

        for sink in self._sinks:
            task = loop.create_task(sink.send_alert(event))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def flush(self) -> None:
        """Schedule any deferred events and await completion of all sink deliveries."""
        with self._lock:
            deferred = list(self._deferred)
            self._deferred.clear()
        for event in deferred:
            self._schedule_delivery(event)
        while self._pending_tasks:
            tasks = list(self._pending_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def recent_alerts(self, limit: int = 50) -> list[AlertEvent]:
        """Return up to limit most recent alert events, newest first."""
        with self._lock:
            return list(reversed(list(self._recent)))[: max(limit, 0)]

    def close(self) -> None:
        """Close the rotating alerts.jsonl writer if open."""
        if self._writer is not None:
            self._writer.close()


def _extract_payload(record: logging.LogRecord) -> dict[str, object]:
    """Extract structured key/value fields from a structlog or stdlib LogRecord."""
    if isinstance(record.msg, Mapping):
        raw = cast("Mapping[object, object]", record.msg)
        return {str(key): value for key, value in raw.items()}
    if isinstance(record.msg, str) and record.msg.lstrip().startswith("{"):
        try:
            parsed: object = json.loads(record.msg)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            mapping = cast("dict[object, object]", parsed)
            return {str(key): value for key, value in mapping.items()}
    return {}


def _extract_exception_info(
    record: logging.LogRecord,
    payload: dict[str, object],
) -> tuple[str | None, str | None]:
    """Extract exception class name and bounded traceback excerpt if present."""
    exc_type_val = payload.get("exception_type")
    exc_type_name = str(exc_type_val) if isinstance(exc_type_val, str) else None
    tb_excerpt: str | None = None

    if record.exc_info and record.exc_info[0] is not None:
        if exc_type_name is None:
            exc_type_name = record.exc_info[0].__name__
        rendered = "".join(traceback.format_exception(*record.exc_info)).strip()
        tb_excerpt = rendered[-_MAX_TRACEBACK_CHARS:]
    elif isinstance(payload.get("exception"), str):
        rendered = cast("str", payload["exception"]).strip()
        tb_excerpt = rendered[-_MAX_TRACEBACK_CHARS:]
    return exc_type_name, tb_excerpt


def _resolve_category(
    record: logging.LogRecord,
    payload: dict[str, object],
    message: str,
    exception_type: str | None,
    *,
    alert_on_timeouts: bool,
) -> AlertCategory | None:
    """Resolve AlertCategory for critical records, explicit alerts, or timeouts."""
    raw_cat = payload.get("alert_category") or getattr(record, "alert_category", None)
    if isinstance(raw_cat, str):
        try:
            category = AlertCategory(raw_cat)
        except ValueError:
            category = None
        if category is not None:
            if (
                category is AlertCategory.TIMEOUT
                and not alert_on_timeouts
                and record.levelno < logging.CRITICAL
            ):
                return None
            return category

    lower_msg = message.lower()
    is_timeout = "timeout" in lower_msg or "timed out" in lower_msg
    if exception_type is not None and "timeout" in exception_type.lower():
        is_timeout = True
    if is_timeout and alert_on_timeouts:
        return AlertCategory.TIMEOUT
    if record.levelno >= logging.CRITICAL:
        return AlertCategory.FATAL_EXCEPTION
    return None


def _opt_str(value: object) -> str | None:
    """Convert non-empty values to str, returning None for empty or missing values."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


class AlertingLogHandler(logging.Handler):
    """Logging handler emitting AlertEvents for critical and timeout/alert errors."""

    def __init__(
        self,
        alert_manager: AlertManagerPort,
        *,
        alert_on_timeouts: bool = True,
    ) -> None:
        super().__init__(level=logging.ERROR)
        self._alert_manager = alert_manager
        self._alert_on_timeouts = alert_on_timeouts

    def emit(self, record: logging.LogRecord) -> None:
        """Inspect ERROR/CRITICAL log records and emit matching AlertEvents."""
        if record.name == "lai.observability.alerts":
            return
        payload = _extract_payload(record)
        raw_msg = payload.get("event")
        message = str(raw_msg) if raw_msg is not None else record.getMessage()
        exception_type, tb_excerpt = _extract_exception_info(record, payload)
        category = _resolve_category(
            record,
            payload,
            message,
            exception_type,
            alert_on_timeouts=self._alert_on_timeouts,
        )
        if category is None:
            return

        component = (
            _opt_str(payload.get("component"))
            or _opt_str(getattr(record, "component", None))
            or record.name
        )
        severity = (
            AlertSeverity.CRITICAL if record.levelno >= logging.CRITICAL else AlertSeverity.ERROR
        )
        event = AlertEvent(
            severity=severity,
            category=category,
            component=component,
            message=message,
            request_id=_opt_str(payload.get("request_id")),
            job_id=_opt_str(payload.get("job_id")),
            session_id=_opt_str(payload.get("session_id")),
            project_id=_opt_str(payload.get("project_id")),
            exception_type=exception_type,
            traceback_excerpt=tb_excerpt,
        )
        self._alert_manager.emit(event)
