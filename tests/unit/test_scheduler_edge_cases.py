"""Additional unit tests for scheduler validation, timeout alerts, and keepalive streams."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from lai.domain.errors import ValidationFailedError
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.observability import AlertManager, configure_observability
from lai.ports.observability import AlertSinkPort
from lai.scheduler import (
    FairPriorityQueue,
    InferenceSpooler,
    await_or_keepalive_json,
    stream_with_sse_keepalive,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from lai.domain.observability import AlertEvent


class _MemoryAlertSink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _req(job_id: str, session_id: str = "ses_1") -> LeaseRequest:
    return LeaseRequest(job_id=job_id, session_id=session_id, project_id="prj_1")


def test_validation_guards_for_queue_and_spooler() -> None:
    """Queue and spooler reject invalid bounds and duplicate job_ids."""
    with pytest.raises(ValidationFailedError):
        FairPriorityQueue(max_queued_jobs=0)
    with pytest.raises(ValidationFailedError):
        InferenceSpooler(acquire_timeout_seconds=0.0)

    queue = FairPriorityQueue(max_queued_jobs=2)
    req = _req("dup_job")
    queue.enqueue(req)
    with pytest.raises(ValidationFailedError):
        queue.enqueue(req)


async def test_spooler_handoff_cancellation() -> None:
    """Cancelling at slot handoff preserves progress and rejects duplicate active job."""
    spooler = InferenceSpooler()
    entered_1, release_1 = asyncio.Event(), asyncio.Event()
    ran: list[str] = []

    async def waiter(job_id: str, *, hold: bool = False) -> None:
        async with spooler.lease(_req(job_id, session_id=f"s_{job_id}")):
            ran.append(job_id)
            if hold:
                entered_1.set()
                await release_1.wait()

    t1 = asyncio.create_task(waiter("h1", hold=True))
    await entered_1.wait()
    with pytest.raises(ValidationFailedError):
        async with spooler.lease(_req("h1")):
            pass

    t2 = asyncio.create_task(waiter("h2"))
    t3 = asyncio.create_task(waiter("h3"))
    await asyncio.sleep(0)
    release_1.set()
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2
    await asyncio.gather(t1, t3)
    assert ran == ["h1", "h3"]
    assert spooler.snapshot().active_job_id is None


async def test_spooler_timeout_emits_alert(tmp_path: Path) -> None:
    """Acquisition timeout evicts waiter and emits AlertCategory.TIMEOUT."""
    sink = _MemoryAlertSink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=False), manager)
    try:
        timeout_spooler = InferenceSpooler(acquire_timeout_seconds=0.02)
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocker() -> None:
            async with timeout_spooler.lease(_req("job_holder", session_id="s_hold")):
                entered.set()
                await release.wait()

        holder = asyncio.create_task(blocker())
        await entered.wait()
        with pytest.raises(TimeoutError):
            async with timeout_spooler.lease(_req("job_timed", session_id="s_wait")):
                pass
        release.set()
        await holder
        await manager.flush()
    finally:
        manager.close()

    assert timeout_spooler.snapshot().total_evicted == 1
    assert len(sink.events) == 1
    assert sink.events[0].category is AlertCategory.TIMEOUT
    assert sink.events[0].component == "scheduler"
    assert sink.events[0].job_id == "job_timed"


async def test_keepalive_error_propagation_and_stream_disconnect() -> None:
    """Keepalive helpers propagate producer exceptions, serialize rich types, and stop cleanly."""

    async def failing_events() -> AsyncIterator[str]:
        yield "ok"
        msg = "stream failed"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="stream failed"):
        _ = [
            item
            async for item in stream_with_sse_keepalive(
                failing_events(),
                keepalive_factory=lambda: "ka",
                interval_seconds=0.5,
            )
        ]

    with pytest.raises(ValidationFailedError):
        async for _ in stream_with_sse_keepalive(
            failing_events(),
            keepalive_factory=lambda: "ka",
            interval_seconds=0.0,
        ):
            pass

    async def rich_coro() -> dict[str, object]:
        await asyncio.sleep(0.03)
        return {
            "ts": datetime(2026, 1, 1, tzinfo=UTC),
            "priority": JobPriority.INTERACTIVE,
            "path": Path("logs/app.log"),
            "tags": {"a"},
        }

    outcome = await await_or_keepalive_json(
        rich_coro(),
        threshold_seconds=0.01,
        interval_seconds=0.5,
    )
    chunks = [chunk async for chunk in cast("AsyncIterator[bytes]", outcome)]
    assert chunks[0] == b"\n"
    assert b"logs/app.log" in chunks[-1] or b"logs\\\\app.log" in chunks[-1]

    disc_event = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_hanging() -> dict[str, str]:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"ok": "never"}

    outcome_2 = await await_or_keepalive_json(
        slow_hanging(),
        threshold_seconds=0.01,
        interval_seconds=5.0,
        is_disconnected=disc_event.is_set,
    )
    stream_2 = cast("AsyncIterator[bytes]", outcome_2)
    assert await anext(stream_2) == b"\n"
    disc_event.set()
    assert [chunk async for chunk in stream_2] == []
    assert cancelled.is_set()
