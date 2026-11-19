"""Unit tests for SessionGate, FairPriorityQueue, InferenceSpooler, and keepalives."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, cast

import pytest

from lai.domain.errors import ClientDisconnectedError, QueueFullError, ValidationFailedError
from lai.domain.observability import LoggingConfig
from lai.domain.scheduler import JobPriority, LeaseRequest, SpoolerSnapshot
from lai.observability import bind_log_context, configure_observability, get_component_logger
from lai.ports.scheduler import InferenceSpoolerPort, SessionGatePort
from lai.scheduler import (
    FairPriorityQueue,
    InferenceSpooler,
    SessionGate,
    await_or_keepalive_json,
    stream_with_sse_keepalive,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator
    from pathlib import Path

TWO_TURNS = 2
THREE_TURNS = 3
EXPECTED_FOUR = 4
EXPECTED_FIVE = 5
TWENTY_JOBS = 20
MAX_QUEUE_TWO = 2
HTTP_SERVICE_UNAVAILABLE = 503
EXPECTED_WAIT_MS = 250.0
MIN_KEEPALIVE_NEWLINES = 2


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _req(
    job_id: str,
    *,
    principal_id: str = "usr_1",
    session_id: str = "ses_1",
    priority: JobPriority = JobPriority.INTERACTIVE,
    bypass_principal_quota: bool = False,
) -> LeaseRequest:
    return LeaseRequest(
        job_id=job_id,
        session_id=session_id,
        project_id="prj_1",
        principal_id=principal_id,
        priority=priority,
        bypass_principal_quota=bypass_principal_quota,
    )


async def test_session_gate_serializes_same_session_and_allows_cross_session() -> None:
    """Same session serializes FIFO and prunes idle locks; distinct sessions run concurrently."""
    gate = SessionGate()
    assert isinstance(gate, SessionGatePort)
    with pytest.raises(ValidationFailedError):
        async with gate.acquire(""):
            pass

    order: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def turn(tag: str) -> None:
        async with gate.acquire("ses_shared"):
            order.append(f"enter:{tag}")
            if tag == "t1":
                entered.set()
                await release.wait()
            order.append(f"exit:{tag}")

    t1 = asyncio.create_task(turn("t1"))
    await entered.wait()
    assert gate.is_locked("ses_shared")
    t2 = asyncio.create_task(turn("t2"))
    t3 = asyncio.create_task(turn("t3"))
    await asyncio.sleep(0)
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2
    release.set()
    await asyncio.gather(t1, t3)
    assert order == ["enter:t1", "exit:t1", "enter:t3", "exit:t3"]
    assert gate.tracked_session_count == 0

    both_active = asyncio.Event()
    active_sessions: set[str] = set()

    async def cross(session_id: str) -> None:
        async with gate.acquire(session_id):
            active_sessions.add(session_id)
            if len(active_sessions) == TWO_TURNS:
                both_active.set()
            await asyncio.wait_for(both_active.wait(), timeout=1.0)

    await asyncio.gather(cross("ses_a"), cross("ses_b"))
    assert gate.tracked_session_count == 0


def test_fair_priority_queue_round_robin_priority_capacity_and_evict() -> None:
    """FairPriorityQueue balances lanes, respects priority bands, caps size, and evicts."""
    queue = FairPriorityQueue(max_queued_jobs=7)
    queue.enqueue(_req("j_a1_1", principal_id="p_a", session_id="s1"))
    queue.enqueue(_req("j_a1_2", principal_id="p_a", session_id="s1"))
    queue.enqueue(_req("j_a1_3", principal_id="p_a", session_id="s1"))
    queue.enqueue(_req("j_b2_1", principal_id="p_b", session_id="s2"))
    queue.enqueue(_req("j_a3_1", principal_id="p_a", session_id="s3"))
    queue.enqueue(
        _req("j_apr", principal_id="p_c", session_id="s4", priority=JobPriority.APPROVAL_RESUME)
    )
    queue.enqueue(
        _req("j_mr", principal_id="p_d", session_id="s5", priority=JobPriority.MAP_REDUCE)
    )

    with pytest.raises(QueueFullError) as full_exc:
        queue.enqueue(_req("j_overflow"))
    assert full_exc.value.status_code == HTTP_SERVICE_UNAVAILABLE
    assert queue.counts_by_priority()["INTERACTIVE"] == EXPECTED_FIVE
    assert queue.evict("j_a1_2") is True
    assert queue.evict("missing") is False
    assert "j_a1_2" not in queue

    popped = [item.job_id for item in iter(queue.dequeue, None)]
    assert popped == ["j_a1_1", "j_b2_1", "j_a3_1", "j_a1_3", "j_apr", "j_mr"]


def test_fair_queue_two_level_principal_round_robin_and_quota() -> None:
    """One principal with N sessions gets one turn per principal round, bounded by quota."""
    queue = FairPriorityQueue(max_queued_jobs=10, max_queued_jobs_per_principal=MAX_QUEUE_TWO)
    queue.enqueue(_req("a_s1", principal_id="p_a", session_id="s1"))
    queue.enqueue(_req("a_s2", principal_id="p_a", session_id="s2"))
    with pytest.raises(QueueFullError) as exc_info:
        queue.enqueue(_req("a_s3", principal_id="p_a", session_id="s3"))
    assert exc_info.value.details == {"principal_id": "p_a", "max_queued_jobs_per_principal": "2"}

    queue.enqueue(_req("b_s1", principal_id="p_b", session_id="s_b"))
    queue.enqueue(
        _req("a_bypass", principal_id="p_a", session_id="s3", bypass_principal_quota=True)
    )
    expected_order = ["a_s1", "b_s1", "a_s2", "a_bypass"]
    assert [item.job_id for item in iter(queue.dequeue, None)] == expected_order


async def test_spooler_preemption_eviction_queue_full_and_log_binding(tmp_path: Path) -> None:
    """Spooler preempts MAP_REDUCE with INTERACTIVE, evicts cancelled jobs, and binds logs."""
    manager = configure_observability(
        LoggingConfig(log_dir=tmp_path, enable_file_logs=True, json_console=True)
    )
    try:
        now = 10.0
        spooler = InferenceSpooler(max_queued_jobs=MAX_QUEUE_TWO, clock=lambda: now)
        assert isinstance(spooler, InferenceSpoolerPort)

        active_started, release_active = asyncio.Event(), asyncio.Event()
        execution_order: list[str] = []

        async def hold(req: LeaseRequest, *, fail: bool = False) -> None:
            async with spooler.lease(req):
                execution_order.append(req.job_id)
                get_component_logger("test").info("lease_body_ran")
                if req.job_id == "job_mr_active":
                    active_started.set()
                    await release_active.wait()
                if fail:
                    msg = "inference crash"
                    raise RuntimeError(msg)

        with bind_log_context(request_id="req_trace"):
            t_active = asyncio.create_task(
                hold(_req("job_mr_active", priority=JobPriority.MAP_REDUCE))
            )
            await active_started.wait()
            t_mr = asyncio.create_task(
                hold(_req("job_mr_wait", session_id="s_mr", priority=JobPriority.MAP_REDUCE))
            )
            t_int = asyncio.create_task(
                hold(_req("job_int_wait", session_id="s_int", priority=JobPriority.INTERACTIVE))
            )
            await asyncio.sleep(0)
            assert spooler.snapshot().queued_count == MAX_QUEUE_TWO
            with pytest.raises(QueueFullError):
                async with spooler.lease(_req("job_rejected")):
                    pass
            t_mr.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t_mr
            t_fail = asyncio.create_task(
                hold(
                    _req("job_bg", session_id="s_bg", priority=JobPriority.MAP_REDUCE),
                    fail=True,
                )
            )
            await asyncio.sleep(0)
            now = 10.25
            release_active.set()
            await asyncio.gather(t_active, t_int)
            with pytest.raises(RuntimeError, match="inference crash"):
                await t_fail

        snap = spooler.snapshot()
        assert execution_order == ["job_mr_active", "job_int_wait", "job_bg"]
        assert (snap.active_job_id, snap.queued_count) == (None, 0)
        assert (snap.total_completed, snap.total_rejected, snap.total_evicted) == (TWO_TURNS, 1, 1)
        assert snap.avg_wait_ms == pytest.approx((EXPECTED_WAIT_MS * 2) / THREE_TURNS)
        entries = [
            json.loads(line)
            for line in (tmp_path / "lai.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        body_logs = [entry for entry in entries if entry.get("event") == "lease_body_ran"]
        assert body_logs[1]["job_id"] == "job_int_wait"
        assert body_logs[1]["request_id"] == "req_trace"
    finally:
        manager.close()


async def test_spooler_max_inflight_one() -> None:
    """Spooler strictly enforces max_inflight == 1 under concurrent mixed-priority load."""
    spooler = InferenceSpooler(max_queued_jobs=32)
    inflight = 0
    max_inflight = 0

    async def worker(idx: int) -> None:
        nonlocal inflight, max_inflight
        req = _req(
            f"job_{idx}",
            principal_id=f"p_{idx % THREE_TURNS}",
            session_id=f"s_{idx % EXPECTED_FIVE}",
            priority=JobPriority.INTERACTIVE if idx % 2 == 0 else JobPriority.MAP_REDUCE,
        )
        async with spooler.lease(req):
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            assert spooler.snapshot().active_job_id == req.job_id
            await asyncio.sleep(0.002)
            inflight -= 1

    await asyncio.gather(*(worker(i) for i in range(TWENTY_JOBS)))
    snap = spooler.snapshot()
    assert max_inflight == 1
    assert snap.total_completed == TWENTY_JOBS
    assert snap.active_job_id is None


async def test_stream_with_sse_keepalive_and_disconnect_cancellation() -> None:
    """SSE keepalive emits heartbeats on idle intervals and cancels producer on disconnect."""
    producer_started = asyncio.Event()

    async def slow_events() -> AsyncIterator[str]:
        producer_started.set()
        yield "ev_1"
        await asyncio.sleep(0.065)
        yield "ev_2"

    collected = [
        item
        async for item in stream_with_sse_keepalive(
            slow_events(),
            keepalive_factory=lambda: "ka",
            interval_seconds=0.02,
        )
    ]
    assert producer_started.is_set()
    assert (collected[0], collected[-1]) == ("ev_1", "ev_2")
    assert collected.count("ka") >= MIN_KEEPALIVE_NEWLINES

    disc_event = asyncio.Event()
    producer_cancelled = asyncio.Event()

    async def infinite_events() -> AsyncIterator[str]:
        try:
            yield "first"
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            producer_cancelled.set()
            raise

    items: list[str] = []
    async for item in stream_with_sse_keepalive(
        infinite_events(),
        keepalive_factory=lambda: "ka",
        interval_seconds=10.0,
        is_disconnected=disc_event.is_set,
    ):
        items.append(item)
        disc_event.set()
    assert items == ["first"]
    assert producer_cancelled.is_set()


async def test_await_or_keepalive_json_fast_slow_and_disconnect() -> None:
    """JSON helper returns direct value before threshold or streams RFC 8259 whitespace + JSON."""

    async def fast_coro() -> dict[str, int]:
        await asyncio.sleep(0.005)
        return {"ok": 1}

    assert await await_or_keepalive_json(fast_coro(), threshold_seconds=0.5) == {"ok": 1}

    async def slow_coro() -> SpoolerSnapshot:
        await asyncio.sleep(0.075)
        return SpoolerSnapshot(total_completed=EXPECTED_FOUR)

    outcome = await await_or_keepalive_json(
        slow_coro(),
        threshold_seconds=0.015,
        interval_seconds=0.02,
    )
    chunks = [chunk async for chunk in cast("AsyncIterator[bytes]", outcome)]
    assert chunks.count(b"\n") >= MIN_KEEPALIVE_NEWLINES
    assert json.loads(b"".join(chunks).decode("utf-8"))["total_completed"] == EXPECTED_FOUR

    cancelled = asyncio.Event()

    async def hanging() -> dict[str, str]:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"never": "reached"}

    disc_event = asyncio.Event()
    asyncio.get_running_loop().call_later(0.02, disc_event.set)
    with pytest.raises(ClientDisconnectedError):
        await await_or_keepalive_json(
            hanging(),
            threshold_seconds=5.0,
            interval_seconds=1.0,
            is_disconnected=disc_event.is_set,
        )
    assert cancelled.is_set()
