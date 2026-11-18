"""Targeted unit tests hardening boundary conditions and mutation coverage."""

from __future__ import annotations

import logging

import pytest

from lai.domain.errors import QueueFullError, RateLimitExceededError
from lai.domain.files import FileChunk, ScoredChunk
from lai.domain.observability import AlertCategory
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.ingestion.detector import estimate_tokens
from lai.ingestion.rrf import reciprocal_rank_fusion
from lai.observability import AlertingLogHandler, AlertManager
from lai.orchestrator.windower import slice_text_into_windows
from lai.scheduler.fair_queue import FairPriorityQueue
from lai.scheduler.session_gate import SessionGate
from lai.security.keys import generate_api_key, hash_api_key, verify_key_hash
from lai.security.ratelimit import TokenBucketRateLimiter

_EXPECTED_RRF_K1_SCORE = 0.5
_EXPECTED_RETRY_AFTER = 0.25
_RATE_LIMIT_120_RPM = 120
_NON_STR_JOB_ID = 123


def _lease(
    job_id: str,
    *,
    principal_id: str = "usr_1",
    session_id: str = "ses_1",
    priority: JobPriority = JobPriority.INTERACTIVE,
) -> LeaseRequest:
    return LeaseRequest(
        job_id=job_id,
        session_id=session_id,
        project_id="prj_1",
        principal_id=principal_id,
        priority=priority,
    )


def _hit(chunk_id: str, start_line: int = 1) -> ScoredChunk:
    return ScoredChunk(
        chunk=FileChunk(
            id=chunk_id,
            file_id="fil_1",
            project_id="prj_1",
            chunk_index=0,
            start_line=start_line,
            end_line=start_line,
            token_count=4,
            text=f"line {chunk_id}",
        ),
        score=1.0,
        source="fts",
    )


def test_keys_empty_inputs_always_return_strict_bool_false() -> None:
    """verify_key_hash returns strict bool False for empty plaintext or empty hash."""
    created = generate_api_key("k", ())
    assert verify_key_hash("", hash_api_key("")) is False
    assert verify_key_hash("", created.record.key_hash) is False
    assert verify_key_hash(created.plaintext_key, "") is False
    assert verify_key_hash("", "") is False
    assert verify_key_hash(created.plaintext_key, created.record.key_hash) is True


async def test_ratelimit_non_unit_rate_exact_bounds_and_eviction() -> None:
    """Rate limiter enforces max_buckets=1, token=0 validation, and 120rpm refill/retry math."""
    with pytest.raises(ValueError, match="max_buckets"):
        TokenBucketRateLimiter(max_buckets=0)

    now = 0.0
    limiter = TokenBucketRateLimiter(
        requests_per_minute=_RATE_LIMIT_120_RPM,
        burst_capacity=2,
        clock=lambda: now,
        max_buckets=1,
    )
    with pytest.raises(ValueError, match="Requested token cost"):
        await limiter.acquire("u1", tokens=0.0)
    with pytest.raises(ValueError, match="Requested token cost"):
        await limiter.acquire("u1", tokens=-1.0)
    with pytest.raises(ValueError, match="Requested token cost"):
        await limiter.acquire("u1", tokens=2.1)

    # Consume 1.5 tokens (0.5 left). Requesting 1.0 leaves deficit=0.5 at 2.0/s -> 0.25s retry.
    await limiter.acquire("u1", tokens=1.5)
    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.acquire("u1", tokens=1.0)
    assert exc_info.value.retry_after_seconds == pytest.approx(_EXPECTED_RETRY_AFTER)

    # Advance 0.5s at 2.0 tokens/sec -> +1.0 token refilled (total 1.5 available)
    now = 0.5
    await limiter.acquire("u1", tokens=1.5)
    with pytest.raises(RateLimitExceededError):
        await limiter.acquire("u1", tokens=0.1)

    # With max_buckets=1, acquiring for u2 evicts u1 so u1 starts fresh at full burst capacity (2.0)
    await limiter.acquire("u2", tokens=2.0)
    await limiter.acquire("u1", tokens=2.0)


async def test_fair_queue_and_session_gate_exact_boundaries() -> None:
    """FairPriorityQueue and SessionGate enforce min capacity, is_empty, __contains__, and lock."""
    queue = FairPriorityQueue(max_queued_jobs=1)
    assert queue.is_empty() is True
    assert ("j1" in queue) is False
    assert (_NON_STR_JOB_ID in queue) is False

    req = _lease("j1", priority=JobPriority.MAP_REDUCE)
    queue.enqueue(req)
    assert queue.is_empty() is False
    assert ("j1" in queue) is True
    with pytest.raises(QueueFullError):
        queue.enqueue(_lease("j2"))

    popped = queue.dequeue()
    assert popped is not None
    assert popped.job_id == "j1"
    assert queue.is_empty() is True
    assert ("j1" in queue) is False
    assert queue.dequeue() is None

    gate = SessionGate()
    assert gate.is_locked("missing_session") is False
    async with gate.acquire("ses_x"):
        assert gate.is_locked("ses_x") is True
    assert gate.is_locked("ses_x") is False


def test_rrf_k_one_limit_one_and_negative_limit() -> None:
    """RRF handles k=1, limit=1, and negative limit boundaries cleanly."""
    hits = [_hit("c1", 1), _hit("c2", 2)]
    fused_one = reciprocal_rank_fusion(hits, [], k=1, limit=1)
    assert len(fused_one) == 1
    assert fused_one[0].chunk.id == "c1"
    assert fused_one[0].score == pytest.approx(_EXPECTED_RRF_K1_SCORE)
    assert reciprocal_rank_fusion(hits, [], k=1, limit=-1) == []


def test_windower_exact_spans_clamped_overlap_and_minimum_budget() -> None:
    """Windower respects window_tokens=1, overlap clamping, and exact token-fit boundaries."""
    min_windows = slice_text_into_windows("f1", "a.log", "hello", window_tokens=1, overlap_tokens=0)
    assert [(w.start_line, w.end_line) for w in min_windows] == [(1, 1)]

    long_single = " ".join(f"w{i}" for i in range(40))
    single_ov = slice_text_into_windows(
        "f1", "a.log", long_single, window_tokens=15, overlap_tokens=10
    )
    assert [(w.start_line, w.end_line) for w in single_ov] == [(1, 1)]

    lines = [f"line {i} alpha beta gamma" for i in range(1, 13)]
    exact_ov = slice_text_into_windows(
        "f1", "app.log", "\n".join(lines), window_tokens=30, overlap_tokens=10
    )
    assert [(w.start_line, w.end_line, w.token_count) for w in exact_ov] == [
        (1, 3, 28),
        (3, 5, 28),
        (5, 7, 28),
        (7, 9, 28),
        (9, 11, 28),
        (11, 12, 21),
    ]

    tiny_lines = ["x" for _ in range(10)]
    clamped = slice_text_into_windows(
        "f1", "a.log", "\n".join(tiny_lines), window_tokens=14, overlap_tokens=14
    )
    assert [(w.start_line, w.end_line, w.token_count) for w in clamped] == [
        (1, 3, 12),
        (2, 4, 12),
        (3, 5, 12),
        (4, 6, 12),
        (5, 7, 12),
        (6, 8, 12),
        (7, 9, 12),
        (8, 10, 12),
    ]

    # Exact fit: 12-token budget holds 2 tiny lines per window
    exact_cand = slice_text_into_windows(
        "f1",
        "a.log",
        "\n".join(tiny_lines[:6]),
        window_tokens=estimate_tokens("[a.log:L1-L3]\nx\nx\nx"),
        overlap_tokens=0,
    )
    assert [(w.start_line, w.end_line) for w in exact_cand] == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]

    # Exact fit when window_tokens equals actual rendered 1-line window tokens
    sample = ["hello world foo bar", "second line here now"]
    fit_tokens = estimate_tokens("[worker.log:L1-L1]\nhello world foo bar")
    fit_ws = slice_text_into_windows(
        "f1", "worker.log", "\n".join(sample), window_tokens=fit_tokens, overlap_tokens=0
    )
    assert [(w.start_line, w.end_line, w.token_count) for w in fit_ws] == [
        (1, 1, fit_tokens),
        (2, 2, estimate_tokens("[worker.log:L2-L2]\nsecond line here now")),
    ]

    # Exercise overlap == window_tokens clamping (25 // 3 = 8, not 25) on multi-line window
    ws25 = slice_text_into_windows(
        "f1", "a.log", "\n".join(["x"] * 12), window_tokens=25, overlap_tokens=25
    )
    assert [(w.start_line, w.end_line) for w in ws25] == [(1, 8), (5, 12)]

    # Exercise exact integer floor division 90 // 4 == 22 (body_budget=68, 4 * 17 == 68)
    line16 = " ".join(["a"] * 16)
    ws90 = slice_text_into_windows(
        "f1", "a.log", "\n".join([line16] * 6), window_tokens=90, overlap_tokens=0
    )
    assert [(w.start_line, w.end_line, w.token_count) for w in ws90] == [
        (1, 4, 74),
        (5, 6, 40),
    ]


def test_alert_handler_detects_timeout_from_exception_type_without_keyword() -> None:
    """AlertingLogHandler classifies TimeoutError from exception_type even if msg lacks it."""
    manager = AlertManager(cooldown_seconds=0.0)
    handler = AlertingLogHandler(manager, alert_on_timeouts=True)
    exc = TimeoutError("deadline")
    handler.emit(
        logging.LogRecord(
            "lai.test", logging.ERROR, __file__, 1, "upstream failed", (), (type(exc), exc, None)
        )
    )
    alerts = manager.recent_alerts()
    assert len(alerts) == 1
    assert alerts[0].category is AlertCategory.TIMEOUT
    assert alerts[0].exception_type == "TimeoutError"
