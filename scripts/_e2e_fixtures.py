"""Deterministic code and log fixtures used by scripts/live_e2e_smoke.py."""

from __future__ import annotations

from typing import Any


def build_oai_payload(msg: str, *, stream: bool = False, max_tokens: int = 120) -> dict[str, Any]:
    """Build an OpenAI /v1/chat/completions request body targeting the Gemma 4 profile."""
    return {
        "model": "gemma4-26b-a4b",
        "messages": [{"role": "user", "content": msg}],
        "stream": stream,
        "enable_thinking": False,
        "max_tokens": max_tokens,
    }


def build_rag_fixtures() -> tuple[bytes, bytes]:
    """Return a tiny Python module and matching 1,490-line service log with a traceback."""
    code = (
        b"def avg_lat(ms: float, active_workers: int) -> float:\n    return ms / active_workers\n"
    )
    lines = [
        f"2026-11-17T07:{(i // 60):02d}:{(i % 60):02d}Z INFO [pool] tick={i}"
        for i in range(1, 1490)
    ]
    lines[740:745] = [
        "2026-11-17T07:12:20Z ERROR [pool] Unhandled exception flushing stats",
        "Traceback (most recent call last):",
        '  File "worker_pool.py", line 2, in avg_lat',
        "    return ms / active_workers",
        "ZeroDivisionError: float division by zero",
    ]
    return code, ("\n".join(lines) + "\n").encode()


def build_oversized_log() -> bytes:
    """Return a >12k-token cluster log with two distinct critical incidents for map-reduce."""
    lines = [
        f"2026-11-17T08:{(i // 60):02d}:{(i % 60):02d}Z INFO [mesh] seq={i} "
        f"shard={(i % 8)} req=req_{i:05d}_trace"
        for i in range(1, 780)
    ]
    lines[20] = "2026-11-17T08:00:20Z ERROR [cache] INCIDENT ALPHA: REDIS_CLUSTER_SPLIT_BRAIN"
    lines[760] = "2026-11-17T08:12:40Z CRITICAL [nvme] INCIDENT OMEGA: NVME_NAMESPACE_READ_ONLY"
    return ("\n".join(lines) + "\n").encode()
