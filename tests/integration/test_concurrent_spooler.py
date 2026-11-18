"""Integration tests for concurrent spooler preemption, session serialization, and eviction."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from lai.config.settings import LaiSettings
from lai.domain import inference as di
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import LoggingConfig
from lai.ports.inference import InferenceProviderPort
from lai.scheduler import SessionGate
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    import httpx
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_EXPECTED_MAP_WINDOWS = 3
_EXPECTED_INTERACTIVE_QUEUED = 2


def _extract_job_tag(request: di.InferenceRequest) -> str:
    last_msg = request.messages[-1].content if request.messages else ""
    for tag in ("mr", "beta", "gamma", "shared_1", "shared_2", "blocker", "disc"):
        if f"[job:{tag}]" in last_msg:
            return tag
    return "unknown"


class _InstrumentedInferenceProvider(InferenceProviderPort):
    """Fake Gemma 4 provider verifying max_concurrent == 1 and optional pause gates."""

    def __init__(self, *, hold_map1: bool = False, hold_tag: str | None = None) -> None:
        self._hold_map1 = hold_map1
        self._hold_tag = hold_tag
        self.active_calls = 0
        self.max_concurrent_calls = 0
        self.map_window_count = 0
        self.call_events: list[str] = []
        self.map1_entered = asyncio.Event()
        self.release_map1 = asyncio.Event()
        self.tag_entered = asyncio.Event()
        self.release_tag = asyncio.Event()

    async def chat_complete(self, request: di.InferenceRequest) -> di.InferenceResponse:
        del request
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        self.map_window_count += 1
        label = f"map_{self.map_window_count}"
        self.call_events.append(f"start:{label}")
        try:
            if self._hold_map1 and self.map_window_count == 1:
                self.map1_entered.set()
                await self.release_map1.wait()
            else:
                await asyncio.sleep(0.01)
            return di.InferenceResponse(
                content=f"window finding {label}",
                usage=TokenUsage(prompt_tokens=15, completion_tokens=5, total_tokens=20),
            )
        finally:
            self.call_events.append(f"end:{label}")
            self.active_calls -= 1

    async def chat_stream(self, request: di.InferenceRequest) -> AsyncGenerator[di.InferenceChunk]:
        tag = _extract_job_tag(request)
        label = f"reduce:{tag}" if tag == "mr" else f"interactive:{tag}"
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        self.call_events.append(f"start:{label}")
        try:
            if self._hold_tag is not None and tag == self._hold_tag:
                self.tag_entered.set()
                await self.release_tag.wait()
            else:
                await asyncio.sleep(0.01)
            yield di.InferenceChunk(content_delta=f"reply:{tag}")
            yield di.InferenceChunk(
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            )
        finally:
            self.call_events.append(f"end:{label}")
            self.active_calls -= 1

    async def health(self) -> di.BackendHealth:
        return di.BackendHealth(healthy=True, status="ok")


def _wrap_keepalive_observer(app: ASGIApp, disc_keepalive_seen: asyncio.Event) -> ASGIApp:
    """Observe SSE : keepalive frames emitted to the cancelling queued client."""

    async def _asgi(scope: Scope, receive: Receive, send: Send) -> None:
        hdrs = cast("list[tuple[bytes, bytes]]", scope.get("headers", []))
        if scope["type"] != "http" or not any(
            k.lower() == b"x-test-observe-keepalive" for k, _ in hdrs
        ):
            await app(scope, receive, send)
            return

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.body" and b": keepalive" in cast(
                "bytes", message.get("body", b"")
            ):
                disc_keepalive_seen.set()
            await send(message)

        await app(scope, receive, _send)

    return _asgi


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5.0):
        for _ in range(1000):
            if predicate():
                return
            await asyncio.sleep(0.005)
    msg = "Timed out waiting for concurrent test condition."
    raise AssertionError(msg)


def _concurrent_settings(tmp_path: Path) -> LaiSettings:
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        direct_file_budget_tokens=256,
        map_window_tokens=256,
        map_window_overlap_tokens=16,
        keepalive_interval_seconds=0.03,
        json_keepalive_threshold_seconds=0.03,
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )


async def _post_oai_stream(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    project_id: str,
    session_id: str,
    tag: str,
) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "gemma4-26b-a4b",
            "project_id": project_id,
            "session_id": session_id,
            "messages": [{"role": "user", "content": f"Chat [job:{tag}]"}],
            "stream": True,
        },
    )


async def test_interactive_requests_preempt_between_map_reduce_windows(tmp_path: Path) -> None:
    """Interactive requests across keys preempt between MAP_REDUCE windows with single-slot GPU."""
    inf = _InstrumentedInferenceProvider(hold_map1=True)
    spec = IntegrationSpec(inference_provider=inf, use_hash_embedder=True)
    async with open_services_harness(_concurrent_settings(tmp_path), spec) as harness:
        h_alpha = await mint_auth_headers(harness.svc, "key-alpha", (ApiScope.ADMIN,))
        h_beta = await mint_auth_headers(harness.svc, "key-beta", (ApiScope.ADMIN,))
        h_gamma = await mint_auth_headers(harness.svc, "key-gamma", (ApiScope.ADMIN,))
        pid = await create_project(harness.client, h_alpha, "conc-proj")
        log_text = "\n".join(
            f"2026-11-17T10:00:{i % 60:02d}Z ERROR worker_{i} timeout" for i in range(110)
        )
        file_id = str(
            (
                await harness.client.post(
                    "/v1/files?wait=true",
                    headers=h_alpha,
                    data={"project_id": pid},
                    files={"file": ("cluster.log", log_text.encode("utf-8"), "text/plain")},
                )
            ).json()["id"]
        )
        ses_mr = await create_session(harness.client, h_alpha, pid, title="mr")
        ses_beta = await create_session(harness.client, h_beta, pid, title="beta")
        ses_gamma = await create_session(harness.client, h_gamma, pid, title="gamma")

        t_mr = asyncio.create_task(
            harness.client.post(
                f"/v1/sessions/{ses_mr}/messages",
                headers=h_alpha,
                json={
                    "content": "Summarize all errors across the entire log [job:mr]",
                    "attached_file_ids": [file_id],
                    "context_mode": "map_reduce",
                    "stream": True,
                },
            )
        )
        await asyncio.wait_for(inf.map1_entered.wait(), timeout=5.0)

        t_beta = asyncio.create_task(
            _post_oai_stream(harness.client, h_beta, pid, ses_beta, "beta")
        )
        t_gamma = asyncio.create_task(
            harness.client.post(
                f"/v1/sessions/{ses_gamma}/messages",
                headers=h_gamma,
                json={"content": "Quick chat [job:gamma]", "stream": False},
            )
        )
        await _wait_until(
            lambda: harness.svc.spooler.snapshot().queued_count >= _EXPECTED_INTERACTIVE_QUEUED
        )

        inf.release_map1.set()
        responses = await asyncio.wait_for(asyncio.gather(t_mr, t_beta, t_gamma), timeout=10.0)

    assert all(res.status_code == HTTPStatus.OK for res in responses)
    assert inf.max_concurrent_calls == 1
    assert inf.map_window_count >= _EXPECTED_MAP_WINDOWS
    assert ": keepalive\n\n" in responses[1].text
    assert "data: [DONE]\n\n" in responses[1].text
    assert inf.call_events[0] == "start:map_1"
    map1_end, map2_start = inf.call_events.index("end:map_1"), inf.call_events.index("start:map_2")
    assert map1_end < inf.call_events.index("start:interactive:beta") < map2_start
    assert map1_end < inf.call_events.index("start:interactive:gamma") < map2_start


async def test_same_session_concurrent_turns_serialize_strictly_via_session_gate(
    tmp_path: Path,
) -> None:
    """Overlapping turns on the same session_id serialize strictly in arrival order."""
    inf = _InstrumentedInferenceProvider(hold_tag="shared_1")
    spec = IntegrationSpec(inference_provider=inf)
    async with open_services_harness(_concurrent_settings(tmp_path), spec) as harness:
        assert isinstance(harness.svc.session_gate, SessionGate)
        gate = harness.svc.session_gate
        headers = await mint_auth_headers(harness.svc, "key-shared", (ApiScope.ADMIN,))
        pid = await create_project(harness.client, headers, "gate-proj")
        ses_shared = await create_session(harness.client, headers, pid, title="shared")

        t_first = asyncio.create_task(
            harness.client.post(
                f"/v1/sessions/{ses_shared}/messages",
                headers=headers,
                json={"content": "Shared turn 1 [job:shared_1]", "stream": False},
            )
        )
        await asyncio.wait_for(inf.tag_entered.wait(), timeout=5.0)
        await _wait_until(lambda: gate.is_locked(ses_shared))

        t_second = asyncio.create_task(
            harness.client.post(
                f"/v1/sessions/{ses_shared}/messages",
                headers=headers,
                json={"content": "Shared turn 2 [job:shared_2]", "stream": False},
            )
        )
        await asyncio.sleep(0.03)
        assert "start:interactive:shared_2" not in inf.call_events

        inf.release_tag.set()
        r1, r2 = await asyncio.wait_for(asyncio.gather(t_first, t_second), timeout=10.0)
        shared_msgs = await harness.svc.session_repo.list_messages(ses_shared)

    assert (r1.status_code, r2.status_code) == (HTTPStatus.OK, HTTPStatus.OK)
    assert inf.max_concurrent_calls == 1
    assert inf.call_events.index("end:interactive:shared_1") < inf.call_events.index(
        "start:interactive:shared_2"
    )
    assert [m.content for m in shared_msgs] == [
        "Shared turn 1 [job:shared_1]",
        "reply:shared_1",
        "Shared turn 2 [job:shared_2]",
        "reply:shared_2",
    ]


async def test_disconnected_queued_sse_client_is_evicted_before_inference(
    tmp_path: Path,
) -> None:
    """Cancelling a queued SSE client emits : keepalive, evicts spooler job, and skips GPU."""
    inf = _InstrumentedInferenceProvider(hold_tag="blocker")
    disc_keepalive_seen = asyncio.Event()
    spec = IntegrationSpec(
        inference_provider=inf,
        asgi_wrapper=lambda app: _wrap_keepalive_observer(app, disc_keepalive_seen),
    )
    async with open_services_harness(_concurrent_settings(tmp_path), spec) as harness:
        h_alpha = await mint_auth_headers(harness.svc, "key-alpha", (ApiScope.ADMIN,))
        h_beta = await mint_auth_headers(harness.svc, "key-beta", (ApiScope.ADMIN,))
        pid = await create_project(harness.client, h_alpha, "evict-proj")
        ses_blocker = await create_session(harness.client, h_alpha, pid, title="blocker")
        ses_disc = await create_session(harness.client, h_beta, pid, title="disc")

        t_blocker = asyncio.create_task(
            harness.client.post(
                f"/v1/sessions/{ses_blocker}/messages",
                headers=h_alpha,
                json={"content": "Block GPU [job:blocker]", "stream": False},
            )
        )
        await asyncio.wait_for(inf.tag_entered.wait(), timeout=5.0)

        t_disc = asyncio.create_task(
            _post_oai_stream(
                harness.client,
                {**h_beta, "X-Test-Observe-Keepalive": "1"},
                pid,
                ses_disc,
                "disc",
            )
        )
        await asyncio.wait_for(disc_keepalive_seen.wait(), timeout=5.0)
        await _wait_until(lambda: harness.svc.spooler.snapshot().queued_count >= 1)

        t_disc.cancel()
        with suppress(asyncio.CancelledError):
            await t_disc
        await _wait_until(lambda: harness.svc.spooler.snapshot().total_evicted >= 1)

        inf.release_tag.set()
        r_blocker = await asyncio.wait_for(t_blocker, timeout=10.0)
        evicted_count = harness.svc.spooler.snapshot().total_evicted

    assert r_blocker.status_code == HTTPStatus.OK
    assert evicted_count >= 1
    assert not any("disc" in ev for ev in inf.call_events)
