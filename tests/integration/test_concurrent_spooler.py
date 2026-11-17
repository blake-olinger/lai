"""End-to-end concurrent HTTP spooler, preemption, session serialization, and eviction test."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain import inference as di
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import LoggingConfig
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.scheduler import SessionGate
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator
    from pathlib import Path

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from lai.api.deps import AppServices

_EXPECTED_MAP_WINDOWS = 3
_EXPECTED_QUEUED_BEFORE_DISCONNECT = 4


def _extract_job_tag(request: di.InferenceRequest) -> str:
    last_msg = request.messages[-1].content if request.messages else ""
    for tag in ("mr", "beta", "gamma", "shared_1", "shared_2", "disc"):
        if f"[job:{tag}]" in last_msg:
            return tag
    return "unknown"


class _InstrumentedInferenceProvider(InferenceProviderPort):
    """Fake Gemma 4 provider verifying max_concurrent == 1 and recording call sequence."""

    def __init__(self) -> None:
        self.active_calls = 0
        self.max_concurrent_calls = 0
        self.map_window_count = 0
        self.call_events: list[str] = []
        self.map1_entered = asyncio.Event()
        self.release_map1 = asyncio.Event()

    async def chat_complete(self, request: di.InferenceRequest) -> di.InferenceResponse:
        del request
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        self.map_window_count += 1
        label = f"map_{self.map_window_count}"
        self.call_events.append(f"start:{label}")
        try:
            if self.map_window_count == 1:
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


@dataclass(slots=True)
class _ConcurrentEnv:
    client: httpx.AsyncClient
    svc: AppServices
    gate: SessionGate
    inf: _InstrumentedInferenceProvider
    disc_keepalive_seen: asyncio.Event
    keys: dict[str, str]
    project_id: str
    file_id: str
    sessions: dict[str, str]


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5.0):
        for _ in range(1000):
            if predicate():
                return
            await asyncio.sleep(0.005)
    msg = "Timed out waiting for concurrent test condition."
    raise AssertionError(msg)


@pytest.fixture
async def conc_env(tmp_path: Path) -> AsyncGenerator[_ConcurrentEnv]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
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
    inf = _InstrumentedInferenceProvider()
    svc = await build_app_services(settings)
    emb = DeterministicHashEmbedder()
    vec = LanceDbVectorStore(data_dir / "lancedb", dimensions=emb.dimensions)
    svc.embedder, svc.vector_store, svc.inference_provider = emb, vec, inf
    svc.ingestion_pipeline = LocalIngestionPipeline(
        blob_store=svc.blob_store, file_repo=svc.file_repo, embedder=emb, vector_store=vec
    )
    svc.orchestrator = TurnOrchestrator(
        settings,
        svc.session_repo,
        svc.file_repo,
        svc.blob_store,
        inf,
        svc.spooler,
        svc.session_gate,
        svc.guardrail_pipeline,
    )
    keys: dict[str, str] = {}
    for name in ("alpha", "beta", "gamma"):
        created = generate_api_key(f"key-{name}", (ApiScope.ADMIN,))
        await svc.api_key_repo.create(created.record)
        keys[name] = created.plaintext_key

    disc_keepalive_seen = asyncio.Event()
    wrapped_app = _wrap_keepalive_observer(create_app(settings, services=svc), disc_keepalive_seen)
    try:
        transport = httpx.ASGITransport(app=cast("Any", wrapped_app))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            alpha_hdr = {"Authorization": f"Bearer {keys['alpha']}"}
            pid = (
                await client.post("/v1/projects", headers=alpha_hdr, json={"name": "conc-proj"})
            ).json()["id"]
            log_text = "\n".join(
                f"2026-11-17T10:00:{i % 60:02d}Z ERROR worker_{i} timeout" for i in range(110)
            )
            f_res = await client.post(
                "/v1/files?wait=true",
                headers=alpha_hdr,
                data={"project_id": pid},
                files={"file": ("cluster.log", log_text.encode("utf-8"), "text/plain")},
            )
            sessions: dict[str, str] = {}
            for sname, kname in (
                ("mr", "alpha"),
                ("shared", "alpha"),
                ("beta", "beta"),
                ("gamma", "gamma"),
                ("disc", "beta"),
            ):
                res = await client.post(
                    "/v1/sessions",
                    headers={"Authorization": f"Bearer {keys[kname]}"},
                    json={"project_id": pid, "title": sname},
                )
                sessions[sname] = res.json()["id"]
            assert isinstance(svc.session_gate, SessionGate)
            yield _ConcurrentEnv(
                client=client,
                svc=svc,
                gate=svc.session_gate,
                inf=inf,
                disc_keepalive_seen=disc_keepalive_seen,
                keys=keys,
                project_id=pid,
                file_id=f_res.json()["id"],
                sessions=sessions,
            )
    finally:
        await close_app_services(svc)


async def _post_oai(
    env: _ConcurrentEnv, key_name: str, session_key: str, tag: str, *, observe: bool = False
) -> httpx.Response:
    hdrs = {"Authorization": f"Bearer {env.keys[key_name]}"}
    if observe:
        hdrs["X-Test-Observe-Keepalive"] = "1"
    return await env.client.post(
        "/v1/chat/completions",
        headers=hdrs,
        json={
            "model": "gemma4-26b-a4b",
            "project_id": env.project_id,
            "session_id": env.sessions[session_key],
            "messages": [{"role": "user", "content": f"Chat [job:{tag}]"}],
            "stream": True,
        },
    )


async def _post_ses(
    env: _ConcurrentEnv, key_name: str, session_key: str, content: str, **extra: object
) -> httpx.Response:
    return await env.client.post(
        f"/v1/sessions/{env.sessions[session_key]}/messages",
        headers={"Authorization": f"Bearer {env.keys[key_name]}"},
        json={"content": content, **extra},
    )


async def test_six_concurrent_requests_interleaving_serialization_and_eviction(
    conc_env: _ConcurrentEnv,
) -> None:
    """Exercise 6 concurrent requests across 3 API keys proving single-slot interleaving."""
    env = conc_env

    # 1. Start MAP_REDUCE synthesis job (Key alpha, ses_mr) and pause inside Window 1
    t_mr = asyncio.create_task(
        _post_ses(
            env,
            "alpha",
            "mr",
            "Summarize all errors across the entire log [job:mr]",
            attached_file_ids=[env.file_id],
            context_mode="map_reduce",
            stream=True,
        )
    )
    await asyncio.wait_for(env.inf.map1_entered.wait(), timeout=5.0)

    # 2-4 & 6. Launch 3 interactive chats across 3 keys + 1 cancelling SSE client
    t_beta = asyncio.create_task(_post_oai(env, "beta", "beta", "beta"))
    t_gamma = asyncio.create_task(
        _post_ses(env, "gamma", "gamma", "Quick chat [job:gamma]", stream=False)
    )
    t_shared_1 = asyncio.create_task(
        _post_ses(env, "alpha", "shared", "Shared turn 1 [job:shared_1]", stream=False)
    )
    t_disc = asyncio.create_task(_post_oai(env, "beta", "disc", "disc", observe=True))

    await _wait_until(
        lambda: (
            env.gate.is_locked(env.sessions["shared"])
            and env.svc.spooler.snapshot().queued_count >= _EXPECTED_QUEUED_BEFORE_DISCONNECT
        ),
    )

    # 5. Overlapping 6th request on same session_id (ses_shared) waits on SessionGate
    t_shared_2 = asyncio.create_task(
        _post_ses(env, "alpha", "shared", "Shared turn 2 [job:shared_2]", stream=False)
    )
    await asyncio.wait_for(env.disc_keepalive_seen.wait(), timeout=5.0)
    await asyncio.sleep(0.05)

    t_disc.cancel()
    with suppress(asyncio.CancelledError):
        await t_disc
    await _wait_until(lambda: env.svc.spooler.snapshot().total_evicted >= 1)

    env.inf.release_map1.set()
    responses = await asyncio.wait_for(
        asyncio.gather(t_mr, t_beta, t_gamma, t_shared_1, t_shared_2), timeout=10.0
    )
    assert all(res.status_code == HTTPStatus.OK for res in responses)
    assert env.inf.max_concurrent_calls == 1
    assert env.inf.map_window_count >= _EXPECTED_MAP_WINDOWS
    assert ": keepalive\n\n" in responses[1].text
    assert "data: [DONE]\n\n" in responses[1].text
    assert env.svc.spooler.snapshot().total_evicted >= 1
    assert not any("disc" in ev for ev in env.inf.call_events)

    events = env.inf.call_events
    assert events[0] == "start:map_1"
    map1_end, map2_start = events.index("end:map_1"), events.index("start:map_2")
    for tag in ("beta", "gamma", "shared_1"):
        assert map1_end < events.index(f"start:interactive:{tag}") < map2_start

    assert events.index("end:interactive:shared_1") < events.index("start:interactive:shared_2")
    shared_msgs = await env.svc.session_repo.list_messages(env.sessions["shared"])
    assert [m.content for m in shared_msgs] == [
        "Shared turn 1 [job:shared_1]",
        "reply:shared_1",
        "Shared turn 2 [job:shared_2]",
        "reply:shared_2",
    ]
