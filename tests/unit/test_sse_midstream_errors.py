"""Tests for SSE mid-stream error framing, priming-window errors, and disconnect cleanup."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

import pytest

from lai.api._streaming import (
    PrimedSseStream,
    build_guardrail_headers,
    guard_openai_sse_frames,
    guard_orchestrator_sse_frames,
    prime_sse_stream,
)
from lai.api.app import create_app
from lai.config.settings import LaiSettings
from lai.domain import chat as dc
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.errors import InferenceBackendError
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.observability import bind_request_id, configure_observability
from lai.security import generate_api_key
from tests._helpers import StubOrchestratorBase, bearer_headers, open_app_client

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from collections.abc import Sequence as Seq
    from pathlib import Path

    import httpx
    from fastapi import FastAPI
    from starlette.types import Message as AsgiMessage
    from starlette.types import Scope

_ERR_MSG = "An unexpected internal server error occurred."
_TWO = 2
_SHORT_ELAPSED = 0.3
_KA_ELAPSED = 0.45
_SES_REQ: dict[str, object] = {"content": "hi", "stream": True}
_OAI_REQ: dict[str, object] = {
    "model": "gemma4-26b-a4b",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
}


class _ScriptedOrchestrator(StubOrchestratorBase):
    """Orchestrator stub yielding scripted events before optionally waiting or raising."""

    def __init__(
        self, events: Seq[dc.OrchestratorEvent] = (), *, error: Exception | None = None
    ) -> None:
        self.events, self.error = list(events), error
        self.wait_event: asyncio.Event | None = None
        self.entered_event: asyncio.Event | None = None
        self.stream_closed = False

    async def run_turn(self, turn_input: dc.TurnInput, principal: AuthPrincipal) -> dc.TurnOutput:
        return _turn_out(turn_input.session_id, turn_input.project_id, principal.principal_id)

    async def run_turn_stream(
        self, turn_input: dc.TurnInput, principal: AuthPrincipal
    ) -> AsyncIterator[dc.OrchestratorEvent]:
        try:
            for event in self.events:
                yield event
            if self.entered_event is not None and principal.principal_id:
                self.entered_event.set()
            if self.wait_event is not None:
                await self.wait_event.wait()
            if self.error is not None:
                raise self.error
        finally:
            self.stream_closed = bool(turn_input.session_id)


def _turn_out(sid: str, pid: str, text: str, *, blocked: bool = False) -> dc.TurnOutput:
    msg = dc.Message(session_id=sid, project_id=pid, role=dc.MessageRole.ASSISTANT, content=text)
    return dc.TurnOutput(
        session_id=sid,
        message=msg,
        context_mode_used=dc.ContextMode.DIRECT,
        usage=dc.TokenUsage(),
        guardrail_blocked=blocked,
    )


async def _never() -> bool:
    return False


async def _collect(
    fn: Callable[[AsyncIterator[str]], AsyncIterator[str]], first: str, exc: BaseException
) -> list[str]:
    async def _gen() -> AsyncIterator[str]:
        yield first
        raise exc

    return [frame async for frame in fn(_gen())]


async def _egen(  # noqa: PLR0913
    items: Seq[dc.OrchestratorEvent] = (),
    *,
    d: float = 0.0,
    w: asyncio.Event | None = None,
    e: Exception | None = None,
    c: set[str] | None = None,
    t: str = "",
) -> AsyncIterator[dc.OrchestratorEvent]:
    try:
        for item in items:
            yield item
        if d > 0:
            await asyncio.sleep(d)
        if w is not None:
            await w.wait()
        if e is not None:
            raise e
    finally:
        if c is not None:
            c.add(t)


async def _prime(
    events: AsyncIterator[dc.OrchestratorEvent],
    *,
    ka: float = 0.5,
    disc: Callable[[], Awaitable[bool]] = _never,
    budget: float = 0.25,
) -> PrimedSseStream:
    return await prime_sse_stream(
        events, keepalive_interval_seconds=ka, is_disconnected=disc, peek_budget_seconds=budget
    )


@asynccontextmanager
async def _open_sse_session(
    tmp_path: Path,
    orch: _ScriptedOrchestrator,
) -> AsyncGenerator[tuple[FastAPI, httpx.AsyncClient, dict[str, str], str, str]]:
    l_cfg = LoggingConfig(log_dir=tmp_path / "l", enable_file_logs=False, alert_cooldown_seconds=0)
    cfg = LaiSettings(data_dir=tmp_path / "d", keepalive_interval_seconds=0.5, logging=l_cfg)
    app = create_app(cfg, orchestrator=orch)
    async with open_app_client(app) as client:
        key = generate_api_key("sse-err-admin", (ApiScope.ADMIN,))
        await app.state.services.api_key_repo.create(key.record)
        hdrs = bearer_headers(key.plaintext_key)
        pid = (await client.post("/v1/projects", headers=hdrs, json={"name": "e"})).json()["id"]
        sid = str(
            (await client.post("/v1/sessions", headers=hdrs, json={"project_id": pid})).json()["id"]
        )
        yield app, client, hdrs, sid, key.plaintext_key


async def test_guard_sse_frame_helpers_lai_unexpected_and_cancelled(tmp_path: Path) -> None:
    """Verify LaiError framing, unexpected Exception alerts/sanitization, and CancelledError."""
    og, ag = guard_orchestrator_sse_frames, guard_openai_sse_frames
    orch_lai = await _collect(og, "data: ok\n\n", InferenceBackendError("boom"))
    assert len(orch_lai) == _TWO
    assert json.loads(orch_lai[1].split("data: ", 1)[1])["error"]["code"] == (
        "inference_backend_error"
    )
    oai_lai = await _collect(ag, ": ka\n\n", InferenceBackendError("boom"))
    assert oai_lai[1].endswith("data: [DONE]\n\n")
    assert json.loads(oai_lai[1].splitlines()[0][6:]) == {
        "error": {"message": "boom", "type": "inference_backend_error"}
    }

    log_cfg = LoggingConfig(log_dir=tmp_path, enable_file_logs=False, alert_cooldown_seconds=0.0)
    mgr = configure_observability(log_cfg)
    try:
        with bind_request_id("req_orch"):
            orch_err = await _collect(og, "data: ok\n\n", ValueError("secret sql"))
        assert "secret sql" not in "".join(orch_err)
        ed = json.loads(orch_err[1].split("data: ", 1)[1])["error"]
        assert (ed["code"], ed["request_id"], ed["details"]) == ("internal_error", "req_orch", {})
        assert ed["message"] == _ERR_MSG
        with bind_request_id("req_oai"):
            oai_err = await _collect(ag, ": ka\n\n", RuntimeError("secret token"))
        assert "secret token" not in "".join(oai_err)
        assert oai_err[1].endswith("data: [DONE]\n\n")
        assert json.loads(oai_err[1].splitlines()[0][6:]) == {
            "error": {"message": _ERR_MSG, "type": "internal_error"}
        }
        cat = AlertCategory.FATAL_EXCEPTION
        seen = {(a.request_id, a.exception_type, a.category) for a in mgr.recent_alerts()}
        assert seen == {("req_oai", "RuntimeError", cat), ("req_orch", "ValueError", cat)}
        assert all(a.message == "unhandled_sse_exception" for a in mgr.recent_alerts())
        for fn in (og, ag):
            with pytest.raises(asyncio.CancelledError):
                await _collect(fn, "data: ok\n\n", asyncio.CancelledError())
        assert len(mgr.recent_alerts()) == _TWO
    finally:
        mgr.close()


async def test_asgi_sse_midstream_lai_error_frames(tmp_path: Path) -> None:
    """Verify mid-stream LaiError emits structured SSE error frames over HTTP."""
    evs = (dc.ProgressEvent(stage="inference", message="starting"),)
    orch = _ScriptedOrchestrator(evs, error=InferenceBackendError("boom", details={"b": "ollama"}))
    async with _open_sse_session(tmp_path, orch) as (_, cli, hdrs, sid, _):
        ses_txt = (await cli.post(f"/v1/sessions/{sid}/messages", headers=hdrs, json=_SES_REQ)).text
        assert "event: progress" in ses_txt
        assert '"code":"inference_backend_error"' in ses_txt
        oai_txt = (await cli.post("/v1/chat/completions", headers=hdrs, json=_OAI_REQ)).text
        assert '"type": "inference_backend_error"' in oai_txt
        assert oai_txt.endswith("data: [DONE]\n\n")


async def test_asgi_sse_midstream_unexpected_error_sanitizes_and_alerts(tmp_path: Path) -> None:
    """Verify mid-stream RuntimeError sanitizes SSE output and emits FATAL_EXCEPTION alerts."""
    evs = (dc.ProgressEvent(stage="inference", message="starting"),)
    orch = _ScriptedOrchestrator(evs, error=RuntimeError("db secret leaked"))
    async with _open_sse_session(tmp_path, orch) as (app, cli, hdrs, sid, _):
        ses_url, oai_url = f"/v1/sessions/{sid}/messages", "/v1/chat/completions"
        for url, rid, req, marker in (
            (ses_url, "r_ses", _SES_REQ, '"code":"internal_error"'),
            (oai_url, "r_oai", _OAI_REQ, '"type": "internal_error"'),
        ):
            resp = await cli.post(url, headers={**hdrs, "X-Request-Id": rid}, json=req)
            assert resp.status_code == HTTPStatus.OK
            assert "db secret leaked" not in resp.text
            assert marker in resp.text
            assert _ERR_MSG in resp.text
            if url == oai_url:
                assert resp.text.endswith("data: [DONE]\n\n")

        alerts = app.state.services.alert_manager.recent_alerts()
        cat = AlertCategory.FATAL_EXCEPTION
        assert {(a.request_id, a.exception_type) for a in alerts if a.category is cat} == {
            ("r_ses", "RuntimeError"),
            ("r_oai", "RuntimeError"),
        }


async def test_asgi_sse_priming_lai_error_returns_http_502(tmp_path: Path) -> None:
    """Verify LaiError raised during SSE priming returns HTTP 502 error envelopes."""
    orch = _ScriptedOrchestrator((), error=InferenceBackendError("down"))
    async with _open_sse_session(tmp_path, orch) as (_, cli, hdrs, sid, _):
        for url, payload in (
            (f"/v1/sessions/{sid}/messages", _SES_REQ),
            ("/v1/chat/completions", _OAI_REQ),
        ):
            r_err = await cli.post(url, headers=hdrs, json=payload)
            assert (r_err.status_code, r_err.json()["error"]["code"]) == (
                HTTPStatus.BAD_GATEWAY,
                "inference_backend_error",
            )


async def test_asgi_sse_priming_disconnect_closes_producer(tmp_path: Path) -> None:
    """Exercise ASGI client disconnect during prime_sse_stream and verify producer cleanup."""
    slow_orch = _ScriptedOrchestrator()
    slow_orch.entered_event, slow_orch.wait_event = asyncio.Event(), asyncio.Event()
    async with _open_sse_session(tmp_path, slow_orch) as (app, _, _, sid, token):
        sent: list[AsgiMessage] = []
        sent_body = False

        async def _receive() -> AsgiMessage:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {
                    "type": "http.request",
                    "body": b'{"content":"hi","stream":true}',
                    "more_body": False,
                }
            assert slow_orch.entered_event is not None
            if not slow_orch.entered_event.is_set():
                await asyncio.sleep(1.0)
            return {"type": "http.disconnect"}

        async def _send(message: AsgiMessage) -> None:
            sent.append(message)

        hdrs = [
            (b"authorization", f"Bearer {token}".encode()),
            (b"content-type", b"application/json"),
        ]
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "path": f"/v1/sessions/{sid}/messages",
            "query_string": b"",
            "headers": hdrs,
        }
        with suppress(asyncio.CancelledError):
            await app(scope, _receive, _send)
        assert (slow_orch.stream_closed, sent[0]["status"]) == (True, HTTPStatus.OK)


async def test_prime_sse_stream_errors_disconnect_and_cleanup() -> None:
    """Verify prime_sse_stream errors, disconnect, pending cancel, and task cancel."""
    hdr_b = build_guardrail_headers(guardrail_blocked=True, guardrail_policy=None)
    assert hdr_b == {"X-LAI-Guardrail-Blocked": "true"}
    closed: set[str] = set()
    ev_p = (dc.ContentDeltaEvent(delta="first"),)
    with pytest.raises(InferenceBackendError, match="boom"):
        await _prime(_egen(ev_p, e=InferenceBackendError("boom"), c=closed, t="err"))
    assert "err" in closed

    polls = 0

    async def _disc_on_second() -> bool:
        nonlocal polls
        polls += 1
        await asyncio.sleep(0)
        return polls > 1

    p_disc = await _prime(_egen(w=asyncio.Event(), c=closed, t="disc"), disc=_disc_on_second)
    assert not hasattr(p_disc, "__dict__")
    assert (p_disc.guardrail_blocked, p_disc.guardrail_policy) == (False, None)
    assert ([ev async for ev in p_disc.events], "disc" in closed) == ([], True)

    t0 = time.monotonic()
    p_slow = await _prime(_egen(ev_p, d=5.0, c=closed, t="early"), ka=10.0, budget=0.05)
    assert (time.monotonic() - t0 < _SHORT_ELAPSED, p_slow.guardrail_blocked) == (True, False)
    assert (await anext(p_slow.events)) == ev_p[0]
    t1 = time.monotonic()
    await cast("AsyncGenerator[dc.OrchestratorEvent]", p_slow.events).aclose()
    assert (time.monotonic() - t1 < _SHORT_ELAPSED, "early" in closed) == (True, True)
    t2 = time.monotonic()
    p_ka = await _prime(_egen(d=0.45), ka=0.6, budget=2.0)
    assert time.monotonic() - t2 < _KA_ELAPSED
    assert [ev async for ev in p_ka.events] == []
    p0 = await _prime(_egen((dc.TurnCompletedEvent(output=_turn_out("s", "p", "x")),)), budget=0.0)
    assert (p0.guardrail_blocked, len([ev async for ev in p0.events])) == (False, 1)
    assert [ev async for ev in (await _prime(_egen(d=0.05), budget=0.01)).events] == []
    task = asyncio.create_task(_prime(_egen(d=5.0, c=closed, t="cancel")))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "cancel" in closed
