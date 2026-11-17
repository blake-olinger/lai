"""Tests for SSE mid-stream error framing, priming-window errors, and disconnect cleanup."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

import httpx
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
from lai.ports.orchestrator import OrchestratorPort
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator
    from collections.abc import Sequence as Seq
    from pathlib import Path

    from fastapi import FastAPI
    from starlette.types import Message as AsgiMessage
    from starlette.types import Scope

_ERR_MSG = "An unexpected internal server error occurred."
_TWO = 2


class _ScriptedOrchestrator(OrchestratorPort):
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


@pytest.fixture(autouse=True)
def _cleanup_root_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


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


async def test_asgi_sse_midstream_and_priming_errors_and_disconnect(tmp_path: Path) -> None:
    """Verify ASGI mid-stream LaiError/RuntimeError, priming HTTP 502, and disconnect cleanup."""
    orch = _ScriptedOrchestrator(
        (dc.ProgressEvent(stage="inference", message="starting"),),
        error=InferenceBackendError("boom", details={"backend": "ollama"}),
    )
    log_cfg = LoggingConfig(
        log_dir=tmp_path / "l", enable_file_logs=False, alert_cooldown_seconds=0.0
    )
    cfg = LaiSettings(data_dir=tmp_path / "d", keepalive_interval_seconds=0.5, logging=log_cfg)
    app = create_app(cfg, orchestrator=orch)

    async with app.router.lifespan_context(app):
        key = generate_api_key("sse-err-admin", (ApiScope.ADMIN,))
        await app.state.services.api_key_repo.create(key.record)
        hdrs = {"Authorization": f"Bearer {key.plaintext_key}"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as cli:
            pid = (await cli.post("/v1/projects", headers=hdrs, json={"name": "e"})).json()["id"]
            r_ses = await cli.post("/v1/sessions", headers=hdrs, json={"project_id": pid})
            sid = str(r_ses.json()["id"])
            ses_url, oai_url = f"/v1/sessions/{sid}/messages", "/v1/chat/completions"
            ses_req = {"content": "hi", "stream": True}
            msgs = [{"role": "user", "content": "hi"}]
            oai_req = {"model": "gemma4-26b-a4b", "messages": msgs, "stream": True}

            ses_txt = (await cli.post(ses_url, headers=hdrs, json=ses_req)).text
            assert "event: progress" in ses_txt
            assert '"code":"inference_backend_error"' in ses_txt
            oai_txt = (await cli.post(oai_url, headers=hdrs, json=oai_req)).text
            assert '"type": "inference_backend_error"' in oai_txt
            assert oai_txt.endswith("data: [DONE]\n\n")

            orch.error = RuntimeError("db secret leaked")
            for url, rid, req, marker in (
                (ses_url, "r_ses", ses_req, '"code":"internal_error"'),
                (oai_url, "r_oai", oai_req, '"type": "internal_error"'),
            ):
                resp = await cli.post(url, headers={**hdrs, "X-Request-Id": rid}, json=req)
                assert resp.status_code == HTTPStatus.OK
                assert "db secret leaked" not in resp.text
                assert marker in resp.text
                assert _ERR_MSG in resp.text
                if url == oai_url:
                    assert resp.text.endswith("data: [DONE]\n\n")

            alerts = app.state.services.alert_manager.recent_alerts()
            fatal = {
                (a.request_id, a.exception_type)
                for a in alerts
                if a.category is AlertCategory.FATAL_EXCEPTION
            }
            assert fatal == {("r_ses", "RuntimeError"), ("r_oai", "RuntimeError")}

            orch.events, orch.error = [], InferenceBackendError("down")
            for url, payload in ((ses_url, ses_req), (oai_url, oai_req)):
                r_err = await cli.post(url, headers=hdrs, json=payload)
                assert r_err.status_code == HTTPStatus.BAD_GATEWAY
                assert r_err.json()["error"]["code"] == "inference_backend_error"

        await _assert_asgi_priming_disconnect(app, sid, key.plaintext_key)


async def _assert_asgi_priming_disconnect(app: FastAPI, sid: str, token: str) -> None:
    """Exercise ASGI client disconnect during prime_sse_stream and verify producer cleanup."""
    slow_orch = _ScriptedOrchestrator()
    slow_orch.entered_event, slow_orch.wait_event = asyncio.Event(), asyncio.Event()
    app.state.services.orchestrator = slow_orch
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

    hdrs = [(b"authorization", f"Bearer {token}".encode()), (b"content-type", b"application/json")]
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
    assert slow_orch.stream_closed is True
    assert sent[0]["status"] == HTTPStatus.OK


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
    assert p_disc.guardrail_blocked is False
    assert p_disc.guardrail_policy is None
    assert [ev async for ev in p_disc.events] == []
    assert "disc" in closed

    t0 = time.monotonic()
    p_slow = await _prime(_egen(ev_p, d=5.0, c=closed, t="early"), ka=10.0, budget=0.05)
    assert time.monotonic() - t0 < 0.3  # noqa: PLR2004
    assert p_slow.guardrail_blocked is False
    first = await anext(p_slow.events)
    assert first == ev_p[0]
    t1 = time.monotonic()
    await cast("AsyncGenerator[dc.OrchestratorEvent]", p_slow.events).aclose()
    assert time.monotonic() - t1 < 0.3  # noqa: PLR2004
    assert "early" in closed

    t2 = time.monotonic()
    p_ka = await _prime(_egen(d=0.45), ka=0.6, budget=2.0)
    assert time.monotonic() - t2 < 0.45  # noqa: PLR2004
    assert [ev async for ev in p_ka.events] == []

    blocked_ev = dc.TurnCompletedEvent(output=_turn_out("s", "p", "x", blocked=True))
    p0 = await _prime(_egen((blocked_ev,)), budget=0.0)
    assert p0.guardrail_blocked is False
    assert len([ev async for ev in p0.events]) == 1
    assert [ev async for ev in (await _prime(_egen(d=0.05), budget=0.01)).events] == []

    task = asyncio.create_task(_prime(_egen(d=5.0, c=closed, t="cancel")))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "cancel" in closed
