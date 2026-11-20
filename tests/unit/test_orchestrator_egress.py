"""Unit/integration tests for Session.system_prompt scan and orchestrator egress gating."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lai.config.settings import LaiSettings
from lai.db import (
    SqliteDatabase,
    SqliteFileRepository,
    SqliteProjectRepository,
    SqliteSessionRepository,
    SqliteUsageAnalyticsRepository,
)
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import (
    ContentDeltaEvent,
    OrchestratorEvent,
    ProgressEvent,
    Session,
    ThinkingDeltaEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnInput,
)
from lai.domain.errors import GuardrailBlockedError
from lai.domain.guardrails import GuardrailBlockMode, GuardrailConfig, GuardrailOutputMode
from lai.domain.inference import BackendHealth, InferenceChunk, InferenceRequest, InferenceResponse
from lai.domain.projects import Project
from lai.domain.tools import ApprovalStatus, PermissionTier, ToolApprovalRecord, ToolResult
from lai.guardrails import CompositeGuardrailPipeline, RegexPolicyGuardrailProvider
from lai.ingestion import LocalBlobStore
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.scheduler import InferenceSpooler, SessionGate

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


class _StubInf(InferenceProviderPort):
    """Minimal fake inference provider streaming configurable thinking and content deltas."""

    def __init__(self, chunks: tuple[str, ...], thinking: str = "") -> None:
        self._chunks, self._thinking, self.calls = chunks, thinking, 0

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="".join(self._chunks))

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        self.calls += 1
        if self._thinking:
            yield InferenceChunk(thinking_delta=self._thinking)
        for idx, part in enumerate(self._chunks):
            is_last = idx == len(self._chunks) - 1
            yield InferenceChunk(
                content_delta=part,
                finish_reason="stop" if is_last else None,
                usage=TokenUsage(prompt_tokens=4, completion_tokens=3, total_tokens=7)
                if is_last
                else None,
            )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


async def _collect_into(
    stream: AsyncGenerator[OrchestratorEvent],
    sink: list[OrchestratorEvent],
) -> None:
    async for ev in stream:
        sink.append(ev)  # noqa: PERF401


def _principal() -> AuthPrincipal:
    return AuthPrincipal(
        principal_id="u1", auth_method=AuthMethod.API_KEY, scopes=(ApiScope.CHAT_WRITE,)
    )


def _build_orch(  # noqa: PLR0913
    tmp_path: Path,
    cfg: GuardrailConfig,
    inf: _StubInf,
    *,
    s_repo: SqliteSessionRepository,
    f_repo: SqliteFileRepository,
    provider: RegexPolicyGuardrailProvider,
    analytics: SqliteUsageAnalyticsRepository | None = None,
) -> tuple[TurnOrchestrator, CompositeGuardrailPipeline]:
    pipe = CompositeGuardrailPipeline(
        cfg, providers=[provider], analytics_repo=analytics, log_dir=tmp_path / "logs"
    )
    orch = TurnOrchestrator(
        LaiSettings(data_dir=tmp_path, guardrails=cfg),
        s_repo,
        f_repo,
        LocalBlobStore(tmp_path / "blobs"),
        inf,
        InferenceSpooler(),
        SessionGate(),
        pipe,
    )
    return orch, pipe


async def test_session_and_turn_system_prompt_scanned_on_ingress(tmp_path: Path) -> None:
    """Both persisted Session.system_prompt and stateless TurnInput.system_prompt are scanned."""
    principal = _principal()
    provider = RegexPolicyGuardrailProvider(rules=(("No Prompt Injection", r"INJECT_SYS"),))
    cfg = GuardrailConfig(refusal_text="System prompt blocked.")

    async with SqliteDatabase(tmp_path / "sys.db") as db:
        await SqliteProjectRepository(db).create(Project(id="p1", name="P1"))
        s_repo, f_repo = SqliteSessionRepository(db), SqliteFileRepository(db)
        await s_repo.create_session(
            Session(id="s_sys", project_id="p1", system_prompt="INJECT_SYS override")
        )
        inf = _StubInf(("should never run",))
        orch, pipe = _build_orch(
            tmp_path, cfg, inf, s_repo=s_repo, f_repo=f_repo, provider=provider
        )
        try:
            out_persisted = await orch.run_turn(
                TurnInput(session_id="s_sys", project_id="p1", user_message="hello"), principal
            )
            assert (
                out_persisted.guardrail_blocked,
                out_persisted.guardrail_policy,
                out_persisted.guardrail_direction,
            ) == (True, "No Prompt Injection", "ingress")

            out_stateless = await orch.run_turn(
                TurnInput(
                    session_id="s_none",
                    project_id="p1",
                    user_message="hello",
                    system_prompt="INJECT_SYS stateless",
                ),
                principal,
            )
            assert (out_stateless.guardrail_blocked, out_stateless.guardrail_direction) == (
                True,
                "ingress",
            )
            assert inf.calls == 0
        finally:
            pipe.close()


async def test_egress_buffered_allow_refusal_and_http_error(tmp_path: Path) -> None:
    """BUFFERED egress flushes safe deltas and blocks unsafe content/thinking before emission."""
    principal = _principal()
    provider = RegexPolicyGuardrailProvider(rules=(("No Dangerous Content", r"BAD_OUT"),))

    async with SqliteDatabase(tmp_path / "buffered.db") as db:
        await SqliteProjectRepository(db).create(Project(id="p1", name="P1"))
        s_repo, f_repo = SqliteSessionRepository(db), SqliteFileRepository(db)
        for sid in ("s_ok", "s_ref", "s_http"):
            await s_repo.create_session(Session(id=sid, project_id="p1"))

        orch_ok, pipe_ok = _build_orch(
            tmp_path,
            GuardrailConfig(),
            _StubInf(("part1 ", "part2"), "step 1"),
            s_repo=s_repo,
            f_repo=f_repo,
            provider=provider,
        )
        try:
            evs_ok = [
                ev
                async for ev in orch_ok.run_turn_stream(
                    TurnInput(session_id="s_ok", project_id="p1", user_message="hi"), principal
                )
            ]
            assert [ev.delta for ev in evs_ok if isinstance(ev, ThinkingDeltaEvent)] == ["step 1"]
            assert [ev.delta for ev in evs_ok if isinstance(ev, ContentDeltaEvent)] == [
                "part1 ",
                "part2",
            ]
            done_ok = next(ev.output for ev in evs_ok if isinstance(ev, TurnCompletedEvent))
            assert (done_ok.guardrail_blocked, done_ok.guardrail_direction) == (False, None)
        finally:
            pipe_ok.close()

        orch_ref, pipe_ref = _build_orch(
            tmp_path,
            GuardrailConfig(refusal_text="Blocked egress."),
            _StubInf(("safe text",), "BAD_OUT"),
            s_repo=s_repo,
            f_repo=f_repo,
            provider=provider,
        )
        try:
            evs_ref = [
                ev
                async for ev in orch_ref.run_turn_stream(
                    TurnInput(session_id="s_ref", project_id="p1", user_message="hi"), principal
                )
            ]
            assert not any(isinstance(ev, ThinkingDeltaEvent) for ev in evs_ref)
            assert [ev.delta for ev in evs_ref if isinstance(ev, ContentDeltaEvent)] == [
                "Blocked egress."
            ]
            done_ref = next(ev.output for ev in evs_ref if isinstance(ev, TurnCompletedEvent))
            assert (
                done_ref.guardrail_blocked,
                done_ref.guardrail_policy,
                done_ref.guardrail_direction,
            ) == (True, "No Dangerous Content", "egress")
            stored_ref = await s_repo.list_messages("s_ref")
            assert all("BAD_OUT" not in (m.thinking or m.content) for m in stored_ref)
            assert stored_ref[-1].content == "Blocked egress."
        finally:
            pipe_ref.close()

        orch_http, pipe_http = _build_orch(
            tmp_path,
            GuardrailConfig(block_mode=GuardrailBlockMode.HTTP_ERROR),
            _StubInf(("BAD_OUT",)),
            s_repo=s_repo,
            f_repo=f_repo,
            provider=provider,
        )
        try:
            leaked: list[OrchestratorEvent] = []
            stream = orch_http.run_turn_stream(
                TurnInput(session_id="s_http", project_id="p1", user_message="hi"), principal
            )
            with pytest.raises(GuardrailBlockedError) as exc_info:
                await _collect_into(stream, leaked)
            assert exc_info.value.violated_policy == "No Dangerous Content"
            assert not any(isinstance(ev, (ContentDeltaEvent, ThinkingDeltaEvent)) for ev in leaked)
            assert await s_repo.list_messages("s_http") == []
        finally:
            pipe_http.close()


async def test_egress_audit_only_off_and_client_tool_resume(tmp_path: Path) -> None:
    """AUDIT_ONLY streams live + records analytics; OFF skips; resume enforces BUFFERED egress."""
    principal = _principal()
    provider = RegexPolicyGuardrailProvider(rules=(("No Dangerous Content", r"BAD_OUT"),))

    async with SqliteDatabase(tmp_path / "modes.db") as db:
        await SqliteProjectRepository(db).create(Project(id="p1", name="P1"))
        s_repo, f_repo = SqliteSessionRepository(db), SqliteFileRepository(db)
        analytics = SqliteUsageAnalyticsRepository(db)
        for sid in ("s_aud", "s_off", "s_res"):
            await s_repo.create_session(Session(id=sid, project_id="p1"))

        orch_aud, pipe_aud = _build_orch(
            tmp_path,
            GuardrailConfig(output_mode=GuardrailOutputMode.AUDIT_ONLY),
            _StubInf(("BAD_OUT",)),
            s_repo=s_repo,
            f_repo=f_repo,
            provider=provider,
            analytics=analytics,
        )
        try:
            out_aud = await orch_aud.run_turn(
                TurnInput(session_id="s_aud", project_id="p1", user_message="hi"), principal
            )
            assert (out_aud.guardrail_blocked, out_aud.message.content) == (False, "BAD_OUT")
        finally:
            pipe_aud.close()

        orch_off, pipe_off = _build_orch(
            tmp_path,
            GuardrailConfig(output_mode=GuardrailOutputMode.OFF),
            _StubInf(("BAD_OUT",)),
            s_repo=s_repo,
            f_repo=f_repo,
            provider=provider,
            analytics=analytics,
        )
        try:
            out_off = await orch_off.run_turn(
                TurnInput(session_id="s_off", project_id="p1", user_message="hi"), principal
            )
            assert (out_off.guardrail_blocked, out_off.message.content) == (False, "BAD_OUT")
        finally:
            pipe_off.close()

        orch_res, pipe_res = _build_orch(
            tmp_path,
            GuardrailConfig(refusal_text="Resume blocked."),
            _StubInf(("BAD_OUT",)),
            s_repo=s_repo,
            f_repo=f_repo,
            provider=provider,
            analytics=analytics,
        )
        try:
            apr = ToolApprovalRecord(
                project_id="p1",
                session_id="s_res",
                tool_name="local_tool",
                arguments_json="{}",
                permission_tier=PermissionTier.READ,
                reason="run",
                status=ApprovalStatus.APPROVED,
                requested_by="u1",
            )
            res_evs = [
                ev
                async for ev in orch_res.resume_client_tool_stream(
                    apr, ToolResult(tool_name="local_tool", success=True, output="ok"), principal
                )
            ]
            assert any(isinstance(ev, ProgressEvent) for ev in res_evs)
            assert [ev.delta for ev in res_evs if isinstance(ev, ContentDeltaEvent)] == [
                "Resume blocked."
            ]
            done_res = next(ev.output for ev in res_evs if isinstance(ev, TurnCompletedEvent))
            assert (done_res.guardrail_blocked, done_res.guardrail_direction) == (True, "egress")
            assert all("BAD_OUT" not in m.content for m in await s_repo.list_messages("s_res"))
        finally:
            pipe_res.close()

        report = await analytics.get_report(window_hours=24)
        assert report.guardrails.total_blocked == 2
