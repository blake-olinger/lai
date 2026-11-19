"""Unit tests for tool timeouts/alerts, TOOL history trimming, and OpenAI client tool DTOs."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pytest

from lai.api.routes._openai_helpers import StreamMeta, build_completion_response
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import (
    ClientToolCallEvent,
    ContextMode,
    InferenceToolCall,
    Message,
    MessageRole,
    Session,
    TokenUsage,
    TurnOutput,
)
from lai.domain.errors import ValidationFailedError
from lai.domain.inference import ModelProfile
from lai.domain.observability import AlertCategory, AlertEvent, AlertSeverity, LoggingConfig
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    ToolApprovalRecord,
    ToolAuditRecord,
    ToolExecutionContext,
    ToolExecutionTarget,
    ToolResult,
    ToolSpec,
)
from lai.observability import AlertManager, configure_observability
from lai.orchestrator._helpers import (
    build_inference_tool_defs,
    compute_history_budget,
    parse_tool_arguments,
    resolve_model_profile,
    trim_history_to_budget,
)
from lai.ports.observability import AlertSinkPort
from lai.ports.repositories import ToolAuditRepositoryPort
from lai.ports.tools import ToolPluginPort
from lai.tools import GuardedToolExecutor, InMemoryToolRegistry
from lai.tools._audit import deserialize_arguments, validate_approval_match

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

EXPECTED_TWO = 2
SHORT_TIMEOUT = 0.05
LONG_SLEEP = 0.4


class _MemoryAlertSink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


class _FakePlugin(ToolPluginPort):
    def __init__(
        self,
        spec: ToolSpec,
        *,
        sleep_seconds: float = 0.0,
        raise_error: Exception | None = None,
    ) -> None:
        self._spec, self._sleep_seconds, self._raise_error = spec, sleep_seconds, raise_error

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        if self._sleep_seconds > 0.0:
            await asyncio.sleep(self._sleep_seconds)
        if self._raise_error is not None:
            raise self._raise_error
        return ToolResult(tool_name=self._spec.name, success=True, output="ok")


class _FakeToolAuditRepo(ToolAuditRepositoryPort):
    def __init__(self) -> None:
        self.approvals: dict[str, ToolApprovalRecord] = {}
        self.audits: list[ToolAuditRecord] = []

    async def create_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        self.approvals[record.id] = record
        return record

    async def get_approval(self, approval_id: str) -> ToolApprovalRecord | None:
        return self.approvals.get(approval_id)

    async def list_pending_approvals(
        self, project_id: str, session_id: str | None = None
    ) -> list[ToolApprovalRecord]:
        del project_id, session_id
        return [r for r in self.approvals.values() if r.status is ApprovalStatus.PENDING]

    async def update_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        self.approvals[record.id] = record
        return record

    async def transition_approval_status(
        self, record: ToolApprovalRecord, *, expected_status: ApprovalStatus
    ) -> bool:
        existing = self.approvals.get(record.id)
        if existing is None or existing.status is not expected_status:
            return False
        self.approvals[record.id] = record
        return True

    async def record_audit(self, record: ToolAuditRecord) -> ToolAuditRecord:
        self.audits.append(record)
        return record

    async def list_audits(
        self, project_id: str, session_id: str | None = None, limit: int = 100
    ) -> list[ToolAuditRecord]:
        del project_id, session_id
        return self.audits[:limit]


def _spec(name: str, tier: PermissionTier, *, timeout: float = 30.0) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        permission_tier=tier,
        timeout_seconds=timeout,
        execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
    )


def _ctx(*scopes: ApiScope) -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id="usr_1",
        principal_scopes=scopes,
    )


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_timeout_emits_alert_and_records_audit(tmp_path: Path) -> None:
    """TimeoutError logs with AlertCategory.TIMEOUT, emits alert, and audits failure."""
    sink = _MemoryAlertSink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(
        LoggingConfig(log_dir=tmp_path, enable_file_logs=False), alert_manager=manager
    )
    try:
        slow = _FakePlugin(
            _spec("slow_read", PermissionTier.READ, timeout=SHORT_TIMEOUT),
            sleep_seconds=LONG_SLEEP,
        )
        boom = _FakePlugin(
            _spec("boom_read", PermissionTier.READ),
            raise_error=RuntimeError("disk error"),
        )
        repo = _FakeToolAuditRepo()
        executor = GuardedToolExecutor(InMemoryToolRegistry((slow, boom)), repo)
        t_res = await executor.invoke("slow_read", {}, _ctx(ApiScope.TOOLS_SAFE))
        b_res = await executor.invoke("boom_read", {}, _ctx(ApiScope.TOOLS_SAFE))
        await manager.flush()
    finally:
        manager.close()

    assert (t_res.success, t_res.error, b_res.success, b_res.error) == (
        False,
        "timeout",
        False,
        "RuntimeError: disk error",
    )
    alerts = manager.recent_alerts()
    assert len(alerts) == EXPECTED_TWO
    assert len(sink.events) == EXPECTED_TWO
    assert [(a.category, a.severity, a.component, a.exception_type) for a in alerts] == [
        (AlertCategory.APPLICATION_ERROR, AlertSeverity.ERROR, "tools", "RuntimeError"),
        (AlertCategory.TIMEOUT, AlertSeverity.ERROR, "tools", "TimeoutError"),
    ]
    assert len(repo.audits) == EXPECTED_TWO
    assert repo.audits[0].error == "timeout"
    assert repo.audits[1].success is False


def test_history_trimming_tool_defs_and_openai_completion_client_tool() -> None:
    """Verify TOOL history retention, orphan stripping, tool defs filter, and OpenAI JSON DTO."""
    tc1 = InferenceToolCall(
        id="tc_1", name="read_local", arguments_json='{"p":"' + ("word " * 30) + '"}'
    )
    tc2 = InferenceToolCall(id="tc_2", name="run_local", arguments_json="{}")
    history = [
        Message(session_id="s", project_id="p", role=MessageRole.USER, content="old " * 40),
        Message(
            session_id="s",
            project_id="p",
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(tc1,),
        ),
        Message(
            session_id="s",
            project_id="p",
            role=MessageRole.TOOL,
            content="r1",
            tool_call_id="tc_1",
            tool_name="read_local",
        ),
        Message(session_id="s", project_id="p", role=MessageRole.ASSISTANT, content="done 1"),
        Message(
            session_id="s",
            project_id="p",
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(tc2,),
        ),
        Message(
            session_id="s",
            project_id="p",
            role=MessageRole.TOOL,
            content="r2",
            tool_call_id="tc_2",
            tool_name="run_local",
        ),
    ]
    trimmed = trim_history_to_budget(history, max_history_tokens=45)
    assert [(m.role, m.tool_call_id, len(m.tool_calls)) for m in trimmed] == [
        (MessageRole.ASSISTANT, None, 0),
        (MessageRole.ASSISTANT, None, 1),
        (MessageRole.TOOL, "tc_2", 0),
    ]

    reg = InMemoryToolRegistry(
        (
            _FakePlugin(
                ToolSpec(name="srv_read", description="r", permission_tier=PermissionTier.READ)
            ),
            _FakePlugin(
                ToolSpec(
                    name="cli_reg",
                    description="c",
                    permission_tier=PermissionTier.READ,
                    execution_target=ToolExecutionTarget.CLIENT,
                )
            ),
        )
    )
    cli_spec = ToolSpec(
        name="cli_turn",
        description="t",
        permission_tier=PermissionTier.EXEC,
        execution_target=ToolExecutionTarget.CLIENT,
    )
    assert [
        d.name for d in build_inference_tool_defs(reg, enable_tools=True, client_tools=(cli_spec,))
    ] == ["srv_read"]
    assert [
        d.name
        for d in build_inference_tool_defs(
            reg, enable_tools=True, client_tools=(cli_spec,), client_app_id="app_1"
        )
    ] == ["srv_read", "cli_turn"]

    ev = ClientToolCallEvent(
        approval_id="apr_1",
        tool_call_id="tc_9",
        tool_name="cli_turn",
        arguments_json='{"k":"v"}',
        requires_approval=True,
    )
    resp = build_completion_response(
        TurnOutput(
            session_id="s",
            message=Message(session_id="s", project_id="p", role=MessageRole.ASSISTANT, content=""),
            context_mode_used=ContextMode.RETRIEVAL,
            usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            pending_approval_id="apr_1",
            pending_client_tool_call=ev,
        ),
        StreamMeta(completion_id="c1", created=1, model_name="gemma4-26b-a4b"),
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls is not None
    assert resp.choices[0].message.tool_calls[0].id == "tc_9"
    assert resp.lai_client_tool_call == ev

    tiny_p = ModelProfile(name="m", hf_repo="r/r", hf_filename="m.gguf", ctx_size=512)
    ses_u = Session(project_id="p", model_profile="unknown-custom")
    assert resolve_model_profile(LaiSettings(), ses_u, tiny_p) is tiny_p
    assert compute_history_budget(tiny_p, reserved_prompt_tokens=400, max_output_tokens=200) == 0
    dedup = build_inference_tool_defs(
        None, enable_tools=True, client_tools=(cli_spec, cli_spec), client_app_id="app_1"
    )
    assert [d.name for d in dedup] == ["cli_turn"]
    assert parse_tool_arguments("   ") == {}
    with pytest.raises(TypeError):
        parse_tool_arguments("[1]")


def test_audit_argument_and_approval_match_helpers() -> None:
    """Verify deserialize_arguments and validate_approval_match error branches."""
    for bad_json in ("{not-json", "[1, 2]"):
        with pytest.raises(ValidationFailedError):
            deserialize_arguments(bad_json)

    assert validate_approval_match(None, _ctx(), "t") == "approval_not_found"
    apr = ToolApprovalRecord(
        project_id="prj_1",
        session_id="ses_other",
        tool_name="t",
        arguments_json="{}",
        permission_tier=PermissionTier.READ,
        reason="needed",
        status=ApprovalStatus.APPROVED,
        requested_by="usr_1",
    )
    assert validate_approval_match(apr, _ctx(), "t") == "approval_context_mismatch"
