"""Unit tests for tool registry, permission policy, approvals, replay guard, and timeouts."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pytest

from lai.domain.auth import ApiScope
from lai.domain.errors import (
    AuthorizationError,
    NotFoundError,
    ToolApprovalRequiredError,
    ValidationFailedError,
)
from lai.domain.observability import AlertCategory, AlertEvent, AlertSeverity, LoggingConfig
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from lai.observability import AlertManager, configure_observability
from lai.ports.observability import AlertSinkPort
from lai.ports.repositories import ToolAuditRepositoryPort
from lai.ports.tools import ToolExecutorPort, ToolPluginPort, ToolPolicyPort, ToolRegistryPort
from lai.tools import DefaultToolPolicy, GuardedToolExecutor, InMemoryToolRegistry

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

EXPECTED_TWO = 2
SHORT_TIMEOUT = 0.05
LONG_SLEEP = 0.4


class _FakePlugin(ToolPluginPort):
    def __init__(
        self,
        spec: ToolSpec,
        *,
        output: str = "ok",
        sleep_seconds: float = 0.0,
        raise_error: Exception | None = None,
    ) -> None:
        self._spec = spec
        self._output = output
        self._sleep_seconds = sleep_seconds
        self._raise_error = raise_error
        self.calls: list[tuple[dict[str, Any], ToolExecutionContext]] = []

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        self.calls.append((dict(arguments), context))
        if self._sleep_seconds > 0.0:
            await asyncio.sleep(self._sleep_seconds)
        if self._raise_error is not None:
            raise self._raise_error
        return ToolResult(tool_name=self._spec.name, success=True, output=self._output)


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

    async def record_audit(self, record: ToolAuditRecord) -> ToolAuditRecord:
        self.audits.append(record)
        return record

    async def list_audits(
        self, project_id: str, session_id: str | None = None, limit: int = 100
    ) -> list[ToolAuditRecord]:
        del project_id, session_id
        return self.audits[:limit]


class _MemoryAlertSink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


def _spec(
    name: str, tier: PermissionTier, *, gated: bool = False, timeout: float = 30.0
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        permission_tier=tier,
        requires_approval_by_default=gated,
        timeout_seconds=timeout,
    )


def _ctx(
    *scopes: ApiScope,
    project_id: str = "prj_1",
    session_id: str = "ses_1",
    approval_id: str | None = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id=project_id,
        session_id=session_id,
        principal_id="usr_1",
        principal_scopes=scopes,
        allowed_file_ids=("fil_1",),
        workspace_root="/repo",
        approved_approval_id=approval_id,
        request_id="req_1",
    )


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def test_registry_and_port_conformance() -> None:
    """InMemoryToolRegistry registers, replaces, looks up, and lists ToolSpecs."""
    read_v1 = _FakePlugin(_spec("read_a", PermissionTier.READ))
    read_v2 = _FakePlugin(
        ToolSpec(name="read_a", description="v2", permission_tier=PermissionTier.READ)
    )
    registry = InMemoryToolRegistry(plugins=(read_v1,))
    assert isinstance(registry, ToolRegistryPort)
    assert isinstance(DefaultToolPolicy(), ToolPolicyPort)
    assert isinstance(GuardedToolExecutor(registry, _FakeToolAuditRepo()), ToolExecutorPort)
    assert registry.get("read_a") is read_v1
    assert registry.get("missing") is None
    registry.register(read_v2)
    assert registry.get("read_a") is read_v2
    assert [s.description for s in registry.list_specs()] == ["v2"]


def test_read_and_privileged_tier_policy_matrix() -> None:
    """Policy enforces READ and WRITE/EXEC/NETWORK scope + approval rules across all scopes."""
    policy = DefaultToolPolicy()
    read_spec = _spec("read_tool", PermissionTier.READ)
    read_allowed = {ApiScope.ADMIN, ApiScope.TOOLS_SAFE, ApiScope.CHAT_WRITE, ApiScope.FILES_READ}
    assert policy.evaluate(read_spec, {}, _ctx()) is PolicyDecision.DENY
    for scope in ApiScope:
        expected = PolicyDecision.ALLOW if scope in read_allowed else PolicyDecision.DENY
        assert policy.evaluate(read_spec, {}, _ctx(scope)) is expected

    for tier in (PermissionTier.WRITE, PermissionTier.EXEC, PermissionTier.NETWORK):
        gated, ungated = _spec("g", tier, gated=True), _spec("u", tier, gated=False)
        for scope in ApiScope:
            if scope not in {ApiScope.ADMIN, ApiScope.TOOLS_EXEC}:
                assert policy.evaluate(gated, {}, _ctx(scope, approval_id="a")) is (
                    PolicyDecision.DENY
                )
                assert policy.evaluate(ungated, {}, _ctx(scope)) is PolicyDecision.DENY
                continue
            assert policy.evaluate(gated, {}, _ctx(scope)) is PolicyDecision.REQUIRE_APPROVAL
            assert policy.evaluate(gated, {}, _ctx(scope, approval_id="  ")) is (
                PolicyDecision.REQUIRE_APPROVAL
            )
            assert policy.evaluate(gated, {}, _ctx(scope, approval_id="a")) is PolicyDecision.ALLOW
            assert policy.evaluate(ungated, {}, _ctx(scope)) is PolicyDecision.ALLOW


async def test_executor_denies_unauthorized_and_missing_tool() -> None:
    """GuardedToolExecutor records DENY audit and raises AuthorizationError / NotFoundError."""
    write_plugin = _FakePlugin(_spec("write_file", PermissionTier.WRITE, gated=True))
    repo = _FakeToolAuditRepo()
    executor = GuardedToolExecutor(InMemoryToolRegistry((write_plugin,)), repo)
    with pytest.raises(NotFoundError):
        await executor.invoke("unknown", {}, _ctx(ApiScope.ADMIN))
    with pytest.raises(AuthorizationError):
        await executor.invoke("write_file", {"path": "x"}, _ctx(ApiScope.TOOLS_SAFE))
    assert write_plugin.calls == []
    assert len(repo.audits) == 1
    assert repo.audits[0].policy_decision is PolicyDecision.DENY
    assert repo.audits[0].success is False


async def test_approval_lifecycle_and_replay_guard() -> None:
    """Approval creation -> approve_and_execute -> EXECUTED blocks replay."""
    write_plugin = _FakePlugin(
        _spec("write_file", PermissionTier.WRITE, gated=True),
        output="wrote Bearer lai_sk_secret12345678",
    )
    repo = _FakeToolAuditRepo()
    executor = GuardedToolExecutor(InMemoryToolRegistry((write_plugin,)), repo)

    with pytest.raises(ToolApprovalRequiredError) as exc_info:
        await executor.invoke("write_file", {"path": "a.py"}, _ctx(ApiScope.TOOLS_EXEC))
    aid = exc_info.value.approval_id
    assert exc_info.value.tool_name == "write_file"
    pending = await repo.get_approval(aid)
    assert pending is not None
    assert pending.status is ApprovalStatus.PENDING

    res = await executor.approve_and_execute(aid, decided_by="usr_admin", approved=True, note="ok")
    assert (res.status, res.decided_by, res.decision_note) == (
        ApprovalStatus.EXECUTED,
        "usr_admin",
        "ok",
    )
    assert len(write_plugin.calls) == 1
    assert write_plugin.calls[0][1].allowed_file_ids == ("fil_1",)
    assert len(repo.audits) == EXPECTED_TWO
    assert repo.audits[1].success is True
    assert "lai_sk_secret12345678" not in repo.audits[1].output_preview
    assert "[REDACTED_SECRET]" in repo.audits[1].output_preview

    with pytest.raises(ValidationFailedError):
        await executor.approve_and_execute(aid, decided_by="usr_admin", approved=True)
    with pytest.raises(AuthorizationError):
        await executor.invoke("write_file", {}, _ctx(ApiScope.TOOLS_EXEC, approval_id=aid))
    assert len(write_plugin.calls) == 1


async def test_approval_deny_mismatch_and_fallback_context() -> None:
    """Denial prevents execution; mismatched project/session/tool cannot consume an approval."""
    write_plugin = _FakePlugin(_spec("write_file", PermissionTier.WRITE, gated=True))
    other_plugin = _FakePlugin(_spec("run_cmd", PermissionTier.EXEC, gated=True))
    repo = _FakeToolAuditRepo()
    registry = InMemoryToolRegistry((write_plugin, other_plugin))
    executor = GuardedToolExecutor(registry, repo)

    with pytest.raises(NotFoundError):
        await executor.approve_and_execute("apr_missing", decided_by="usr_admin", approved=True)
    with pytest.raises(ValidationFailedError):
        await executor.approve_and_execute("apr_missing", decided_by="  ", approved=True)
    with pytest.raises(ToolApprovalRequiredError) as exc_deny:
        await executor.invoke("write_file", {"path": "b.py"}, _ctx(ApiScope.TOOLS_EXEC))
    denied = await executor.approve_and_execute(
        exc_deny.value.approval_id, decided_by="usr_admin", approved=False, note="no"
    )
    assert denied.status is ApprovalStatus.DENIED
    with pytest.raises(AuthorizationError):
        await executor.invoke("write_file", {}, _ctx(ApiScope.TOOLS_EXEC, approval_id=denied.id))

    rec = ToolApprovalRecord(
        project_id="prj_1",
        session_id="ses_1",
        tool_name="write_file",
        arguments_json='{"path": "c.py"}',
        permission_tier=PermissionTier.WRITE,
        reason="needs approval",
        status=ApprovalStatus.APPROVED,
        requested_by="usr_1",
        decided_by="usr_admin",
    )
    await repo.create_approval(rec)
    rid = rec.id
    for bad_tool, bad_ctx in (
        ("write_file", _ctx(ApiScope.TOOLS_EXEC, approval_id="apr_missing")),
        ("write_file", _ctx(ApiScope.TOOLS_EXEC, project_id="prj_other", approval_id=rid)),
        ("write_file", _ctx(ApiScope.TOOLS_EXEC, session_id="ses_other", approval_id=rid)),
        ("run_cmd", _ctx(ApiScope.TOOLS_EXEC, approval_id=rid)),
    ):
        with pytest.raises(AuthorizationError):
            await executor.invoke(bad_tool, {}, bad_ctx)

    ok_res = await executor.invoke("write_file", {}, _ctx(ApiScope.TOOLS_EXEC, approval_id=rid))
    assert ok_res.success is True
    consumed = await repo.get_approval(rid)
    assert consumed is not None
    assert consumed.status is ApprovalStatus.EXECUTED

    with pytest.raises(ToolApprovalRequiredError) as exc_fresh:
        await executor.invoke("write_file", {"path": "d.py"}, _ctx(ApiScope.TOOLS_EXEC))
    done = await GuardedToolExecutor(registry, repo).approve_and_execute(
        exc_fresh.value.approval_id, decided_by="usr_admin", approved=True
    )
    assert done.status is ApprovalStatus.EXECUTED
    assert len(write_plugin.calls) == EXPECTED_TWO


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
            _spec("boom_read", PermissionTier.READ), raise_error=RuntimeError("disk error")
        )
        repo = _FakeToolAuditRepo()
        executor = GuardedToolExecutor(InMemoryToolRegistry((slow, boom)), repo)
        timeout_res = await executor.invoke("slow_read", {}, _ctx(ApiScope.TOOLS_SAFE))
        boom_res = await executor.invoke("boom_read", {}, _ctx(ApiScope.TOOLS_SAFE))
        await manager.flush()
    finally:
        manager.close()

    assert (timeout_res.success, timeout_res.error) == (False, "timeout")
    assert (boom_res.success, boom_res.error) == (False, "RuntimeError: disk error")
    alerts = manager.recent_alerts()
    assert len(alerts) == 1
    assert len(sink.events) == 1
    alert = alerts[0]
    assert (alert.category, alert.severity, alert.component, alert.exception_type) == (
        AlertCategory.TIMEOUT,
        AlertSeverity.ERROR,
        "tools",
        "TimeoutError",
    )
    assert (alert.project_id, alert.session_id, alert.request_id) == ("prj_1", "ses_1", "req_1")
    assert len(repo.audits) == EXPECTED_TWO
    assert repo.audits[0].error == "timeout"
    assert repo.audits[1].success is False
