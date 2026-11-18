"""Unit tests for dual-plane tool policy, host ban, confinement, approvals, and replay."""

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
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
    ToolExecutionContext,
    ToolExecutionTarget,
    ToolResult,
    ToolSpec,
)
from lai.ports.repositories import ToolAuditRepositoryPort
from lai.ports.tools import ToolExecutorPort, ToolPluginPort, ToolPolicyPort, ToolRegistryPort
from lai.tools import DefaultToolPolicy, GuardedToolExecutor, InMemoryToolRegistry

if TYPE_CHECKING:
    from collections.abc import Generator


class _FakePlugin(ToolPluginPort):
    def __init__(
        self,
        spec: ToolSpec,
        *,
        output: str = "ok",
        sleep_seconds: float = 0.0,
        raise_error: Exception | None = None,
    ) -> None:
        self._spec, self._output = spec, output
        self._sleep_seconds, self._raise_error = sleep_seconds, raise_error
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


class _AllowAllPolicy(ToolPolicyPort):
    def evaluate(
        self, spec: ToolSpec, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> PolicyDecision:
        del spec, arguments, context
        return PolicyDecision.ALLOW


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


def _spec(
    name: str,
    tier: PermissionTier,
    *,
    gated: bool = False,
    timeout: float = 30.0,
    target: ToolExecutionTarget = ToolExecutionTarget.SERVER_ATTACHMENT,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        permission_tier=tier,
        requires_approval_by_default=gated,
        timeout_seconds=timeout,
        execution_target=target,
    )


def _ctx(
    *scopes: ApiScope,
    approval_id: str | None = None,
    app_id: str | None = None,
    tiers: tuple[PermissionTier, ...] = (),
) -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id="usr_1",
        principal_scopes=scopes,
        allowed_file_ids=("fil_1",),
        workspace_root="/repo",
        approved_approval_id=approval_id,
        request_id="req_1",
        client_app_id=app_id,
        client_allowed_tiers=tiers,
        tool_call_id="tc_1",
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
    r1, r2 = (
        _FakePlugin(_spec("r", PermissionTier.READ)),
        _FakePlugin(_spec("r", PermissionTier.READ)),
    )
    reg = InMemoryToolRegistry(plugins=(r1,))
    assert isinstance(reg, ToolRegistryPort)
    assert isinstance(DefaultToolPolicy(), ToolPolicyPort)
    assert isinstance(GuardedToolExecutor(reg, _FakeToolAuditRepo()), ToolExecutorPort)
    assert (reg.get("r"), reg.get("missing")) == (r1, None)
    reg.register(r2)
    assert reg.get("r") is r2
    assert [s.name for s in reg.list_specs()] == ["r"]


def test_dual_plane_policy_matrix() -> None:
    """Policy forbids server host tiers and enforces client app tiers, scopes, and approvals."""
    p = DefaultToolPolicy()
    al, dn, rq = PolicyDecision.ALLOW, PolicyDecision.DENY, PolicyDecision.REQUIRE_APPROVAL
    r_srv, r_srv_g = _spec("rs", PermissionTier.READ), _spec("rsg", PermissionTier.READ, gated=True)
    r_scopes = {ApiScope.ADMIN, ApiScope.TOOLS_SAFE, ApiScope.CHAT_WRITE, ApiScope.FILES_READ}
    assert p.evaluate(r_srv, {}, _ctx()) is dn
    for sc in ApiScope:
        assert p.evaluate(r_srv, {}, _ctx(sc)) is (al if sc in r_scopes else dn)
        if sc in r_scopes:
            assert p.evaluate(r_srv_g, {}, _ctx(sc)) is rq
            assert p.evaluate(r_srv_g, {}, _ctx(sc, approval_id=" ")) is rq
            assert p.evaluate(r_srv_g, {}, _ctx(sc, approval_id="a")) is al

    for tier in (PermissionTier.WRITE, PermissionTier.EXEC, PermissionTier.NETWORK):
        s_bad = _spec("sb", tier, gated=False)
        for sc in ApiScope:
            assert p.evaluate(s_bad, {}, _ctx(sc, approval_id="a")) is dn

    cr = _spec("cr", PermissionTier.READ, target=ToolExecutionTarget.CLIENT)
    crg = _spec("crg", PermissionTier.READ, gated=True, target=ToolExecutionTarget.CLIENT)
    assert p.evaluate(cr, {}, _ctx(ApiScope.TOOLS_SAFE, tiers=(PermissionTier.READ,))) is dn
    assert (
        p.evaluate(cr, {}, _ctx(ApiScope.TOOLS_SAFE, app_id=" ", tiers=(PermissionTier.READ,)))
        is dn
    )
    assert p.evaluate(cr, {}, _ctx(ApiScope.TOOLS_SAFE, app_id="app_1", tiers=())) is dn
    assert (
        p.evaluate(cr, {}, _ctx(ApiScope.TOOLS_SAFE, app_id="app_1", tiers=(PermissionTier.READ,)))
        is al
    )
    assert (
        p.evaluate(crg, {}, _ctx(ApiScope.TOOLS_SAFE, app_id="app_1", tiers=(PermissionTier.READ,)))
        is rq
    )
    assert (
        p.evaluate(
            crg,
            {},
            _ctx(
                ApiScope.TOOLS_SAFE, approval_id="a", app_id="app_1", tiers=(PermissionTier.READ,)
            ),
        )
        is al
    )

    for tier in (PermissionTier.WRITE, PermissionTier.EXEC, PermissionTier.NETWORK):
        cu = _spec("cu", tier, gated=False, target=ToolExecutionTarget.CLIENT)
        cg = _spec("cg", tier, gated=True, target=ToolExecutionTarget.CLIENT)
        assert (
            p.evaluate(
                cu, {}, _ctx(ApiScope.TOOLS_EXEC, app_id="app_1", tiers=(PermissionTier.READ,))
            )
            is dn
        )
        for sc in ApiScope:
            c_ctx = _ctx(sc, app_id="app_1", tiers=(tier,))
            if sc not in {ApiScope.ADMIN, ApiScope.TOOLS_EXEC}:
                assert p.evaluate(cu, {}, c_ctx) is dn
                assert p.evaluate(cg, {}, c_ctx) is dn
            else:
                assert p.evaluate(cu, {}, c_ctx) is rq
                assert p.evaluate(cg, {}, c_ctx) is rq
                c_ok = _ctx(sc, approval_id="a", app_id="app_1", tiers=(tier,))
                assert p.evaluate(cu, {}, c_ok) is al
                assert p.evaluate(cg, {}, c_ok) is al


async def test_executor_server_host_ban_and_unauthorized() -> None:
    """Executor unconditionally forbids non-READ server plugins even under custom policy."""
    write_p = _FakePlugin(_spec("write_file", PermissionTier.WRITE, gated=True))
    exec_p = _FakePlugin(_spec("run_ps", PermissionTier.EXEC))
    read_p = _FakePlugin(_spec("read_f", PermissionTier.READ, gated=True))
    repo = _FakeToolAuditRepo()
    reg = InMemoryToolRegistry((write_p, exec_p, read_p))
    executor = GuardedToolExecutor(reg, repo, policy=_AllowAllPolicy())

    with pytest.raises(NotFoundError):
        await executor.invoke("unknown", {}, _ctx(ApiScope.ADMIN))
    for bad in ("write_file", "run_ps"):
        with pytest.raises(AuthorizationError, match="forbidden"):
            await executor.invoke(bad, {"path": "x"}, _ctx(ApiScope.ADMIN, ApiScope.TOOLS_EXEC))
    assert write_p.calls == []
    assert exec_p.calls == []
    assert all(
        a.policy_decision is PolicyDecision.DENY and a.error == "server_host_tool_forbidden"
        for a in repo.audits
    )

    strict_exec = GuardedToolExecutor(reg, repo)
    with pytest.raises(AuthorizationError):
        await strict_exec.invoke("read_f", {}, _ctx(ApiScope.TOOLS_EXEC))
    assert repo.audits[-1].error == "policy_denied"


async def test_server_approval_confinement_restore_cas_and_replay() -> None:
    """Gated READ server plugin restores confinement across restart and blocks replay."""
    read_p = _FakePlugin(
        _spec("read_secret", PermissionTier.READ, gated=True),
        output="read Bearer lai_sk_secret12345678",
    )
    repo = _FakeToolAuditRepo()
    reg = InMemoryToolRegistry((read_p,))
    executor = GuardedToolExecutor(reg, repo)

    with pytest.raises(ToolApprovalRequiredError) as exc_info:
        await executor.invoke("read_secret", {"file_id": "fil_1"}, _ctx(ApiScope.TOOLS_SAFE))
    aid = exc_info.value.approval_id
    pending = await repo.get_approval(aid)
    assert pending is not None
    assert (pending.status, pending.allowed_file_ids, pending.principal_scopes) == (
        ApprovalStatus.PENDING,
        ("fil_1",),
        (ApiScope.TOOLS_SAFE,),
    )

    restarted = GuardedToolExecutor(reg, repo)
    res = await restarted.approve_and_execute(aid, decided_by="usr_admin", approved=True, note="ok")
    assert (res.status, res.decided_by, res.decision_note) == (
        ApprovalStatus.EXECUTED,
        "usr_admin",
        "ok",
    )
    assert len(read_p.calls) == 1
    assert read_p.calls[0][1].allowed_file_ids == ("fil_1",)
    assert read_p.calls[0][1].principal_scopes == (ApiScope.TOOLS_SAFE,)
    assert "[REDACTED_SECRET]" in repo.audits[-1].output_preview

    with pytest.raises(ValidationFailedError):
        await restarted.approve_and_execute(aid, decided_by="usr_admin", approved=True)
    with pytest.raises(AuthorizationError):
        await restarted.invoke("read_secret", {}, _ctx(ApiScope.TOOLS_SAFE, approval_id=aid))
    with pytest.raises(NotFoundError):
        await restarted.approve_and_execute("apr_missing", decided_by="usr_admin", approved=True)
    with pytest.raises(ValidationFailedError):
        await restarted.approve_and_execute(aid, decided_by=" ", approved=True)

    rec_cli = ToolApprovalRecord(
        project_id="prj_1",
        session_id="ses_1",
        tool_name="read_secret",
        arguments_json="{}",
        permission_tier=PermissionTier.READ,
        reason="client",
        status=ApprovalStatus.APPROVED,
        requested_by="usr_1",
        execution_target=ToolExecutionTarget.CLIENT,
        client_app_id="app_1",
    )
    await repo.create_approval(rec_cli)
    with pytest.raises(AuthorizationError):
        await restarted.invoke("read_secret", {}, _ctx(ApiScope.TOOLS_SAFE, approval_id=rec_cli.id))
    assert repo.audits[-1].error == "approval_target_mismatch"
