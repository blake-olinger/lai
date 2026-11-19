"""Mutation-killing tests for tool dispatch, executor caps, RRF, and fair queue."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from lai.domain.auth import ApiScope
from lai.domain.errors import (
    AuthorizationError,
    ClientToolDispatchError,
    ToolApprovalRequiredError,
    ValidationFailedError,
)
from lai.domain.files import FileChunk, ScoredChunk
from lai.domain.scheduler import JobPriority, LeaseRequest
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
from lai.ingestion.rrf import reciprocal_rank_fusion
from lai.ports.repositories import ToolAuditRepositoryPort
from lai.ports.tools import ToolPluginPort
from lai.scheduler import FairPriorityQueue
from lai.tools import GuardedToolExecutor, InMemoryToolRegistry
from lai.tools.plugins._scoping import optional_int

if TYPE_CHECKING:
    from collections.abc import Generator

_CACHE_CAP = 1024
_DEFAULT_LIMIT = 10
_SEVEN = 7


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


class _MemAuditRepo(ToolAuditRepositoryPort):
    """Minimal in-memory ToolAuditRepositoryPort with configurable second-get None."""

    def __init__(self, *, return_none_on_second_get: bool = False) -> None:
        self.approvals: dict[str, ToolApprovalRecord] = {}
        self.audits: list[ToolAuditRecord] = []
        self._return_none_on_second_get = return_none_on_second_get
        self._get_calls = 0

    async def create_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        self.approvals[record.id] = record
        return record

    async def get_approval(self, approval_id: str) -> ToolApprovalRecord | None:
        self._get_calls += 1
        if self._return_none_on_second_get and self._get_calls >= 3:
            return None
        return self.approvals.get(approval_id)

    async def list_pending_approvals(
        self,
        project_id: str,
        session_id: str | None = None,
    ) -> list[ToolApprovalRecord]:
        del project_id, session_id
        return [r for r in self.approvals.values() if r.status is ApprovalStatus.PENDING]

    async def update_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        self.approvals[record.id] = record
        return record

    async def transition_approval_status(
        self,
        record: ToolApprovalRecord,
        *,
        expected_status: ApprovalStatus,
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
        self,
        project_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolAuditRecord]:
        del project_id, session_id
        return self.audits[:limit]


class _ServerTool(ToolPluginPort):
    """Configurable server tool plugin capturing context and optionally failing."""

    def __init__(
        self,
        name: str,
        *,
        tier: PermissionTier = PermissionTier.READ,
        requires_approval: bool = False,
        fail_result: bool = False,
    ) -> None:
        self._name = name
        self._tier = tier
        self._requires_approval = requires_approval
        self._fail_result = fail_result
        self.seen_allowed_files: list[tuple[str, ...]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"Test tool {self._name}",
            permission_tier=self._tier,
            requires_approval_by_default=self._requires_approval,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments
        self.seen_allowed_files.append(context.allowed_file_ids)
        if self._fail_result:
            return ToolResult(tool_name=self._name, success=False, output="partial", error="boom")
        return ToolResult(tool_name=self._name, success=True, output="ok")


def _ctx(
    *scopes: ApiScope,
    approval_id: str | None = None,
    allowed_file_ids: tuple[str, ...] = ("fil_1",),
) -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id="usr_1",
        principal_scopes=scopes,
        allowed_file_ids=allowed_file_ids,
        approved_approval_id=approval_id,
        client_app_id="app_1",
        client_allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
        tool_call_id="tc_1",
    )


def _client_specs() -> dict[str, ToolSpec]:
    return {
        "cli_read": ToolSpec(
            name="cli_read",
            description="Client read",
            permission_tier=PermissionTier.READ,
            execution_target=ToolExecutionTarget.CLIENT,
        ),
        "cli_exec": ToolSpec(
            name="cli_exec",
            description="Client exec",
            permission_tier=PermissionTier.EXEC,
            execution_target=ToolExecutionTarget.CLIENT,
        ),
    }


async def test_client_dispatch_and_server_audit_booleans() -> None:
    """Assert audit.success booleans across client dispatch and server tool outcomes."""
    repo = _MemAuditRepo()
    registry = InMemoryToolRegistry()
    registry.register(_ServerTool("srv_exec", tier=PermissionTier.EXEC))
    registry.register(_ServerTool("srv_read", tier=PermissionTier.READ))
    registry.register(_ServerTool("srv_fail", tier=PermissionTier.READ, fail_result=True))
    registry.register(_ServerTool("srv_gated", tier=PermissionTier.READ, requires_approval=True))

    client_tools = _client_specs()
    executor = GuardedToolExecutor(registry, repo)

    with pytest.raises(AuthorizationError):
        await executor.invoke("cli_exec", {}, _ctx(ApiScope.TOOLS_SAFE), client_tools=client_tools)
    assert repo.audits[-1].success is False

    with pytest.raises(AuthorizationError):
        await executor.invoke(
            "cli_exec",
            {},
            _ctx(ApiScope.TOOLS_EXEC, approval_id="apr_forged"),
            client_tools=client_tools,
        )
    assert repo.audits[-1].success is False

    with pytest.raises(ClientToolDispatchError):
        await executor.invoke("cli_exec", {}, _ctx(ApiScope.TOOLS_EXEC), client_tools=client_tools)
    assert repo.audits[-1].success is False

    with pytest.raises(ClientToolDispatchError):
        await executor.invoke("cli_read", {}, _ctx(ApiScope.TOOLS_SAFE), client_tools=client_tools)
    assert repo.audits[-1].success is True

    with pytest.raises(AuthorizationError):
        await executor.invoke("srv_exec", {}, _ctx(ApiScope.TOOLS_EXEC))
    assert repo.audits[-1].success is False
    assert repo.audits[-1].error == "server_host_tool_forbidden"

    with pytest.raises(AuthorizationError):
        await executor.invoke("srv_read", {}, _ctx())
    assert repo.audits[-1].success is False
    assert repo.audits[-1].error == "policy_denied"

    fail_res = await executor.invoke("srv_fail", {}, _ctx(ApiScope.TOOLS_SAFE))
    assert fail_res.success is False
    assert repo.audits[-1].success is False
    assert repo.audits[-1].error == "boom"

    with pytest.raises(ToolApprovalRequiredError):
        await executor.invoke("srv_gated", {}, _ctx(ApiScope.TOOLS_SAFE))
    assert repo.audits[-1].success is False


async def test_executor_fallback_and_1024_context_eviction() -> None:
    """Assert approve_and_execute post-get None fallback and exact 1024-entry LRU eviction."""
    fallback_repo = _MemAuditRepo(return_none_on_second_get=True)
    fb_registry = InMemoryToolRegistry()
    fb_registry.register(_ServerTool("srv_gated", tier=PermissionTier.READ, requires_approval=True))
    fallback_exec = GuardedToolExecutor(fb_registry, fallback_repo)
    with pytest.raises(ToolApprovalRequiredError) as fb_exc:
        await fallback_exec.invoke("srv_gated", {}, _ctx(ApiScope.TOOLS_SAFE))
    approved = await fallback_exec.approve_and_execute(
        fb_exc.value.approval_id,
        decided_by="admin_1",
        approved=True,
    )
    assert approved is not None
    assert approved.status is ApprovalStatus.EXECUTED

    cap_repo = _MemAuditRepo()
    cap_tool = _ServerTool("srv_cap", tier=PermissionTier.READ, requires_approval=True)
    cap_registry = InMemoryToolRegistry()
    cap_registry.register(cap_tool)
    cap_exec = GuardedToolExecutor(cap_registry, cap_repo)
    approval_ids: list[str] = []
    for _ in range(_CACHE_CAP + 1):
        with pytest.raises(ToolApprovalRequiredError) as exc_info:
            await cap_exec.invoke(
                "srv_cap",
                {},
                _ctx(ApiScope.TOOLS_SAFE, allowed_file_ids=("fil_from_ctx",)),
            )
        approval_ids.append(exc_info.value.approval_id)

    for aid in (approval_ids[0], approval_ids[1]):
        cap_repo.approvals[aid] = cap_repo.approvals[aid].model_copy(
            update={"allowed_file_ids": ("fil_from_db",)},
        )
    await cap_exec.approve_and_execute(approval_ids[0], decided_by="adm", approved=True)
    await cap_exec.approve_and_execute(approval_ids[1], decided_by="adm", approved=True)
    assert cap_tool.seen_allowed_files[0] == ("fil_from_db",)
    assert cap_tool.seen_allowed_files[1] == ("fil_from_ctx",)


def test_optional_int_rrf_and_fair_queue_boundaries() -> None:
    """Kill _scoping, RRF limit=0/1, and FairPriorityQueue empty/decrement boundaries."""
    assert optional_int({}, "limit", _DEFAULT_LIMIT) == _DEFAULT_LIMIT
    assert optional_int({"limit": _SEVEN}, "limit", _DEFAULT_LIMIT) == _SEVEN
    with pytest.raises(ValidationFailedError):
        optional_int({"limit": True}, "limit", _DEFAULT_LIMIT)
    with pytest.raises(ValidationFailedError):
        optional_int({"limit": "7"}, "limit", _DEFAULT_LIMIT)

    chunk1 = FileChunk(
        id="c1",
        file_id="f1",
        project_id="p1",
        chunk_index=0,
        start_line=1,
        end_line=2,
        token_count=4,
        text="one",
    )
    chunk2 = FileChunk(
        id="c2",
        file_id="f1",
        project_id="p1",
        chunk_index=1,
        start_line=3,
        end_line=4,
        token_count=4,
        text="two",
    )
    hits = [
        ScoredChunk(chunk=chunk1, score=1.0, source="fts"),
        ScoredChunk(chunk=chunk2, score=0.5, source="fts"),
    ]
    assert reciprocal_rank_fusion(hits, [], limit=0) == []
    assert len(reciprocal_rank_fusion(hits, [], limit=1)) == 1

    queue = FairPriorityQueue(max_queued_jobs=8, max_queued_jobs_per_principal=4)
    assert queue.dequeue() is None
    queue.enqueue(
        LeaseRequest(
            job_id="j_mr",
            session_id="s1",
            project_id="p1",
            principal_id="u1",
            priority=JobPriority.MAP_REDUCE,
        ),
    )
    dequeued_mr = queue.dequeue()
    assert dequeued_mr is not None
    assert dequeued_mr.job_id == "j_mr"
    assert queue.queued_count_for_principal("u1") == 0

    for jid in ("j1", "j2"):
        queue.enqueue(LeaseRequest(job_id=jid, session_id="s1", project_id="p1", principal_id="u1"))
    assert queue.dequeue() is not None
    assert queue.queued_count_for_principal("u1") == 1
    assert queue.dequeue() is not None
    assert queue.queued_count_for_principal("u1") == 0
