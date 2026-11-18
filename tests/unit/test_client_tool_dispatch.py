"""Unit tests for client tool dispatch, result validation, guardrails, and expiry."""

from __future__ import annotations

import hashlib
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from lai.domain.auth import ApiScope
from lai.domain.base import utc_now
from lai.domain.errors import (
    AuthorizationError,
    ClientToolDispatchError,
    GuardrailBlockedError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationFailedError,
)
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailVerdict,
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
from lai.guardrails import CompositeGuardrailPipeline
from lai.ports.guardrails import GuardrailProviderPort
from lai.ports.repositories import ToolAuditRepositoryPort
from lai.tools import GuardedToolExecutor, InMemoryToolRegistry

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


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


class _SubstringBlocker(GuardrailProviderPort):
    @property
    def name(self) -> str:
        return "substr-blocker"

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if "INJECT_SYSTEM" in prompt:
            return GuardrailDecision(
                verdict=GuardrailVerdict.BLOCK,
                provider=self.name,
                violated_policy="indirect_prompt_injection",
                reason="Detected prompt injection in tool output.",
                prompt_sha256=sha,
                prompt_chars=len(prompt),
                scrubbed_prompt=prompt,
            )
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            prompt_sha256=sha,
            prompt_chars=len(prompt),
            scrubbed_prompt=prompt,
        )


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _cli_ctx(*scopes: ApiScope, approval_id: str | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id="usr_1",
        principal_scopes=scopes,
        allowed_file_ids=("fil_1",),
        approved_approval_id=approval_id,
        client_app_id="app_1",
        client_allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
        tool_call_id="tc_cli_1",
    )


def _client_specs() -> dict[str, ToolSpec]:
    return {
        "read_local": ToolSpec(
            name="read_local",
            description="Read local workspace",
            permission_tier=PermissionTier.READ,
            execution_target=ToolExecutionTarget.CLIENT,
        ),
        "run_local": ToolSpec(
            name="run_local",
            description="Run local command",
            permission_tier=PermissionTier.EXEC,
            execution_target=ToolExecutionTarget.CLIENT,
        ),
    }


async def test_client_dispatch_lease_scrub_truncation_and_payload_cap() -> None:
    """Exercise client dispatch, replay guard, lease issuance, 413 cap, and scrubbing."""
    repo = _FakeToolAuditRepo()
    client_map = _client_specs()
    executor = GuardedToolExecutor(
        InMemoryToolRegistry(),
        repo,
        lease_ttl_seconds=60.0,
        max_output_chars=25,
        max_payload_bytes=64,
    )

    with pytest.raises(AuthorizationError):
        await executor.invoke(
            "run_local", {"cmd": "id"}, _cli_ctx(ApiScope.TOOLS_SAFE), client_tools=client_map
        )
    assert repo.audits[-1].policy_decision is PolicyDecision.DENY

    with pytest.raises(AuthorizationError):
        await executor.invoke(
            "run_local",
            {"cmd": "id"},
            _cli_ctx(ApiScope.TOOLS_EXEC, approval_id="apr_forged"),
            client_tools=client_map,
        )
    assert repo.audits[-1].error == "client_approval_replay_forbidden"

    with pytest.raises(ClientToolDispatchError) as read_exc:
        await executor.invoke(
            "read_local", {"path": "a.py"}, _cli_ctx(ApiScope.TOOLS_SAFE), client_tools=client_map
        )
    assert read_exc.value.requires_approval is False
    assert read_exc.value.expires_at is not None
    assert repo.audits[-1].error == "client_lease_issued"
    lease_id = read_exc.value.approval_id

    with pytest.raises(PayloadTooLargeError):
        await executor.submit_client_result(
            lease_id,
            "app_1",
            ToolResult(tool_name="read_local", success=True, output="x" * 65),
            principal_id="usr_1",
        )

    ok_res = await executor.submit_client_result(
        lease_id,
        "app_1",
        ToolResult(
            tool_name="read_local",
            success=True,
            output="Bearer lai_sk_1234567890abcd extra tail",
            error="err user@example.com",
        ),
        principal_id="usr_1",
    )
    assert ok_res.truncated is True
    assert len(ok_res.output) <= 25  # noqa: PLR2004
    assert "lai_sk_1234567890abcd" not in ok_res.output
    assert "[REDACTED_EMAIL]" in (ok_res.error or "")

    with pytest.raises(ValidationFailedError):
        await executor.submit_client_result(
            lease_id,
            "app_1",
            ToolResult(tool_name="read_local", success=True, output="again"),
            principal_id="usr_1",
        )


async def test_client_result_guardrails_errors_and_expiry(tmp_path: Path) -> None:
    """Verify guardrail block, missing/non-client/wrong-app/pending/expired result checks."""
    repo = _FakeToolAuditRepo()
    client_map = _client_specs()
    with CompositeGuardrailPipeline(
        GuardrailConfig(enabled=True, block_mode=GuardrailBlockMode.REFUSAL_MESSAGE),
        providers=[_SubstringBlocker()],
        log_dir=tmp_path / "glogs",
    ) as pipeline:
        executor = GuardedToolExecutor(
            InMemoryToolRegistry(), repo, guardrail_pipeline=pipeline, lease_ttl_seconds=60.0
        )

        with pytest.raises(ClientToolDispatchError) as exec_exc:
            await executor.invoke(
                "run_local", {"cmd": "cat"}, _cli_ctx(ApiScope.TOOLS_EXEC), client_tools=client_map
            )
        assert exec_exc.value.requires_approval is True
        exec_id = exec_exc.value.approval_id

        with pytest.raises(ValidationFailedError):
            await executor.submit_client_result(
                exec_id,
                "app_1",
                ToolResult(tool_name="run_local", success=True, output="ok"),
                principal_id="usr_1",
            )
        await executor.approve_and_execute(exec_id, decided_by="adm", approved=True)
        with pytest.raises(GuardrailBlockedError):
            await executor.submit_client_result(
                exec_id,
                "app_1",
                ToolResult(tool_name="run_local", success=True, output="INJECT_SYSTEM"),
                principal_id="usr_1",
            )
        blocked_rec = await repo.get_approval(exec_id)
        assert blocked_rec is not None
        assert (blocked_rec.status, blocked_rec.decision_note) == (
            ApprovalStatus.EXECUTED,
            "blocked_by_guardrail",
        )
        assert repo.audits[-1].error == "output_blocked_by_guardrail"

        with pytest.raises(NotFoundError):
            await executor.submit_client_result(
                "apr_missing",
                "app_1",
                ToolResult(tool_name="x", success=True, output="ok"),
                principal_id="usr_1",
            )
        srv_rec = ToolApprovalRecord(
            project_id="prj_1",
            session_id="ses_1",
            tool_name="s",
            arguments_json="{}",
            permission_tier=PermissionTier.READ,
            reason="srv",
            status=ApprovalStatus.APPROVED,
            requested_by="usr_1",
        )
        await repo.create_approval(srv_rec)
        with pytest.raises(ValidationFailedError):
            await executor.submit_client_result(
                srv_rec.id,
                "app_1",
                ToolResult(tool_name="s", success=True, output="ok"),
                principal_id="usr_1",
            )

        exp_rec = ToolApprovalRecord(
            project_id="prj_1",
            session_id="ses_1",
            tool_name="read_local",
            arguments_json="{}",
            permission_tier=PermissionTier.READ,
            reason="exp",
            status=ApprovalStatus.APPROVED,
            requested_by="usr_1",
            execution_target=ToolExecutionTarget.CLIENT,
            client_app_id="app_1",
            expires_at=utc_now() - timedelta(seconds=1),
        )
        await repo.create_approval(exp_rec)
        with pytest.raises(AuthorizationError):
            await executor.submit_client_result(
                exp_rec.id,
                "app_other",
                ToolResult(tool_name="read_local", success=True, output="ok"),
                principal_id="usr_1",
            )
        with pytest.raises(ValidationFailedError):
            await executor.submit_client_result(
                exp_rec.id,
                "app_1",
                ToolResult(tool_name="read_local", success=True, output="ok"),
                principal_id="usr_1",
            )
        expired_after = await repo.get_approval(exp_rec.id)
        assert expired_after is not None
        assert expired_after.status is ApprovalStatus.EXPIRED
        assert repo.audits[-1].error == "approval_expired"
