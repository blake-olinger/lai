"""SQLite repository for tool approvals and immutable execution audit trails."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from lai.db._codecs import clamp_limit, format_dt, format_opt_dt, parse_dt, parse_opt_dt
from lai.domain.auth import ApiScope
from lai.domain.errors import NotFoundError
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
    ToolExecutionTarget,
    ToolResult,
)
from lai.ports.repositories import ToolAuditRepositoryPort

if TYPE_CHECKING:
    import aiosqlite

    from lai.db.connection import SqliteDatabase

_APPROVAL_COLUMNS = (
    "id, project_id, session_id, tool_name, arguments_json, permission_tier, reason, "
    "status, requested_by, decided_by, decision_note, created_at, decided_at, "
    "execution_target, client_app_id, tool_call_id, allowed_file_ids_json, "
    "principal_scopes_json, expires_at, execution_result_json"
)
_AUDIT_COLUMNS = (
    "id, project_id, session_id, principal_id, tool_name, permission_tier, "
    "policy_decision, arguments_json, success, output_preview, error, "
    "approval_id, duration_ms, created_at"
)


def _row_to_approval(row: aiosqlite.Row) -> ToolApprovalRecord:
    """Map a tool_approvals SQLite row to a ToolApprovalRecord domain model."""
    raw_file_ids = cast("list[object]", json.loads(str(row["allowed_file_ids_json"])))
    raw_scopes = cast("list[str]", json.loads(str(row["principal_scopes_json"])))
    raw_result = cast("str | None", row["execution_result_json"])
    execution_result = (
        ToolResult.model_validate_json(raw_result) if raw_result is not None else None
    )
    return ToolApprovalRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        session_id=str(row["session_id"]),
        tool_name=str(row["tool_name"]),
        arguments_json=str(row["arguments_json"]),
        permission_tier=PermissionTier(str(row["permission_tier"])),
        reason=str(row["reason"]),
        status=ApprovalStatus(str(row["status"])),
        requested_by=str(row["requested_by"]),
        decided_by=cast("str | None", row["decided_by"]),
        decision_note=cast("str | None", row["decision_note"]),
        created_at=parse_dt(str(row["created_at"])),
        decided_at=parse_opt_dt(cast("str | None", row["decided_at"])),
        execution_target=ToolExecutionTarget(str(row["execution_target"])),
        client_app_id=cast("str | None", row["client_app_id"]),
        tool_call_id=cast("str | None", row["tool_call_id"]),
        allowed_file_ids=tuple(str(item) for item in raw_file_ids),
        principal_scopes=tuple(ApiScope(item) for item in raw_scopes),
        expires_at=parse_opt_dt(cast("str | None", row["expires_at"])),
        execution_result=execution_result,
    )


def _approval_mutation_params(
    record: ToolApprovalRecord,
) -> tuple[str, str | None, str | None, str | None, str | None, str | None]:
    """Build parameter tuple for updating mutable approval decision and result columns."""
    result_json = (
        record.execution_result.model_dump_json() if record.execution_result is not None else None
    )
    return (
        record.status.value,
        record.decided_by,
        record.decision_note,
        format_opt_dt(record.decided_at),
        format_opt_dt(record.expires_at),
        result_json,
    )


def _row_to_audit(row: aiosqlite.Row) -> ToolAuditRecord:
    """Map a tool_audits SQLite row to a ToolAuditRecord domain model."""
    return ToolAuditRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        session_id=str(row["session_id"]),
        principal_id=str(row["principal_id"]),
        tool_name=str(row["tool_name"]),
        permission_tier=PermissionTier(str(row["permission_tier"])),
        policy_decision=PolicyDecision(str(row["policy_decision"])),
        arguments_json=str(row["arguments_json"]),
        success=bool(int(row["success"])),
        output_preview=str(row["output_preview"]),
        error=cast("str | None", row["error"]),
        approval_id=cast("str | None", row["approval_id"]),
        duration_ms=float(row["duration_ms"]),
        created_at=parse_dt(str(row["created_at"])),
    )


class SqliteToolAuditRepository(ToolAuditRepositoryPort):
    """SQLite persistence adapter implementing ToolAuditRepositoryPort."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def create_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        """Persist a pending tool approval request."""
        file_ids_json = json.dumps(list(record.allowed_file_ids))
        scopes_json = json.dumps([scope.value for scope in record.principal_scopes])
        result_json = (
            record.execution_result.model_dump_json()
            if record.execution_result is not None
            else None
        )
        await self._db.execute_write(
            """
            INSERT INTO tool_approvals (
                id, project_id, session_id, tool_name, arguments_json,
                permission_tier, reason, status, requested_by,
                decided_by, decision_note, created_at, decided_at,
                execution_target, client_app_id, tool_call_id, allowed_file_ids_json,
                principal_scopes_json, expires_at, execution_result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.id,
                record.project_id,
                record.session_id,
                record.tool_name,
                record.arguments_json,
                record.permission_tier.value,
                record.reason,
                record.status.value,
                record.requested_by,
                record.decided_by,
                record.decision_note,
                format_dt(record.created_at),
                format_opt_dt(record.decided_at),
                record.execution_target.value,
                record.client_app_id,
                record.tool_call_id,
                file_ids_json,
                scopes_json,
                format_opt_dt(record.expires_at),
                result_json,
            ),
        )
        return record

    async def get_approval(self, approval_id: str) -> ToolApprovalRecord | None:
        """Fetch a tool approval record by ID."""
        row = await self._db.fetch_one(
            f"SELECT {_APPROVAL_COLUMNS} FROM tool_approvals WHERE id = ?;",  # noqa: S608
            (approval_id,),
        )
        return _row_to_approval(row) if row is not None else None

    async def list_pending_approvals(
        self,
        project_id: str,
        session_id: str | None = None,
    ) -> list[ToolApprovalRecord]:
        """List pending approval requests for a project or session."""
        rows = await self._db.fetch_all(
            f"SELECT {_APPROVAL_COLUMNS} FROM tool_approvals "  # noqa: S608
            "WHERE project_id = ? AND status = ? "
            "AND (? IS NULL OR session_id = ?) "
            "ORDER BY created_at ASC, rowid ASC;",
            (project_id, ApprovalStatus.PENDING.value, session_id, session_id),
        )
        return [_row_to_approval(row) for row in rows]

    async def update_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        """Persist an updated approval decision state."""
        updated = await self._db.execute_write(
            """
            UPDATE tool_approvals
            SET status = ?, decided_by = ?, decision_note = ?, decided_at = ?,
                expires_at = ?, execution_result_json = ?
            WHERE id = ?;
            """,
            (*_approval_mutation_params(record), record.id),
        )
        if updated == 0:
            msg = f"Tool approval record not found: {record.id}"
            raise NotFoundError(msg, details={"approval_id": record.id})
        return record

    async def transition_approval_status(
        self,
        record: ToolApprovalRecord,
        *,
        expected_status: ApprovalStatus,
    ) -> bool:
        """Atomically update approval state only when current status matches expected_status."""
        updated = await self._db.execute_write(
            """
            UPDATE tool_approvals
            SET status = ?, decided_by = ?, decision_note = ?, decided_at = ?,
                expires_at = ?, execution_result_json = ?
            WHERE id = ? AND status = ?;
            """,
            (*_approval_mutation_params(record), record.id, expected_status.value),
        )
        return updated == 1

    async def record_audit(self, record: ToolAuditRecord) -> ToolAuditRecord:
        """Append an immutable tool execution audit entry."""
        await self._db.execute_write(
            """
            INSERT INTO tool_audits (
                id, project_id, session_id, principal_id, tool_name,
                permission_tier, policy_decision, arguments_json, success,
                output_preview, error, approval_id, duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.id,
                record.project_id,
                record.session_id,
                record.principal_id,
                record.tool_name,
                record.permission_tier.value,
                record.policy_decision.value,
                record.arguments_json,
                1 if record.success else 0,
                record.output_preview,
                record.error,
                record.approval_id,
                record.duration_ms,
                format_dt(record.created_at),
            ),
        )
        return record

    async def list_audits(
        self,
        project_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[ToolAuditRecord]:
        """List recent tool audit entries for a project or session."""
        rows = await self._db.fetch_all(
            f"SELECT {_AUDIT_COLUMNS} FROM tool_audits "  # noqa: S608
            "WHERE project_id = ? AND (? IS NULL OR session_id = ?) "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?;",
            (project_id, session_id, session_id, clamp_limit(limit)),
        )
        return [_row_to_audit(row) for row in rows]
