"""SQLite repository for tool approvals and immutable execution audit trails."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lai.db._codecs import clamp_limit, format_dt, format_opt_dt, parse_dt, parse_opt_dt
from lai.domain.errors import NotFoundError
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
)
from lai.ports.repositories import ToolAuditRepositoryPort

if TYPE_CHECKING:
    import aiosqlite

    from lai.db.connection import SqliteDatabase

_APPROVAL_COLUMNS = (
    "id, project_id, session_id, tool_name, arguments_json, permission_tier, reason, "
    "status, requested_by, decided_by, decision_note, created_at, decided_at"
)
_AUDIT_COLUMNS = (
    "id, project_id, session_id, principal_id, tool_name, permission_tier, "
    "policy_decision, arguments_json, success, output_preview, error, "
    "approval_id, duration_ms, created_at"
)


def _row_to_approval(row: aiosqlite.Row) -> ToolApprovalRecord:
    """Map a tool_approvals SQLite row to a ToolApprovalRecord domain model."""
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
        await self._db.execute_write(
            """
            INSERT INTO tool_approvals (
                id, project_id, session_id, tool_name, arguments_json,
                permission_tier, reason, status, requested_by,
                decided_by, decision_note, created_at, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            SET status = ?, decided_by = ?, decision_note = ?, decided_at = ?
            WHERE id = ?;
            """,
            (
                record.status.value,
                record.decided_by,
                record.decision_note,
                format_opt_dt(record.decided_at),
                record.id,
            ),
        )
        if updated == 0:
            msg = f"Tool approval record not found: {record.id}"
            raise NotFoundError(msg, details={"approval_id": record.id})
        return record

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
