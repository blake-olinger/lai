"""Client tool result validation, scrubbing, guardrail scanning, and CAS consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.base import utc_now
from lai.domain.errors import (
    AuthorizationError,
    GuardrailBlockedError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationFailedError,
)
from lai.domain.guardrails import GuardrailDirection, GuardrailVerdict
from lai.domain.tools import (
    ApprovalStatus,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
    ToolExecutionTarget,
    ToolResult,
)
from lai.observability.logging import get_request_id
from lai.observability.scrubber import scrub_sensitive_text

if TYPE_CHECKING:
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.repositories import ToolAuditRepositoryPort

_MAX_AUDIT_PREVIEW_CHARS = 500


async def _record_client_audit(  # noqa: PLR0913
    audit_repo: ToolAuditRepositoryPort,
    record: ToolApprovalRecord,
    *,
    principal_id: str,
    success: bool,
    output_preview: str = "",
    error: str | None = None,
    duration_ms: float = 0.0,
) -> None:
    """Append an immutable ToolAuditRecord for a client tool lifecycle transition."""
    await audit_repo.record_audit(
        ToolAuditRecord(
            project_id=record.project_id,
            session_id=record.session_id,
            principal_id=principal_id,
            tool_name=record.tool_name,
            permission_tier=record.permission_tier,
            policy_decision=PolicyDecision.ALLOW,
            arguments_json=record.arguments_json,
            success=success,
            output_preview=output_preview,
            error=error,
            approval_id=record.id,
            duration_ms=duration_ms,
        )
    )


async def _load_approved_client_record(
    audit_repo: ToolAuditRepositoryPort,
    *,
    approval_id: str,
    client_app_id: str,
    principal_id: str,
    is_admin: bool = False,
) -> ToolApprovalRecord:
    """Load and validate an active, non-expired APPROVED client approval record."""
    record = await audit_repo.get_approval(approval_id)
    if record is None:
        msg = f"Tool approval {approval_id!r} not found."
        raise NotFoundError(msg, details={"approval_id": approval_id})
    if record.execution_target is not ToolExecutionTarget.CLIENT:
        msg = f"Tool approval {approval_id!r} is not a client tool invocation."
        raise ValidationFailedError(
            msg,
            details={"approval_id": approval_id, "execution_target": record.execution_target.value},
        )
    if not client_app_id or record.client_app_id != client_app_id:
        msg = (
            f"Client app {client_app_id!r} is not authorized to submit result "
            f"for approval {approval_id!r}."
        )
        raise AuthorizationError(
            msg,
            details={"approval_id": approval_id, "client_app_id": client_app_id},
        )
    if not is_admin and record.requested_by != principal_id:
        msg = (
            f"Principal {principal_id!r} is not authorized to submit result "
            f"for approval {approval_id!r}."
        )
        raise AuthorizationError(
            msg,
            details={"approval_id": approval_id, "principal_id": principal_id},
        )
    if record.status is not ApprovalStatus.APPROVED:
        msg = f"Tool approval {approval_id!r} is {record.status.value!r}, expected 'approved'."
        raise ValidationFailedError(
            msg,
            details={"approval_id": approval_id, "status": record.status.value},
        )
    if record.expires_at is not None and utc_now() >= record.expires_at:
        expired = record.model_copy(update={"status": ApprovalStatus.EXPIRED})
        await audit_repo.transition_approval_status(
            expired, expected_status=ApprovalStatus.APPROVED
        )
        await _record_client_audit(
            audit_repo,
            record,
            principal_id=principal_id,
            success=False,
            error="approval_expired",
        )
        msg = f"Tool approval {approval_id!r} has expired."
        raise ValidationFailedError(
            msg,
            details={"approval_id": approval_id, "status": ApprovalStatus.EXPIRED.value},
        )
    return record


def _sanitize_client_result(
    result: ToolResult,
    *,
    tool_name: str,
    max_output_chars: int,
    max_payload_bytes: int,
    approval_id: str,
) -> ToolResult:
    """Enforce payload byte cap, truncate to max_output_chars, and scrub secrets/PII."""
    payload_bytes = len(result.output.encode("utf-8")) + (
        len(result.error.encode("utf-8")) if result.error is not None else 0
    )
    if payload_bytes > max_payload_bytes:
        msg = (
            f"Client tool result payload ({payload_bytes} bytes) exceeds "
            f"limit ({max_payload_bytes} bytes)."
        )
        raise PayloadTooLargeError(
            msg,
            details={"approval_id": approval_id, "max_payload_bytes": str(max_payload_bytes)},
        )

    scrubbed_full = scrub_sensitive_text(result.output, max_chars=max(len(result.output) * 4, 1))
    truncated = result.truncated or len(scrubbed_full) > max_output_chars
    scrubbed_output = (
        scrub_sensitive_text(result.output, max_chars=max_output_chars)
        if len(scrubbed_full) > max_output_chars
        else scrubbed_full
    )
    scrubbed_error = (
        scrub_sensitive_text(result.error, max_chars=max_output_chars)
        if result.error is not None
        else None
    )
    return result.model_copy(
        update={
            "tool_name": tool_name,
            "output": scrubbed_output,
            "error": scrubbed_error,
            "truncated": truncated,
        }
    )


async def _scan_client_result_guardrails(
    guardrail_pipeline: GuardrailPipelinePort | None,
    audit_repo: ToolAuditRepositoryPort,
    record: ToolApprovalRecord,
    sanitized_result: ToolResult,
    *,
    principal_id: str,
) -> None:
    """Scan scrubbed client stdout + error text against guardrails to block prompt injection."""
    if guardrail_pipeline is None:
        return
    scan_text = "\n".join(
        part for part in (sanitized_result.output, sanitized_result.error) if part
    )
    if not scan_text:
        return

    blocked_exc: GuardrailBlockedError | None = None
    try:
        decision = await guardrail_pipeline.check_prompt(
            scan_text,
            direction=GuardrailDirection.TOOL_OUTPUT,
            request_id=get_request_id(),
            session_id=record.session_id,
            project_id=record.project_id,
            principal_id=principal_id,
        )
        if decision.verdict is GuardrailVerdict.BLOCK:
            blocked_exc = GuardrailBlockedError(
                "Client tool output blocked by safety guardrail policy.",
                violated_policy=decision.violated_policy,
                details={
                    "approval_id": record.id,
                    "provider": decision.provider,
                    "direction": GuardrailDirection.TOOL_OUTPUT.value,
                },
            )
    except GuardrailBlockedError as exc:
        blocked_exc = exc

    if blocked_exc is None:
        return

    blocked_record = record.model_copy(
        update={
            "status": ApprovalStatus.EXECUTED,
            "decision_note": "blocked_by_guardrail",
        }
    )
    transitioned = await audit_repo.transition_approval_status(
        blocked_record, expected_status=ApprovalStatus.APPROVED
    )
    if not transitioned:
        msg = f"Tool approval {record.id!r} was already consumed."
        raise ValidationFailedError(msg, details={"approval_id": record.id}) from blocked_exc
    await _record_client_audit(
        audit_repo,
        record,
        principal_id=principal_id,
        success=False,
        error="output_blocked_by_guardrail",
        duration_ms=sanitized_result.duration_ms,
    )
    raise blocked_exc


async def submit_client_tool_result(  # noqa: PLR0913
    *,
    audit_repo: ToolAuditRepositoryPort,
    guardrail_pipeline: GuardrailPipelinePort | None,
    approval_id: str,
    client_app_id: str,
    result: ToolResult,
    principal_id: str,
    is_admin: bool = False,
    max_output_chars: int,
    max_payload_bytes: int,
) -> ToolResult:
    """Validate, sanitize, guardrail-scan, and CAS-consume an approved client tool result."""
    record = await _load_approved_client_record(
        audit_repo,
        approval_id=approval_id,
        client_app_id=client_app_id,
        principal_id=principal_id,
        is_admin=is_admin,
    )
    sanitized = _sanitize_client_result(
        result,
        tool_name=record.tool_name,
        max_output_chars=max_output_chars,
        max_payload_bytes=max_payload_bytes,
        approval_id=approval_id,
    )
    await _scan_client_result_guardrails(
        guardrail_pipeline,
        audit_repo,
        record,
        sanitized,
        principal_id=principal_id,
    )
    executed_record = record.model_copy(
        update={
            "status": ApprovalStatus.EXECUTED,
            "execution_result": sanitized,
        }
    )
    transitioned = await audit_repo.transition_approval_status(
        executed_record, expected_status=ApprovalStatus.APPROVED
    )
    if not transitioned:
        msg = f"Tool approval {approval_id!r} was already consumed."
        raise ValidationFailedError(msg, details={"approval_id": approval_id})

    preview = scrub_sensitive_text(sanitized.output, max_chars=_MAX_AUDIT_PREVIEW_CHARS)
    await _record_client_audit(
        audit_repo,
        record,
        principal_id=principal_id,
        success=sanitized.success,
        output_preview=preview,
        error=sanitized.error,
        duration_ms=sanitized.duration_ms,
    )
    return sanitized
