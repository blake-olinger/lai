"""Pure argument serialization, approval matching, and audit record helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from lai.domain.errors import ValidationFailedError
from lai.domain.tools import ApprovalStatus, ToolAuditRecord
from lai.observability.scrubber import scrub_sensitive_text

if TYPE_CHECKING:
    from lai.domain.tools import (
        PolicyDecision,
        ToolApprovalRecord,
        ToolExecutionContext,
        ToolSpec,
    )

_MAX_OUTPUT_PREVIEW_CHARS = 500
_MS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class InvocationMeta:
    """Immutable metadata shared across policy, approval, and audit recording."""

    context: ToolExecutionContext
    spec: ToolSpec
    decision: PolicyDecision
    arguments_json: str
    approval_id: str | None


def normalize_optional_id(value: str | None) -> str | None:
    """Strip optional identifiers and treat whitespace-only strings as None."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def serialize_arguments(arguments: dict[str, Any]) -> str:
    """Deterministically serialize tool arguments to JSON."""
    return json.dumps(arguments, sort_keys=True, default=str)


def deserialize_arguments(arguments_json: str) -> dict[str, Any]:
    """Parse stored approval arguments JSON and ensure it is a mapping."""
    try:
        parsed: object = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        msg = "Stored approval arguments_json is not valid JSON."
        raise ValidationFailedError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "Stored approval arguments_json must decode to a JSON object."
        raise ValidationFailedError(msg)
    mapping = cast("dict[object, object]", parsed)
    return {str(key): value for key, value in mapping.items()}


def elapsed_ms(started: float) -> float:
    """Compute elapsed milliseconds since perf_counter start timestamp."""
    return max((time.perf_counter() - started) * _MS_PER_SECOND, 0.0)


def validate_approval_match(
    approval: ToolApprovalRecord | None,
    context: ToolExecutionContext,
    tool_name: str,
) -> str | None:
    """Return an error code if approval is missing, mismatched, or not APPROVED."""
    if approval is None:
        return "approval_not_found"
    if (approval.project_id, approval.session_id, approval.tool_name) != (
        context.project_id,
        context.session_id,
        tool_name,
    ):
        return "approval_context_mismatch"
    if approval.status is not ApprovalStatus.APPROVED:
        return f"approval_invalid_status:{approval.status.value}"
    return None


def build_audit_record(
    meta: InvocationMeta,
    *,
    success: bool,
    output_preview: str = "",
    error: str | None = None,
    duration_ms: float = 0.0,
) -> ToolAuditRecord:
    """Construct an immutable ToolAuditRecord with scrubbed output preview."""
    scrubbed_preview = (
        scrub_sensitive_text(output_preview, max_chars=_MAX_OUTPUT_PREVIEW_CHARS)
        if output_preview
        else ""
    )
    return ToolAuditRecord(
        project_id=meta.context.project_id,
        session_id=meta.context.session_id,
        principal_id=meta.context.principal_id,
        tool_name=meta.spec.name,
        permission_tier=meta.spec.permission_tier,
        policy_decision=meta.decision,
        arguments_json=meta.arguments_json,
        success=success,
        output_preview=scrubbed_preview,
        error=error,
        approval_id=meta.approval_id,
        duration_ms=duration_ms,
    )
