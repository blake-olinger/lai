"""Policy-enforcing, approval-gated, timeout-guarded, and audited tool executor."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from lai.domain.auth import ApiScope
from lai.domain.base import utc_now
from lai.domain.errors import (
    AuthorizationError,
    LaiError,
    NotFoundError,
    ToolApprovalRequiredError,
    ValidationFailedError,
)
from lai.domain.observability import AlertCategory
from lai.domain.tools import (
    ApprovalStatus,
    PolicyDecision,
    ToolApprovalRecord,
    ToolExecutionContext,
    ToolResult,
)
from lai.observability.logging import bind_log_context, get_component_logger
from lai.ports.tools import ToolExecutorPort
from lai.tools._audit import (
    InvocationMeta,
    build_audit_record,
    deserialize_arguments,
    elapsed_ms,
    normalize_optional_id,
    serialize_arguments,
    validate_approval_match,
)
from lai.tools.policy import DefaultToolPolicy

if TYPE_CHECKING:
    from lai.ports.repositories import ToolAuditRepositoryPort
    from lai.ports.tools import ToolPluginPort, ToolPolicyPort, ToolRegistryPort

_MAX_CACHED_CONTEXTS = 1024


class GuardedToolExecutor(ToolExecutorPort):
    """Executes registered tool plugins after policy, approval, and timeout checks."""

    def __init__(
        self,
        registry: ToolRegistryPort,
        audit_repo: ToolAuditRepositoryPort,
        policy: ToolPolicyPort | None = None,
    ) -> None:
        self._registry = registry
        self._audit_repo = audit_repo
        self._policy: ToolPolicyPort = policy or DefaultToolPolicy()
        self._pending_contexts: dict[str, ToolExecutionContext] = {}

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Evaluate policy, validate approval if provided, execute, and audit."""
        plugin = self._registry.get(tool_name)
        if plugin is None:
            msg = f"Tool {tool_name!r} is not registered."
            raise NotFoundError(msg, details={"tool_name": tool_name})

        spec = plugin.spec
        meta = InvocationMeta(
            context=context,
            spec=spec,
            decision=self._policy.evaluate(spec, arguments, context),
            arguments_json=serialize_arguments(arguments),
            approval_id=normalize_optional_id(context.approved_approval_id),
        )

        if meta.decision is PolicyDecision.DENY:
            await self._record(meta, success=False, error="policy_denied")
            msg = f"Principal {context.principal_id!r} is denied access to tool {tool_name!r}."
            raise AuthorizationError(
                msg,
                details={"tool_name": tool_name, "permission_tier": spec.permission_tier.value},
            )

        if meta.decision is PolicyDecision.REQUIRE_APPROVAL:
            await self._raise_approval_required(meta)

        if meta.approval_id is not None:
            await self._consume_approved_request(meta.approval_id, meta)

        return await self._execute_plugin(plugin, arguments, meta)

    async def approve_and_execute(
        self,
        approval_id: str,
        decided_by: str,
        approved: bool,
        note: str | None = None,
    ) -> ToolApprovalRecord:
        """Resolve a PENDING approval request and execute the tool if approved."""
        approver = decided_by.strip()
        if not approver:
            msg = "decided_by must be a non-empty principal identifier."
            raise ValidationFailedError(msg)

        record = await self._audit_repo.get_approval(approval_id)
        if record is None:
            msg = f"Tool approval {approval_id!r} not found."
            raise NotFoundError(msg, details={"approval_id": approval_id})
        if record.status is not ApprovalStatus.PENDING:
            msg = f"Tool approval {approval_id!r} is already {record.status.value!r}."
            raise ValidationFailedError(
                msg,
                details={"approval_id": approval_id, "status": record.status.value},
            )

        decided_at = utc_now()
        if not approved:
            self._pending_contexts.pop(approval_id, None)
            denied = record.model_copy(
                update={
                    "status": ApprovalStatus.DENIED,
                    "decided_by": approver,
                    "decision_note": note,
                    "decided_at": decided_at,
                }
            )
            return await self._audit_repo.update_approval(denied)

        arguments = deserialize_arguments(record.arguments_json)
        approved_record = record.model_copy(
            update={
                "status": ApprovalStatus.APPROVED,
                "decided_by": approver,
                "decision_note": note,
                "decided_at": decided_at,
            }
        )
        saved_approved = await self._audit_repo.update_approval(approved_record)

        exec_context = self._resolve_execution_context(saved_approved)
        await self.invoke(saved_approved.tool_name, arguments, exec_context)
        refreshed = await self._audit_repo.get_approval(approval_id)
        return refreshed or saved_approved.model_copy(update={"status": ApprovalStatus.EXECUTED})

    def _resolve_execution_context(self, record: ToolApprovalRecord) -> ToolExecutionContext:
        """Reuse the original pending context or construct a scoped fallback context."""
        cached = self._pending_contexts.pop(record.id, None)
        if cached is not None:
            return cached.model_copy(update={"approved_approval_id": record.id})
        return ToolExecutionContext(
            project_id=record.project_id,
            session_id=record.session_id,
            principal_id=record.requested_by,
            principal_scopes=(ApiScope.TOOLS_EXEC, ApiScope.TOOLS_SAFE),
            approved_approval_id=record.id,
        )

    async def _raise_approval_required(self, meta: InvocationMeta) -> None:
        """Persist a pending approval + audit record and raise ToolApprovalRequiredError."""
        spec = meta.spec
        approval = ToolApprovalRecord(
            project_id=meta.context.project_id,
            session_id=meta.context.session_id,
            tool_name=spec.name,
            arguments_json=meta.arguments_json,
            permission_tier=spec.permission_tier,
            reason=f"Tool {spec.name!r} ({spec.permission_tier.value}) requires explicit approval.",
            status=ApprovalStatus.PENDING,
            requested_by=meta.context.principal_id,
        )
        saved = await self._audit_repo.create_approval(approval)
        self._remember_pending_context(saved.id, meta.context)
        pending_meta = InvocationMeta(
            context=meta.context,
            spec=spec,
            decision=meta.decision,
            arguments_json=meta.arguments_json,
            approval_id=saved.id,
        )
        await self._record(pending_meta, success=False, error="approval_required")
        msg = f"Tool {spec.name!r} requires explicit approval."
        raise ToolApprovalRequiredError(
            msg,
            approval_id=saved.id,
            tool_name=spec.name,
            details={"permission_tier": spec.permission_tier.value},
        )

    async def _consume_approved_request(
        self,
        approval_id: str,
        meta: InvocationMeta,
    ) -> None:
        """Verify an approval record matches context and transition APPROVED -> EXECUTED."""
        approval = await self._audit_repo.get_approval(approval_id)
        mismatch_error = validate_approval_match(approval, meta.context, meta.spec.name)
        if mismatch_error is not None or approval is None:
            error_code = mismatch_error or "approval_not_found"
            await self._record(meta, success=False, error=error_code)
            msg = f"Approval {approval_id!r} cannot authorize {meta.spec.name!r}: {error_code}."
            raise AuthorizationError(msg, details={"approval_id": approval_id})

        self._pending_contexts.pop(approval_id, None)
        await self._audit_repo.update_approval(
            approval.model_copy(update={"status": ApprovalStatus.EXECUTED})
        )

    async def _execute_plugin(
        self,
        plugin: ToolPluginPort,
        arguments: dict[str, Any],
        meta: InvocationMeta,
    ) -> ToolResult:
        """Run plugin.execute under asyncio.wait_for and write the resulting audit record."""
        spec = plugin.spec
        ctx = meta.context
        started = time.perf_counter()
        with bind_log_context(
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            project_id=ctx.project_id,
            principal_id=ctx.principal_id,
            component="tools",
        ):
            try:
                raw_result = await asyncio.wait_for(
                    plugin.execute(arguments, ctx),
                    timeout=spec.timeout_seconds,
                )
            except TimeoutError:
                get_component_logger("tools").exception(
                    "tool_execution_timeout",
                    tool_name=spec.name,
                    timeout_seconds=spec.timeout_seconds,
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    principal_id=ctx.principal_id,
                    request_id=ctx.request_id,
                    alert_category=AlertCategory.TIMEOUT.value,
                    exception_type="TimeoutError",
                )
                return await self._record_failed_result(meta, started, "timeout")
            except LaiError as exc:
                await self._record(
                    meta,
                    success=False,
                    error=exc.message,
                    duration_ms=elapsed_ms(started),
                )
                raise
            except Exception as exc:  # noqa: BLE001
                get_component_logger("tools").exception(
                    "tool_execution_failed",
                    tool_name=spec.name,
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    principal_id=ctx.principal_id,
                    request_id=ctx.request_id,
                    exception_type=type(exc).__name__,
                )
                return await self._record_failed_result(
                    meta,
                    started,
                    f"{type(exc).__name__}: {exc}",
                )

        duration_ms = elapsed_ms(started)
        result = (
            raw_result
            if raw_result.duration_ms > 0.0
            else raw_result.model_copy(update={"duration_ms": duration_ms})
        )
        await self._record(
            meta,
            success=result.success,
            output_preview=result.output,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        return result

    async def _record_failed_result(
        self,
        meta: InvocationMeta,
        started: float,
        error: str,
    ) -> ToolResult:
        """Audit a failed tool execution and return a normalized failed ToolResult."""
        duration_ms = elapsed_ms(started)
        await self._record(meta, success=False, error=error, duration_ms=duration_ms)
        return ToolResult(
            tool_name=meta.spec.name,
            success=False,
            output="",
            error=error,
            duration_ms=duration_ms,
        )

    async def _record(
        self,
        meta: InvocationMeta,
        *,
        success: bool,
        output_preview: str = "",
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Persist an audit record built from InvocationMeta."""
        record = build_audit_record(
            meta,
            success=success,
            output_preview=output_preview,
            error=error,
            duration_ms=duration_ms,
        )
        await self._audit_repo.record_audit(record)

    def _remember_pending_context(
        self,
        approval_id: str,
        context: ToolExecutionContext,
    ) -> None:
        """Cache execution context for pending approvals with bounded capacity."""
        if len(self._pending_contexts) >= _MAX_CACHED_CONTEXTS:
            oldest_key = next(iter(self._pending_contexts))
            del self._pending_contexts[oldest_key]
        self._pending_contexts[approval_id] = context
