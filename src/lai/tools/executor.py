"""Policy-enforcing, approval-gated, timeout-guarded, and audited dual-plane tool executor."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from lai.domain.base import utc_now
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
    ToolExecutionContext,
    ToolExecutionTarget,
    ToolResult,
)
from lai.ports.tools import ToolExecutorPort
from lai.tools._audit import (
    InvocationMeta,
    deserialize_arguments,
    normalize_optional_id,
    serialize_arguments,
    validate_approval_match,
)
from lai.tools._client_dispatch import dispatch_client_tool
from lai.tools._client_results import submit_client_tool_result
from lai.tools._plugin_runner import execute_server_plugin, record_tool_audit
from lai.tools.policy import DefaultToolPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lai.domain.tools import ToolSpec
    from lai.ports.guardrails import GuardrailPipelinePort
    from lai.ports.repositories import ToolAuditRepositoryPort
    from lai.ports.tools import ToolPluginPort, ToolPolicyPort, ToolRegistryPort

_MAX_CACHED_CONTEXTS = 1024
DEFAULT_LEASE_TTL_SECONDS = 300.0
DEFAULT_MAX_OUTPUT_CHARS = 16_000
DEFAULT_MAX_PAYLOAD_BYTES = 262_144


class GuardedToolExecutor(ToolExecutorPort):
    """Executes server attachment READ plugins and dispatches CLIENT tools under policy."""

    def __init__(  # noqa: PLR0913
        self,
        registry: ToolRegistryPort,
        audit_repo: ToolAuditRepositoryPort,
        policy: ToolPolicyPort | None = None,
        *,
        guardrail_pipeline: GuardrailPipelinePort | None = None,
        lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self._registry = registry
        self._audit_repo = audit_repo
        self._policy: ToolPolicyPort = policy or DefaultToolPolicy()
        self._guardrail_pipeline = guardrail_pipeline
        self._lease_ttl_seconds = lease_ttl_seconds
        self._max_output_chars = max_output_chars
        self._max_payload_bytes = max_payload_bytes
        self._pending_contexts: dict[str, ToolExecutionContext] = {}

    def _resolve_target(
        self,
        tool_name: str,
        client_tools: Mapping[str, ToolSpec] | None,
    ) -> tuple[ToolPluginPort | None, ToolSpec]:
        plugin = self._registry.get(tool_name)
        if plugin is not None:
            return plugin, plugin.spec
        client_spec = client_tools.get(tool_name) if client_tools is not None else None
        if client_spec is not None:
            return None, client_spec
        msg = f"Tool {tool_name!r} is not registered."
        raise NotFoundError(msg, details={"tool_name": tool_name})

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        *,
        client_tools: Mapping[str, ToolSpec] | None = None,
    ) -> ToolResult:
        """Evaluate policy, enforce dual-plane boundaries, audit, and run or dispatch."""
        plugin, spec = self._resolve_target(tool_name, client_tools)
        arguments_json = serialize_arguments(arguments)

        if (
            spec.execution_target is ToolExecutionTarget.SERVER_ATTACHMENT
            and spec.permission_tier is not PermissionTier.READ
        ):
            deny_meta = InvocationMeta(
                context=context,
                spec=spec,
                decision=PolicyDecision.DENY,
                arguments_json=arguments_json,
                approval_id=normalize_optional_id(context.approved_approval_id),
            )
            await record_tool_audit(
                self._audit_repo, deny_meta, success=False, error="server_host_tool_forbidden"
            )
            msg = (
                f"Server execution of {spec.permission_tier.value!r} tool "
                f"{tool_name!r} is forbidden."
            )
            raise AuthorizationError(
                msg,
                details={
                    "tool_name": tool_name,
                    "permission_tier": spec.permission_tier.value,
                    "execution_target": spec.execution_target.value,
                },
            )

        if spec.execution_target is ToolExecutionTarget.CLIENT:
            await dispatch_client_tool(
                spec=spec,
                arguments=arguments,
                arguments_json=arguments_json,
                context=context,
                policy=self._policy,
                audit_repo=self._audit_repo,
                lease_ttl_seconds=self._lease_ttl_seconds,
            )

        if plugin is None:
            msg = f"Server plugin {tool_name!r} is not registered."
            raise AuthorizationError(msg, details={"tool_name": tool_name})

        meta = InvocationMeta(
            context=context,
            spec=spec,
            decision=self._policy.evaluate(spec, arguments, context),
            arguments_json=arguments_json,
            approval_id=normalize_optional_id(context.approved_approval_id),
        )
        if meta.decision is PolicyDecision.DENY:
            await record_tool_audit(self._audit_repo, meta, success=False, error="policy_denied")
            msg = f"Principal {context.principal_id!r} is denied access to tool {tool_name!r}."
            details = {"tool_name": tool_name, "permission_tier": spec.permission_tier.value}
            raise AuthorizationError(msg, details=details)

        if meta.decision is PolicyDecision.REQUIRE_APPROVAL:
            await self._raise_approval_required(meta)

        if meta.approval_id is not None:
            await self._consume_approved_request(meta.approval_id, meta)

        return await execute_server_plugin(plugin, arguments, meta, self._audit_repo)

    async def _cas_pending_transition(self, updated: ToolApprovalRecord) -> None:
        if not await self._audit_repo.transition_approval_status(
            updated, expected_status=ApprovalStatus.PENDING
        ):
            msg = f"Tool approval {updated.id!r} was already resolved."
            raise ValidationFailedError(msg, details={"approval_id": updated.id})

    async def approve_and_execute(
        self,
        approval_id: str,
        decided_by: str,
        approved: bool,
        note: str | None = None,
    ) -> ToolApprovalRecord:
        """Atomically transition PENDING approval and run server tool or issue client lease."""
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
            details = {"approval_id": approval_id, "status": record.status.value}
            raise ValidationFailedError(msg, details=details)

        decided_at = utc_now()
        if not approved:
            denied = record.model_copy(
                update={
                    "status": ApprovalStatus.DENIED,
                    "decided_by": approver,
                    "decision_note": note,
                    "decided_at": decided_at,
                }
            )
            await self._cas_pending_transition(denied)
            self._pending_contexts.pop(approval_id, None)
            return denied

        if record.execution_target is ToolExecutionTarget.CLIENT:
            expires_at = decided_at + timedelta(seconds=self._lease_ttl_seconds)
            client_approved = record.model_copy(
                update={
                    "status": ApprovalStatus.APPROVED,
                    "decided_by": approver,
                    "decision_note": note,
                    "decided_at": decided_at,
                    "expires_at": expires_at,
                }
            )
            await self._cas_pending_transition(client_approved)
            self._pending_contexts.pop(approval_id, None)
            return client_approved

        approved_record = record.model_copy(
            update={
                "status": ApprovalStatus.APPROVED,
                "decided_by": approver,
                "decision_note": note,
                "decided_at": decided_at,
            }
        )
        await self._cas_pending_transition(approved_record)
        return await self._execute_approved_server_tool(approved_record)

    async def submit_client_result(
        self,
        approval_id: str,
        client_app_id: str,
        result: ToolResult,
        *,
        principal_id: str,
        is_admin: bool = False,
    ) -> ToolResult:
        """Validate, sanitize, guardrail-scan, and consume an approved client tool lease."""
        return await submit_client_tool_result(
            audit_repo=self._audit_repo,
            guardrail_pipeline=self._guardrail_pipeline,
            approval_id=approval_id,
            client_app_id=client_app_id,
            result=result,
            principal_id=principal_id,
            is_admin=is_admin,
            max_output_chars=self._max_output_chars,
            max_payload_bytes=self._max_payload_bytes,
        )

    async def _execute_approved_server_tool(
        self, approved_record: ToolApprovalRecord
    ) -> ToolApprovalRecord:
        arguments = deserialize_arguments(approved_record.arguments_json)
        exec_context = self._resolve_execution_context(approved_record)
        await self.invoke(approved_record.tool_name, arguments, exec_context)
        refreshed = await self._audit_repo.get_approval(approved_record.id)
        return refreshed or approved_record.model_copy(update={"status": ApprovalStatus.EXECUTED})

    def _resolve_execution_context(self, record: ToolApprovalRecord) -> ToolExecutionContext:
        """Reuse original pending context or restore persisted confinement fields."""
        cached = self._pending_contexts.pop(record.id, None)
        if cached is not None:
            return cached.model_copy(update={"approved_approval_id": record.id})
        return ToolExecutionContext(
            project_id=record.project_id,
            session_id=record.session_id,
            principal_id=record.requested_by,
            principal_scopes=record.principal_scopes,
            allowed_file_ids=record.allowed_file_ids,
            approved_approval_id=record.id,
            tool_call_id=record.tool_call_id,
        )

    async def _raise_approval_required(self, meta: InvocationMeta) -> None:
        """Persist a pending server approval + audit record and raise ToolApprovalRequiredError."""
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
            execution_target=spec.execution_target,
            tool_call_id=meta.context.tool_call_id,
            allowed_file_ids=meta.context.allowed_file_ids,
            principal_scopes=meta.context.principal_scopes,
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
        await record_tool_audit(
            self._audit_repo, pending_meta, success=False, error="approval_required"
        )
        msg = f"Tool {spec.name!r} requires explicit approval."
        details = {"permission_tier": spec.permission_tier.value}
        raise ToolApprovalRequiredError(
            msg, approval_id=saved.id, tool_name=spec.name, details=details
        )

    async def _consume_approved_request(
        self,
        approval_id: str,
        meta: InvocationMeta,
    ) -> None:
        """Verify context match and atomically transition APPROVED -> EXECUTED."""
        approval = await self._audit_repo.get_approval(approval_id)
        mismatch_error = validate_approval_match(approval, meta.context, meta.spec.name)
        if mismatch_error is not None or approval is None:
            error_code = mismatch_error or "approval_not_found"
            await record_tool_audit(self._audit_repo, meta, success=False, error=error_code)
            msg = f"Approval {approval_id!r} cannot authorize {meta.spec.name!r}: {error_code}."
            raise AuthorizationError(msg, details={"approval_id": approval_id})

        self._pending_contexts.pop(approval_id, None)
        consumed = approval.model_copy(update={"status": ApprovalStatus.EXECUTED})
        if not await self._audit_repo.transition_approval_status(
            consumed, expected_status=ApprovalStatus.APPROVED
        ):
            await record_tool_audit(
                self._audit_repo, meta, success=False, error="approval_already_consumed"
            )
            msg = f"Approval {approval_id!r} was already consumed."
            raise AuthorizationError(msg, details={"approval_id": approval_id})

    def _remember_pending_context(
        self,
        approval_id: str,
        context: ToolExecutionContext,
    ) -> None:
        if len(self._pending_contexts) >= _MAX_CACHED_CONTEXTS:
            oldest_key = next(iter(self._pending_contexts))
            del self._pending_contexts[oldest_key]
        self._pending_contexts[approval_id] = context
