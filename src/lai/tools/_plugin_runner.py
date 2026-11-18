"""Timeout-guarded server plugin runner and audit recording helper."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory
from lai.domain.tools import ToolResult
from lai.observability.logging import bind_log_context, get_component_logger
from lai.tools._audit import InvocationMeta, build_audit_record, elapsed_ms

if TYPE_CHECKING:
    from lai.ports.repositories import ToolAuditRepositoryPort
    from lai.ports.tools import ToolPluginPort


async def record_tool_audit(  # noqa: PLR0913
    audit_repo: ToolAuditRepositoryPort,
    meta: InvocationMeta,
    *,
    success: bool,
    output_preview: str = "",
    error: str | None = None,
    duration_ms: float = 0.0,
) -> None:
    """Persist an immutable ToolAuditRecord derived from InvocationMeta."""
    record = build_audit_record(
        meta,
        success=success,
        output_preview=output_preview,
        error=error,
        duration_ms=duration_ms,
    )
    await audit_repo.record_audit(record)


async def _record_failed_result(
    audit_repo: ToolAuditRepositoryPort,
    meta: InvocationMeta,
    started: float,
    error: str,
) -> ToolResult:
    duration_ms = elapsed_ms(started)
    await record_tool_audit(audit_repo, meta, success=False, error=error, duration_ms=duration_ms)
    return ToolResult(
        tool_name=meta.spec.name,
        success=False,
        output="",
        error=error,
        duration_ms=duration_ms,
    )


async def execute_server_plugin(
    plugin: ToolPluginPort,
    arguments: dict[str, Any],
    meta: InvocationMeta,
    audit_repo: ToolAuditRepositoryPort,
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
            return await _record_failed_result(audit_repo, meta, started, "timeout")
        except LaiError as exc:
            await record_tool_audit(
                audit_repo,
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
            return await _record_failed_result(
                audit_repo,
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
    await record_tool_audit(
        audit_repo,
        meta,
        success=result.success,
        output_preview=result.output,
        error=result.error,
        duration_ms=result.duration_ms,
    )
    return result
