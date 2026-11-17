"""Tool plugin, registry, policy, and guarded executor protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lai.domain.tools import (
        PolicyDecision,
        ToolApprovalRecord,
        ToolExecutionContext,
        ToolResult,
        ToolSpec,
    )


@runtime_checkable
class ToolPluginPort(Protocol):
    """Individual tool implementation exposed to the model."""

    @property
    def spec(self) -> ToolSpec:
        """Return the declarative specification for this tool."""
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute the tool with validated arguments and project context."""
        ...


@runtime_checkable
class ToolRegistryPort(Protocol):
    """Catalog of registered tool plugins."""

    def register(self, plugin: ToolPluginPort) -> None:
        """Register or replace a tool plugin by name."""
        ...

    def get(self, name: str) -> ToolPluginPort | None:
        """Look up a registered tool plugin by name."""
        ...

    def list_specs(self) -> list[ToolSpec]:
        """Return specifications for all registered tools."""
        ...


@runtime_checkable
class ToolPolicyPort(Protocol):
    """Security policy evaluator governing tool execution."""

    def evaluate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> PolicyDecision:
        """Determine whether a tool invocation is allowed, denied, or needs approval."""
        ...


@runtime_checkable
class ToolExecutorPort(Protocol):
    """Policy-enforcing and audited tool runner."""

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Evaluate policy, audit, and execute or raise ToolApprovalRequiredError."""
        ...

    async def approve_and_execute(
        self,
        approval_id: str,
        decided_by: str,
        approved: bool,
        note: str | None = None,
    ) -> ToolApprovalRecord:
        """Resolve a pending approval request and execute if approved."""
        ...
