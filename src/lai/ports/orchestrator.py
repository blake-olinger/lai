"""Turn orchestrator protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import OrchestratorEvent, TurnInput, TurnOutput
    from lai.domain.tools import ToolApprovalRecord, ToolResult


@runtime_checkable
class OrchestratorPort(Protocol):
    """Coordinates retrieval, map-reduce, tool loops, and inference turns."""

    def run_turn_stream(
        self,
        turn_input: TurnInput,
        principal: AuthPrincipal,
    ) -> AsyncIterator[OrchestratorEvent]:
        """Execute a turn and yield streaming orchestrator events."""
        ...

    async def run_turn(
        self,
        turn_input: TurnInput,
        principal: AuthPrincipal,
    ) -> TurnOutput:
        """Execute a turn to completion and return the final TurnOutput."""
        ...

    def resume_client_tool_stream(
        self,
        approval: ToolApprovalRecord,
        tool_result: ToolResult,
        principal: AuthPrincipal,
    ) -> AsyncIterator[OrchestratorEvent]:
        """Resume a paused turn after client tool execution and yield events."""
        ...

    async def resume_client_tool(
        self,
        approval: ToolApprovalRecord,
        tool_result: ToolResult,
        principal: AuthPrincipal,
    ) -> TurnOutput:
        """Resume a paused turn after client tool execution and return TurnOutput."""
        ...
