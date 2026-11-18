"""Tool catalog, approval resolution, and client tool result API schemas."""

from __future__ import annotations

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.tools import ToolSpec


class ToolListResponse(StrictModel):
    """Catalog of registered tool specifications."""

    items: list[ToolSpec]
    count: int = Field(ge=0)


class ResolveApprovalRequest(StrictModel):
    """Payload for approving or denying a pending high-risk tool invocation."""

    approved: bool
    note: str | None = Field(default=None, max_length=1000)


class SubmitToolResultRequest(StrictModel):
    """Payload posted by a client application to return client-executed tool output."""

    output: str
    success: bool = True
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    stream: bool = False
