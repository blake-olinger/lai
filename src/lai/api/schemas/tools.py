"""Tool catalog and approval resolution API schemas."""

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
