"""Phase 1 get_file_overview tool plugin returning deterministic file summaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lai.domain.tools import PermissionTier, ToolResult, ToolSpec
from lai.ports.tools import ToolPluginPort
from lai.tools.plugins._scoping import load_scoped_file, require_non_empty_str

if TYPE_CHECKING:
    from lai.domain.tools import ToolExecutionContext
    from lai.ports.repositories import FileRepositoryPort


class GetFileOverviewPlugin(ToolPluginPort):
    """Read-only inspector returning file metadata and structural summaries."""

    def __init__(
        self,
        file_repo: FileRepositoryPort,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._file_repo = file_repo
        self._spec = ToolSpec(
            name="get_file_overview",
            description=(
                "Return metadata plus deterministic LogFileSummary or CodeFileSummary "
                "for an ingested project file."
            ),
            permission_tier=PermissionTier.READ,
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "minLength": 1},
                },
                "required": ["file_id"],
                "additionalProperties": False,
            },
            requires_approval_by_default=False,
            timeout_seconds=timeout_seconds,
        )

    @property
    def spec(self) -> ToolSpec:
        """Return the declarative specification for get_file_overview."""
        return self._spec

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Verify project/file scoping and serialize metadata and structural summaries."""
        file_id = require_non_empty_str(arguments, "file_id")
        file_record = await load_scoped_file(self._file_repo, file_id, context)
        payload = {
            "file_id": file_record.id,
            "project_id": file_record.project_id,
            "filename": file_record.filename,
            "content_type": file_record.content_type,
            "kind": file_record.kind.value,
            "status": file_record.status.value,
            "size_bytes": file_record.size_bytes,
            "sha256": file_record.sha256,
            "token_count": file_record.token_count,
            "chunk_count": file_record.chunk_count,
            "error_message": file_record.error_message,
            "created_at": file_record.created_at.isoformat(),
            "updated_at": file_record.updated_at.isoformat(),
            "log_summary": (
                file_record.log_summary.model_dump(mode="json")
                if file_record.log_summary is not None
                else None
            ),
            "code_summary": (
                file_record.code_summary.model_dump(mode="json")
                if file_record.code_summary is not None
                else None
            ),
        }
        return ToolResult(
            tool_name=self._spec.name,
            success=True,
            output=json.dumps(payload),
        )
