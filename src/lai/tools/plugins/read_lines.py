"""Phase 1 read_file_lines tool plugin with line-span clamping and scoping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lai.domain.errors import NotFoundError, ValidationFailedError
from lai.domain.tools import PermissionTier, ToolResult, ToolSpec
from lai.ports.tools import ToolPluginPort
from lai.tools.plugins._scoping import (
    load_scoped_file,
    optional_int,
    require_non_empty_str,
)

if TYPE_CHECKING:
    from lai.domain.tools import ToolExecutionContext
    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.storage import BlobStorePort

DEFAULT_MAX_LINES_PER_CALL = 250


class ReadFileLinesPlugin(ToolPluginPort):
    """Read-only bounded line-range reader backed by BlobStorePort."""

    def __init__(
        self,
        file_repo: FileRepositoryPort,
        blob_store: BlobStorePort,
        *,
        max_lines: int = DEFAULT_MAX_LINES_PER_CALL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._file_repo = file_repo
        self._blob_store = blob_store
        self._max_lines = max(max_lines, 1)
        self._spec = ToolSpec(
            name="read_file_lines",
            description=(
                "Read an inclusive 1-based line span from a project file blob, "
                f"clamped to at most {self._max_lines} lines per call."
            ),
            permission_tier=PermissionTier.READ,
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "minLength": 1},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["file_id"],
                "additionalProperties": False,
            },
            requires_approval_by_default=False,
            timeout_seconds=timeout_seconds,
        )

    @property
    def spec(self) -> ToolSpec:
        """Return the declarative specification for read_file_lines."""
        return self._spec

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Verify file scoping, clamp line span, and read text lines from blob storage."""
        file_id = require_non_empty_str(arguments, "file_id")
        start_line = optional_int(arguments, "start_line", 1)
        end_line = optional_int(arguments, "end_line", None)

        if start_line is None or start_line < 1:
            msg = "Argument 'start_line' must be >= 1."
            raise ValidationFailedError(msg, details={"field": "start_line"})
        if end_line is not None and end_line < start_line:
            msg = "Argument 'end_line' must be >= 'start_line'."
            raise ValidationFailedError(msg, details={"field": "end_line"})

        max_allowed_end = start_line + self._max_lines - 1
        requested_clamped = end_line is not None and end_line > max_allowed_end
        effective_end = max_allowed_end if end_line is None else min(end_line, max_allowed_end)

        file_record = await load_scoped_file(self._file_repo, file_id, context)
        try:
            content = await self._blob_store.read_text_range(
                Path(file_record.blob_path),
                start_line=start_line,
                end_line=effective_end,
            )
        except FileNotFoundError as exc:
            msg = f"Blob content for file {file_id!r} was not found."
            raise NotFoundError(msg, details={"file_id": file_id}) from exc

        truncated = requested_clamped or (
            end_line is None and len(content.splitlines()) >= self._max_lines
        )
        payload = {
            "file_id": file_record.id,
            "filename": file_record.filename,
            "start_line": start_line,
            "end_line": effective_end,
            "truncated": truncated,
            "content": content,
        }
        return ToolResult(
            tool_name=self._spec.name,
            success=True,
            output=json.dumps(payload),
            truncated=truncated,
        )
