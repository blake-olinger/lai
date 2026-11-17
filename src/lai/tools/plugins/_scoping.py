"""Shared project/file-scope validation and argument helpers for read plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lai.domain.errors import AuthorizationError, NotFoundError, ValidationFailedError

if TYPE_CHECKING:
    from lai.domain.files import FileRecord
    from lai.domain.tools import ToolExecutionContext
    from lai.ports.repositories import FileRepositoryPort


def require_non_empty_str(arguments: dict[str, Any], field_name: str) -> str:
    """Extract and validate a required non-empty string argument."""
    raw = arguments.get(field_name)
    if not isinstance(raw, str) or not raw.strip():
        msg = f"Argument {field_name!r} must be a non-empty string."
        raise ValidationFailedError(msg, details={"field": field_name})
    return raw.strip()


def optional_int(arguments: dict[str, Any], field_name: str, default: int | None) -> int | None:
    """Extract an optional integer argument, rejecting booleans and non-integers."""
    if field_name not in arguments or arguments[field_name] is None:
        return default
    raw = arguments[field_name]
    if isinstance(raw, bool) or not isinstance(raw, int):
        msg = f"Argument {field_name!r} must be an integer."
        raise ValidationFailedError(msg, details={"field": field_name})
    return raw


def validate_allowed_file_id(file_id: str, context: ToolExecutionContext) -> None:
    """Ensure file_id is permitted when context.allowed_file_ids is non-empty."""
    if context.allowed_file_ids and file_id not in context.allowed_file_ids:
        msg = f"File {file_id!r} is outside the allowed file scope for this session."
        raise AuthorizationError(msg, details={"file_id": file_id})


async def load_scoped_file(
    file_repo: FileRepositoryPort,
    file_id: str,
    context: ToolExecutionContext,
) -> FileRecord:
    """Fetch a FileRecord and enforce allowed_file_ids and project_id isolation."""
    validate_allowed_file_id(file_id, context)
    record = await file_repo.get_file(file_id)
    if record is None:
        msg = f"File {file_id!r} not found."
        raise NotFoundError(msg, details={"file_id": file_id})
    if record.project_id != context.project_id:
        msg = f"File {file_id!r} does not belong to project {context.project_id!r}."
        raise AuthorizationError(
            msg,
            details={"file_id": file_id, "project_id": context.project_id},
        )
    return record
