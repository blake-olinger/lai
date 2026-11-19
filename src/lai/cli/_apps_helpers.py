"""Internal CLI helpers for loading static tool specs and validating app update flags."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import typer
from pydantic import ValidationError

from lai.domain.errors import ValidationFailedError
from lai.domain.tools import ToolSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lai.domain.client_apps import ClientAppRecord


def load_registered_tools_file(tools_path: Path) -> tuple[ToolSpec, ...]:
    """Load and validate a JSON array of ToolSpec objects from disk."""
    if not tools_path.is_file():
        msg = f"Registered tools file not found: {tools_path}"
        raise ValidationFailedError(msg, details={"path": str(tools_path)})
    try:
        raw: object = json.loads(tools_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to parse registered tools JSON {tools_path}: {exc}"
        raise ValidationFailedError(msg, details={"path": str(tools_path)}) from exc
    if not isinstance(raw, list):
        msg = f"Registered tools JSON must be an array of tool specs: {tools_path}"
        raise ValidationFailedError(msg, details={"path": str(tools_path)})
    specs: list[ToolSpec] = []
    for index, item in enumerate(cast("list[object]", raw)):
        try:
            specs.append(ToolSpec.model_validate(item))
        except ValidationError as exc:
            msg = f"Invalid tool spec at index {index} in {tools_path}: {exc.errors()[0]['msg']}"
            raise ValidationFailedError(msg, details={"index": str(index)}) from exc
    return tuple(specs)


def resolve_update_collections(  # noqa: PLR0913
    *,
    project_id: list[str] | None,
    clear_projects: bool,
    registered_tools: Path | None,
    clear_tools: bool,
    expires_days: int | None,
    clear_expiry: bool,
) -> tuple[tuple[str, ...] | None, tuple[ToolSpec, ...] | None]:
    """Validate mutually exclusive update flags and resolve project/tool updates."""
    if project_id and clear_projects:
        msg = "Cannot combine --project-id with --clear-projects."
        raise ValidationFailedError(msg)
    if registered_tools is not None and clear_tools:
        msg = "Cannot combine --registered-tools with --clear-tools."
        raise ValidationFailedError(msg)
    if expires_days is not None and clear_expiry:
        msg = "Cannot combine --expires-days with --clear-expiry."
        raise ValidationFailedError(msg)

    resolved_projects: tuple[str, ...] | None = None
    if clear_projects:
        resolved_projects = ()
    elif project_id is not None:
        resolved_projects = tuple(project_id)

    resolved_tools: tuple[ToolSpec, ...] | None = None
    if clear_tools:
        resolved_tools = ()
    elif registered_tools is not None:
        resolved_tools = load_registered_tools_file(registered_tools)
    return resolved_projects, resolved_tools


def echo_app_summary(verb: str, record: ClientAppRecord) -> None:
    """Print a concise summary line pair for a created or updated client application."""
    tier_csv = ",".join(item.value for item in record.allowed_tiers)
    projects_csv = ",".join(record.allowed_project_ids) or "*"
    typer.echo(f"{verb} client app: id={record.id} oauth_client_id={record.oauth_client_id}")
    typer.echo(
        f"Allowed tiers: {tier_csv} | Projects: {projects_csv} | "
        f"Tools: {len(record.registered_tools)}"
    )
