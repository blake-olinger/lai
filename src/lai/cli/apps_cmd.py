"""CLI subcommands for registering, updating, listing, and revoking client applications."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from lai.cli._apps_helpers import (
    echo_app_summary,
    load_registered_tools_file,
    resolve_update_collections,
)
from lai.cli._common import open_client_app_repo, resolve_db_path, resolve_settings
from lai.domain.base import utc_now
from lai.domain.errors import LaiError, NotFoundError
from lai.domain.tools import PermissionTier, ToolSpec
from lai.security import build_client_app, update_client_app

if TYPE_CHECKING:
    from lai.domain.client_apps import ClientAppRecord

apps_app = typer.Typer(
    name="apps",
    no_args_is_help=True,
    help="Manage registered client applications.",
)


async def _create_app_async(  # noqa: PLR0913
    db_path: Path,
    name: str,
    *,
    oauth_client_id: str,
    tiers: tuple[PermissionTier, ...],
    project_ids: tuple[str, ...],
    registered_tools: tuple[ToolSpec, ...],
    expires_days: int | None,
) -> ClientAppRecord:
    expires_at = utc_now() + timedelta(days=expires_days) if expires_days is not None else None
    record = build_client_app(
        name=name,
        oauth_client_id=oauth_client_id,
        allowed_tiers=tiers,
        registered_tools=registered_tools,
        allowed_project_ids=project_ids,
        expires_at=expires_at,
    )
    async with open_client_app_repo(db_path) as repo:
        return await repo.create(record)


async def _update_app_async(  # noqa: PLR0913
    db_path: Path,
    app_id: str,
    *,
    name: str | None,
    tiers: tuple[PermissionTier, ...] | None,
    project_ids: tuple[str, ...] | None,
    registered_tools: tuple[ToolSpec, ...] | None,
    expires_days: int | None,
    clear_expiry: bool,
) -> ClientAppRecord:
    update_expires = clear_expiry or expires_days is not None
    expires_at = utc_now() + timedelta(days=expires_days) if expires_days is not None else None
    async with open_client_app_repo(db_path) as repo:
        existing = await repo.get_by_id(app_id)
        if existing is None:
            msg = f"Client app {app_id!r} not found."
            raise NotFoundError(msg, details={"app_id": app_id})
        updated = update_client_app(
            existing,
            name=name,
            allowed_tiers=tiers,
            registered_tools=registered_tools,
            allowed_project_ids=project_ids,
            expires_at=expires_at,
            update_expires_at=update_expires,
        )
        return await repo.update(updated)


async def _list_apps_async(db_path: Path) -> list[ClientAppRecord]:
    async with open_client_app_repo(db_path) as repo:
        return await repo.list_all()


async def _revoke_app_async(db_path: Path, app_id: str) -> bool:
    async with open_client_app_repo(db_path) as repo:
        return await repo.revoke(app_id, utc_now())


@apps_app.command("create")
def create_app(  # noqa: PLR0913, PLR0917
    name: Annotated[
        str, typer.Option("--name", "-n", help="Human-readable label for the client app.")
    ],
    oauth_client_id: Annotated[
        str,
        typer.Option("--oauth-client-id", help="OIDC client ID bound to this application."),
    ],
    tier: Annotated[
        list[PermissionTier] | None,
        typer.Option("--tier", "-t", help="Allowed tool permission tier (repeatable)."),
    ] = None,
    project_id: Annotated[
        list[str] | None,
        typer.Option("--project-id", "-p", help="Optional allowed project ID (repeatable)."),
    ] = None,
    registered_tools: Annotated[
        Path | None,
        typer.Option(
            "--registered-tools",
            help="Optional JSON file containing an array of static ToolSpec definitions.",
        ),
    ] = None,
    expires_days: Annotated[
        int | None,
        typer.Option("--expires-days", min=1, help="Optional expiration lifetime in days."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Optional explicit SQLite DB path override."),
    ] = None,
) -> None:
    """Register a new client application bound to an OIDC client ID."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)

    effective_tiers = tuple(tier) if tier else (PermissionTier.READ,)
    effective_projects = tuple(project_id) if project_id else ()
    try:
        loaded_tools = (
            load_registered_tools_file(registered_tools) if registered_tools is not None else ()
        )
        record = asyncio.run(
            _create_app_async(
                target_db,
                name,
                oauth_client_id=oauth_client_id,
                tiers=effective_tiers,
                project_ids=effective_projects,
                registered_tools=loaded_tools,
                expires_days=expires_days,
            )
        )
    except LaiError as exc:
        typer.echo(f"Error ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    echo_app_summary("Created", record)


@apps_app.command("update")
def update_app_command(  # noqa: PLR0913, PLR0917
    app_id: Annotated[str, typer.Argument(help="Client app ID (app_...) to update.")],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Updated human-readable label."),
    ] = None,
    tier: Annotated[
        list[PermissionTier] | None,
        typer.Option("--tier", "-t", help="Replacement allowed tool tier (repeatable)."),
    ] = None,
    project_id: Annotated[
        list[str] | None,
        typer.Option("--project-id", "-p", help="Replacement allowed project ID (repeatable)."),
    ] = None,
    clear_projects: Annotated[
        bool,
        typer.Option("--clear-projects", help="Clear allowed_project_ids restriction (*)."),
    ] = False,
    registered_tools: Annotated[
        Path | None,
        typer.Option(
            "--registered-tools",
            help="Replacement JSON file containing an array of static ToolSpec definitions.",
        ),
    ] = None,
    clear_tools: Annotated[
        bool,
        typer.Option("--clear-tools", help="Clear all static registered_tools."),
    ] = False,
    expires_days: Annotated[
        int | None,
        typer.Option("--expires-days", min=1, help="New expiration lifetime in days from now."),
    ] = None,
    clear_expiry: Annotated[
        bool,
        typer.Option("--clear-expiry", help="Remove expiration limit (never expires)."),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Optional explicit SQLite DB path override."),
    ] = None,
) -> None:
    """Update mutable settings on an active (non-revoked) client application."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)
    try:
        resolved_projects, resolved_tools = resolve_update_collections(
            project_id=project_id,
            clear_projects=clear_projects,
            registered_tools=registered_tools,
            clear_tools=clear_tools,
            expires_days=expires_days,
            clear_expiry=clear_expiry,
        )
        record = asyncio.run(
            _update_app_async(
                target_db,
                app_id,
                name=name,
                tiers=tuple(tier) if tier is not None else None,
                project_ids=resolved_projects,
                registered_tools=resolved_tools,
                expires_days=expires_days,
                clear_expiry=clear_expiry,
            )
        )
    except LaiError as exc:
        typer.echo(f"Error ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    echo_app_summary("Updated", record)


@apps_app.command("list")
def list_apps(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Optional explicit SQLite DB path override."),
    ] = None,
) -> None:
    """List registered client applications and their bound OIDC client IDs."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)
    records = asyncio.run(_list_apps_async(target_db))

    if not records:
        typer.echo("No client apps found.")
        return

    now = utc_now()
    for record in records:
        if record.revoked_at is not None:
            status_label = "revoked"
        elif record.expires_at is not None and now >= record.expires_at:
            status_label = "expired"
        else:
            status_label = "active"
        tier_csv = ",".join(item.value for item in record.allowed_tiers)
        projects_csv = ",".join(record.allowed_project_ids) or "*"
        typer.echo(
            f"{record.id}\t{record.name}\t{record.oauth_client_id}\t{status_label}"
            f"\t[{tier_csv}]\t[{projects_csv}]\ttools={len(record.registered_tools)}"
        )


@apps_app.command("revoke")
def revoke_app(
    app_id: Annotated[str, typer.Argument(help="Client app ID (app_...) to revoke.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Optional explicit SQLite DB path override."),
    ] = None,
) -> None:
    """Revoke an active client application by ID."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)
    revoked = asyncio.run(_revoke_app_async(target_db, app_id))
    if not revoked:
        typer.echo(f"Client app not found: {app_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Revoked client app: {app_id}")
