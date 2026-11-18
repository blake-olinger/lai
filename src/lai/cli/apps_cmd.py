"""CLI subcommands for registering, listing, and revoking client applications."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path  # noqa: TC003 - evaluated at runtime by Typer
from typing import TYPE_CHECKING, Annotated

import typer

from lai.cli._common import open_client_app_repo, resolve_db_path, resolve_settings
from lai.domain.base import utc_now
from lai.domain.errors import LaiError
from lai.domain.tools import PermissionTier
from lai.security import generate_client_app

if TYPE_CHECKING:
    from lai.domain.client_apps import ClientAppRecord, CreatedClientApp

apps_app = typer.Typer(
    name="apps",
    no_args_is_help=True,
    help="Manage registered client applications.",
)


async def _create_app_async(
    db_path: Path,
    name: str,
    tiers: tuple[PermissionTier, ...],
    project_ids: tuple[str, ...],
    expires_days: int | None,
) -> CreatedClientApp:
    expires_at = utc_now() + timedelta(days=expires_days) if expires_days is not None else None
    created = generate_client_app(
        name=name,
        allowed_tiers=tiers,
        allowed_project_ids=project_ids,
        expires_at=expires_at,
    )
    async with open_client_app_repo(db_path) as repo:
        await repo.create(created.record)
    return created


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
    tier: Annotated[
        list[PermissionTier] | None,
        typer.Option("--tier", "-t", help="Allowed tool permission tier (repeatable)."),
    ] = None,
    project_id: Annotated[
        list[str] | None,
        typer.Option("--project-id", "-p", help="Optional allowed project ID (repeatable)."),
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
    """Register a new client application and print its one-time secret."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)

    effective_tiers = tuple(tier) if tier else (PermissionTier.READ,)
    effective_projects = tuple(project_id) if project_id else ()
    try:
        created = asyncio.run(
            _create_app_async(target_db, name, effective_tiers, effective_projects, expires_days)
        )
    except LaiError as exc:
        typer.echo(f"Error ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    tier_csv = ",".join(item.value for item in created.record.allowed_tiers)
    projects_csv = ",".join(created.record.allowed_project_ids) or "*"
    typer.echo(f"Created client app: id={created.record.id} prefix={created.record.secret_prefix}")
    typer.echo(f"Allowed tiers: {tier_csv} | Projects: {projects_csv}")
    typer.echo(f"Secret (store securely, shown once): {created.plaintext_secret}")


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
    """List registered client applications without revealing secrets."""
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
            f"{record.id}\t{record.name}\t{record.secret_prefix}\t{status_label}"
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
