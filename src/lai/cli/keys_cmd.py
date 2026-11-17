"""CLI subcommands for minting, listing, and revoking API keys."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path  # noqa: TC003 - evaluated at runtime by Typer
from typing import Annotated

import typer

from lai.cli._common import open_api_key_repo, resolve_db_path, resolve_settings
from lai.domain.auth import ApiKeyRecord, ApiScope, CreatedApiKey
from lai.domain.base import utc_now
from lai.security import generate_api_key

keys_app = typer.Typer(name="keys", no_args_is_help=True, help="Manage LAI API keys.")

_DEFAULT_USER_SCOPES: tuple[ApiScope, ...] = (
    ApiScope.PROJECTS_READ,
    ApiScope.FILES_READ,
    ApiScope.FILES_WRITE,
    ApiScope.CHAT_READ,
    ApiScope.CHAT_WRITE,
    ApiScope.TOOLS_SAFE,
)


async def _create_key_async(
    db_path: Path,
    name: str,
    scopes: tuple[ApiScope, ...],
    expires_days: int | None,
) -> CreatedApiKey:
    expires_at = utc_now() + timedelta(days=expires_days) if expires_days is not None else None
    created = generate_api_key(name=name, scopes=scopes, expires_at=expires_at)
    async with open_api_key_repo(db_path) as repo:
        await repo.create(created.record)
    return created


async def _list_keys_async(db_path: Path) -> list[ApiKeyRecord]:
    async with open_api_key_repo(db_path) as repo:
        return await repo.list_all()


async def _revoke_key_async(db_path: Path, key_id: str) -> bool:
    async with open_api_key_repo(db_path) as repo:
        return await repo.revoke(key_id, utc_now())


@keys_app.command("create")
def create_key(  # noqa: PLR0913, PLR0917
    name: Annotated[str, typer.Option("--name", "-n", help="Human-readable label for the key.")],
    scope: Annotated[
        list[ApiScope] | None,
        typer.Option("--scope", "-s", help="Granted API scope (repeatable)."),
    ] = None,
    admin: Annotated[
        bool,
        typer.Option("--admin", help="Grant full admin scope to this key."),
    ] = False,
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
    """Mint a new API key and print the one-time plaintext secret."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)

    scopes_list: list[ApiScope] = list(scope) if scope else []
    if admin:
        scopes_list.append(ApiScope.ADMIN)
    effective_scopes = tuple(scopes_list) if scopes_list else _DEFAULT_USER_SCOPES

    created = asyncio.run(_create_key_async(target_db, name, effective_scopes, expires_days))
    scope_csv = ",".join(item.value for item in created.record.scopes)
    typer.echo(f"Created API key: id={created.record.id} prefix={created.record.key_prefix}")
    typer.echo(f"Scopes: {scope_csv}")
    typer.echo(f"Secret (store securely, shown once): {created.plaintext_key}")


@keys_app.command("list")
def list_keys(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Optional explicit SQLite DB path override."),
    ] = None,
) -> None:
    """List persisted API key metadata without revealing secrets."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)
    records = asyncio.run(_list_keys_async(target_db))

    if not records:
        typer.echo("No API keys found.")
        return

    now = utc_now()
    for record in records:
        if record.revoked_at is not None:
            status_label = "revoked"
        elif record.expires_at is not None and now >= record.expires_at:
            status_label = "expired"
        else:
            status_label = "active"
        scope_csv = ",".join(item.value for item in record.scopes)
        typer.echo(
            f"{record.id}\t{record.name}\t{record.key_prefix}\t{status_label}\t[{scope_csv}]"
        )


@keys_app.command("revoke")
def revoke_key(
    key_id: Annotated[str, typer.Argument(help="API key ID (key_...) to revoke.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    db_path: Annotated[
        Path | None,
        typer.Option("--db-path", help="Optional explicit SQLite DB path override."),
    ] = None,
) -> None:
    """Revoke an active API key by ID."""
    settings = resolve_settings(config)
    target_db = resolve_db_path(settings, db_path)
    revoked = asyncio.run(_revoke_key_async(target_db, key_id))
    if not revoked:
        typer.echo(f"API key not found: {key_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Revoked API key: {key_id}")
