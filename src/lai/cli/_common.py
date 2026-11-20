"""Shared configuration, database, and path helpers for CLI commands."""

from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from lai.config.settings import LaiSettings, load_settings
from lai.db import SqliteApiKeyRepository, SqliteClientAppRepository, SqliteDatabase
from lai.domain.errors import LaiError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def resolve_settings(
    config_path: Path | None,
    **overrides: object,
) -> LaiSettings:
    """Load LaiSettings from optional YAML config and overrides, exiting cleanly on LaiError."""
    try:
        return load_settings(config_path, **overrides)
    except LaiError as exc:
        typer.echo(f"Configuration error ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc


def resolve_log_dir(settings: LaiSettings) -> Path:
    """Resolve the active log directory honoring custom data_dir overrides."""
    if settings.logging.log_dir == Path("data/logs") and settings.data_dir != Path("data"):
        return settings.data_dir / "logs"
    return settings.logging.log_dir


def resolve_db_path(settings: LaiSettings, db_path: Path | None = None) -> Path:
    """Resolve SQLite database path from explicit override or settings.data_dir."""
    return db_path if db_path is not None else settings.data_dir / "lai.db"


def chmod_posix(path: Path, mode: int) -> None:
    """Apply POSIX file or directory permissions on a best-effort basis."""
    if os.name == "posix":
        with suppress(OSError):
            path.chmod(mode)


def resolve_llama_executable(bin_dir: Path) -> Path:
    """Locate Linux llama-server under bin_dir/llama/current, bin_dir/llama, or PATH."""
    llama_dir = bin_dir / "llama"
    binary_name = "llama-server"
    current = llama_dir / "current" / binary_name
    if current.exists():
        return current
    direct = llama_dir / binary_name
    if direct.exists():
        return direct
    if llama_dir.exists():
        found = next(llama_dir.rglob(binary_name), None)
        if found is not None:
            return found
    on_path = shutil.which(binary_name)
    if on_path is not None:
        return Path(on_path)
    return current


@asynccontextmanager
async def open_api_key_repo(db_path: Path) -> AsyncGenerator[SqliteApiKeyRepository]:
    """Open and initialize SQLite database and yield SqliteApiKeyRepository."""
    db = SqliteDatabase(db_path)
    await db.initialize()
    try:
        yield SqliteApiKeyRepository(db)
    finally:
        await db.close()


@asynccontextmanager
async def open_client_app_repo(db_path: Path) -> AsyncGenerator[SqliteClientAppRepository]:
    """Open and initialize SQLite database and yield SqliteClientAppRepository."""
    db = SqliteDatabase(db_path)
    await db.initialize()
    try:
        yield SqliteClientAppRepository(db)
    finally:
        await db.close()
