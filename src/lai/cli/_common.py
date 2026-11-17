"""Shared configuration, database, and path helpers for CLI commands."""

from __future__ import annotations

import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from lai.config.settings import LaiSettings, load_settings
from lai.db import SqliteApiKeyRepository, SqliteDatabase
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


def resolve_llama_executable(bin_dir: Path) -> Path:
    """Locate the llama-server executable under bin_dir/llama or PATH."""
    llama_dir = bin_dir / "llama"
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    direct = llama_dir / binary_name
    if direct.exists():
        return direct
    if llama_dir.exists():
        found = next(llama_dir.rglob(binary_name), None)
        if found is not None:
            return found
    on_path = shutil.which(binary_name) or shutil.which("llama-server")
    if on_path is not None:
        return Path(on_path)
    return direct


@asynccontextmanager
async def open_api_key_repo(db_path: Path) -> AsyncGenerator[SqliteApiKeyRepository]:
    """Open and initialize SQLite database and yield SqliteApiKeyRepository."""
    db = SqliteDatabase(db_path)
    await db.initialize()
    try:
        yield SqliteApiKeyRepository(db)
    finally:
        await db.close()
