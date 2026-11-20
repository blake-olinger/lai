"""CLI command initializing local workspace directories, SQLite DB, and config YAML."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import yaml

from lai.cli._common import chmod_posix
from lai.config.profiles import DEFAULT_MODEL_PROFILE_NAME
from lai.db import SqliteDatabase

_POSIX_DIR_MODE = 0o700
_POSIX_FILE_MODE = 0o600


async def _init_database(db_path: Path) -> None:
    """Initialize SQLite schema idempotently."""
    db = SqliteDatabase(db_path)
    await db.initialize()
    await db.close()


def _build_initial_config_payload(
    *,
    host: str,
    port: int,
    data_dir: Path,
    models_dir: Path,
    bin_dir: Path,
) -> dict[str, object]:
    """Build default configuration dictionary serialized to config.yaml."""
    return {
        "host": host,
        "port": port,
        "llama_host": "127.0.0.1",
        "llama_port": 8081,
        "data_dir": str(data_dir),
        "models_dir": str(models_dir),
        "bin_dir": str(bin_dir),
        "default_model_profile": DEFAULT_MODEL_PROFILE_NAME,
        "auth_providers": ["api_key"],
        "model_integrity": {
            "strict_allowlist": True,
            "verify_sha256_on_startup": True,
        },
        "guardrails": {
            "enabled": True,
            "block_mode": "refusal_message",
            "output_mode": "buffered",
            "allow_runtime_disable": False,
            "max_violation_threshold": 0.95,
            "host": "127.0.0.1",
            "port": 8082,
        },
        "logging": {
            "log_dir": str(data_dir / "logs"),
            "level": "INFO",
            "retention_days": 14,
            "error_retention_days": 30,
        },
    }


def init_command(  # noqa: PLR0913, PLR0917
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Target YAML config path to write."),
    ] = Path("config/config.yaml"),
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", help="Directory for SQLite DB, blobs, backups, logs, and LanceDB."
        ),
    ] = Path("data"),
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir", help="Directory for downloaded GGUF models."),
    ] = Path("models"),
    bin_dir: Annotated[
        Path,
        typer.Option("--bin-dir", help="Directory for downloaded llama.cpp binaries."),
    ] = Path("bin"),
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host for the API server."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8400,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing config file if present."),
    ] = False,
) -> None:
    """Initialize local LAI workspace directories, SQLite database, and YAML config."""
    private_dirs = (
        data_dir,
        data_dir / "blobs",
        data_dir / "lancedb",
        data_dir / "backups",
    )
    for directory in (*private_dirs, data_dir / "logs", models_dir, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for private_dir in private_dirs:
        chmod_posix(private_dir, _POSIX_DIR_MODE)

    db_path = data_dir / "lai.db"
    asyncio.run(_init_database(db_path))
    chmod_posix(db_path, _POSIX_FILE_MODE)

    config_status = "kept existing"
    if not config.exists() or force:
        config.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_initial_config_payload(
            host=host,
            port=port,
            data_dir=data_dir,
            models_dir=models_dir,
            bin_dir=bin_dir,
        )
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        chmod_posix(config, _POSIX_FILE_MODE)
        config_status = "wrote"

    typer.echo(f"Initialized database: {db_path}")
    typer.echo(f"Config ({config_status}): {config}")
    typer.echo(f"Data directory: {data_dir}")
