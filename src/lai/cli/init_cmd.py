"""CLI command initializing local workspace directories, SQLite DB, and config YAML."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer
import yaml

from lai.cli._common import chmod_posix
from lai.config.profiles import DEFAULT_MODEL_PROFILE_NAME
from lai.config.settings import load_yaml_mapping
from lai.db import SqliteDatabase
from lai.domain.errors import LaiError

_POSIX_DIR_MODE = 0o700
_POSIX_FILE_MODE = 0o600


def _optional_path(mapping: dict[str, object], key: str) -> Path | None:
    """Return a Path from a mapping key when present as a non-empty string."""
    value = mapping.get(key)
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def _nested_mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    """Return a nested string-keyed mapping or an empty dict."""
    value = mapping.get(key)
    if isinstance(value, dict):
        nested = cast("dict[object, object]", value)
        return {str(k): v for k, v in nested.items()}
    return {}


def _resolve_init_paths(
    *,
    config: Path,
    force: bool,
    data_dir: Path | None,
    models_dir: Path | None,
    bin_dir: Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve workspace directories from CLI flags, existing YAML config, or defaults."""
    yaml_cfg: dict[str, object] = {}
    if config.exists() and not force:
        try:
            yaml_cfg = load_yaml_mapping(config)
        except LaiError as exc:
            typer.echo(f"Configuration error ({exc.code}): {exc.message}", err=True)
            raise typer.Exit(code=1) from exc

    cfg_data_dir = _optional_path(yaml_cfg, "data_dir")
    cfg_models_dir = _optional_path(yaml_cfg, "models_dir")
    cfg_bin_dir = _optional_path(yaml_cfg, "bin_dir")
    cfg_log_dir = _optional_path(_nested_mapping(yaml_cfg, "logging"), "log_dir")
    cfg_backups_dir = _optional_path(_nested_mapping(yaml_cfg, "updates"), "backups_dir")

    resolved_data_dir = data_dir if data_dir is not None else (cfg_data_dir or Path("data"))
    resolved_models_dir = (
        models_dir if models_dir is not None else (cfg_models_dir or Path("models"))
    )
    resolved_bin_dir = bin_dir if bin_dir is not None else (cfg_bin_dir or Path("bin"))
    resolved_log_dir = (
        cfg_log_dir if data_dir is None and cfg_log_dir is not None else resolved_data_dir / "logs"
    )
    resolved_backups_dir = (
        cfg_backups_dir
        if data_dir is None and cfg_backups_dir is not None
        else resolved_data_dir / "backups"
    )
    return (
        resolved_data_dir,
        resolved_models_dir,
        resolved_bin_dir,
        resolved_log_dir,
        resolved_backups_dir,
    )


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
        typer.Option("--config", "-c", help="Target YAML config path to read or write."),
    ] = Path("config/config.yaml"),
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Directory for SQLite DB, blobs, backups, logs, and LanceDB "
            "(defaults to existing config value or ./data).",
        ),
    ] = None,
    models_dir: Annotated[
        Path | None,
        typer.Option(
            "--models-dir",
            help="Directory for downloaded GGUF models "
            "(defaults to existing config value or ./models).",
        ),
    ] = None,
    bin_dir: Annotated[
        Path | None,
        typer.Option(
            "--bin-dir",
            help="Directory for downloaded llama.cpp binaries "
            "(defaults to existing config value or ./bin).",
        ),
    ] = None,
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
    (
        resolved_data_dir,
        resolved_models_dir,
        resolved_bin_dir,
        resolved_log_dir,
        resolved_backups_dir,
    ) = _resolve_init_paths(
        config=config,
        force=force,
        data_dir=data_dir,
        models_dir=models_dir,
        bin_dir=bin_dir,
    )
    private_dirs = (
        resolved_data_dir,
        resolved_data_dir / "blobs",
        resolved_data_dir / "lancedb",
        resolved_backups_dir,
    )
    for directory in (*private_dirs, resolved_log_dir, resolved_models_dir, resolved_bin_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for private_dir in private_dirs:
        chmod_posix(private_dir, _POSIX_DIR_MODE)

    db_path = resolved_data_dir / "lai.db"
    asyncio.run(_init_database(db_path))
    chmod_posix(db_path, _POSIX_FILE_MODE)

    config_status = "kept existing"
    if not config.exists() or force:
        config.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_initial_config_payload(
            host=host,
            port=port,
            data_dir=resolved_data_dir,
            models_dir=resolved_models_dir,
            bin_dir=resolved_bin_dir,
        )
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        chmod_posix(config, _POSIX_FILE_MODE)
        config_status = "wrote"

    typer.echo(f"Initialized database: {db_path}")
    typer.echo(f"Config ({config_status}): {config}")
    typer.echo(f"Data directory: {resolved_data_dir}")
