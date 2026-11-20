"""CLI subcommands for checking, applying, and rolling back verified LAI releases."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from lai.api._runtime_config_sections import merge_persisted_runtime_overrides
from lai.cli._common import resolve_db_path, resolve_log_dir, resolve_settings
from lai.cli._upgrade_helpers import refresh_slot_binaries
from lai.db import SqliteDatabase, SqliteDatabaseBackupAdapter, SqliteRuntimeConfigRepository
from lai.domain.errors import LaiError
from lai.domain.updates import UpgradeCheckResult, UpgradeResult, UpgradeStatus
from lai.domain.version import PACKAGE_VERSION
from lai.observability import AlertManager, configure_observability
from lai.updates import (
    GitHubReleaseSource,
    HttpHealthProbe,
    LocalWheelSource,
    ReleaseSlotManager,
    SystemdServiceController,
    UpgradeCoordinator,
)

if TYPE_CHECKING:
    from lai.config.settings import LaiSettings
    from lai.ports.updates import ReleaseSourcePort

upgrade_app = typer.Typer(
    name="upgrade",
    no_args_is_help=True,
    help="Check, apply, and roll back SHA-256-verified LAI releases.",
)


async def load_effective_upgrade_settings(settings: LaiSettings) -> LaiSettings:
    """Merge persisted SQLite runtime_settings overrides onto YAML/env settings."""
    db_path = resolve_db_path(settings)
    if not db_path.is_file():
        return settings
    db = SqliteDatabase(db_path)
    try:
        await db.connect()
        repo = SqliteRuntimeConfigRepository(db)
        persisted = await repo.get_all_sections()
    except LaiError:
        return settings
    finally:
        await db.close()
    if not persisted:
        return settings
    return merge_persisted_runtime_overrides(
        settings,
        persisted,
        boot_guardrails_config=settings.guardrails,
        boot_smtp_host=settings.logging.alert_smtp.host,
        boot_smtp_port=settings.logging.alert_smtp.port,
    )


def _init_upgrade_observability(settings: LaiSettings) -> AlertManager:
    """Initialize journal/console logging and AlertManager without opening root-owned log files."""
    log_cfg = settings.logging.model_copy(
        update={"log_dir": resolve_log_dir(settings), "enable_file_logs": False}
    )
    return configure_observability(log_cfg, fallback_admin_emails=settings.oidc_admin_emails)


async def _shutdown_upgrade_observability(alert_manager: AlertManager) -> None:
    """Flush AlertManager and cleanly detach root log handlers before CLI exit."""
    await alert_manager.flush()
    alert_manager.close()
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def build_upgrade_coordinator(  # noqa: PLR0913
    settings: LaiSettings,
    *,
    wheel: Path | None = None,
    sha256: str | None = None,
    requirements: Path | None = None,
    requirements_sha256: str | None = None,
    github_token: str | None = None,
) -> UpgradeCoordinator:
    """Construct UpgradeCoordinator wired to GitHub/local wheel source, slots, systemd, and DB."""
    source: ReleaseSourcePort = (
        LocalWheelSource(
            wheel,
            expected_sha256=sha256,
            requirements_path=requirements,
            requirements_sha256=requirements_sha256,
        )
        if wheel is not None
        else GitHubReleaseSource(settings.updates.github_repo, token=github_token)
    )
    slots = ReleaseSlotManager(
        settings.updates.releases_dir,
        settings.updates.current_symlink,
    )
    return UpgradeCoordinator(
        config=settings.updates,
        source=source,
        slots=slots,
        service=SystemdServiceController(),
        health_probe=HttpHealthProbe(),
        backup=SqliteDatabaseBackupAdapter(),
    )


async def _run_check_async(
    base_settings: LaiSettings,
    *,
    github_token: str | None,
) -> UpgradeCheckResult:
    settings = await load_effective_upgrade_settings(base_settings)
    alert_manager = _init_upgrade_observability(settings)
    try:
        coordinator = build_upgrade_coordinator(settings, github_token=github_token)
        return await coordinator.check(PACKAGE_VERSION)
    finally:
        await _shutdown_upgrade_observability(alert_manager)


async def _run_apply_async(  # noqa: PLR0913
    base_settings: LaiSettings,
    *,
    config_path: Path | None,
    version: str | None,
    wheel: Path | None,
    sha256: str | None,
    requirements: Path | None,
    requirements_sha256: str | None,
    with_binaries: bool,
    if_enabled: bool,
    dry_run: bool,
    force: bool,
    github_token: str | None,
) -> tuple[UpgradeResult, Path | None]:
    settings = await load_effective_upgrade_settings(base_settings)
    alert_manager = _init_upgrade_observability(settings)
    binary_path: Path | None = None
    try:
        if if_enabled and not settings.updates.enabled:
            return (
                UpgradeResult(status=UpgradeStatus.SKIPPED_DISABLED, from_version=PACKAGE_VERSION),
                None,
            )
        coordinator = build_upgrade_coordinator(
            settings,
            wheel=wheel,
            sha256=sha256,
            requirements=requirements,
            requirements_sha256=requirements_sha256,
            github_token=github_token,
        )

        async def _refresh_hook(slot_path: Path) -> Path | None:
            nonlocal binary_path
            binary_path = await refresh_slot_binaries(
                slot_path,
                config_path=config_path,
                bin_dir=settings.bin_dir,
            )
            return binary_path

        db_path = resolve_db_path(settings)
        result = await coordinator.apply(
            PACKAGE_VERSION,
            target_version=version,
            dry_run=dry_run,
            db_path=db_path if db_path.is_file() else None,
            respect_enabled=if_enabled,
            force=force,
            pre_activate_hook=_refresh_hook if (with_binaries and not dry_run) else None,
        )
        return result, binary_path
    finally:
        await _shutdown_upgrade_observability(alert_manager)


async def _run_rollback_async(
    base_settings: LaiSettings,
    *,
    backup: Path | None,
) -> UpgradeResult:
    settings = await load_effective_upgrade_settings(base_settings)
    alert_manager = _init_upgrade_observability(settings)
    try:
        coordinator = build_upgrade_coordinator(settings)
        db_path = resolve_db_path(settings)
        return await coordinator.rollback(
            PACKAGE_VERSION,
            db_path=db_path if db_path.is_file() else None,
            backup_path=backup,
        )
    finally:
        await _shutdown_upgrade_observability(alert_manager)


@upgrade_app.command("check")
def check_upgrade(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", envvar="GITHUB_TOKEN", help="Optional GitHub API token."),
    ] = None,
) -> None:
    """Check GitHub Releases for a newer verified LAI wheel."""
    settings = resolve_settings(config)
    try:
        result = asyncio.run(_run_check_async(settings, github_token=token))
    except LaiError as exc:
        typer.echo(f"Upgrade check failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Current version: {result.current_version}")
    typer.echo(f"Latest version:  {result.latest_version}")
    typer.echo(f"Update available: {'yes' if result.update_available else 'no'}")


@upgrade_app.command("apply")
def apply_upgrade(  # noqa: PLR0913, PLR0917
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    version: Annotated[
        str | None,
        typer.Option("--version", help="Specific release version (e.g. 0.2.0) to install."),
    ] = None,
    wheel: Annotated[
        Path | None,
        typer.Option("--wheel", help="Path to a staged local wheel file for CI/CD push deploys."),
    ] = None,
    sha256: Annotated[
        str | None,
        typer.Option("--sha256", help="Expected 64-hex SHA-256 digest when using --wheel."),
    ] = None,
    requirements: Annotated[
        Path | None,
        typer.Option(
            "--requirements",
            help="Path to hashed requirements-lock.txt when using --wheel.",
        ),
    ] = None,
    requirements_sha256: Annotated[
        str | None,
        typer.Option(
            "--requirements-sha256",
            help="Expected 64-hex SHA-256 digest for --requirements.",
        ),
    ] = None,
    with_binaries: Annotated[
        bool,
        typer.Option("--with-binaries", help="Also refresh pinned Linux llama.cpp CUDA binaries."),
    ] = False,
    if_enabled: Annotated[
        bool,
        typer.Option("--if-enabled", help="No-op cleanly unless updates.enabled is True."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Check target release without modifying slots or service."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Install target release even if not newer than current."),
    ] = False,
    token: Annotated[
        str | None,
        typer.Option("--token", envvar="GITHUB_TOKEN", help="Optional GitHub API token."),
    ] = None,
) -> None:
    """Download, SHA-256 verify, back up SQLite, activate slot, and verify health."""
    if wheel is None and (
        sha256 is not None or requirements is not None or requirements_sha256 is not None
    ):
        typer.echo(
            "--sha256, --requirements, and --requirements-sha256 require --wheel.",
            err=True,
        )
        raise typer.Exit(code=2)
    settings = resolve_settings(config)
    try:
        result, binary_path = asyncio.run(
            _run_apply_async(
                settings,
                config_path=config,
                version=version,
                wheel=wheel,
                sha256=sha256,
                requirements=requirements,
                requirements_sha256=requirements_sha256,
                with_binaries=with_binaries,
                if_enabled=if_enabled,
                dry_run=dry_run,
                force=force,
                github_token=token,
            )
        )
    except LaiError as exc:
        typer.echo(f"Upgrade failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    if binary_path is not None:
        typer.echo(f"llama_server: {binary_path}")
    typer.echo(f"Status: {result.status.value} ({result.from_version} -> {result.to_version})")
    if result.slot_path is not None:
        typer.echo(f"Slot: {result.slot_path}")
    if result.backup_path is not None:
        typer.echo(f"Backup: {result.backup_path}")
    if result.status in {UpgradeStatus.FAILED, UpgradeStatus.ROLLED_BACK}:
        typer.echo(f"Error: {result.error or 'Upgrade did not succeed.'}", err=True)
        raise typer.Exit(code=1)


@upgrade_app.command("rollback")
def rollback_upgrade(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    backup: Annotated[
        Path | None,
        typer.Option("--backup", help="Optional SQLite backup file to restore atomically."),
    ] = None,
) -> None:
    """Roll back /opt/lai/current to the previous release slot and restart lai.service."""
    settings = resolve_settings(config)
    try:
        result = asyncio.run(_run_rollback_async(settings, backup=backup))
    except LaiError as exc:
        typer.echo(f"Rollback failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Status: {result.status.value} ({result.from_version} -> {result.to_version})")
    if result.slot_path is not None:
        typer.echo(f"Slot: {result.slot_path}")
    if result.status is not UpgradeStatus.ROLLED_BACK:
        typer.echo(f"Error: {result.error or 'Rollback failed.'}", err=True)
        raise typer.Exit(code=1)
