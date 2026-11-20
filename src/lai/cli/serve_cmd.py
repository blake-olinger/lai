"""CLI command supervising Gemma 4 (:8081), ShieldGemma (:8082), and Uvicorn (:8400)."""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TC003 - evaluated at runtime by Typer
from typing import TYPE_CHECKING, Annotated

import typer
import uvicorn

from lai.api.app import create_app
from lai.cli._common import resolve_llama_executable, resolve_log_dir, resolve_settings
from lai.config.profiles import get_model_profile
from lai.domain.errors import LaiError
from lai.guardrails import ShieldGemmaSupervisor
from lai.inference import LlamaCppSupervisor

if TYPE_CHECKING:
    from lai.config.settings import LaiSettings
    from lai.domain.guardrails import GuardrailConfig
    from lai.domain.inference import ModelProfile
    from lai.ports.inference import InferenceSupervisorPort

__all__ = ["build_supervisors", "create_app", "serve_async", "serve_command"]


async def _start_supervisors(
    llama_supervisor: InferenceSupervisorPort,
    guardrail_supervisor: ShieldGemmaSupervisor | None,
    profile: ModelProfile,
    guardrail_config: GuardrailConfig,
) -> None:
    """Start primary GPU and CPU ShieldGemma supervisors concurrently, cancelling on failure."""
    if guardrail_supervisor is None:
        await llama_supervisor.ensure_running(profile)
        return
    tasks: tuple[asyncio.Task[object], asyncio.Task[object]] = (
        asyncio.create_task(llama_supervisor.ensure_running(profile)),
        asyncio.create_task(guardrail_supervisor.ensure_running(guardrail_config)),
    )
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def build_supervisors(
    settings: LaiSettings,
) -> tuple[InferenceSupervisorPort, ShieldGemmaSupervisor | None]:
    """Construct primary llama-server supervisor and optional CPU ShieldGemma supervisor."""
    log_dir = resolve_log_dir(settings)
    exe_path = resolve_llama_executable(settings.bin_dir)
    verify_sha = settings.model_integrity.verify_sha256_on_startup
    llama_supervisor = LlamaCppSupervisor(
        executable_path=exe_path,
        models_dir=settings.models_dir,
        log_dir=log_dir,
        host=settings.llama_host,
        port=settings.llama_port,
        verify_sha256=verify_sha,
    )
    guardrail_supervisor: ShieldGemmaSupervisor | None = None
    if settings.guardrails.enabled:
        guardrail_model_path = settings.models_dir / settings.guardrails.hf_filename
        guardrail_supervisor = ShieldGemmaSupervisor(
            executable_path=exe_path,
            model_path=guardrail_model_path,
            log_dir=log_dir,
            retention_days=settings.logging.retention_days,
            compress_rotated=settings.logging.compress_rotated,
            expected_sha256=settings.guardrails.sha256,
            verify_sha256=verify_sha,
        )
    return llama_supervisor, guardrail_supervisor


async def serve_async(
    settings: LaiSettings,
    *,
    supervise: bool = True,
    config_path: Path | None = None,
) -> None:
    """Start supervised backend instances and run the Uvicorn API server."""
    llama_supervisor: InferenceSupervisorPort | None = None
    guardrail_supervisor: ShieldGemmaSupervisor | None = None

    try:
        if supervise:
            llama_supervisor, guardrail_supervisor = build_supervisors(settings)
            profile = get_model_profile(
                settings.default_model_profile,
                settings.custom_profiles,
            )
            await _start_supervisors(
                llama_supervisor,
                guardrail_supervisor,
                profile,
                settings.guardrails,
            )

        app = create_app(settings, config_path=config_path)
        uv_config = uvicorn.Config(
            app=app,
            host=settings.host,
            port=settings.port,
            log_config=None,
        )
        server = uvicorn.Server(uv_config)
        await server.serve()
    finally:
        if guardrail_supervisor is not None:
            await guardrail_supervisor.stop()
        if llama_supervisor is not None:
            await llama_supervisor.stop()


def serve_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", help="Override API listen host."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="Override API listen port."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Override active model profile."),
    ] = None,
    no_supervise: Annotated[
        bool,
        typer.Option(
            "--no-supervise",
            help="Skip spawning local inference/guardrail subprocesses.",
        ),
    ] = False,
) -> None:
    """Start llama-server inference, CPU ShieldGemma-2B guardrail, and FastAPI server (:8400)."""
    overrides: dict[str, object] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if profile is not None:
        overrides["default_model_profile"] = profile

    settings = resolve_settings(config, **overrides)
    try:
        asyncio.run(serve_async(settings, supervise=not no_supervise, config_path=config))
    except LaiError as exc:
        typer.echo(f"Server startup failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
