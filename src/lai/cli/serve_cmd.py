"""CLI command supervising Gemma 4 (:8081), ShieldGemma (:8082), and Uvicorn (:8400)."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path  # noqa: TC003 - evaluated at runtime by Typer
from typing import TYPE_CHECKING, Annotated, Protocol, cast

import typer
import uvicorn

from lai.cli._common import resolve_llama_executable, resolve_log_dir, resolve_settings
from lai.config.profiles import get_model_profile
from lai.domain.errors import LaiError
from lai.guardrails import ShieldGemmaSupervisor
from lai.inference import LlamaCppSupervisor

if TYPE_CHECKING:
    from starlette.types import ASGIApp

    from lai.config.settings import LaiSettings


class _AppFactory(Protocol):
    def __call__(self, settings: LaiSettings) -> ASGIApp: ...


def _load_create_app() -> _AppFactory:
    """Load lai.api.app.create_app dynamically to preserve layer independence."""
    module = importlib.import_module("lai.api.app")
    factory = module.create_app
    return cast("_AppFactory", factory)


def build_supervisors(
    settings: LaiSettings,
) -> tuple[LlamaCppSupervisor, ShieldGemmaSupervisor | None]:
    """Construct primary Gemma 4 and optional CPU ShieldGemma supervisors."""
    exe_path = resolve_llama_executable(settings.bin_dir)
    log_dir = resolve_log_dir(settings)

    llama_supervisor = LlamaCppSupervisor(
        executable_path=exe_path,
        models_dir=settings.models_dir,
        log_dir=log_dir,
        host=settings.llama_host,
        port=settings.llama_port,
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
        )
    return llama_supervisor, guardrail_supervisor


async def serve_async(settings: LaiSettings, *, supervise: bool = True) -> None:
    """Start supervised llama-server instances and run the Uvicorn API server."""
    llama_supervisor: LlamaCppSupervisor | None = None
    guardrail_supervisor: ShieldGemmaSupervisor | None = None

    try:
        if supervise:
            llama_supervisor, guardrail_supervisor = build_supervisors(settings)
            profile = get_model_profile(
                settings.default_model_profile,
                settings.custom_profiles,
            )
            await llama_supervisor.ensure_running(profile)
            if guardrail_supervisor is not None:
                await guardrail_supervisor.ensure_running(settings.guardrails)

        app = _load_create_app()(settings)
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
            help="Skip spawning local llama-server and ShieldGemma subprocesses.",
        ),
    ] = False,
) -> None:
    """Start Gemma 4 (:8081), ShieldGemma-2B (:8082), and the FastAPI server (:8400)."""
    overrides: dict[str, object] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if profile is not None:
        overrides["default_model_profile"] = profile

    settings = resolve_settings(config, **overrides)
    try:
        asyncio.run(serve_async(settings, supervise=not no_supervise))
    except LaiError as exc:
        typer.echo(f"Server startup failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
