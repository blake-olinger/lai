"""CLI subcommands for downloading llama.cpp CUDA binaries and GGUF models."""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TC003 - evaluated at runtime by Typer
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from lai.cli._common import resolve_settings
from lai.config.profiles import BUILTIN_PROFILES, get_model_profile
from lai.domain.errors import LaiError
from lai.inference import ModelDownloader, setup_ollama_models

if TYPE_CHECKING:
    from lai.config.settings import LaiSettings

models_app = typer.Typer(
    name="models",
    no_args_is_help=True,
    help="Download and inspect local llama.cpp binaries and GGUF models.",
)


def build_downloader(settings: LaiSettings) -> ModelDownloader:
    """Construct a ModelDownloader configured from settings."""
    return ModelDownloader(bin_dir=settings.bin_dir, models_dir=settings.models_dir)


async def _setup_models_async(  # noqa: PLR0913
    settings: LaiSettings,
    *,
    profile_name: str,
    tag: str | None,
    cuda_version: str | None,
    hf_token: str | None,
    skip_binaries: bool,
    skip_guardrail: bool,
    force: bool,
) -> dict[str, Path]:
    """Download CUDA llama-server binaries, primary Gemma 4 GGUF, and ShieldGemma-2B GGUF."""
    downloader = build_downloader(settings)
    profile = get_model_profile(profile_name, settings.custom_profiles)
    results: dict[str, Path] = {}

    if not skip_binaries:
        results["llama_server"] = await downloader.download_llama_binaries(
            tag=tag,
            cuda_version=cuda_version,
            force=force,
        )

    results["primary_model"] = await downloader.download_model(
        profile,
        force=force,
        token=hf_token,
    )

    if settings.guardrails.enabled and not skip_guardrail:
        results["guardrail_model"] = await downloader.download_model(
            settings.guardrails,
            force=force,
            token=hf_token,
        )

    return results


async def _setup_ollama_async(
    settings: LaiSettings,
    *,
    profile_name: str,
    skip_guardrail: bool,
) -> dict[str, str]:
    """Pull HF GGUFs via Ollama and create tuned local aliases."""
    profile = get_model_profile(profile_name, settings.custom_profiles)
    return await setup_ollama_models(
        profile,
        settings.guardrails,
        base_url=settings.ollama_base_url,
        skip_guardrail=skip_guardrail,
    )


@models_app.command("setup")
def setup_models(  # noqa: PLR0913, PLR0917
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Model profile name to download."),
    ] = None,
    backend: Annotated[
        Literal["llamacpp", "ollama"] | None,
        typer.Option("--backend", "-b", help="Override inference backend (llamacpp or ollama)."),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", help="Optional llama.cpp GitHub release tag."),
    ] = None,
    cuda_version: Annotated[
        str | None,
        typer.Option("--cuda-version", help="Preferred CUDA toolkit filter (e.g. 12.4)."),
    ] = None,
    hf_token: Annotated[
        str | None,
        typer.Option("--hf-token", envvar="HF_TOKEN", help="Optional HuggingFace Hub token."),
    ] = None,
    skip_binaries: Annotated[
        bool,
        typer.Option("--skip-binaries", help="Skip downloading llama.cpp CUDA binaries."),
    ] = False,
    skip_guardrail: Annotated[
        bool,
        typer.Option("--skip-guardrail", help="Skip downloading ShieldGemma-2B GGUF."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download artifacts even if already present."),
    ] = False,
) -> None:
    """Download CUDA llama-server or pull Ollama models + ShieldGemma-2B."""
    settings = resolve_settings(config)
    target_profile = profile or settings.default_model_profile
    effective_backend = backend or settings.inference_backend
    try:
        if effective_backend == "ollama":
            resolved: dict[str, Path] | dict[str, str] = asyncio.run(
                _setup_ollama_async(
                    settings,
                    profile_name=target_profile,
                    skip_guardrail=skip_guardrail,
                )
            )
        else:
            resolved = asyncio.run(
                _setup_models_async(
                    settings,
                    profile_name=target_profile,
                    tag=tag,
                    cuda_version=cuda_version,
                    hf_token=hf_token,
                    skip_binaries=skip_binaries,
                    skip_guardrail=skip_guardrail,
                    force=force,
                )
            )
    except LaiError as exc:
        typer.echo(f"Model setup failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    for key, val in resolved.items():
        typer.echo(f"{key}: {val}")


@models_app.command("list")
def list_profiles(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
) -> None:
    """List built-in and custom model profiles."""
    settings = resolve_settings(config)
    combined = {**BUILTIN_PROFILES, **settings.custom_profiles}
    for name, item in combined.items():
        marker = "*" if name == settings.default_model_profile else " "
        typer.echo(f"{marker} {name}\t{item.hf_repo}/{item.hf_filename}\tctx={item.ctx_size}")
