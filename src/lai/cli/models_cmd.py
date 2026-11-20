"""CLI subcommands for downloading Linux llama.cpp CUDA binaries, GGUF models, and embeddings."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from lai.cli._common import resolve_settings
from lai.config.profiles import BUILTIN_PROFILES, get_model_profile
from lai.domain.errors import LaiError
from lai.inference import ModelDownloader
from lai.ingestion import FastEmbedCpuAdapter

if TYPE_CHECKING:
    from lai.config.settings import LaiSettings

models_app = typer.Typer(
    name="models",
    no_args_is_help=True,
    help="Download and inspect Linux llama.cpp binaries, GGUF models, and CPU embeddings.",
)


def build_downloader(settings: LaiSettings) -> ModelDownloader:
    """Construct a ModelDownloader configured from settings."""
    return ModelDownloader(bin_dir=settings.bin_dir, models_dir=settings.models_dir)


def build_embedder(settings: LaiSettings) -> FastEmbedCpuAdapter:
    """Construct a CPU-only FastEmbed adapter caching ONNX artifacts under models_dir/fastembed."""
    return FastEmbedCpuAdapter(cache_dir=settings.models_dir / "fastembed")


async def _setup_models_async(  # noqa: PLR0913
    settings: LaiSettings,
    *,
    profile_name: str,
    tag: str | None,
    cuda_version: str | None,
    hf_token: str | None,
    skip_binaries: bool,
    skip_guardrail: bool,
    skip_embeddings: bool,
    force: bool,
) -> dict[str, Path]:
    """Download llama-server CUDA binaries, Gemma 4 GGUF, ShieldGemma-2B GGUF, and FastEmbed."""
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

    if not skip_embeddings:
        embed_dir = settings.models_dir / "fastembed"
        embed_dir.mkdir(parents=True, exist_ok=True)
        embedder = build_embedder(settings)
        try:
            await embedder.embed_query("warmup")
        except (OSError, RuntimeError, ValueError) as exc:
            msg = f"Failed to pre-warm FastEmbed CPU model in {embed_dir}: {exc}"
            raise LaiError(msg, code="embeddings_setup_failed", status_code=500) from exc
        results["embeddings_cache"] = embed_dir

    return results


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
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag", help="Optional llama.cpp GitHub release tag (default: manifest pin)."
        ),
    ] = None,
    cuda_version: Annotated[
        str | None,
        typer.Option("--cuda-version", help="Preferred CUDA toolkit version (e.g. 12.8 or 13.5)."),
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
    skip_embeddings: Annotated[
        bool,
        typer.Option("--skip-embeddings", help="Skip pre-warming FastEmbed CPU ONNX embeddings."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download artifacts even if already present."),
    ] = False,
) -> None:
    """Download verified Linux llama-server binaries, GGUF models, and FastEmbed ONNX cache."""
    settings = resolve_settings(config)
    target_profile = profile or settings.default_model_profile
    try:
        resolved = asyncio.run(
            _setup_models_async(
                settings,
                profile_name=target_profile,
                tag=tag,
                cuda_version=cuda_version,
                hf_token=hf_token,
                skip_binaries=skip_binaries,
                skip_guardrail=skip_guardrail,
                skip_embeddings=skip_embeddings,
                force=force,
            )
        )
    except LaiError as exc:
        typer.echo(f"Model setup failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    for key, val in resolved.items():
        typer.echo(f"{key}: {val}")


@models_app.command("refresh-binaries")
def refresh_binaries(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML configuration path."),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag", help="Optional llama.cpp GitHub release tag (default: manifest pin)."
        ),
    ] = None,
    cuda_version: Annotated[
        str | None,
        typer.Option("--cuda-version", help="Preferred CUDA toolkit version (e.g. 12.8 or 13.5)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download binaries even if already present."),
    ] = False,
) -> None:
    """Download and activate SHA-256-verified Linux llama.cpp CUDA binaries from this slot's pin."""
    settings = resolve_settings(config)
    downloader = build_downloader(settings)
    try:
        binary_path = asyncio.run(
            downloader.download_llama_binaries(tag=tag, cuda_version=cuda_version, force=force)
        )
    except LaiError as exc:
        typer.echo(f"Binary refresh failed ({exc.code}): {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"llama_server: {binary_path}")


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
