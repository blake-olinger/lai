"""HuggingFace Hub GGUF model download and SHA-256 verification helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import httpx
from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
from huggingface_hub.errors import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
)

from lai.domain.errors import InferenceBackendError
from lai.domain.observability import AlertCategory
from lai.observability.integrity import verify_file_sha256
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    import structlog

    from lai.domain.guardrails import GuardrailConfig
    from lai.domain.inference import ModelProfile


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


class HfDownloader(Protocol):
    """Callable protocol matching huggingface_hub.hf_hub_download."""

    def __call__(  # noqa: PLR0913
        self,
        *,
        repo_id: str,
        filename: str,
        local_dir: str,
        force_download: bool,
        token: str | None,
        revision: str | None = None,
    ) -> object:
        """Download a file from a HuggingFace Hub repository."""
        ...


def default_hf_download(  # noqa: PLR0913
    *,
    repo_id: str,
    filename: str,
    local_dir: str,
    force_download: bool,
    token: str | None,
    revision: str | None = None,
) -> str:
    """Invoke huggingface_hub.hf_hub_download with strict types."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        force_download=force_download,
        token=token,
        revision=revision,
    )


def _optional_str_attr(obj: object, name: str) -> str | None:
    raw: object = getattr(obj, name, None)
    return raw if isinstance(raw, str) else None


def _unlink_quietly(path: Path) -> None:
    path.unlink(missing_ok=True)


async def _verify_model_digest_or_unlink(
    path: Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> None:
    if expected_sha256 is None:
        return
    try:
        await asyncio.to_thread(verify_file_sha256, path, expected_sha256, label=label)
    except InferenceBackendError:
        await asyncio.to_thread(_unlink_quietly, path)
        raise


def _log_download_failure(
    filename: str,
    repos: list[str],
    last_error: Exception | None,
) -> None:
    if isinstance(last_error, httpx.TimeoutException):
        _logger().error(
            "gguf_download_timeout",
            filename=filename,
            alert_category=AlertCategory.TIMEOUT.value,
            exception_type=type(last_error).__name__,
        )
        return
    _logger().error(
        "gguf_download_failed",
        filename=filename,
        repos=repos,
        error=str(last_error),
    )


async def download_hf_model(
    profile_or_config: ModelProfile | GuardrailConfig,
    *,
    models_dir: Path,
    hf_downloader: HfDownloader,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download a GGUF model from HuggingFace Hub with revision pinning and SHA-256 checks."""
    expected_sha256 = _optional_str_attr(profile_or_config, "sha256")
    hf_revision = _optional_str_attr(profile_or_config, "hf_revision")
    dest_path = models_dir / profile_or_config.hf_filename
    if not force and dest_path.exists():
        await _verify_model_digest_or_unlink(
            dest_path, expected_sha256, label=profile_or_config.hf_filename
        )
        return dest_path

    await asyncio.to_thread(models_dir.mkdir, parents=True, exist_ok=True)
    repos = [profile_or_config.hf_repo]
    fallback = profile_or_config.fallback_hf_repo
    if fallback and fallback not in repos:
        repos.append(fallback)

    last_error: Exception | None = None
    for repo_id in repos:
        repo_revision = hf_revision if repo_id == profile_or_config.hf_repo else None
        try:
            result = await asyncio.to_thread(
                hf_downloader,
                repo_id=repo_id,
                filename=profile_or_config.hf_filename,
                local_dir=str(models_dir),
                force_download=force,
                token=token,
                revision=repo_revision,
            )
        except (
            httpx.HTTPError,
            OSError,
            EntryNotFoundError,
            LocalEntryNotFoundError,
            HFValidationError,
            ValueError,
        ) as exc:
            last_error = exc
            _logger().warning(
                "gguf_download_attempt_failed",
                repo_id=repo_id,
                filename=profile_or_config.hf_filename,
                error=str(exc),
            )
            continue

        if not isinstance(result, str):
            msg = f"Unexpected HuggingFace download return type: {type(result).__name__}"
            raise InferenceBackendError(msg)
        resolved = Path(result)
        await _verify_model_digest_or_unlink(
            resolved,
            expected_sha256,
            label=f"{repo_id}/{profile_or_config.hf_filename}",
        )
        _logger().info("gguf_model_ready", repo_id=repo_id, path=str(resolved))
        return resolved

    _log_download_failure(profile_or_config.hf_filename, repos, last_error)
    msg = (
        f"Failed to download {profile_or_config.hf_filename!r} "
        f"from repositories {repos}: {last_error}"
    )
    raise InferenceBackendError(msg) from last_error
