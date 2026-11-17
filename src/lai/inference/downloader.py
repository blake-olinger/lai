"""Downloader for llama.cpp Windows CUDA binaries and HuggingFace GGUF models."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import httpx
from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
from huggingface_hub.errors import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
)

from lai.domain.errors import InferenceBackendError
from lai.domain.observability import AlertCategory
from lai.inference._archive import (
    parse_release_assets,
    safe_extract_zip,
    select_llama_cuda_assets,
)
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import structlog

    from lai.domain.guardrails import GuardrailConfig
    from lai.domain.inference import ModelProfile

_DEFAULT_GITHUB_REPO = "ggml-org/llama.cpp"
_GITHUB_API_BASE = "https://api.github.com/repos"
_HTTP_ERROR_THRESHOLD = 400
_DOWNLOAD_CHUNK_BYTES = 65536


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


class HfDownloader(Protocol):
    """Callable protocol matching huggingface_hub.hf_hub_download."""

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        local_dir: str,
        force_download: bool,
        token: str | None,
    ) -> object:
        """Download a file from a HuggingFace Hub repository."""
        ...


def _default_hf_download(
    *,
    repo_id: str,
    filename: str,
    local_dir: str,
    force_download: bool,
    token: str | None,
) -> str:
    """Invoke huggingface_hub.hf_hub_download with strict types."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        force_download=force_download,
        token=token,
    )


class ModelDownloader:
    """Fetches official llama.cpp CUDA binaries and GGUF model files."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        bin_dir: Path,
        models_dir: Path,
        github_repo: str = _DEFAULT_GITHUB_REPO,
        timeout_seconds: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
        hf_downloader: HfDownloader | None = None,
    ) -> None:
        self._bin_dir = bin_dir
        self._models_dir = models_dir
        self._github_repo = github_repo
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._hf_downloader: HfDownloader = hf_downloader or _default_hf_download

    @asynccontextmanager
    async def _client_ctx(self) -> AsyncGenerator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_seconds),
            follow_redirects=True,
        ) as client:
            yield client

    async def _fetch_release_assets(
        self,
        client: httpx.AsyncClient,
        tag: str | None,
    ) -> list[tuple[str, str]]:
        suffix = f"tags/{tag}" if tag else "latest"
        url = f"{_GITHUB_API_BASE}/{self._github_repo}/releases/{suffix}"
        payload = await self._get_github_json(client, url)
        assets = parse_release_assets(payload)
        if not assets:
            msg = "GitHub release metadata contained no downloadable assets."
            raise InferenceBackendError(msg)
        return assets

    async def _get_github_json(self, client: httpx.AsyncClient, url: str) -> object:
        """Fetch JSON metadata from the GitHub Releases API with timeout alerting."""
        try:
            response = await client.get(
                url,
                headers={"Accept": "application/vnd.github+json"},
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            _logger().exception(
                "github_release_timeout",
                url=url,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = f"GitHub release lookup timed out: {exc}"
            raise InferenceBackendError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"GitHub release lookup failed: {exc}"
            raise InferenceBackendError(msg) from exc

        if response.status_code >= _HTTP_ERROR_THRESHOLD:
            msg = f"GitHub release lookup returned HTTP {response.status_code}"
            raise InferenceBackendError(msg)
        return response.json()

    async def _resolve_release_assets(
        self,
        client: httpx.AsyncClient,
        *,
        tag: str | None,
        cuda_version: str | None,
    ) -> tuple[tuple[str, str], tuple[str, str] | None]:
        """Resolve Windows CUDA assets from latest/tag release, falling back to recent releases."""
        latest_assets = await self._fetch_release_assets(client, tag)
        try:
            return select_llama_cuda_assets(latest_assets, cuda_version=cuda_version)
        except InferenceBackendError:
            if tag is not None:
                raise

        list_url = f"{_GITHUB_API_BASE}/{self._github_repo}/releases?per_page=10"
        raw_releases = await self._get_github_json(client, list_url)
        if isinstance(raw_releases, list):
            for release in cast("list[object]", raw_releases):
                candidate_assets = parse_release_assets(release)
                if not candidate_assets:
                    continue
                try:
                    return select_llama_cuda_assets(
                        candidate_assets,
                        cuda_version=cuda_version,
                    )
                except InferenceBackendError:
                    continue
        return select_llama_cuda_assets(latest_assets, cuda_version=cuda_version)

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        path.unlink(missing_ok=True)

    async def _download_and_extract(
        self,
        client: httpx.AsyncClient,
        url: str,
        zip_path: Path,
        dest_dir: Path,
    ) -> None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with client.stream("GET", url, timeout=self._timeout_seconds) as response:
                if response.status_code >= _HTTP_ERROR_THRESHOLD:
                    msg = f"Binary asset download failed with HTTP {response.status_code}: {url}"
                    raise InferenceBackendError(msg)
                with zip_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                        handle.write(chunk)
        except httpx.TimeoutException as exc:
            await asyncio.to_thread(self._unlink_quietly, zip_path)
            _logger().exception(
                "binary_download_timeout",
                url=url,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(exc).__name__,
            )
            msg = f"Binary asset download timed out: {exc}"
            raise InferenceBackendError(msg) from exc
        except httpx.HTTPError as exc:
            await asyncio.to_thread(self._unlink_quietly, zip_path)
            msg = f"Binary asset download failed: {exc}"
            raise InferenceBackendError(msg) from exc

        try:
            await asyncio.to_thread(safe_extract_zip, zip_path, dest_dir)
        finally:
            await asyncio.to_thread(self._unlink_quietly, zip_path)

    async def download_llama_binaries(
        self,
        *,
        tag: str | None = None,
        cuda_version: str | None = None,
        force: bool = False,
    ) -> Path:
        """Fetch official llama.cpp Windows CUDA x64 release binaries into bin/llama/."""
        llama_dir = self._bin_dir / "llama"
        existing = llama_dir / "llama-server.exe"
        if not force and existing.exists():
            return existing

        async with self._client_ctx() as client:
            (main_name, main_url), cudart = await self._resolve_release_assets(
                client,
                tag=tag,
                cuda_version=cuda_version,
            )
            await self._download_and_extract(
                client,
                main_url,
                llama_dir / f".{main_name}.tmp",
                llama_dir,
            )
            exe_path = (
                existing if existing.exists() else next(llama_dir.rglob("llama-server.exe"), None)
            )
            if exe_path is None:
                msg = "Downloaded llama.cpp archive did not contain llama-server.exe."
                raise InferenceBackendError(msg)

            if cudart is not None:
                cudart_name, cudart_url = cudart
                await self._download_and_extract(
                    client,
                    cudart_url,
                    llama_dir / f".{cudart_name}.tmp",
                    exe_path.parent,
                )

        _logger().info("llama_binaries_ready", executable_path=str(exe_path), asset=main_name)
        return exe_path

    async def download_model(
        self,
        profile_or_config: ModelProfile | GuardrailConfig,
        *,
        force: bool = False,
        token: str | None = None,
    ) -> Path:
        """Download a GGUF model from HuggingFace Hub with fallback repo support."""
        dest_path = self._models_dir / profile_or_config.hf_filename
        if not force and dest_path.exists():
            return dest_path

        self._models_dir.mkdir(parents=True, exist_ok=True)
        repos = [profile_or_config.hf_repo]
        fallback = profile_or_config.fallback_hf_repo
        if fallback and fallback not in repos:
            repos.append(fallback)

        last_error: Exception | None = None
        for repo_id in repos:
            try:
                result = await asyncio.to_thread(
                    self._hf_downloader,
                    repo_id=repo_id,
                    filename=profile_or_config.hf_filename,
                    local_dir=str(self._models_dir),
                    force_download=force,
                    token=token,
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
            _logger().info("gguf_model_ready", repo_id=repo_id, path=str(resolved))
            return resolved

        if isinstance(last_error, httpx.TimeoutException):
            _logger().error(
                "gguf_download_timeout",
                filename=profile_or_config.hf_filename,
                alert_category=AlertCategory.TIMEOUT.value,
                exception_type=type(last_error).__name__,
            )
        else:
            _logger().error(
                "gguf_download_failed",
                filename=profile_or_config.hf_filename,
                repos=repos,
                error=str(last_error),
            )
        msg = (
            f"Failed to download {profile_or_config.hf_filename!r} "
            f"from repositories {repos}: {last_error}"
        )
        raise InferenceBackendError(msg) from last_error
