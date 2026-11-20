"""Downloader for Linux x64 llama.cpp CUDA binaries and HuggingFace GGUF models."""

from __future__ import annotations

import asyncio
import platform
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from lai.domain.errors import InferenceBackendError
from lai.inference._archive import (
    atomic_swap_symlink,
    resolve_platform_spec,
    validate_release_tag,
)
from lai.inference._github_release import (
    download_verify_and_extract_archive,
    resolve_release_assets,
)
from lai.inference._hf_download import HfDownloader, default_hf_download, download_hf_model
from lai.inference._runtime_helpers import detect_nvidia_cuda_version
from lai.inference.manifest import (
    DEFAULT_LLAMA_CPP_TAG,
    LLAMA_CPP_LINUX_X64_MANIFEST,
    SUPPORTED_CUDA_VERSIONS,
    lookup_manifest_sha256,
    select_compatible_cuda_version,
)
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping

    import structlog

    from lai.domain.guardrails import GuardrailConfig
    from lai.domain.inference import LlamaCppBinarySpec, ModelProfile

_DEFAULT_GITHUB_REPO = "ggml-org/llama.cpp"
_POSIX_EXEC_BITS = 0o111

__all__ = ["HfDownloader", "ModelDownloader"]


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


def _locate_extracted_binary(staging_dir: Path, binary_name: str) -> Path:
    direct_exe = staging_dir / binary_name
    exe_path = direct_exe if direct_exe.exists() else next(staging_dir.rglob(binary_name), None)
    if exe_path is None:
        msg = f"Downloaded llama.cpp archive did not contain {binary_name}."
        raise InferenceBackendError(msg)
    if sys.platform != "win32":
        exe_path.chmod(exe_path.stat().st_mode | _POSIX_EXEC_BITS)
    return exe_path


class ModelDownloader:
    """Fetches pinned Linux x64 llama.cpp CUDA binaries and HuggingFace GGUF models."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        bin_dir: Path,
        models_dir: Path,
        github_repo: str = _DEFAULT_GITHUB_REPO,
        timeout_seconds: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
        hf_downloader: HfDownloader | None = None,
        platform_name: str | None = None,
        machine_arch: str | None = None,
        cuda_detector: Callable[[], tuple[int, int] | None] | None = None,
        manifest: Mapping[str, LlamaCppBinarySpec] | None = None,
    ) -> None:
        self._bin_dir = bin_dir
        self._models_dir = models_dir
        self._github_repo = github_repo
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._hf_downloader: HfDownloader = hf_downloader or default_hf_download
        self._platform_name = platform_name if platform_name is not None else sys.platform
        self._machine_arch = machine_arch if machine_arch is not None else platform.machine()
        self._cuda_detector: Callable[[], tuple[int, int] | None] = (
            cuda_detector if cuda_detector is not None else detect_nvidia_cuda_version
        )
        self._manifest: Mapping[str, LlamaCppBinarySpec] = (
            manifest if manifest is not None else LLAMA_CPP_LINUX_X64_MANIFEST
        )

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

    @staticmethod
    def _rmtree_quietly(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def _cached_binary_if_ready(
        self,
        llama_dir: Path,
        binary_name: str,
        *,
        tag: str | None,
        force: bool,
    ) -> Path | None:
        if force:
            return None
        current_link = llama_dir / "current"
        current_exe = current_link / binary_name
        if not current_exe.exists():
            return None
        if tag is None:
            return current_exe
        if tag == "latest":
            return None
        target_release = (llama_dir / "releases" / validate_release_tag(tag)).resolve()
        if current_link.is_symlink() and current_link.resolve() == target_release:
            return current_exe
        return None

    async def download_llama_binaries(
        self,
        *,
        tag: str | None = None,
        cuda_version: str | None = None,
        force: bool = False,
    ) -> Path:
        """Fetch official Linux x64 llama.cpp CUDA release binaries and swap current symlink."""
        _label, _main_re, _cudart_re, binary_name = resolve_platform_spec(
            self._platform_name,
            self._machine_arch,
        )
        llama_dir = self._bin_dir / "llama"
        cached = self._cached_binary_if_ready(llama_dir, binary_name, tag=tag, force=force)
        if cached is not None:
            return cached

        supported_cudas = tuple(self._manifest.keys()) or SUPPORTED_CUDA_VERSIONS
        effective_cuda = (
            cuda_version
            if cuda_version is not None
            else select_compatible_cuda_version(self._cuda_detector(), supported=supported_cudas)
        )
        requested_tag = DEFAULT_LLAMA_CPP_TAG if tag is None else tag

        async with self._client_ctx() as client:
            release_tag, main_asset, cudart_asset, digests = await resolve_release_assets(
                client,
                github_repo=self._github_repo,
                requested_tag=requested_tag,
                cuda_version=effective_cuda,
                platform_name=self._platform_name,
                machine_arch=self._machine_arch,
                timeout_seconds=self._timeout_seconds,
            )
            return await self._install_release_slot(
                client,
                llama_dir=llama_dir,
                binary_name=binary_name,
                release_tag=release_tag,
                main_asset=main_asset,
                cudart_asset=cudart_asset,
                digests=digests,
            )

    async def _install_release_slot(  # noqa: PLR0913
        self,
        client: httpx.AsyncClient,
        *,
        llama_dir: Path,
        binary_name: str,
        release_tag: str,
        main_asset: tuple[str, str],
        cudart_asset: tuple[str, str] | None,
        digests: dict[str, str],
    ) -> Path:
        releases_dir = llama_dir / "releases"
        releases_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = releases_dir / f".{release_tag}.{uuid.uuid4().hex}.staging"
        main_name, main_url = main_asset
        main_sha = lookup_manifest_sha256(main_name, manifest=self._manifest) or digests.get(
            main_name
        )

        try:
            await download_verify_and_extract_archive(
                client,
                url=main_url,
                archive_path=llama_dir / f".{main_name}.tmp",
                dest_dir=staging_dir,
                timeout_seconds=self._timeout_seconds,
                expected_sha256=main_sha,
                label=main_name,
            )
            exe_in_staging = _locate_extracted_binary(staging_dir, binary_name)
            if cudart_asset is not None:
                cudart_name, cudart_url = cudart_asset
                cudart_sha = lookup_manifest_sha256(
                    cudart_name, manifest=self._manifest
                ) or digests.get(cudart_name)
                await download_verify_and_extract_archive(
                    client,
                    url=cudart_url,
                    archive_path=llama_dir / f".{cudart_name}.tmp",
                    dest_dir=exe_in_staging.parent,
                    timeout_seconds=self._timeout_seconds,
                    expected_sha256=cudart_sha,
                    label=cudart_name,
                )
            rel_exe = exe_in_staging.relative_to(staging_dir)
            release_dir = releases_dir / release_tag
            if release_dir.exists():
                await asyncio.to_thread(self._rmtree_quietly, release_dir)
            staging_dir.replace(release_dir)
        except Exception:
            await asyncio.to_thread(self._rmtree_quietly, staging_dir)
            raise

        current_link = llama_dir / "current"
        atomic_swap_symlink(current_link, Path("releases") / release_tag)
        active_exe = current_link / rel_exe
        _logger().info(
            "llama_binaries_ready",
            executable_path=str(active_exe),
            release_tag=release_tag,
            asset=main_name,
        )
        return active_exe

    async def download_model(
        self,
        profile_or_config: ModelProfile | GuardrailConfig,
        *,
        force: bool = False,
        token: str | None = None,
    ) -> Path:
        """Download a GGUF model from HuggingFace Hub with revision pinning and SHA-256 checks."""
        return await download_hf_model(
            profile_or_config,
            models_dir=self._models_dir,
            hf_downloader=self._hf_downloader,
            force=force,
            token=token,
        )
