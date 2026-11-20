"""GitHub Releases API querying and streaming archive download helpers for llama.cpp."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import httpx

from lai.domain.errors import InferenceBackendError
from lai.domain.observability import AlertCategory
from lai.inference._archive import (
    extract_tag_from_asset_name,
    parse_release_asset_digests,
    parse_release_assets,
    parse_release_tag,
    safe_extract_archive,
    select_llama_cuda_assets,
    validate_release_tag,
)
from lai.observability.integrity import verify_file_sha256
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from pathlib import Path

    import structlog

_GITHUB_API_BASE = "https://api.github.com/repos"
_HTTP_ERROR_THRESHOLD = 400
_DOWNLOAD_CHUNK_BYTES = 65536

type ResolvedRelease = tuple[str, tuple[str, str], tuple[str, str] | None, dict[str, str]]


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("inference")


def _unlink_quietly(path: Path) -> None:
    path.unlink(missing_ok=True)


async def get_github_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_seconds: float,
) -> object:
    """Fetch JSON metadata from the GitHub Releases API with timeout alerting."""
    try:
        response = await client.get(
            url,
            headers={"Accept": "application/vnd.github+json"},
            timeout=timeout_seconds,
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


def _select_from_payload(
    payload: object,
    *,
    cuda_version: str,
    fallback_tag: str,
    platform_name: str,
    machine_arch: str,
) -> ResolvedRelease:
    assets = parse_release_assets(payload)
    if not assets:
        msg = "GitHub release metadata contained no downloadable assets."
        raise InferenceBackendError(msg)
    main_asset, cudart_asset = select_llama_cuda_assets(
        assets,
        cuda_version=cuda_version,
        platform_name=platform_name,
        machine_arch=machine_arch,
    )
    tag_candidate = (
        parse_release_tag(payload) or extract_tag_from_asset_name(main_asset[0]) or fallback_tag
    )
    release_tag = validate_release_tag(tag_candidate)
    return release_tag, main_asset, cudart_asset, parse_release_asset_digests(payload)


async def resolve_release_assets(  # noqa: PLR0913
    client: httpx.AsyncClient,
    *,
    github_repo: str,
    requested_tag: str,
    cuda_version: str,
    platform_name: str,
    machine_arch: str,
    timeout_seconds: float,
) -> ResolvedRelease:
    """Resolve Linux CUDA tarballs for a pinned tag or rolling latest release."""
    suffix = "latest" if requested_tag == "latest" else f"tags/{requested_tag}"
    payload = await get_github_json(
        client,
        f"{_GITHUB_API_BASE}/{github_repo}/releases/{suffix}",
        timeout_seconds=timeout_seconds,
    )
    fallback_tag = "release" if requested_tag == "latest" else requested_tag
    try:
        return _select_from_payload(
            payload,
            cuda_version=cuda_version,
            fallback_tag=fallback_tag,
            platform_name=platform_name,
            machine_arch=machine_arch,
        )
    except InferenceBackendError:
        if requested_tag != "latest":
            raise

    raw_releases = await get_github_json(
        client,
        f"{_GITHUB_API_BASE}/{github_repo}/releases?per_page=10",
        timeout_seconds=timeout_seconds,
    )
    if isinstance(raw_releases, list):
        for release in cast("list[object]", raw_releases):
            try:
                return _select_from_payload(
                    release,
                    cuda_version=cuda_version,
                    fallback_tag="release",
                    platform_name=platform_name,
                    machine_arch=machine_arch,
                )
            except InferenceBackendError:
                continue
    return _select_from_payload(
        payload,
        cuda_version=cuda_version,
        fallback_tag="release",
        platform_name=platform_name,
        machine_arch=machine_arch,
    )


async def download_verify_and_extract_archive(  # noqa: PLR0913
    client: httpx.AsyncClient,
    *,
    url: str,
    archive_path: Path,
    dest_dir: Path,
    timeout_seconds: float,
    expected_sha256: str | None,
    label: str,
) -> None:
    """Stream-download a .tar.gz archive, verify SHA-256 when known, and extract safely."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with client.stream("GET", url, timeout=timeout_seconds) as response:
            if response.status_code >= _HTTP_ERROR_THRESHOLD:
                msg = f"Binary asset download failed with HTTP {response.status_code}: {url}"
                raise InferenceBackendError(msg)
            with archive_path.open("wb") as handle:
                async for chunk in response.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
    except httpx.TimeoutException as exc:
        await asyncio.to_thread(_unlink_quietly, archive_path)
        _logger().exception(
            "binary_download_timeout",
            url=url,
            alert_category=AlertCategory.TIMEOUT.value,
            exception_type=type(exc).__name__,
        )
        msg = f"Binary asset download timed out: {exc}"
        raise InferenceBackendError(msg) from exc
    except httpx.HTTPError as exc:
        await asyncio.to_thread(_unlink_quietly, archive_path)
        msg = f"Binary asset download failed: {exc}"
        raise InferenceBackendError(msg) from exc

    try:
        if expected_sha256 is not None:
            await asyncio.to_thread(
                verify_file_sha256,
                archive_path,
                expected_sha256,
                label=label,
            )
        await asyncio.to_thread(
            safe_extract_archive,
            archive_path,
            dest_dir,
            strip_single_top_dir=True,
        )
    finally:
        await asyncio.to_thread(_unlink_quietly, archive_path)
