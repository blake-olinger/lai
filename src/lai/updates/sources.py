"""Release artifact sources (GitHub Releases and local wheels) with strict SHA-256 verification."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

import httpx

from lai.domain.observability import AlertCategory
from lai.domain.updates import ReleaseArtifact
from lai.ports.updates import ReleaseSourcePort
from lai.updates._checksums import (
    CHUNK_BYTES,
    REQUIREMENTS_LOCK_FILENAME,
    compute_sha256,
    make_upgrade_error,
    parse_semver_tuple,
    parse_sha256sums,
    prepare_dest_path,
    updates_logger,
    verify_file_sha256,
    verify_wheel_sha256,
)
from lai.updates._local_source import LocalWheelSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

__all__ = [
    "GitHubReleaseSource",
    "LocalWheelSource",
    "compute_sha256",
    "parse_semver_tuple",
    "parse_sha256sums",
    "verify_file_sha256",
    "verify_wheel_sha256",
]

_GITHUB_API_BASE = "https://api.github.com/repos"
_HTTP_ERR = 400
_HTTP_404 = 404


def _as_dict(value: object, ctx: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise make_upgrade_error("upgrade_invalid_release", f"Expected JSON object for {ctx}.", 502)
    return cast("dict[str, object]", value)


def _iter_asset_pairs(release_data: dict[str, object]) -> list[tuple[str, str]]:
    raw_assets = release_data.get("assets")
    if not isinstance(raw_assets, list):
        raise make_upgrade_error(
            "upgrade_asset_missing", "GitHub release missing assets list.", 404
        )
    pairs: list[tuple[str, str]] = []
    for item in cast("list[object]", raw_assets):
        if not isinstance(item, dict):
            continue
        asset = cast("dict[str, object]", item)
        name, url = asset.get("name"), asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str):
            pairs.append((name, url))
    return pairs


def _find_release_assets(
    release_data: dict[str, object],
) -> tuple[str, str, str, str | None, str | None]:
    wheel_name: str | None = None
    wheel_url: str | None = None
    sums_url: str | None = None
    req_name: str | None = None
    req_url: str | None = None
    for name, url in _iter_asset_pairs(release_data):
        if name == "SHA256SUMS.txt":
            sums_url = url
        elif name == REQUIREMENTS_LOCK_FILENAME:
            req_name, req_url = name, url
        elif wheel_name is None and name.startswith("lai-") and name.endswith(".whl"):
            wheel_name, wheel_url = name, url
    if wheel_name is None or wheel_url is None:
        raise make_upgrade_error("upgrade_asset_missing", "Release lacks a lai-*.whl asset.", 404)
    if sums_url is None:
        raise make_upgrade_error("upgrade_checksums_missing", "Release lacks SHA256SUMS.txt.")
    return wheel_name, wheel_url, sums_url, req_name, req_url


def _check_http_ok(resp: httpx.Response, label: str) -> None:
    if resp.status_code >= _HTTP_ERR:
        raise make_upgrade_error(
            "upgrade_download_failed", f"Fetch {label} failed: HTTP {resp.status_code}", 502
        )


def _require_checksum(checksums: dict[str, str], filename: str, *, event: str) -> str:
    digest = checksums.get(filename)
    if digest is None:
        updates_logger().error(
            event,
            wheel_filename=filename,
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        raise make_upgrade_error(
            "upgrade_checksum_missing", f"{filename!r} absent from SHA256SUMS.txt."
        )
    return digest


class GitHubReleaseSource(ReleaseSourcePort):
    """ReleaseSourcePort backed by the GitHub Releases API and SHA256SUMS.txt."""

    def __init__(  # noqa: PLR0913
        self,
        github_repo: str = "blake-olinger/lai",
        *,
        token: str | None = None,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        api_base_url: str = _GITHUB_API_BASE,
        require_requirements_lock: bool = True,
    ) -> None:
        self._repo, self._token, self._timeout = github_repo, token, timeout_seconds
        self._http_client = http_client
        self._api_base = api_base_url.rstrip("/")
        self._require_requirements_lock = require_requirements_lock

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "lai-updater"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @asynccontextmanager
    async def _client_ctx(self) -> AsyncGenerator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            yield client

    async def _get_json(self, client: httpx.AsyncClient, url: str) -> object:
        try:
            resp = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise make_upgrade_error(
                "upgrade_source_unreachable", f"GitHub API failed: {exc}", 502
            ) from exc
        if resp.status_code == _HTTP_404:
            raise make_upgrade_error("upgrade_release_not_found", f"Release not found: {url}", 404)
        if resp.status_code >= _HTTP_ERR:
            raise make_upgrade_error(
                "upgrade_source_error", f"GitHub API HTTP {resp.status_code}", 502
            )
        return resp.json()

    async def _build_artifact(
        self, client: httpx.AsyncClient, release: dict[str, object]
    ) -> ReleaseArtifact:
        raw_tag = release.get("tag_name")
        if not isinstance(raw_tag, str) or not raw_tag.strip():
            raise make_upgrade_error("upgrade_invalid_release", "Release missing tag_name.", 502)
        tag = raw_tag.strip()
        wheel_name, wheel_url, sums_url, req_name, req_url = _find_release_assets(release)
        try:
            sums_resp = await client.get(sums_url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise make_upgrade_error(
                "upgrade_download_failed", f"SHA256SUMS fetch failed: {exc}", 502
            ) from exc
        _check_http_ok(sums_resp, "SHA256SUMS.txt")
        checksums = parse_sha256sums(sums_resp.text)
        sha256 = _require_checksum(
            checksums, wheel_name, event="upgrade_wheel_missing_from_checksums"
        )
        req_sha256: str | None = None
        if req_name is None or req_url is None:
            if self._require_requirements_lock:
                raise make_upgrade_error(
                    "upgrade_requirements_missing",
                    f"Release lacks {REQUIREMENTS_LOCK_FILENAME} asset.",
                    404,
                )
        else:
            req_sha256 = _require_checksum(
                checksums, req_name, event="upgrade_requirements_missing_from_checksums"
            )
        return ReleaseArtifact(
            version=tag.lstrip("vV"),
            tag=tag,
            wheel_filename=wheel_name,
            sha256=sha256,
            download_url=wheel_url,
            requirements_filename=req_name,
            requirements_sha256=req_sha256,
            requirements_download_url=req_url,
        )

    async def get_latest(self, *, allow_prerelease: bool = False) -> ReleaseArtifact:
        """Fetch metadata for the latest release (optionally including prereleases)."""
        async with self._client_ctx() as client:
            if not allow_prerelease:
                url = f"{self._api_base}/{self._repo}/releases/latest"
                data = _as_dict(await self._get_json(client, url), "latest release")
                return await self._build_artifact(client, data)
            raw_list = await self._get_json(client, f"{self._api_base}/{self._repo}/releases")
            if not isinstance(raw_list, list):
                raise make_upgrade_error(
                    "upgrade_invalid_release", "Releases payload is not a list.", 502
                )
            for item in cast("list[object]", raw_list):
                if isinstance(item, dict) and not cast("dict[str, object]", item).get("draft"):
                    return await self._build_artifact(client, cast("dict[str, object]", item))
            raise make_upgrade_error(
                "upgrade_release_not_found", f"No releases for {self._repo}.", 404
            )

    async def get_version(self, version: str) -> ReleaseArtifact:
        """Fetch metadata for a specific tagged release version."""
        clean = version.strip().lstrip("vV")
        url = f"{self._api_base}/{self._repo}/releases/tags/v{clean}"
        async with self._client_ctx() as client:
            data = _as_dict(await self._get_json(client, url), f"release v{clean}")
            return await self._build_artifact(client, data)

    async def _stream_verified_asset(
        self,
        *,
        url: str,
        filename: str,
        expected_sha256: str,
        dest_dir: Path,
    ) -> Path:
        dest_path = await asyncio.to_thread(prepare_dest_path, dest_dir, filename)
        try:
            async with (
                self._client_ctx() as client,
                client.stream("GET", url, headers=self._headers()) as resp,
            ):
                _check_http_ok(resp, filename)
                with dest_path.open("wb") as handle:
                    async for chunk in resp.aiter_bytes(chunk_size=CHUNK_BYTES):
                        handle.write(chunk)
            await asyncio.to_thread(verify_file_sha256, dest_path, expected_sha256)
        except httpx.HTTPError as exc:
            dest_path.unlink(missing_ok=True)
            raise make_upgrade_error(
                "upgrade_download_failed", f"Download error: {exc}", 502
            ) from exc
        except Exception:
            dest_path.unlink(missing_ok=True)
            raise
        return dest_path

    async def download_wheel(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path:
        """Stream artifact wheel into dest_dir and strictly verify SHA-256 (unlinking on error)."""
        if not artifact.download_url:
            raise make_upgrade_error("upgrade_download_failed", "Artifact lacks download_url.", 400)
        return await self._stream_verified_asset(
            url=artifact.download_url,
            filename=artifact.wheel_filename,
            expected_sha256=artifact.sha256,
            dest_dir=dest_dir,
        )

    async def download_requirements(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path | None:
        """Stream and SHA-256-verify requirements-lock.txt into dest_dir."""
        if not artifact.requirements_download_url or not artifact.requirements_filename:
            if self._require_requirements_lock:
                raise make_upgrade_error(
                    "upgrade_requirements_missing",
                    f"Artifact lacks {REQUIREMENTS_LOCK_FILENAME} download URL.",
                    400,
                )
            return None
        if not artifact.requirements_sha256:
            raise make_upgrade_error(
                "upgrade_checksum_missing",
                f"Artifact lacks SHA-256 checksum for {artifact.requirements_filename}.",
            )
        return await self._stream_verified_asset(
            url=artifact.requirements_download_url,
            filename=artifact.requirements_filename,
            expected_sha256=artifact.requirements_sha256,
            dest_dir=dest_dir,
        )
