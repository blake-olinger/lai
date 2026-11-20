"""Release artifact sources (GitHub Releases and local wheels) with strict SHA-256 verification."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory
from lai.domain.updates import ReleaseArtifact
from lai.observability.logging import get_component_logger
from lai.ports.updates import ReleaseSourcePort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import structlog

__all__ = [
    "GitHubReleaseSource",
    "LocalWheelSource",
    "compute_sha256",
    "parse_semver_tuple",
    "parse_sha256sums",
    "verify_wheel_sha256",
]

_GITHUB_API_BASE = "https://api.github.com/repos"
_HTTP_ERR = 400
_HTTP_404 = 404
_CHUNK_BYTES = 65536
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)(?:[.\-+A-Za-z0-9]*)?$")
_WHEEL_RE = re.compile(r"^lai-(\d+\.\d+\.\d+[A-Za-z0-9._+]*)-.+\.whl$")


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("updates")


def _err(code: str, msg: str, status: int = 500) -> LaiError:
    return LaiError(msg, code=code, status_code=status)


def parse_semver_tuple(version: str) -> tuple[int, int, int]:
    """Parse a SemVer string (with optional leading 'v' or suffix) into (major, minor, patch)."""
    match = _SEMVER_RE.fullmatch(version.strip())
    if match is None:
        msg = f"Invalid SemVer version: {version!r}"
        raise ValueError(msg)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse sha256sum-formatted lines into a {filename: lowercase_hex_digest} mapping."""
    checksums: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:  # noqa: PLR2004
            raise _err("upgrade_invalid_checksums", f"Malformed SHA256SUMS line: {raw_line!r}")
        digest = parts[0].strip().lower()
        filename = Path(parts[1].strip().lstrip("*")).name
        if not _SHA256_RE.fullmatch(digest) or not filename:
            raise _err("upgrade_invalid_checksums", f"Invalid SHA256SUMS entry: {raw_line!r}")
        checksums[filename] = digest
    return checksums


def compute_sha256(path: Path) -> str:
    """Compute the lowercase SHA-256 hex digest of a file using streaming reads."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_wheel_sha256(wheel_path: Path, expected_sha256: str) -> None:
    """Verify wheel_path matches expected_sha256 using constant-time comparison."""
    expected = expected_sha256.strip().lower()
    actual = compute_sha256(wheel_path)
    if not _SHA256_RE.fullmatch(expected) or not hmac.compare_digest(actual, expected):
        _logger().error(
            "upgrade_wheel_checksum_mismatch",
            wheel_path=str(wheel_path),
            expected_sha256=expected,
            actual_sha256=actual,
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        msg = f"SHA-256 mismatch for {wheel_path.name}: expected {expected}, got {actual}."
        raise _err("upgrade_checksum_mismatch", msg)


def _as_dict(value: object, ctx: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _err("upgrade_invalid_release", f"Expected JSON object for {ctx}.", 502)
    return cast("dict[str, object]", value)


def _find_release_assets(release_data: dict[str, object]) -> tuple[str, str, str]:
    raw_assets = release_data.get("assets")
    if not isinstance(raw_assets, list):
        raise _err("upgrade_asset_missing", "GitHub release missing assets list.", 404)
    wheel_name: str | None = None
    wheel_url: str | None = None
    sums_url: str | None = None
    for item in cast("list[object]", raw_assets):
        if not isinstance(item, dict):
            continue
        asset = cast("dict[str, object]", item)
        name, url = asset.get("name"), asset.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        if name == "SHA256SUMS.txt":
            sums_url = url
        elif wheel_name is None and name.startswith("lai-") and name.endswith(".whl"):
            wheel_name, wheel_url = name, url
    if wheel_name is None or wheel_url is None:
        raise _err("upgrade_asset_missing", "Release lacks a lai-*.whl asset.", 404)
    if sums_url is None:
        raise _err("upgrade_checksums_missing", "Release lacks SHA256SUMS.txt.")
    return wheel_name, wheel_url, sums_url


def _prepare_dest_path(dest_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise _err("upgrade_invalid_wheel", f"Unsafe wheel filename: {filename!r}", 400)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / filename


def _check_http_ok(resp: httpx.Response, label: str) -> None:
    if resp.status_code >= _HTTP_ERR:
        raise _err("upgrade_download_failed", f"Fetch {label} failed: HTTP {resp.status_code}", 502)


class GitHubReleaseSource(ReleaseSourcePort):
    """ReleaseSourcePort backed by the GitHub Releases API and SHA256SUMS.txt."""

    def __init__(
        self,
        github_repo: str = "blake-olinger/lai",
        *,
        token: str | None = None,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        api_base_url: str = _GITHUB_API_BASE,
    ) -> None:
        self._repo, self._token, self._timeout = github_repo, token, timeout_seconds
        self._http_client = http_client
        self._api_base = api_base_url.rstrip("/")

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
            raise _err("upgrade_source_unreachable", f"GitHub API failed: {exc}", 502) from exc
        if resp.status_code == _HTTP_404:
            raise _err("upgrade_release_not_found", f"Release not found: {url}", 404)
        if resp.status_code >= _HTTP_ERR:
            raise _err("upgrade_source_error", f"GitHub API HTTP {resp.status_code}", 502)
        return resp.json()

    async def _build_artifact(
        self, client: httpx.AsyncClient, release: dict[str, object]
    ) -> ReleaseArtifact:
        raw_tag = release.get("tag_name")
        if not isinstance(raw_tag, str) or not raw_tag.strip():
            raise _err("upgrade_invalid_release", "Release missing tag_name.", 502)
        tag = raw_tag.strip()
        wheel_name, wheel_url, sums_url = _find_release_assets(release)
        try:
            sums_resp = await client.get(sums_url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise _err("upgrade_download_failed", f"SHA256SUMS fetch failed: {exc}", 502) from exc
        _check_http_ok(sums_resp, "SHA256SUMS.txt")
        sha256 = parse_sha256sums(sums_resp.text).get(wheel_name)
        if sha256 is None:
            _logger().error(
                "upgrade_wheel_missing_from_checksums",
                wheel_filename=wheel_name,
                alert_category=AlertCategory.SECURITY_ALERT.value,
            )
            raise _err("upgrade_checksum_missing", f"{wheel_name!r} absent from SHA256SUMS.txt.")
        return ReleaseArtifact(
            version=tag.lstrip("vV"),
            tag=tag,
            wheel_filename=wheel_name,
            sha256=sha256,
            download_url=wheel_url,
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
                raise _err("upgrade_invalid_release", "Releases payload is not a list.", 502)
            for item in cast("list[object]", raw_list):
                if isinstance(item, dict) and not cast("dict[str, object]", item).get("draft"):
                    return await self._build_artifact(client, cast("dict[str, object]", item))
            raise _err("upgrade_release_not_found", f"No releases for {self._repo}.", 404)

    async def get_version(self, version: str) -> ReleaseArtifact:
        """Fetch metadata for a specific tagged release version."""
        clean = version.strip().lstrip("vV")
        url = f"{self._api_base}/{self._repo}/releases/tags/v{clean}"
        async with self._client_ctx() as client:
            data = _as_dict(await self._get_json(client, url), f"release v{clean}")
            return await self._build_artifact(client, data)

    async def download_wheel(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path:
        """Stream artifact wheel into dest_dir and strictly verify SHA-256 (unlinking on error)."""
        if not artifact.download_url:
            raise _err("upgrade_download_failed", "Artifact lacks download_url.", 400)
        dest_path = await asyncio.to_thread(_prepare_dest_path, dest_dir, artifact.wheel_filename)
        try:
            async with (
                self._client_ctx() as client,
                client.stream("GET", artifact.download_url, headers=self._headers()) as resp,
            ):
                _check_http_ok(resp, artifact.wheel_filename)
                with dest_path.open("wb") as handle:
                    async for chunk in resp.aiter_bytes(chunk_size=_CHUNK_BYTES):
                        handle.write(chunk)
            await asyncio.to_thread(verify_wheel_sha256, dest_path, artifact.sha256)
        except httpx.HTTPError as exc:
            dest_path.unlink(missing_ok=True)
            raise _err("upgrade_download_failed", f"Download error: {exc}", 502) from exc
        except Exception:
            dest_path.unlink(missing_ok=True)
            raise
        return dest_path


def _parse_sidecar_sha256(sidecar: Path, filename: str) -> str:
    text = sidecar.read_text(encoding="utf-8").strip()
    if _SHA256_RE.fullmatch(text.lower()):
        return text.lower()
    digest = parse_sha256sums(text).get(filename)
    if digest is None:
        raise _err("upgrade_checksum_missing", f"Sidecar {sidecar} omits {filename!r}.")
    return digest


def _stage_local_wheel(src: Path, dest_dir: Path, artifact: ReleaseArtifact) -> Path:
    if not src.is_file():
        raise _err("upgrade_invalid_wheel", f"Local wheel missing: {src}", 404)
    verify_wheel_sha256(src, artifact.sha256)
    dest = _prepare_dest_path(dest_dir, artifact.wheel_filename)
    if dest.resolve() != src.resolve():
        shutil.copyfile(src, dest)
    return dest


class LocalWheelSource(ReleaseSourcePort):
    """ReleaseSourcePort reading a staged local wheel file for CI/CD push deployments."""

    def __init__(
        self,
        wheel_path: Path,
        *,
        expected_sha256: str | None = None,
        sha256_path: Path | None = None,
    ) -> None:
        self._wheel_path = wheel_path
        self._expected_sha256, self._sha256_path = expected_sha256, sha256_path

    def _resolve_sha256(self) -> str:
        if self._expected_sha256 is not None:
            return self._expected_sha256.strip().lower()
        if self._sha256_path is not None:
            if not self._sha256_path.is_file():
                raise _err("upgrade_checksums_missing", f"Missing: {self._sha256_path}", 404)
            return _parse_sidecar_sha256(self._sha256_path, self._wheel_path.name)
        sidecar = self._wheel_path.with_name(f"{self._wheel_path.name}.sha256")
        if sidecar.is_file():
            return _parse_sidecar_sha256(sidecar, self._wheel_path.name)
        sums = self._wheel_path.parent / "SHA256SUMS.txt"
        if sums.is_file():
            found = parse_sha256sums(sums.read_text(encoding="utf-8")).get(self._wheel_path.name)
            if found is not None:
                return found
        return compute_sha256(self._wheel_path)

    def _build_local_artifact(self) -> ReleaseArtifact:
        if not self._wheel_path.is_file():
            raise _err("upgrade_invalid_wheel", f"Local wheel missing: {self._wheel_path}", 404)
        match = _WHEEL_RE.fullmatch(self._wheel_path.name)
        if match is None:
            raise _err("upgrade_invalid_wheel", f"Invalid wheel name: {self._wheel_path.name}", 400)
        ver = match.group(1)
        return ReleaseArtifact(
            version=ver,
            tag=f"v{ver}",
            wheel_filename=self._wheel_path.name,
            sha256=self._resolve_sha256(),
            local_path=self._wheel_path,
        )

    async def get_latest(self, *, allow_prerelease: bool = False) -> ReleaseArtifact:
        """Return artifact metadata for the configured local wheel."""
        del allow_prerelease
        return await asyncio.to_thread(self._build_local_artifact)

    async def get_version(self, version: str) -> ReleaseArtifact:
        """Return artifact metadata if the local wheel matches the requested version."""
        artifact = await asyncio.to_thread(self._build_local_artifact)
        if artifact.version != version.strip().lstrip("vV"):
            raise _err("upgrade_release_not_found", f"Wheel {artifact.version} != {version}", 404)
        return artifact

    async def download_wheel(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path:
        """Verify local wheel SHA-256 strictly and stage a verified copy into dest_dir."""
        return await asyncio.to_thread(_stage_local_wheel, self._wheel_path, dest_dir, artifact)
