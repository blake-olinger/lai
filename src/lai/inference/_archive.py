"""GitHub release asset matching and traversal-safe archive extraction."""

from __future__ import annotations

import re
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from lai.domain.errors import InferenceBackendError

_WIN_LLAMA_CUDA_ASSET_RE = re.compile(r"^llama-.+-bin-win-cuda-(.+)-x64\.zip$")
_WIN_CUDART_ASSET_RE = re.compile(
    r"^cudart-llama(?:-[^-]+)?-bin-win-(?:cuda-)?(.+)-x64\.zip$",
)
_LINUX_LLAMA_CUDA_ASSET_RE = re.compile(r"^llama-.+-bin-ubuntu-cuda-(.+)-x64\.tar\.gz$")
_LINUX_CUDART_ASSET_RE = re.compile(
    r"^cudart-llama(?:-[^-]+)?-bin-ubuntu-(?:cuda-)?(.+)-x64\.tar\.gz$",
)
_EXEC_PERM_MASK = 0o111
_DEFAULT_FILE_MODE = 0o644
_EXEC_FILE_MODE = 0o755


def resolve_platform_spec(
    platform_name: str | None = None,
) -> tuple[str, re.Pattern[str], re.Pattern[str], str]:
    """Return (platform_label, main_asset_re, cudart_asset_re, binary_name) for target OS."""
    target = (platform_name or sys.platform).lower()
    if target == "win32":
        return (
            "Windows CUDA x64",
            _WIN_LLAMA_CUDA_ASSET_RE,
            _WIN_CUDART_ASSET_RE,
            "llama-server.exe",
        )
    if target.startswith("linux"):
        return (
            "Linux CUDA x64",
            _LINUX_LLAMA_CUDA_ASSET_RE,
            _LINUX_CUDART_ASSET_RE,
            "llama-server",
        )
    msg = (
        f"Automated llama.cpp CUDA binary downloads are supported on Windows and Linux x64 "
        f"(got {target!r}). Install llama-server manually with --skip-binaries or use "
        "--backend ollama."
    )
    raise InferenceBackendError(msg)


def _cuda_version_sort_key(version_str: str) -> tuple[tuple[int, ...], str]:
    """Produce a natural numeric sort key for CUDA version strings like cu12.4 or 13.5."""
    nums = tuple(int(part) for part in re.findall(r"\d+", version_str))
    return (nums, version_str)


def parse_release_assets(release_payload: object) -> list[tuple[str, str]]:
    """Extract (asset_name, download_url) pairs from a GitHub Release JSON object."""
    if not isinstance(release_payload, Mapping):
        return []
    release_map = cast("Mapping[str, object]", release_payload)
    raw_assets = release_map.get("assets")
    if not isinstance(raw_assets, list):
        return []
    assets: list[tuple[str, str]] = []
    for item in cast("list[object]", raw_assets):
        if not isinstance(item, Mapping):
            continue
        item_map = cast("Mapping[str, object]", item)
        name = item_map.get("name")
        url = item_map.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str) and name and url:
            assets.append((name, url))
    return assets


def _find_matching_cudart(
    assets: list[tuple[str, str]],
    cuda_tag: str,
    cudart_re: re.Pattern[str],
) -> tuple[str, str] | None:
    """Find companion cudart archive matching the selected CUDA version if present."""
    cudarts: list[tuple[str, str, str]] = []
    for name, url in assets:
        match = cudart_re.match(name)
        if match is not None:
            cudarts.append((match.group(1), name, url))
    if not cudarts:
        return None
    for ver, name, url in cudarts:
        if ver == cuda_tag or cuda_tag in ver or ver in cuda_tag:
            return (name, url)
    _, name, url = max(cudarts, key=lambda item: _cuda_version_sort_key(item[0]))
    return (name, url)


def select_llama_cuda_assets(
    assets: list[tuple[str, str]],
    *,
    cuda_version: str | None = None,
    platform_name: str | None = None,
) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """Select the newest CUDA x64 llama.cpp archive and optional cudart archive for platform."""
    label, main_re, cudart_re, _binary = resolve_platform_spec(platform_name)
    candidates: list[tuple[str, str, str]] = []
    for name, url in assets:
        match = main_re.match(name)
        if match is not None:
            candidates.append((match.group(1), name, url))

    if not candidates:
        msg = f"No official llama.cpp {label} binary archive found in release assets."
        raise InferenceBackendError(msg)

    if cuda_version is not None:
        filtered = [c for c in candidates if cuda_version in c[0]]
        if not filtered:
            msg = f"No llama.cpp {label} asset matches cuda_version={cuda_version!r}."
            raise InferenceBackendError(msg)
        candidates = filtered

    selected_cuda, main_name, main_url = max(
        candidates,
        key=lambda item: _cuda_version_sort_key(item[0]),
    )
    return (main_name, main_url), _find_matching_cudart(assets, selected_cuda, cudart_re)


def _validate_archive_target(archive_name: str, member_name: str, dest_root: Path) -> Path:
    """Resolve member_name under dest_root while blocking path traversal."""
    member_path = Path(member_name)
    if (
        member_path.is_absolute()
        or member_name.startswith(("/", "\\"))
        or ".." in member_path.parts
    ):
        msg = f"Unsafe archive path detected in {archive_name}: {member_name}"
        raise InferenceBackendError(msg)
    target = (dest_root / member_path).resolve()
    if not target.is_relative_to(dest_root):
        msg = f"Archive entry escapes target directory: {member_name}"
        raise InferenceBackendError(msg)
    return target


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive into dest_dir while blocking Zip-Slip traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = _validate_archive_target(zip_path.name, info.filename, dest_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _extract_safe_tar_symlink(
    archive_name: str,
    member: tarfile.TarInfo,
    target: Path,
    dest_root: Path,
) -> None:
    """Validate that a tar symlink target stays strictly within dest_root and create it."""
    link_path = Path(member.linkname)
    if link_path.is_absolute() or member.linkname.startswith(("/", "\\")):
        msg = f"Unsafe absolute symlink in {archive_name}: {member.name} -> {member.linkname}"
        raise InferenceBackendError(msg)
    resolved_link = (target.parent / link_path).resolve()
    if not resolved_link.is_relative_to(dest_root):
        msg = f"Unsafe symlink escapes target directory in {archive_name}: {member.name}"
        raise InferenceBackendError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(member.linkname)
    except OSError:
        if sys.platform != "win32":
            raise


def safe_extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into dest_dir while blocking Tar-Slip and unsafe entries."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = _validate_archive_target(tar_path.name, member.name, dest_root)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                _extract_safe_tar_symlink(tar_path.name, member, target, dest_root)
                continue
            if not member.isfile():
                msg = f"Unsupported tar entry type in {tar_path.name}: {member.name}"
                raise InferenceBackendError(msg)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as dst:
                shutil.copyfileobj(extracted, dst)
            mode = _EXEC_FILE_MODE if (member.mode & _EXEC_PERM_MASK) else _DEFAULT_FILE_MODE
            if sys.platform != "win32":
                target.chmod(mode)


def safe_extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Dispatch extraction safely for .zip or .tar.gz/.tgz binary archives."""
    normalized = archive_path.name.lower().removesuffix(".tmp")
    if normalized.endswith((".tar.gz", ".tgz")):
        safe_extract_tar_gz(archive_path, dest_dir)
        return
    safe_extract_zip(archive_path, dest_dir)
