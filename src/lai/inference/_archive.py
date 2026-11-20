"""Linux x64 GitHub release asset matching, safe tar.gz extraction, and atomic symlink swap."""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import tarfile
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from lai.domain.errors import InferenceBackendError

_LINUX_X64_ARCHES: frozenset[str] = frozenset({"x86_64", "amd64"})
_LINUX_LLAMA_CUDA_ASSET_RE = re.compile(r"^llama-(.+)-bin-ubuntu-cuda-(.+)-x64\.tar\.gz$")
_LINUX_CUDART_ASSET_RE = re.compile(
    r"^cudart-llama(?:-[^-]+)?-bin-ubuntu-(?:cuda-)?(.+)-x64\.tar\.gz$",
)
_SHA256_PREFIX = "sha256:"
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_EXEC_PERM_MASK = 0o111
_DEFAULT_FILE_MODE = 0o644
_EXEC_FILE_MODE = 0o755


def resolve_platform_spec(
    platform_name: str | None = None,
    machine_arch: str | None = None,
) -> tuple[str, re.Pattern[str], re.Pattern[str], str]:
    """Return (platform_label, main_asset_re, cudart_asset_re, binary_name) for Linux x64."""
    target = (platform_name if platform_name is not None else sys.platform).lower()
    arch = (machine_arch if machine_arch is not None else platform.machine()).lower()
    if not target.startswith("linux"):
        msg = (
            f"Automated llama.cpp CUDA binary downloads are supported only on Linux x86_64 "
            f"(got platform {target!r}). Install llama-server manually with --skip-binaries."
        )
        raise InferenceBackendError(msg)
    if arch not in _LINUX_X64_ARCHES:
        msg = (
            f"Automated llama.cpp CUDA binary downloads require Linux x86_64/amd64 "
            f"(got architecture {arch!r}). Install llama-server manually with --skip-binaries."
        )
        raise InferenceBackendError(msg)
    return (
        "Linux CUDA x64",
        _LINUX_LLAMA_CUDA_ASSET_RE,
        _LINUX_CUDART_ASSET_RE,
        "llama-server",
    )


def _cuda_version_sort_key(version_str: str) -> tuple[tuple[int, ...], str]:
    """Produce a natural numeric sort key for CUDA version strings like 12.8 or 13.5."""
    nums = tuple(int(part) for part in re.findall(r"\d+", version_str))
    return (nums, version_str)


def parse_release_tag(release_payload: object) -> str | None:
    """Extract tag_name from a GitHub Release JSON mapping if present."""
    if not isinstance(release_payload, Mapping):
        return None
    release_map = cast("Mapping[str, object]", release_payload)
    tag = release_map.get("tag_name")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    return None


def extract_tag_from_asset_name(asset_name: str) -> str | None:
    """Extract release build tag (e.g. 'b12552') from a Linux llama.cpp tarball name."""
    match = _LINUX_LLAMA_CUDA_ASSET_RE.match(asset_name)
    return match.group(1) if match is not None else None


def validate_release_tag(tag: str) -> str:
    """Validate that a release tag is safe for use as a single path component."""
    cleaned = tag.strip()
    if not cleaned or cleaned in {".", ".."} or _SAFE_TAG_RE.fullmatch(cleaned) is None:
        msg = f"Unsafe llama.cpp release tag: {tag!r}"
        raise InferenceBackendError(msg)
    return cleaned


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


def parse_release_asset_digests(release_payload: object) -> dict[str, str]:
    """Extract asset_name -> lowercase sha256 digest pairs from GitHub Release JSON."""
    if not isinstance(release_payload, Mapping):
        return {}
    release_map = cast("Mapping[str, object]", release_payload)
    raw_assets = release_map.get("assets")
    if not isinstance(raw_assets, list):
        return {}
    digests: dict[str, str] = {}
    for item in cast("list[object]", raw_assets):
        if not isinstance(item, Mapping):
            continue
        item_map = cast("Mapping[str, object]", item)
        name = item_map.get("name")
        raw_digest = item_map.get("digest")
        if not isinstance(name, str) or not isinstance(raw_digest, str):
            continue
        if raw_digest.lower().startswith(_SHA256_PREFIX):
            hex_part = raw_digest[len(_SHA256_PREFIX) :].strip().lower()
            if _SHA256_HEX_RE.fullmatch(hex_part) is not None:
                digests[name] = hex_part
    return digests


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
    machine_arch: str | None = None,
) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """Select the matching Linux x64 llama.cpp CUDA archive and companion cudart archive."""
    label, main_re, cudart_re, _binary = resolve_platform_spec(platform_name, machine_arch)
    candidates: list[tuple[str, str, str]] = []
    for name, url in assets:
        match = main_re.match(name)
        if match is not None:
            candidates.append((match.group(2), name, url))

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


def _validate_raw_member_name(archive_name: str, member_name: str) -> tuple[str, ...]:
    """Validate raw archive path for traversal/absoluteness and return POSIX parts."""
    normalized = member_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    raw_path = Path(member_name)
    if (
        raw_path.is_absolute()
        or posix_path.is_absolute()
        or member_name.startswith(("/", "\\"))
        or ".." in posix_path.parts
        or ".." in raw_path.parts
    ):
        msg = f"Unsafe archive path detected in {archive_name}: {member_name}"
        raise InferenceBackendError(msg)
    return tuple(part for part in posix_path.parts if part not in {"", "."})


def _validate_archive_target(archive_name: str, member_name: str, dest_root: Path) -> Path:
    """Resolve member_name under dest_root while blocking path traversal."""
    parts = _validate_raw_member_name(archive_name, member_name)
    target = (dest_root.joinpath(*parts)).resolve()
    if not target.is_relative_to(dest_root):
        msg = f"Archive entry escapes target directory: {member_name}"
        raise InferenceBackendError(msg)
    return target


def _detect_common_top_dir(members: list[tarfile.TarInfo], archive_name: str) -> str | None:
    """Return shared top-level directory prefix if all archive entries reside under one folder."""
    non_empty: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
    for member in members:
        parts = _validate_raw_member_name(archive_name, member.name)
        if parts:
            non_empty.append((member, parts))
    if not non_empty:
        return None
    top_names = {parts[0] for _member, parts in non_empty}
    if len(top_names) != 1:
        return None
    has_non_dir_root_file = any(
        len(parts) == 1 and not member.isdir() for member, parts in non_empty
    )
    if has_non_dir_root_file:
        return None
    return next(iter(top_names))


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


def safe_extract_tar_gz(
    tar_path: Path,
    dest_dir: Path,
    *,
    strip_single_top_dir: bool = True,
) -> None:
    """Extract a .tar.gz archive into dest_dir while blocking Tar-Slip and unsafe symlinks."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        common_top = (
            _detect_common_top_dir(members, tar_path.name) if strip_single_top_dir else None
        )
        for member in members:
            parts = _validate_raw_member_name(tar_path.name, member.name)
            if common_top is not None and parts and parts[0] == common_top:
                parts = parts[1:]
            if not parts:
                continue
            target = _validate_archive_target(tar_path.name, "/".join(parts), dest_root)
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


def safe_extract_archive(
    archive_path: Path,
    dest_dir: Path,
    *,
    strip_single_top_dir: bool = True,
) -> None:
    """Extract a Linux .tar.gz/.tgz binary archive safely into dest_dir."""
    normalized = archive_path.name.lower().removesuffix(".tmp")
    if not normalized.endswith((".tar.gz", ".tgz")):
        msg = f"Unsupported archive format (expected .tar.gz/.tgz): {archive_path.name}"
        raise InferenceBackendError(msg)
    safe_extract_tar_gz(archive_path, dest_dir, strip_single_top_dir=strip_single_top_dir)


def atomic_swap_symlink(link_path: Path, target_path: Path) -> None:
    """Atomically point link_path to target_path via a staging symlink and Path.replace."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    staging = link_path.parent / f".{link_path.name}.{uuid.uuid4().hex}.tmp"
    staging.symlink_to(target_path, target_is_directory=True)
    try:
        staging.replace(link_path)
    except PermissionError:
        if os.name != "nt":
            staging.unlink(missing_ok=True)
            raise
        if link_path.is_symlink():
            link_path.unlink()
        staging.replace(link_path)
    except OSError:
        staging.unlink(missing_ok=True)
        raise
