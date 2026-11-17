"""GitHub release asset matching and Zip-Slip-safe archive extraction."""

from __future__ import annotations

import re
import shutil
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from lai.domain.errors import InferenceBackendError

_LLAMA_CUDA_ASSET_RE = re.compile(r"^llama-.+-bin-win-cuda-(.+)-x64\.zip$")
_CUDART_ASSET_RE = re.compile(r"^cudart-llama-bin-win-(?:cuda-)?(.+)-x64\.zip$")


def _cuda_version_sort_key(version_str: str) -> tuple[tuple[int, ...], str]:
    """Produce a natural numeric sort key for CUDA version strings like cu12.4 or 13.3."""
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
) -> tuple[str, str] | None:
    """Find companion cudart zip matching the selected CUDA version if present."""
    cudarts: list[tuple[str, str, str]] = []
    for name, url in assets:
        match = _CUDART_ASSET_RE.match(name)
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
) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """Select the main Windows CUDA x64 llama.cpp zip and optional cudart zip."""
    candidates: list[tuple[str, str, str]] = []
    for name, url in assets:
        match = _LLAMA_CUDA_ASSET_RE.match(name)
        if match is not None:
            candidates.append((match.group(1), name, url))

    if not candidates:
        msg = "No official llama.cpp Windows CUDA x64 binary zip found in release assets."
        raise InferenceBackendError(msg)

    if cuda_version is not None:
        filtered = [c for c in candidates if cuda_version in c[0]]
        if not filtered:
            msg = f"No llama.cpp Windows CUDA x64 asset matches cuda_version={cuda_version!r}."
            raise InferenceBackendError(msg)
        candidates = filtered

    selected_cuda, main_name, main_url = max(
        candidates,
        key=lambda item: _cuda_version_sort_key(item[0]),
    )
    return (main_name, main_url), _find_matching_cudart(assets, selected_cuda)


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive into dest_dir while blocking Zip-Slip traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_path = Path(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                msg = f"Unsafe archive path detected in {zip_path.name}: {info.filename}"
                raise InferenceBackendError(msg)
            target = (dest_root / member_path).resolve()
            if not target.is_relative_to(dest_root):
                msg = f"Archive entry escapes target directory: {info.filename}"
                raise InferenceBackendError(msg)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
