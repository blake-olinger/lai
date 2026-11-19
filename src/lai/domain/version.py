"""Single-source-of-truth package version resolution from distribution metadata."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path
from typing import cast

_FALLBACK_DEV_VERSION = "0.0.0.dev0"
_REPO_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def read_pyproject_version(pyproject_path: Path = _REPO_PYPROJECT) -> str | None:
    """Read `[project].version` from a local pyproject.toml if present and valid."""
    if not pyproject_path.is_file():
        return None
    try:
        raw = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project: object = raw.get("project")
    if not isinstance(project, dict):
        return None
    version_obj = cast("dict[str, object]", project).get("version")
    if isinstance(version_obj, str) and version_obj.strip():
        return version_obj.strip()
    return None


def get_package_version(
    distribution_name: str = "lai",
    *,
    pyproject_path: Path = _REPO_PYPROJECT,
) -> str:
    """Resolve package version from installed metadata, falling back to pyproject.toml."""
    try:
        resolved = importlib.metadata.version(distribution_name).strip()
    except importlib.metadata.PackageNotFoundError:
        resolved = ""
    if resolved:
        return resolved
    from_pyproject = read_pyproject_version(pyproject_path)
    return from_pyproject if from_pyproject is not None else _FALLBACK_DEV_VERSION


PACKAGE_VERSION: str = get_package_version()
