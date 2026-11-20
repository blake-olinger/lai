"""Bump SemVer version in pyproject.toml and sync uv.lock and OpenAPI docs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lai.domain.version import get_package_version, read_pyproject_version  # noqa: E402

_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_PROJECT_SECTION_RE = re.compile(r"(?m)^\[project\][ \t]*$")
_NEXT_SECTION_RE = re.compile(r"(?m)^\[[^\]]+\][ \t]*$")
_VERSION_LINE_RE = re.compile(r'(?m)^(version[ \t]*=[ \t]*")([^"]+)(")[ \t]*$')


def validate_semver(version: str) -> str:
    """Validate strict 3-part SemVer (MAJOR.MINOR.PATCH) and return the stripped value."""
    cleaned = version.strip()
    if _SEMVER_RE.fullmatch(cleaned) is None:
        msg = f"Invalid SemVer version {version!r}; expected MAJOR.MINOR.PATCH (e.g. 1.0.2)."
        raise ValueError(msg)
    return cleaned


def update_pyproject_version(pyproject_path: Path, new_version: str) -> str:
    """Replace `[project].version` in pyproject.toml and return the previous version."""
    text = pyproject_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    project_obj: object = parsed.get("project")
    if not isinstance(project_obj, dict):
        msg = f"Missing [project] table in {pyproject_path}."
        raise TypeError(msg)
    old_version = cast("dict[str, object]", project_obj).get("version")
    if not isinstance(old_version, str):
        msg = f"Missing [project].version in {pyproject_path}."
        raise TypeError(msg)

    sec_match = _PROJECT_SECTION_RE.search(text)
    if sec_match is None:
        msg = f"Could not locate [project] header in {pyproject_path}."
        raise ValueError(msg)
    next_match = _NEXT_SECTION_RE.search(text, pos=sec_match.end())
    sec_end = next_match.start() if next_match is not None else len(text)
    section_body = text[sec_match.end() : sec_end]

    updated_body, count = _VERSION_LINE_RE.subn(
        rf"\g<1>{new_version}\g<3>",
        section_body,
        count=1,
    )
    if count != 1:
        msg = f"Could not update version line inside [project] in {pyproject_path}."
        raise ValueError(msg)

    updated = text[: sec_match.end()] + updated_body + text[sec_end:]
    pyproject_path.write_text(updated, encoding="utf-8", newline="\n")
    return old_version


def check_release_tag(
    tag: str,
    *,
    pyproject_path: Path = REPO_ROOT / "pyproject.toml",
    installed_version: str | None = None,
) -> int:
    """Verify git tag vX.Y.Z matches pyproject.toml and installed package metadata."""
    cleaned_tag = tag.strip()
    if cleaned_tag.startswith("refs/tags/"):
        cleaned_tag = cleaned_tag.removeprefix("refs/tags/")
    expected_version = read_pyproject_version(pyproject_path)
    if expected_version is None:
        sys.stderr.write(f"ERROR: Could not read [project].version from {pyproject_path}.\n")
        return 1
    try:
        validate_semver(expected_version)
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    expected_tag = f"v{expected_version}"
    if cleaned_tag != expected_tag:
        sys.stderr.write(
            f"ERROR: Release tag {cleaned_tag!r} does not match pyproject.toml "
            f"version {expected_version!r} (expected tag {expected_tag!r}).\n"
        )
        return 1

    runtime_version = (
        installed_version
        if installed_version is not None
        else get_package_version(pyproject_path=pyproject_path)
    )
    if runtime_version != expected_version:
        sys.stderr.write(
            f"ERROR: Runtime package version {runtime_version!r} does not match "
            f"pyproject.toml version {expected_version!r}. Run `uv sync --locked`.\n"
        )
        return 1

    sys.stdout.write(
        f"OK: release tag {cleaned_tag} matches pyproject.toml and runtime version "
        f"({expected_version}).\n"
    )
    return 0


def _run_checked(cmd: list[str], cwd: Path) -> None:
    """Run a sync subprocess and raise RuntimeError on non-zero exit."""
    sys.stdout.write(f">>> {' '.join(cmd)}\n")
    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=cwd, check=False)  # noqa: S603
    if completed.returncode != 0:
        msg = f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}"
        raise RuntimeError(msg)


def bump_repository_version(
    new_version: str,
    *,
    repo_root: Path = REPO_ROOT,
    skip_lock: bool = False,
    skip_openapi: bool = False,
) -> tuple[str, str]:
    """Bump version in pyproject.toml and resync uv.lock and OpenAPI docs."""
    validated_version = validate_semver(new_version)
    pyproject_path = repo_root / "pyproject.toml"
    old_version = update_pyproject_version(pyproject_path, validated_version)
    if not skip_lock:
        _run_checked(["uv", "lock"], cwd=repo_root)
        _run_checked(["uv", "sync", "--locked"], cwd=repo_root)
    if not skip_openapi:
        _run_checked(
            [sys.executable, str(repo_root / "scripts" / "export_openapi.py")],
            cwd=repo_root,
        )
    return old_version, validated_version


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for version bumping and release tag verification."""
    parser = argparse.ArgumentParser(
        description="Bump SemVer version and synchronize lockfile and OpenAPI docs.",
    )
    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help="Target SemVer version (MAJOR.MINOR.PATCH, e.g. 0.2.0).",
    )
    parser.add_argument(
        "--skip-lock",
        action="store_true",
        help="Skip running `uv lock` and `uv sync --locked`.",
    )
    parser.add_argument(
        "--skip-openapi",
        action="store_true",
        help="Skip regenerating openapi.json and docs/api-reference.md.",
    )
    parser.add_argument(
        "--check-tag",
        default=None,
        help="Verify that a git tag (e.g. v0.1.0) matches pyproject.toml and package metadata.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for bumping version or checking release tag parity."""
    args = build_parser().parse_args()
    check_tag = cast("str | None", args.check_tag)
    if check_tag is not None:
        return check_release_tag(check_tag)

    raw_version = cast("str | None", args.version)
    if raw_version is None:
        sys.stderr.write("ERROR: Provide a target version (e.g. 0.2.0) or --check-tag vX.Y.Z.\n")
        return 2
    try:
        old_ver, new_ver = bump_repository_version(
            raw_version,
            skip_lock=cast("bool", args.skip_lock),
            skip_openapi=cast("bool", args.skip_openapi),
        )
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    sys.stdout.write(f"OK: bumped version {old_ver} -> {new_ver}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
