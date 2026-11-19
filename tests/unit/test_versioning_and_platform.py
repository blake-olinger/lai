"""Unit tests for version SSOT, bump_version.py, and cross-platform llama.cpp downloads."""

from __future__ import annotations

import importlib.metadata
import io
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from lai import __version__
from lai.domain.errors import InferenceBackendError
from lai.domain.version import get_package_version, read_pyproject_version
from lai.inference import ModelDownloader
from lai.inference._archive import resolve_platform_spec, safe_extract_tar_gz

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bump_version  # noqa: E402

_HTTP_OK = 200


def _make_tar_gz_bytes(
    entries: Sequence[tuple[str, bytes | None, int, str | None]],
) -> bytes:
    """Build an in-memory .tar.gz archive of (name, content_or_None, mode, symlink_target)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data, mode, linkname in entries:
            info = tarfile.TarInfo(name=name)
            info.mode = mode
            if linkname is not None:
                info.type = tarfile.SYMTYPE
                info.linkname = linkname
                tf.addfile(info)
            elif data is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_package_version_ssot_and_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package version resolves from metadata, falls back to pyproject.toml, then dev0."""
    pyproject_ver = read_pyproject_version(REPO_ROOT / "pyproject.toml")
    assert pyproject_ver is not None
    assert __version__ == pyproject_ver

    custom_pyproject = tmp_path / "pyproject.toml"
    custom_pyproject.write_text('[project]\nname = "demo"\nversion = "2.3.4"\n', encoding="utf-8")
    assert read_pyproject_version(custom_pyproject) == "2.3.4"
    assert read_pyproject_version(tmp_path / "missing.toml") is None

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("[project\n", encoding="utf-8")
    assert read_pyproject_version(bad_toml) is None

    def _missing_dist(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing_dist)
    assert get_package_version("lai", pyproject_path=custom_pyproject) == "2.3.4"
    assert get_package_version("lai", pyproject_path=tmp_path / "none.toml") == "0.0.0.dev0"


def test_bump_version_helpers_and_check_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """bump_version validates 3-part SemVer, updates pyproject + CHANGELOG, and checks tags."""
    assert bump_version.validate_semver(" 1.2.3 ") == "1.2.3"
    for bad_ver in ("1.0.2.0", "v1.2.3", "01.2.3", "1.2", "abc"):
        with pytest.raises(ValueError, match="SemVer"):
            bump_version.validate_semver(bad_ver)

    assert bump_version.validate_release_date("2026-12-01") == "2026-12-01"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        bump_version.validate_release_date("12/01/2026")

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "lai"\nversion = "0.1.0"\n\n[tool.demo]\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Feature X.\n\n"
        "## [0.1.0] - 2026-11-19\n\n- Init.\n\n"
        "[Unreleased]: https://github.com/blake-olinger/lai/compare/v0.1.0...HEAD\n"
        "[0.1.0]: https://github.com/blake-olinger/lai/releases/tag/v0.1.0\n",
        encoding="utf-8",
    )

    old_v, new_v = bump_version.bump_repository_version(
        "0.2.0",
        repo_root=tmp_path,
        release_date="2026-12-01",
        skip_lock=True,
        skip_openapi=True,
    )
    assert (old_v, new_v) == ("0.1.0", "0.2.0")
    assert 'version = "0.2.0"' in pyproject.read_text(encoding="utf-8")
    assert 'version = "9.9.9"' in pyproject.read_text(encoding="utf-8")

    cl_text = changelog.read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [0.2.0] - 2026-12-01\n\n### Added" in cl_text
    assert "[Unreleased]: https://github.com/blake-olinger/lai/compare/v0.2.0...HEAD" in cl_text
    assert "[0.2.0]: https://github.com/blake-olinger/lai/compare/v0.1.0...v0.2.0" in cl_text

    with pytest.raises(ValueError, match="already exists"):
        bump_version.update_changelog(
            changelog,
            "0.2.0",
            previous_version="0.1.0",
            release_date="2026-12-02",
        )

    assert (
        bump_version.check_release_tag(
            "refs/tags/v0.2.0",
            pyproject_path=pyproject,
            installed_version="0.2.0",
        )
        == 0
    )
    assert "OK: release tag v0.2.0" in capsys.readouterr().out
    assert (
        bump_version.check_release_tag(
            "v0.1.0",
            pyproject_path=pyproject,
            installed_version="0.2.0",
        )
        == 1
    )
    assert "does not match pyproject.toml" in capsys.readouterr().err
    assert (
        bump_version.check_release_tag(
            "v0.2.0",
            pyproject_path=pyproject,
            installed_version="0.1.0",
        )
        == 1
    )
    assert "Runtime package version" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["bump_version.py", "--check-tag", f"v{__version__}"])
    assert bump_version.main() == 0


async def test_downloader_linux_cuda_tar_gz_and_tar_slip_protection(tmp_path: Path) -> None:
    """ModelDownloader fetches Linux CUDA .tar.gz + cudart and blocks Tar-Slip/unsafe links."""
    with pytest.raises(InferenceBackendError, match="Windows and Linux"):
        resolve_platform_spec("darwin")

    main_tar = _make_tar_gz_bytes(
        [
            ("build/", None, 0o755, None),
            ("build/llama-server", b"ELF_MAIN", 0o755, None),
            ("build/libllama.so.0", b"SO_DATA", 0o644, None),
            ("build/libllama.so", None, 0o777, "libllama.so.0"),
        ]
    )
    cudart_tar = _make_tar_gz_bytes([("libcudart.so.13", b"CUDART_SO", 0o644, None)])
    release_json = {
        "assets": [
            {
                "name": "llama-b12532-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "browser_download_url": "https://dl.test/old-lin.tar.gz",
            },
            {
                "name": "llama-b12532-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "browser_download_url": "https://dl.test/main-lin.tar.gz",
            },
            {
                "name": "cudart-llama-b12532-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "browser_download_url": "https://dl.test/cudart-lin.tar.gz",
            },
        ]
    }
    downloader = ModelDownloader(
        bin_dir=tmp_path / "bin",
        models_dir=tmp_path / "models",
        platform_name="linux",
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").mock(
            return_value=httpx.Response(_HTTP_OK, json=release_json)
        )
        mock.get("https://dl.test/main-lin.tar.gz").mock(
            return_value=httpx.Response(_HTTP_OK, content=main_tar)
        )
        mock.get("https://dl.test/cudart-lin.tar.gz").mock(
            return_value=httpx.Response(_HTTP_OK, content=cudart_tar)
        )
        exe_path = await downloader.download_llama_binaries()

    assert exe_path.name == "llama-server"
    assert exe_path.read_bytes() == b"ELF_MAIN"
    assert (exe_path.parent / "libcudart.so.13").read_bytes() == b"CUDART_SO"

    for bad_name, bad_entries, expected_err in (
        ("traversal.tar.gz", [("../escape.txt", b"x", 0o644, None)], "Unsafe archive path"),
        ("abs_sym.tar.gz", [("bad", None, 0o777, "/etc/passwd")], "Unsafe absolute symlink"),
        ("esc_sym.tar.gz", [("sub/bad", None, 0o777, "../../../etc/passwd")], "Unsafe symlink"),
    ):
        tar_path = tmp_path / bad_name
        tar_path.write_bytes(_make_tar_gz_bytes(bad_entries))
        with pytest.raises(InferenceBackendError, match=expected_err):
            safe_extract_tar_gz(tar_path, tmp_path / f"out_{bad_name}")
