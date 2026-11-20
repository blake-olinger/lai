"""Unit tests for version SSOT, bump_version.py, and Linux x64 llama.cpp downloads."""

from __future__ import annotations

import hashlib
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
from lai.domain.inference import LlamaCppBinarySpec
from lai.domain.version import get_package_version, read_pyproject_version
from lai.inference import DEFAULT_LLAMA_CPP_TAG, ModelDownloader
from lai.inference._archive import (
    parse_release_asset_digests,
    parse_release_assets,
    parse_release_tag,
    resolve_platform_spec,
    safe_extract_archive,
    safe_extract_tar_gz,
    select_llama_cuda_assets,
    validate_release_tag,
)

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
    """bump_version validates 3-part SemVer, updates pyproject.toml, and checks tags."""
    assert bump_version.validate_semver(" 1.2.3 ") == "1.2.3"
    for bad_ver in ("1.0.2.0", "v1.2.3", "01.2.3", "1.2", "abc"):
        with pytest.raises(ValueError, match="SemVer"):
            bump_version.validate_semver(bad_ver)

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "lai"\nversion = "0.1.0"\n\n[tool.demo]\nversion = "9.9.9"\n',
        encoding="utf-8",
    )

    old_v, new_v = bump_version.bump_repository_version(
        "0.2.0",
        repo_root=tmp_path,
        skip_lock=True,
        skip_openapi=True,
    )
    assert (old_v, new_v) == ("0.1.0", "0.2.0")
    assert 'version = "0.2.0"' in pyproject.read_text(encoding="utf-8")
    assert 'version = "9.9.9"' in pyproject.read_text(encoding="utf-8")

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


def _detect_cuda_135() -> tuple[int, int] | None:
    return (13, 5)


async def test_downloader_linux_x64_nested_cudart_strip_and_tar_slip(tmp_path: Path) -> None:
    """Downloader enforces Linux x86_64, flattens nested cudart tarballs, and blocks Tar-Slip."""
    for bad_os in ("darwin", "win32"):
        with pytest.raises(InferenceBackendError, match="only on Linux x86_64"):
            resolve_platform_spec(bad_os, "x86_64")
    for bad_arch in ("aarch64", "arm64"):
        with pytest.raises(InferenceBackendError, match="x86_64/amd64"):
            resolve_platform_spec("linux", bad_arch)
    assert resolve_platform_spec("linux", "amd64")[3] == "llama-server"

    main_tar = _make_tar_gz_bytes(
        [
            ("llama-b12552/", None, 0o755, None),
            ("llama-b12552/llama-server", b"ELF_MAIN", 0o755, None),
            ("llama-b12552/libllama.so.0", b"SO_DATA", 0o644, None),
            ("llama-b12552/libllama.so", None, 0o777, "libllama.so.0"),
        ]
    )
    cudart_prefix = "cudart-llama-b12552-bin-ubuntu-cuda-13.5-x64"
    cudart_tar = _make_tar_gz_bytes(
        [
            (f"{cudart_prefix}/", None, 0o755, None),
            (f"{cudart_prefix}/libcudart.so.13", b"CUDART_SO", 0o644, None),
            (f"{cudart_prefix}/libcublas.so.13", b"CUBLAS_SO", 0o644, None),
            (f"{cudart_prefix}/libcublasLt.so.13", b"CUBLASLT_SO", 0o644, None),
            (f"{cudart_prefix}/libcudart.so", None, 0o777, "libcudart.so.13"),
        ]
    )
    test_manifest = {
        "13.5": LlamaCppBinarySpec(
            tag=DEFAULT_LLAMA_CPP_TAG,
            cuda_version="13.5",
            asset_name="llama-b12552-bin-ubuntu-cuda-13.5-x64.tar.gz",
            sha256=hashlib.sha256(main_tar).hexdigest(),
            cudart_asset_name="cudart-llama-b12552-bin-ubuntu-cuda-13.5-x64.tar.gz",
            cudart_sha256=hashlib.sha256(cudart_tar).hexdigest(),
        ),
    }
    release_json = {
        "tag_name": DEFAULT_LLAMA_CPP_TAG,
        "assets": [
            {
                "name": "llama-b12552-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "browser_download_url": "https://dl.test/old-lin.tar.gz",
            },
            {
                "name": "llama-b12552-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "browser_download_url": "https://dl.test/main-lin.tar.gz",
            },
            {
                "name": "cudart-llama-b12552-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "browser_download_url": "https://dl.test/cudart-lin.tar.gz",
            },
        ],
    }
    downloader = ModelDownloader(
        bin_dir=tmp_path / "bin",
        models_dir=tmp_path / "models",
        platform_name="linux",
        machine_arch="x86_64",
        cuda_detector=_detect_cuda_135,
        manifest=test_manifest,
    )
    tag_url = (
        f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{DEFAULT_LLAMA_CPP_TAG}"
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(tag_url).mock(return_value=httpx.Response(_HTTP_OK, json=release_json))
        mock.get("https://dl.test/main-lin.tar.gz").mock(
            return_value=httpx.Response(_HTTP_OK, content=main_tar)
        )
        mock.get("https://dl.test/cudart-lin.tar.gz").mock(
            return_value=httpx.Response(_HTTP_OK, content=cudart_tar)
        )
        exe_path = await downloader.download_llama_binaries()

    assert exe_path == tmp_path / "bin" / "llama" / "current" / "llama-server"
    assert exe_path.read_bytes() == b"ELF_MAIN"
    assert (exe_path.parent / "libcudart.so.13").read_bytes() == b"CUDART_SO"
    assert (exe_path.parent / "libcublas.so.13").read_bytes() == b"CUBLAS_SO"
    assert (exe_path.parent / "libcublasLt.so.13").read_bytes() == b"CUBLASLT_SO"
    assert not (exe_path.parent / cudart_prefix).exists()


def test_archive_strip_options_tar_slip_and_metadata_edge_cases(tmp_path: Path) -> None:
    """safe_extract_tar_gz handles strip_single_top_dir, Tar-Slip, and invalid metadata."""
    cudart_prefix = "cudart-llama-b12552-bin-ubuntu-cuda-13.5-x64"
    cudart_tar = _make_tar_gz_bytes(
        [
            (f"{cudart_prefix}/", None, 0o755, None),
            (f"{cudart_prefix}/libcudart.so.13", b"CUDART_SO", 0o644, None),
        ]
    )
    nested_tar = tmp_path / "nested.tar.gz"
    nested_tar.write_bytes(cudart_tar)
    no_strip_out = tmp_path / "no_strip"
    safe_extract_tar_gz(nested_tar, no_strip_out, strip_single_top_dir=False)
    assert (no_strip_out / cudart_prefix / "libcudart.so.13").read_bytes() == b"CUDART_SO"

    single_file_tar = tmp_path / "single.tar.gz"
    single_file_tar.write_bytes(_make_tar_gz_bytes([("libcudart.so.13", b"RAW", 0o644, None)]))
    single_out = tmp_path / "single_out"
    safe_extract_tar_gz(single_file_tar, single_out, strip_single_top_dir=True)
    assert (single_out / "libcudart.so.13").read_bytes() == b"RAW"

    zip_path = tmp_path / "legacy.zip"
    zip_path.write_bytes(b"PK")
    with pytest.raises(InferenceBackendError, match="Unsupported archive format"):
        safe_extract_archive(zip_path, tmp_path / "zip_out")

    for bad_name, bad_entries, expected_err in (
        ("traversal.tar.gz", [("../escape.txt", b"x", 0o644, None)], "Unsafe archive path"),
        ("nested_trav.tar.gz", [("top/../../escape.txt", b"x", 0o644, None)], "Unsafe archive"),
        ("abs_sym.tar.gz", [("top/bad", None, 0o777, "/etc/passwd")], "Unsafe absolute symlink"),
        ("esc_sym.tar.gz", [("top/sub/bad", None, 0o777, "../../../etc/passwd")], "Unsafe symlink"),
    ):
        tar_path = tmp_path / bad_name
        tar_path.write_bytes(_make_tar_gz_bytes(bad_entries))
        with pytest.raises(InferenceBackendError, match=expected_err):
            safe_extract_tar_gz(tar_path, tmp_path / f"out_{bad_name}")

    assert parse_release_tag("not-a-dict") is None
    assert parse_release_assets("not-a-dict") == []
    assert parse_release_asset_digests({"assets": "bad"}) == {}
    with pytest.raises(InferenceBackendError, match=r"Unsafe llama\.cpp release tag"):
        validate_release_tag("../evil")
    with pytest.raises(InferenceBackendError, match="cuda_version"):
        select_llama_cuda_assets(
            [("llama-b12552-bin-ubuntu-cuda-12.8-x64.tar.gz", "https://dl.test/a.tar.gz")],
            cuda_version="13.5",
            platform_name="linux",
            machine_arch="x86_64",
        )
    fifo_tar = tmp_path / "fifo.tar.gz"
    fifo_buf = io.BytesIO()
    with tarfile.open(fileobj=fifo_buf, mode="w:gz") as tf:
        fifo_info = tarfile.TarInfo(name="pipe")
        fifo_info.type = tarfile.FIFOTYPE
        tf.addfile(fifo_info)
    fifo_tar.write_bytes(fifo_buf.getvalue())
    with pytest.raises(InferenceBackendError, match="Unsupported tar entry type"):
        safe_extract_tar_gz(fifo_tar, tmp_path / "fifo_out")
