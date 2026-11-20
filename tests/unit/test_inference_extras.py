"""Unit tests for LlamaCppProvider, Linux ModelDownloader, manifest, CUDA, and integrity."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx
from huggingface_hub.errors import RepositoryNotFoundError

from lai.domain.chat import MessageRole
from lai.domain.errors import InferenceBackendError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceToolDef,
    LlamaCppBinarySpec,
    ModelProfile,
)
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.inference import (
    DEFAULT_CUDA_VERSION,
    DEFAULT_LLAMA_CPP_TAG,
    LlamaCppProvider,
    ModelDownloader,
    detect_nvidia_cuda_version,
    query_gpu_telemetry,
    select_compatible_cuda_version,
)
from lai.inference._runtime_helpers import parse_nvidia_smi_cuda_version, parse_nvidia_smi_output
from lai.observability.logging import configure_observability
from lai.ports.inference import InferenceProviderPort

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
EXPECTED_PORT = 8081
EXPECTED_TOTAL_TOKENS = 20
EXPECTED_SLOTS_TOTAL = 2
EXPECTED_GPU_USED_MB = 2220
EXPECTED_GPU_TOTAL_MB = 16303
EXPECTED_GPU_UTIL = 11
EXPECTED_GPU_TEMP = 35.32


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _tar(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _asset(name: str, url: str, data: bytes | None = None) -> dict[str, str]:
    item = {"name": name, "browser_download_url": url}
    if data is not None:
        item["digest"] = f"sha256:{hashlib.sha256(data).hexdigest()}"
    return item


async def test_llamacpp_provider_complete_and_health() -> None:
    """LlamaCppProvider serializes tools, parses thinking/tool calls, and checks health."""
    request = InferenceRequest(
        model="gemma4-26b-a4b",
        messages=(InferenceMessage(role=MessageRole.USER, content="Find errors"),),
        tools=(InferenceToolDef(name="search_files", description="Search indexed files"),),
        enable_thinking=True,
    )
    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_files", "arguments": '{"q":"err"}'},
    }
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<thought>check logs</thought>Calling tool",
                    "reasoning_content": "plan:",
                    "tool_calls": [tool],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    async with LlamaCppProvider(host="127.0.0.1", port=EXPECTED_PORT) as provider:
        assert isinstance(provider, InferenceProviderPort)
        with respx.mock(assert_all_called=True) as mock:
            comp_route = mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                return_value=httpx.Response(HTTP_OK, json=body)
            )
            mock.get("http://127.0.0.1:8081/health").mock(
                side_effect=[
                    httpx.Response(
                        HTTP_OK, json={"status": "ok", "slots_idle": 2, "slots_processing": 0}
                    ),
                    httpx.Response(HTTP_UNAVAILABLE, json={"error": {"message": "Loading model"}}),
                ]
            )
            resp = await provider.chat_complete(request)
            sent = json.loads(cast("httpx.Request", comp_route.calls[0].request).content)
            assert sent["stream"] is False
            assert sent["chat_template_kwargs"] == {"enable_thinking": True}
            assert resp.thinking == "plan:check logs"
            assert resp.content == "Calling tool"
            assert resp.tool_calls[0].name == "search_files"
            assert resp.usage.total_tokens == EXPECTED_TOTAL_TOKENS
            ok = await provider.health()
            loading = await provider.health()
            assert ok.healthy is True
            assert ok.slots_total == EXPECTED_SLOTS_TOTAL
            assert loading.healthy is False
            assert loading.status == "starting"


def _spec(tag: str, cuda: str, main_sha: str, cu_sha: str) -> LlamaCppBinarySpec:
    return LlamaCppBinarySpec(
        tag=tag,
        cuda_version=cuda,
        asset_name=f"llama-{tag}-bin-ubuntu-cuda-{cuda}-x64.tar.gz",
        sha256=main_sha,
        cudart_asset_name=f"cudart-llama-{tag}-bin-ubuntu-cuda-{cuda}-x64.tar.gz",
        cudart_sha256=cu_sha,
    )


async def test_downloader_cuda_auto_select_sha256_and_atomic_swap(tmp_path: Path) -> None:
    """ModelDownloader auto-selects CUDA, enforces SHA-256, and swaps current symlink."""
    m12 = _tar({"llama-b12552/llama-server": b"ELF_12_8"})
    c12 = _tar({"cudart-12.8/libcudart.so.12": b"CU12"})
    m13 = _tar({"llama-b12556/llama-server": b"ELF_13_5"})
    c13 = _tar({"cudart-13.5/libcudart.so.13": b"CU13"})
    tag = DEFAULT_LLAMA_CPP_TAG
    manifest = {
        "12.8": _spec(
            tag, "12.8", hashlib.sha256(m12).hexdigest(), hashlib.sha256(c12).hexdigest()
        ),
        "13.5": _spec(tag, "13.5", "f" * 64, "e" * 64),
    }
    rel_552 = {
        "tag_name": tag,
        "assets": [
            _asset(manifest["12.8"].asset_name, "https://dl.test/m12.tar.gz"),
            _asset(manifest["12.8"].cudart_asset_name, "https://dl.test/c12.tar.gz"),
            _asset(manifest["13.5"].asset_name, "https://dl.test/bad13.tar.gz"),
        ],
    }
    rel_556 = {
        "tag_name": "b12556",
        "assets": [
            _asset(
                "llama-b12556-bin-ubuntu-cuda-13.5-x64.tar.gz", "https://dl.test/m13.tar.gz", m13
            ),
            _asset(
                "cudart-llama-b12556-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "https://dl.test/c13.tar.gz",
                c13,
            ),
        ],
    }
    driver_cuda: tuple[int, int] | None = None
    manager = configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=False))
    try:
        dl = ModelDownloader(
            bin_dir=tmp_path / "bin",
            models_dir=tmp_path / "models",
            platform_name="linux",
            machine_arch="x86_64",
            cuda_detector=lambda: driver_cuda,
            manifest=manifest,
        )
        base = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags"
        dl_u = "https://dl.test"
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{base}/{tag}").mock(return_value=httpx.Response(HTTP_OK, json=rel_552))
            mock.get(f"{dl_u}/m12.tar.gz").mock(return_value=httpx.Response(HTTP_OK, content=m12))
            mock.get(f"{dl_u}/c12.tar.gz").mock(return_value=httpx.Response(HTTP_OK, content=c12))
            exe_12 = await dl.download_llama_binaries()
        assert exe_12 == tmp_path / "bin" / "llama" / "current" / "llama-server"
        assert exe_12.read_bytes() == b"ELF_12_8"
        assert (exe_12.parent / "libcudart.so.12").read_bytes() == b"CU12"
        assert await dl.download_llama_binaries() == exe_12

        driver_cuda = (13, 5)
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{base}/{tag}").mock(return_value=httpx.Response(HTTP_OK, json=rel_552))
            mock.get(f"{dl_u}/bad13.tar.gz").mock(return_value=httpx.Response(HTTP_OK, content=m13))
            with pytest.raises(InferenceBackendError, match="SHA-256 integrity mismatch"):
                await dl.download_llama_binaries(force=True)
        assert not list((tmp_path / "bin" / "llama").glob(".*.tmp"))
        assert any(a.category is AlertCategory.SECURITY_ALERT for a in manager.recent_alerts())
        assert exe_12.read_bytes() == b"ELF_12_8"

        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{base}/b12556").mock(return_value=httpx.Response(HTTP_OK, json=rel_556))
            mock.get(f"{dl_u}/m13.tar.gz").mock(return_value=httpx.Response(HTTP_OK, content=m13))
            mock.get(f"{dl_u}/c13.tar.gz").mock(return_value=httpx.Response(HTTP_OK, content=c13))
            exe_13 = await dl.download_llama_binaries(tag="b12556")
        assert exe_13.read_bytes() == b"ELF_13_5"
        assert (exe_13.parent / "libcudart.so.13").read_bytes() == b"CU13"
        assert await dl.download_llama_binaries(tag="b12556") == exe_13
    finally:
        manager.close()


async def test_downloader_gguf_revision_sha256_cache_and_fallback(tmp_path: Path) -> None:
    """ModelDownloader pins hf_revision, verifies cached & fresh SHA-256, and falls back."""
    calls: list[tuple[str, str, bool, str | None, str | None]] = []

    def _fake_hf(  # noqa: PLR0913
        *,
        repo_id: str,
        filename: str,
        local_dir: str,
        force_download: bool,
        token: str | None,
        revision: str | None = None,
    ) -> str:
        calls.append((repo_id, filename, force_download, token, revision))
        if repo_id.startswith("QuantFactory/"):
            req = httpx.Request("GET", "https://huggingface.co/QuantFactory/shieldgemma-2b-GGUF")
            raise RepositoryNotFoundError("missing", response=httpx.Response(404, request=req))
        if repo_id.startswith("timeout/"):
            msg = "hf timeout"
            raise httpx.ReadTimeout(msg)
        if repo_id.startswith("broken/"):
            msg = "network fail"
            raise OSError(msg)
        out = Path(local_dir) / filename
        out.write_bytes(b"CORRUPT" if repo_id.startswith("tampered/") else b"GGUF")
        return str(out)

    good_sha = hashlib.sha256(b"GGUF").hexdigest()
    dl = ModelDownloader(
        bin_dir=tmp_path / "bin", models_dir=tmp_path / "models", hf_downloader=_fake_hf
    )
    profile = ModelProfile(
        name="gemma4",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        fallback_hf_repo="bartowski/google_gemma-4-26B-A4B-it-GGUF",
        hf_filename="gemma4.gguf",
        hf_revision="abc1234",
        sha256=good_sha,
    )
    gemma_path = await dl.download_model(profile)
    assert gemma_path.read_bytes() == b"GGUF"
    assert calls[0] == (profile.hf_repo, profile.hf_filename, False, None, "abc1234")
    assert await dl.download_model(profile) == gemma_path
    assert len(calls) == 1
    gemma_path.write_bytes(b"BITFLIP")
    with pytest.raises(InferenceBackendError, match="SHA-256 integrity mismatch"):
        await dl.download_model(profile)
    assert not gemma_path.exists()

    tampered = profile.model_copy(
        update={"hf_repo": "tampered/repo", "hf_filename": "bad.gguf", "sha256": good_sha}
    )
    with pytest.raises(InferenceBackendError, match="SHA-256 integrity mismatch"):
        await dl.download_model(tampered)
    assert not (tmp_path / "models" / "bad.gguf").exists()

    shield = GuardrailConfig()
    assert (await dl.download_model(shield)).name == shield.hf_filename
    assert (shield.fallback_hf_repo, shield.hf_filename, False, None, None) in calls
    for upd in (
        {"hf_repo": "timeout/a", "fallback_hf_repo": None, "hf_filename": "t.gguf"},
        {"hf_repo": "broken/a", "fallback_hf_repo": "broken/b", "hf_filename": "x.gguf"},
    ):
        with pytest.raises(InferenceBackendError, match="Failed to download"):
            await dl.download_model(profile.model_copy(update=upd))


def test_gpu_telemetry_and_cuda_version_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """query_gpu_telemetry and detect_nvidia_cuda_version parse nvidia-smi and handle errors."""
    banner = "| NVIDIA-SMI 570.124.06   Driver Version: 570.124.06   CUDA Version: 13.5 |\n"
    assert parse_nvidia_smi_cuda_version(banner) == (13, 5)
    assert parse_nvidia_smi_cuda_version("no header") is None
    assert parse_nvidia_smi_output("") is None
    assert parse_nvidia_smi_output("bad,csv") is None
    assert parse_nvidia_smi_output("RTX 5080, 100, 16000, 5, [N/A]\n") is not None
    assert select_compatible_cuda_version((13, 5)) == "13.5"
    assert select_compatible_cuda_version((12, 9)) == "12.8"
    assert select_compatible_cuda_version((12, 4)) == DEFAULT_CUDA_VERSION
    assert select_compatible_cuda_version(None) == DEFAULT_CUDA_VERSION

    def _ok_run(cmd: Sequence[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        out = banner if len(cmd) == 1 else "NVIDIA GeForce RTX 5080, 2220, 16303, 11, 35.32\n"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=out, stderr="")

    def _which_nv(name: str) -> str | None:
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def _which_none(_name: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", _which_nv)
    monkeypatch.setattr(subprocess, "run", _ok_run)
    assert detect_nvidia_cuda_version() == (13, 5)
    info = query_gpu_telemetry()
    assert info is not None
    assert (info.memory_used_mb, info.memory_total_mb) == (
        EXPECTED_GPU_USED_MB,
        EXPECTED_GPU_TOTAL_MB,
    )
    assert (info.utilization_percent, info.temperature_c) == (EXPECTED_GPU_UTIL, EXPECTED_GPU_TEMP)

    def _missing_run(cmd: Sequence[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        del cmd
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", _missing_run)
    assert query_gpu_telemetry() is None
    assert detect_nvidia_cuda_version() is None
    monkeypatch.setattr(shutil, "which", _which_none)
    assert query_gpu_telemetry() is None
    assert detect_nvidia_cuda_version() is None
