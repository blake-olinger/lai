"""Unit tests for OpenAICompatProvider, ModelDownloader, GPU telemetry, and Win32 jobs."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx
from huggingface_hub.errors import RepositoryNotFoundError

from lai.domain.chat import MessageRole
from lai.domain.errors import InferenceBackendError, ValidationFailedError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceToolDef,
    ModelProfile,
)
from lai.inference import (
    LlamaCppProvider,
    ModelDownloader,
    OpenAICompatProvider,
    query_gpu_telemetry,
)
from lai.inference._winjob import WindowsJobObject
from lai.ports.inference import InferenceProviderPort

if TYPE_CHECKING:
    from collections.abc import Sequence

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
HTTP_SERVER_ERROR = 500
EXPECTED_PORT = 8081
EXPECTED_PROMPT_TOKENS = 12
EXPECTED_COMPLETION_TOKENS = 8
EXPECTED_TOTAL_TOKENS = 20
EXPECTED_SLOTS_TOTAL = 2
EXPECTED_GPU_USED_MB = 2220
EXPECTED_GPU_TOTAL_MB = 16303
EXPECTED_GPU_UTIL = 11
EXPECTED_GPU_TEMP = 35.32


def _make_zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


async def test_llamacpp_provider_complete_and_health() -> None:
    """LlamaCppProvider serializes tools, parses thinking/tool calls, and checks health."""
    request = InferenceRequest(
        model="gemma4-26b-a4b",
        messages=(InferenceMessage(role=MessageRole.USER, content="Find errors"),),
        tools=(InferenceToolDef(name="search_files", description="Search indexed files"),),
        enable_thinking=True,
    )
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_files", "arguments": '{"q":"err"}'},
    }
    response_body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<thought>check logs</thought>Calling tool",
                    "reasoning_content": "plan:",
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "completion_tokens": EXPECTED_COMPLETION_TOKENS,
            "total_tokens": EXPECTED_TOTAL_TOKENS,
        },
    }
    async with LlamaCppProvider(host="127.0.0.1", port=EXPECTED_PORT) as provider:
        assert isinstance(provider, InferenceProviderPort)
        with respx.mock(assert_all_called=True) as mock:
            comp_route = mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                return_value=httpx.Response(HTTP_OK, json=response_body)
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


async def test_openai_compat_provider_auth_streaming_and_health() -> None:
    """OpenAICompatProvider normalizes /v1 URLs, sends Bearer auth, and streams deltas."""
    with pytest.raises(ValidationFailedError):
        OpenAICompatProvider("ftp://invalid.local")

    request = InferenceRequest(
        model="gpt-4o-mini",
        messages=(InferenceMessage(role=MessageRole.USER, content="Ping"),),
    )
    sse_data = (
        'data: {"choices":[{"delta":{"content":"Pong"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    complete_json = {
        "choices": [{"message": {"role": "assistant", "content": "Pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    async with OpenAICompatProvider(
        "http://127.0.0.1:11434",
        api_key="secret-token",
    ) as provider:
        assert isinstance(provider, InferenceProviderPort)
        with respx.mock(assert_all_called=True) as mock:
            models_route = mock.get("http://127.0.0.1:11434/v1/models").mock(
                return_value=httpx.Response(HTTP_OK, json={"data": [{"id": "gpt-4o-mini"}]})
            )
            comp_route = mock.post("http://127.0.0.1:11434/v1/chat/completions").mock(
                side_effect=[
                    httpx.Response(HTTP_OK, json=complete_json),
                    httpx.Response(HTTP_OK, content=sse_data.encode("utf-8")),
                    httpx.Response(HTTP_SERVER_ERROR, text="upstream error"),
                ]
            )
            health = await provider.health()
            assert health.healthy is True
            assert health.model_loaded == "gpt-4o-mini"
            auth_hdr = cast("httpx.Request", models_route.calls[0].request).headers["authorization"]
            assert auth_hdr == "Bearer secret-token"

            resp = await provider.chat_complete(request)
            assert resp.content == "Pong"
            sent_body = cast("httpx.Request", comp_route.calls[0].request).content
            assert b"chat_template_kwargs" not in sent_body

            chunks = [chunk async for chunk in provider.chat_stream(request)]
            assert "".join(c.content_delta for c in chunks) == "Pong"

            with pytest.raises(InferenceBackendError, match="HTTP 500"):
                await provider.chat_complete(request)


async def test_downloader_llama_cuda_binaries_and_zip_slip(tmp_path: Path) -> None:
    """ModelDownloader fetches newest Windows CUDA x64 binary + cudart and blocks Zip-Slip."""
    main_zip = _make_zip_bytes({"llama-server.exe": b"MZ_MAIN", "ggml.dll": b"DLL"})
    cudart_zip = _make_zip_bytes({"cudart64_13.dll": b"CUDART"})
    release_json = {
        "assets": [
            {
                "name": "llama-b9000-bin-win-cuda-12.4-x64.zip",
                "browser_download_url": "https://dl.test/old.zip",
            },
            {
                "name": "llama-b9897-bin-win-cuda-13.3-x64.zip",
                "browser_download_url": "https://dl.test/main.zip",
            },
            {
                "name": "cudart-llama-bin-win-cuda-13.3-x64.zip",
                "browser_download_url": "https://dl.test/cudart.zip",
            },
        ]
    }
    downloader = ModelDownloader(bin_dir=tmp_path / "bin", models_dir=tmp_path / "models")
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").mock(
            return_value=httpx.Response(HTTP_OK, json=release_json)
        )
        mock.get("https://dl.test/main.zip").mock(
            return_value=httpx.Response(HTTP_OK, content=main_zip)
        )
        mock.get("https://dl.test/cudart.zip").mock(
            return_value=httpx.Response(HTTP_OK, content=cudart_zip)
        )
        exe_path = await downloader.download_llama_binaries()

    assert exe_path == tmp_path / "bin" / "llama" / "llama-server.exe"
    assert exe_path.read_bytes() == b"MZ_MAIN"
    assert (exe_path.parent / "cudart64_13.dll").read_bytes() == b"CUDART"
    # Second call uses cached binary without hitting network.
    assert await downloader.download_llama_binaries() == exe_path

    evil_zip = _make_zip_bytes({"../escape.txt": b"boom"})
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").mock(
            return_value=httpx.Response(
                HTTP_OK,
                json={
                    "assets": [
                        {
                            "name": "llama-b9897-bin-win-cuda-13.3-x64.zip",
                            "browser_download_url": "https://dl.test/evil.zip",
                        }
                    ]
                },
            )
        )
        mock.get("https://dl.test/evil.zip").mock(
            return_value=httpx.Response(HTTP_OK, content=evil_zip)
        )
        with pytest.raises(InferenceBackendError, match="Unsafe archive path"):
            await downloader.download_llama_binaries(force=True)


async def test_downloader_gguf_model_and_shieldgemma_fallback(tmp_path: Path) -> None:
    """ModelDownloader supports ModelProfile, GuardrailConfig, and fallback repos."""
    calls: list[tuple[str, str, bool, str | None]] = []

    def _fake_hf(
        *,
        repo_id: str,
        filename: str,
        local_dir: str,
        force_download: bool,
        token: str | None,
    ) -> str:
        calls.append((repo_id, filename, force_download, token))
        if repo_id.startswith("QuantFactory/"):
            msg = "missing primary"
            req = httpx.Request("GET", "https://huggingface.co/QuantFactory/shieldgemma-2b-GGUF")
            raise RepositoryNotFoundError(msg, response=httpx.Response(404, request=req))
        if repo_id.startswith("broken/"):
            msg = "network fail"
            raise OSError(msg)
        out = Path(local_dir) / filename
        out.write_bytes(b"GGUF")
        return str(out)

    downloader = ModelDownloader(
        bin_dir=tmp_path / "bin",
        models_dir=tmp_path / "models",
        hf_downloader=_fake_hf,
    )
    profile = ModelProfile(
        name="gemma4",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        fallback_hf_repo="bartowski/google_gemma-4-26B-A4B-it-GGUF",
        hf_filename="gemma4.gguf",
    )
    gemma_path = await downloader.download_model(profile)
    assert gemma_path.read_bytes() == b"GGUF"

    shield = GuardrailConfig()
    shield_path = await downloader.download_model(shield)
    assert shield_path.name == shield.hf_filename
    assert (shield.hf_repo, shield.hf_filename, False, None) in calls
    assert (shield.fallback_hf_repo, shield.hf_filename, False, None) in calls

    broken = profile.model_copy(
        update={"hf_repo": "broken/a", "fallback_hf_repo": "broken/b", "hf_filename": "x.gguf"}
    )
    with pytest.raises(InferenceBackendError, match="Failed to download"):
        await downloader.download_model(broken)


def test_query_gpu_telemetry_parsing_and_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    """query_gpu_telemetry resolves nvidia-smi, parses float temps, and handles absence."""
    seen_cmds: list[list[str]] = []

    def _ok_run(cmd: Sequence[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(list(cmd))
        out = "NVIDIA GeForce RTX 5080, 2220, 16303, 11, 35.32\n"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=out, stderr="")

    def _which_nvidia(name: str) -> str | None:
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def _which_none(_name: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", _which_nvidia)
    monkeypatch.setattr(subprocess, "run", _ok_run)
    info = query_gpu_telemetry()
    assert seen_cmds[0][0] == "/usr/bin/nvidia-smi"
    assert info is not None
    assert info.gpu_name == "NVIDIA GeForce RTX 5080"
    assert (info.memory_used_mb, info.memory_total_mb) == (
        EXPECTED_GPU_USED_MB,
        EXPECTED_GPU_TOTAL_MB,
    )
    assert (info.utilization_percent, info.temperature_c) == (EXPECTED_GPU_UTIL, EXPECTED_GPU_TEMP)

    def _missing_run(cmd: Sequence[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(list(cmd))
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", _missing_run)
    assert query_gpu_telemetry() is None
    assert seen_cmds[-1][0] == "/usr/bin/nvidia-smi"
    monkeypatch.setattr(shutil, "which", _which_none)
    assert query_gpu_telemetry() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects are Windows-only")
def test_windows_job_object_kills_assigned_process_on_close() -> None:
    """Closing a WindowsJobObject terminates its assigned child process immediately."""
    job = WindowsJobObject()
    assert job.active is True
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assigned = job.assign_pid(proc.pid)
        assert assigned is True
        job.close()
        rc = proc.wait(timeout=5.0)
        assert rc is not None
    finally:
        job.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        time.sleep(0.01)
