"""Additional edge-case unit tests for inference parser, clients, supervisor, and downloader."""

from __future__ import annotations

import io
import json
import logging
import subprocess
import tarfile
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

from lai.domain.chat import MessageRole
from lai.domain.errors import InferenceBackendError, ValidationFailedError
from lai.domain.inference import (
    BackendHealth,
    InferenceMessage,
    InferenceRequest,
    InferenceToolCall,
    ModelProfile,
)
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.inference import (
    DEFAULT_LLAMA_CPP_TAG,
    LlamaCppProvider,
    LlamaCppSupervisor,
    ModelDownloader,
    ThinkingStreamParser,
)
from lai.observability.logging import configure_observability

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

HTTP_OK = 200
HTTP_ERROR = 500


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _make_tar_gz_files(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_thinking_parser_mid_thought_eof_and_multi_blocks() -> None:
    """ThinkingStreamParser handles mid-thought EOF, tag-in-reasoning, and multiple blocks."""
    parser = ThinkingStreamParser()
    d1 = parser.feed(
        "<thought>a</thought>X<thought>b</thought>Y<thought>c</th", "<thought>r</thought>"
    )
    assert parser.in_thought is True
    assert d1.thinking_delta == "rabc"
    assert d1.content_delta == "XY"

    d2 = parser.finalize()
    assert d2.thinking_delta == "</th"
    assert d2.content_delta == ""


async def test_llamacpp_client_edge_cases_and_stream_errors(tmp_path: Path) -> None:
    """Cover validation, unreachable health, tool history serialization, and stream errors."""
    with pytest.raises(ValidationFailedError):
        LlamaCppProvider(host="", port=8081)
    with pytest.raises(ValidationFailedError):
        LlamaCppProvider(host="127.0.0.1", port=0)

    req = InferenceRequest(
        model="gemma4",
        messages=(
            InferenceMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=(
                    InferenceToolCall(id="tc_1", name="search_files", arguments_json="{}"),
                ),
            ),
            InferenceMessage(
                role=MessageRole.TOOL,
                name="search_files",
                tool_call_id="tc_1",
                content="hit",
            ),
        ),
    )

    manager = configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=False))
    try:
        async with LlamaCppProvider(host="127.0.0.1", port=8081) as provider:
            with respx.mock(assert_all_called=True) as mock:
                mock.get("http://127.0.0.1:8081/health").mock(
                    side_effect=httpx.ConnectError("refused")
                )
                unreachable = await provider.health()
                assert unreachable.healthy is False
                assert unreachable.status == "unreachable"

            with respx.mock(assert_all_called=True) as mock:
                route = mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                    side_effect=[
                        httpx.Response(HTTP_ERROR, text="bad gateway"),
                        httpx.Response(
                            HTTP_OK,
                            content=b'data: {"error":{"message":"context overflow"}}\n\n',
                        ),
                        httpx.ReadTimeout("stream stalled"),
                    ]
                )
                with pytest.raises(InferenceBackendError, match="HTTP 500"):
                    async for _ in provider.chat_stream(req):
                        pass

                sent = json.loads(cast("httpx.Request", route.calls[0].request).content)
                assert sent["messages"][0]["tool_calls"][0]["id"] == "tc_1"
                assert sent["messages"][1]["tool_call_id"] == "tc_1"

                with pytest.raises(InferenceBackendError, match="context overflow"):
                    async for _ in provider.chat_stream(req):
                        pass

                with pytest.raises(InferenceBackendError, match="timed out"):
                    async for _ in provider.chat_stream(req):
                        pass

        assert any(a.category is AlertCategory.TIMEOUT for a in manager.recent_alerts())
    finally:
        manager.close()


class _QuickProc:
    """Minimal Popen double supporting startup crash and profile switch tests."""

    def __init__(self, returncode: int | None = None) -> None:
        self.pid = 7777
        self.stdout = io.BytesIO(b"fatal: out of memory\n")
        self._returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def kill(self) -> None:
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode


async def test_supervisor_missing_files_startup_crash_and_profile_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor rejects missing paths, alerts on startup crash, and switches profiles."""
    exe = tmp_path / "llama-server"
    models = tmp_path / "models"
    models.mkdir()
    log_dir = tmp_path / "logs"

    p1 = ModelProfile(name="p1", hf_repo="r/a", hf_filename="a.gguf")
    p2 = ModelProfile(name="p2", hf_repo="r/b", hf_filename="b.gguf")

    sup = LlamaCppSupervisor(
        executable_path=exe,
        models_dir=models,
        log_dir=log_dir,
        startup_timeout_seconds=0.5,
        poll_interval_seconds=0.01,
        health_probe=_ok_health,
    )
    with pytest.raises(InferenceBackendError, match="executable not found"):
        await sup.ensure_running(p1)

    exe.write_bytes(b"")
    with pytest.raises(InferenceBackendError, match="GGUF model file not found"):
        await sup.ensure_running(p1)

    (models / "a.gguf").write_bytes(b"A")
    (models / "b.gguf").write_bytes(b"B")

    manager = configure_observability(LoggingConfig(log_dir=log_dir, enable_file_logs=False))
    try:
        crashed_proc = _QuickProc(returncode=1)

        def _spawn_crashed(_cmd: list[str], **_kw: object) -> _QuickProc:
            return crashed_proc

        monkeypatch.setattr(subprocess, "Popen", _spawn_crashed)
        with pytest.raises(InferenceBackendError, match="exited unexpectedly"):
            await sup.ensure_running(p1)
        assert any(a.category is AlertCategory.SUBPROCESS_CRASH for a in manager.recent_alerts())
        assert "fatal: out of memory" in (log_dir / "llama-server.log").read_text("utf-8")

        procs: list[_QuickProc] = [_QuickProc(), _QuickProc()]
        idx = 0

        def _spawn(_cmd: list[str], **_kw: object) -> _QuickProc:
            nonlocal idx
            proc = procs[idx]
            idx += 1
            return proc

        monkeypatch.setattr(subprocess, "Popen", _spawn)
        h1 = await sup.ensure_running(p1)
        assert h1.model_loaded == "p1"
        h1_again = await sup.ensure_running(p1)
        assert h1_again.model_loaded == "p1"
        assert idx == 1

        h2 = await sup.ensure_running(p2)
        assert procs[0].terminated is True
        assert h2.model_loaded == "p2"
    finally:
        await sup.stop()
        manager.close()


async def _ok_health() -> BackendHealth:
    return BackendHealth(healthy=True, status="ok")


def _no_cuda() -> tuple[int, int] | None:
    return None


def _rel(name: str, url: str, tag: str = DEFAULT_LLAMA_CPP_TAG) -> dict[str, object]:
    return {"tag_name": tag, "assets": [{"name": name, "browser_download_url": url}]}


async def test_downloader_timeout_and_missing_cuda_asset(tmp_path: Path) -> None:
    """ModelDownloader alerts on timeouts and fails cleanly on missing Linux CUDA assets."""
    manager = configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=False))
    try:
        downloader = ModelDownloader(
            bin_dir=tmp_path / "bin",
            models_dir=tmp_path / "models",
            platform_name="linux",
            machine_arch="x86_64",
            cuda_detector=_no_cuda,
        )
        tag_url = (
            f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{DEFAULT_LLAMA_CPP_TAG}"
        )
        cu12 = "llama-b12552-bin-ubuntu-cuda-12.8-x64.tar.gz"
        with respx.mock(assert_all_called=True) as mock:
            mock.get(tag_url).mock(
                side_effect=[
                    httpx.ReadTimeout("github timeout"),
                    httpx.Response(HTTP_ERROR, text="server error"),
                    httpx.Response(HTTP_OK, json=_rel("src.tgz", "u")),
                    httpx.Response(HTTP_OK, json=_rel(cu12, "https://dl.test/fail.tar.gz")),
                    httpx.Response(HTTP_OK, json=_rel(cu12, "https://dl.test/slow.tar.gz")),
                ]
            )
            mock.get("https://dl.test/fail.tar.gz").mock(
                return_value=httpx.Response(HTTP_ERROR, text="download fail")
            )
            mock.get("https://dl.test/slow.tar.gz").mock(side_effect=httpx.ReadTimeout("dl stall"))
            for expected in ("timed out", "HTTP 500", "Linux CUDA x64", "HTTP 500", "timed out"):
                with pytest.raises(InferenceBackendError, match=expected):
                    await downloader.download_llama_binaries()

        assert any(a.category is AlertCategory.TIMEOUT for a in manager.recent_alerts())
    finally:
        manager.close()


async def test_downloader_latest_falls_back_to_rolling_releases(tmp_path: Path) -> None:
    """When tag='latest' lacks Linux CUDA tarballs, download_llama_binaries scans /releases."""
    tar_bytes = _make_tar_gz_files({"llama-b12478/llama-server": b"ELF_FALLBACK"})
    downloader = ModelDownloader(
        bin_dir=tmp_path / "bin",
        models_dir=tmp_path / "models",
        platform_name="linux",
        machine_arch="x86_64",
        cuda_detector=_no_cuda,
    )
    asset = "llama-b12478-bin-ubuntu-cuda-12.8-x64.tar.gz"
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").mock(
            return_value=httpx.Response(HTTP_OK, json=_rel("v1.0.0-source.tar.gz", "u", "v1.0.0"))
        )
        mock.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10").mock(
            return_value=httpx.Response(
                HTTP_OK,
                json=[
                    _rel("notes.txt", "n", "notes"),
                    _rel(asset, "https://dl.test/b12478.tar.gz", "b12478"),
                ],
            )
        )
        mock.get("https://dl.test/b12478.tar.gz").mock(
            return_value=httpx.Response(HTTP_OK, content=tar_bytes)
        )
        exe = await downloader.download_llama_binaries(tag="latest")
    assert exe == tmp_path / "bin" / "llama" / "current" / "llama-server"
    assert exe.read_bytes() == b"ELF_FALLBACK"
    assert (tmp_path / "bin" / "llama" / "releases" / "b12478" / "llama-server").exists()
