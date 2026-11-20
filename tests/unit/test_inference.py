"""Unit tests for llama.cpp flags, thinking parser, client, and POSIX supervisor."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import subprocess
import sys
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from lai.domain.chat import MessageRole
from lai.domain.errors import InferenceBackendError, ValidationFailedError
from lai.domain.inference import (
    BackendHealth,
    InferenceMessage,
    InferenceRequest,
    ModelProfile,
)
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.inference import (
    LlamaCppProvider,
    LlamaCppSupervisor,
    ThinkingStreamParser,
    build_llama_server_args,
    split_thinking_content,
)
from lai.observability.logging import configure_observability
from lai.ports.inference import InferenceSupervisorPort

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator
    from pathlib import Path

HTTP_OK = 200
EXPECTED_PORT = 8081
EXPECTED_STREAM_TOKENS = 9


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _sample_profile(**overrides: object) -> ModelProfile:
    base = ModelProfile(
        name="gemma4-26b-a4b",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        fallback_hf_repo="bartowski/google_gemma-4-26B-A4B-it-GGUF",
        hf_filename="gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
        reasoning_format="deepseek",
    )
    return base.model_copy(update=overrides)


def test_build_llama_server_args_and_validation(tmp_path: Path) -> None:
    """build_llama_server_args emits expected Gemma 4 flags and validates host/port."""
    model_path = tmp_path / "model.gguf"
    profile = _sample_profile()
    args = build_llama_server_args(profile, model_path, "127.0.0.1", EXPECTED_PORT)

    expected_tail = (
        "--host 127.0.0.1 --port 8081 --parallel 1 --ctx-size 24576 "
        "--flash-attn on --jinja -ctk q8_0 -ctv q8_0 -ngl 999 "
        "--n-cpu-moe 3 -b 2048 -ub 512 --no-mmap --reasoning-format deepseek"
    )
    assert args[:2] == ["--model", str(model_path)]
    assert " ".join(args[2:]) == expected_tail

    minimal = _sample_profile(
        flash_attn=False, jinja=False, cpu_moe_layers=0, reasoning_format=None
    )
    min_args = build_llama_server_args(minimal, model_path, "127.0.0.1", EXPECTED_PORT)
    for omitted in ("--flash-attn", "--jinja", "--n-cpu-moe", "--reasoning-format"):
        assert omitted not in min_args

    with pytest.raises(ValidationFailedError):
        build_llama_server_args(profile, model_path, "  ", EXPECTED_PORT)
    with pytest.raises(ValidationFailedError):
        build_llama_server_args(profile, model_path, "127.0.0.1", 70000)


def test_thinking_parser_streaming_and_non_streaming() -> None:
    """ThinkingStreamParser never leaks split <thought> tags into content deltas."""
    parser = ThinkingStreamParser()
    fragments = [
        ("Pre ", None),
        ("<tho", None),
        ("ught>step 1 ", "extra "),
        ("step 2</tho", None),
        ("ught>Ans", None),
        ("wer <th", None),
    ]
    contents: list[str] = []
    thoughts: list[str] = []
    for content_part, reasoning_part in fragments:
        delta = parser.feed(content_part, reasoning_part)
        assert "<thought>" not in delta.content_delta
        assert "</thought>" not in delta.content_delta
        contents.append(delta.content_delta)
        thoughts.append(delta.thinking_delta)

    tail = parser.finalize()
    assert "".join([*thoughts, tail.thinking_delta]) == "extra step 1 step 2"
    assert "".join([*contents, tail.content_delta]) == "Pre Answer <th"

    parsed = split_thinking_content(
        "<thought>alpha</thought>beta</thought>gamma", reasoning_content="pre-"
    )
    assert parsed.thinking_delta == "pre-alpha"
    assert parsed.content_delta == "betagamma"


async def test_llamacpp_provider_stream_tool_calls_and_timeout_alert(tmp_path: Path) -> None:
    """chat_stream assembles split tool call deltas and emits TIMEOUT alerts."""
    usage_frame = (
        'data: {"choices":[],"usage":'
        '{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}\n\n'
    )
    sse_payload = (
        'data: {"choices":[{"delta":{"content":"<tho"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"ught>think</thought>Hi ","tool_calls":'
        '[{"index":0,"id":"call_9","function":{"name":"read_lines","arguments":"{\\"line"}}]'
        '},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"arguments":"\\":1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        + usage_frame
        + "data: [DONE]\n\n"
    )
    request = InferenceRequest(
        model="gemma4-26b-a4b",
        messages=(InferenceMessage(role=MessageRole.USER, content="hi"),),
    )

    manager = configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=False))
    try:
        async with LlamaCppProvider(host="127.0.0.1", port=EXPECTED_PORT) as provider:
            with respx.mock(assert_all_called=True) as mock:
                mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                    return_value=httpx.Response(HTTP_OK, content=sse_payload.encode("utf-8"))
                )
                chunks = [chunk async for chunk in provider.chat_stream(request)]

            assert "".join(c.thinking_delta for c in chunks) == "think"
            assert "".join(c.content_delta for c in chunks) == "Hi "
            tool_chunks = [c for c in chunks if c.tool_calls]
            assert len(tool_chunks) == 1
            assert tool_chunks[0].tool_calls[0].id == "call_9"
            assert tool_chunks[0].tool_calls[0].arguments_json == '{"line":1}'
            assert chunks[-1].usage is not None
            assert chunks[-1].usage.total_tokens == EXPECTED_STREAM_TOKENS

            with respx.mock(assert_all_called=True) as mock:
                mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                    side_effect=httpx.ReadTimeout("timed out")
                )
                with pytest.raises(InferenceBackendError):
                    await provider.chat_complete(request)

        alerts = manager.recent_alerts()
        assert len(alerts) == 1
        assert alerts[0].category is AlertCategory.TIMEOUT
    finally:
        manager.close()


class _SlowClosingStream(httpx.AsyncByteStream):
    """Fake streaming byte source tracking immediate upstream aclose calls."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'
        await asyncio.sleep(30.0)

    async def aclose(self) -> None:
        self.closed = True


async def test_llamacpp_provider_cancellation_and_close_abort_upstream() -> None:
    """Closing the generator or cancelling its task immediately closes the HTTP stream."""
    request = InferenceRequest(
        model="gemma4-26b-a4b",
        messages=(InferenceMessage(role=MessageRole.USER, content="hi"),),
    )

    async with LlamaCppProvider(host="127.0.0.1", port=EXPECTED_PORT) as provider:
        stream_one = _SlowClosingStream()
        with respx.mock(assert_all_called=True) as mock:
            mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                return_value=httpx.Response(HTTP_OK, stream=stream_one)
            )
            gen = provider.chat_stream(request)
            assert (await anext(gen)).content_delta == "first"
            await gen.aclose()
        assert stream_one.closed is True

        stream_two = _SlowClosingStream()
        with respx.mock(assert_all_called=True) as mock:
            mock.post("http://127.0.0.1:8081/v1/chat/completions").mock(
                return_value=httpx.Response(HTTP_OK, stream=stream_two)
            )

            async def _consume() -> None:
                async for _ in provider.chat_stream(request):
                    pass

            task = asyncio.create_task(_consume())
            await asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert stream_two.closed is True


class _FakeProcess:
    """Minimal Popen double simulating llama-server lifecycle and crashes."""

    def __init__(self, output_bytes: bytes = b"server ready\n") -> None:
        self.pid = 424242
        self.stdout = io.BytesIO(output_bytes)
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def kill(self) -> None:
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def crash(self, code: int = 1) -> None:
        self._returncode = code


async def test_llama_cpp_supervisor_lifecycle_pdeathsig_and_alerts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LlamaCppSupervisor attaches PR_SET_PDEATHSIG on Linux and alerts on crash/timeout."""
    exe_path = tmp_path / "llama-server"
    exe_path.write_bytes(b"ELF")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_bytes = b"GGUF_WEIGHTS"
    profile = _sample_profile(sha256=hashlib.sha256(model_bytes).hexdigest())
    (models_dir / profile.hf_filename).write_bytes(model_bytes)
    log_dir = tmp_path / "logs"

    captured_cmds: list[list[str]] = []
    captured_kwargs: list[dict[str, object]] = []
    fake_proc = _FakeProcess(b"llama_model_loader: loaded\n")

    def _fake_popen(cmd: list[str], **kw: object) -> _FakeProcess:
        captured_cmds.append(cmd)
        captured_kwargs.append(dict(kw))
        return fake_proc

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    ready = True

    async def _probe() -> BackendHealth:
        return BackendHealth(healthy=ready, status="ok" if ready else "loading")

    manager = configure_observability(
        LoggingConfig(log_dir=log_dir, enable_file_logs=False, alert_cooldown_seconds=0.0)
    )
    supervisor = LlamaCppSupervisor(
        executable_path=exe_path,
        models_dir=models_dir,
        log_dir=log_dir,
        startup_timeout_seconds=0.2,
        poll_interval_seconds=0.01,
        verify_sha256=True,
        health_probe=_probe,
    )
    try:
        assert isinstance(supervisor, InferenceSupervisorPort)
        health = await supervisor.ensure_running(profile)
        assert health.healthy is True
        assert supervisor.active_profile() == profile
        assert captured_cmds[0][0] == str(exe_path)
        assert "--n-cpu-moe" in captured_cmds[0]
        assert captured_kwargs[0]["start_new_session"] is True
        assert callable(captured_kwargs[0]["preexec_fn"])

        fake_proc.crash(code=139)
        crashed = await supervisor.health()
        assert crashed.healthy is False
        assert crashed.status == "crashed"
        assert "llama_model_loader: loaded" in (log_dir / "llama-server.log").read_text("utf-8")

        ready = False
        fake_proc = _FakeProcess(b"loading...\n")
        with pytest.raises(InferenceBackendError, match="timed out"):
            await supervisor.ensure_running(profile)

        categories = [alert.category for alert in manager.recent_alerts()]
        assert AlertCategory.SUBPROCESS_CRASH in categories
        assert AlertCategory.TIMEOUT in categories
    finally:
        await supervisor.stop()
        manager.close()


async def test_llama_cpp_supervisor_rejects_corrupt_model_sha256(tmp_path: Path) -> None:
    """LlamaCppSupervisor blocks spawn and emits SECURITY_ALERT when GGUF SHA-256 mismatches."""
    exe_path = tmp_path / "llama-server"
    exe_path.write_bytes(b"ELF")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    profile = _sample_profile(sha256="0" * 64)
    (models_dir / profile.hf_filename).write_bytes(b"TAMPERED")

    manager = configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=False))
    supervisor = LlamaCppSupervisor(
        executable_path=exe_path,
        models_dir=models_dir,
        log_dir=tmp_path / "logs",
        verify_sha256=True,
    )
    try:
        with pytest.raises(InferenceBackendError, match="SHA-256 integrity mismatch"):
            await supervisor.ensure_running(profile)
        assert any(a.category is AlertCategory.SECURITY_ALERT for a in manager.recent_alerts())
    finally:
        await supervisor.stop()
        manager.close()
