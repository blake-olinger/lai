"""Unit and mutation-hardening tests for pluggable Ollama backend and guardrails."""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx
import uvicorn
import yaml
from typer.testing import CliRunner

from lai.api.runtime import build_app_services, close_app_services
from lai.cli import serve_cmd
from lai.cli.app import app as cli_app
from lai.config.profiles import GEMMA4_26B_A4B_PROFILE
from lai.config.settings import LaiSettings
from lai.domain import (
    AlertCategory,
    AlertEvent,
    GuardrailConfig,
    GuardrailVerdict,
    LoggingConfig,
    SafetyPrinciple,
)
from lai.domain.chat import MessageRole
from lai.domain.errors import InferenceBackendError
from lai.domain.inference import InferenceMessage, InferenceRequest
from lai.guardrails import OllamaShieldGemmaProvider, compute_yes_probability
from lai.inference import (
    OllamaSupervisor,
    OpenAICompatProvider,
    build_ollama_modelfile,
    build_ollama_serve_env,
    derive_hf_quant_tag,
    ollama_manager,
    setup_ollama_models,
)
from lai.observability import AlertManager, configure_observability
from lai.ports import AlertSinkPort, GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from starlette.types import ASGIApp

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
EXPECTED_CTX = 24576
EXPECTED_GUARD_CTX = 4096
EXPECTED_THRESH = 0.80
TWO_CALLS = 2


class _MemorySink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _lp(p_yes: float, p_no: float) -> list[dict[str, object]]:
    return [
        {
            "top_logprobs": [
                {"token": "Yes", "logprob": math.log(p_yes)},
                {"token": "No", "logprob": math.log(p_no)},
            ]
        }
    ]


def test_compute_yes_probability_edge_cases_and_exact_math() -> None:
    """Exercise compute_yes_probability boundary inputs and normalization for gremlins."""
    bad_top = [
        "bad-entry",
        {"token": 123, "logprob": -0.1},
        {"token": "Yes", "logprob": "bad"},
        {"token": "Yes", "logprob": True},
        {"token": "Maybe", "logprob": -0.2},
    ]
    bad_top_case: object = [{"top_logprobs": bad_top}]
    for bad in (None, [], ["x"], [{"top_logprobs": "bad"}], bad_top_case):
        assert compute_yes_probability(bad) is None

    payload = [
        {
            "top_logprobs": [
                {"token": " Yes ", "logprob": math.log(0.4)},
                {"token": "yes", "logprob": math.log(0.2)},
                {"token": "No", "logprob": math.log(0.3)},
                {"token": " NO ", "logprob": math.log(0.1)},
            ]
        }
    ]
    assert compute_yes_probability(payload) == pytest.approx(0.6)


async def test_ollama_guardrail_allow_marginal_threshold_and_block() -> None:
    """OllamaShieldGemmaProvider calibrates P(Yes) threshold and short-circuits blocks."""
    cfg = GuardrailConfig(
        backend="ollama",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="lai-shieldgemma-2b",
        violation_threshold=EXPECTED_THRESH,
        timeout_seconds=5.0,
    )
    principles = (
        SafetyPrinciple(name="No Hate Speech", description="No hate."),
        SafetyPrinciple(name="No Harassment", description="No harassment."),
    )
    provider = OllamaShieldGemmaProvider(cfg, principles=principles)
    assert isinstance(provider, GuardrailProviderPort)
    assert provider.name == "shieldgemma-2b-ollama"

    url = "http://127.0.0.1:11434/api/generate"
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(url).mock(
            side_effect=[
                httpx.Response(HTTP_OK, json={"response": "Yes", "logprobs": _lp(0.6, 0.4)}),
                httpx.Response(HTTP_OK, json={"response": "Yes", "logprobs": _lp(0.6, 0.4)}),
                httpx.Response(HTTP_OK, json={"response": "No"}),
                httpx.Response(HTTP_OK, json={"response": "Yes", "logprobs": _lp(0.99, 0.01)}),
            ]
        )
        allow_res = await provider.evaluate("How do I kill a hung process by PID?")
        assert allow_res.verdict is GuardrailVerdict.ALLOW
        block_res = await provider.evaluate("Targeted abusive harassment")
        assert block_res.verdict is GuardrailVerdict.BLOCK
        assert block_res.violated_policy == "No Harassment"
        assert route.call_count == TWO_CALLS * TWO_CALLS


async def test_ollama_guardrail_errors_timeout_alert_and_health(tmp_path: Path) -> None:
    """OllamaShieldGemmaProvider handles fallbacks, timeouts, errors, and /api/tags health."""
    sink = _MemorySink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    try:
        cfg = GuardrailConfig(backend="ollama", ollama_base_url="http://127.0.0.1:11434/v1")
        provider = OllamaShieldGemmaProvider(
            cfg,
            principles=(SafetyPrinciple(name="Safe", description="Stay safe"),),
        )
        gen_url = "http://127.0.0.1:11434/api/generate"
        tags_url = "http://127.0.0.1:11434/api/tags"

        with respx.mock(assert_all_called=True) as mock:
            mock.post(gen_url).mock(
                side_effect=[
                    httpx.ReadTimeout("timed out"),
                    httpx.ConnectError("down"),
                    httpx.Response(HTTP_OK, json={"response": 42}),
                    httpx.Response(HTTP_OK, json={"response": "Maybe"}),
                ]
            )
            for pattern in ("timed out", "request failed", "missing string", "Ambiguous"):
                with pytest.raises(InferenceBackendError, match=pattern):
                    await provider.evaluate("hello")

            mock.get(tags_url).mock(
                side_effect=[
                    httpx.Response(
                        HTTP_OK, json={"models": [{"name": "lai-shieldgemma-2b:latest"}]}
                    ),
                    httpx.Response(HTTP_OK, json={"models": [{"name": "other:latest"}]}),
                    httpx.Response(HTTP_UNAVAILABLE, text="busy"),
                    httpx.ReadTimeout("slow"),
                    httpx.ConnectError("refused"),
                ]
            )
            ok = await provider.health()
            assert ok.healthy is True
            assert ok.model_loaded == "lai-shieldgemma-2b"
            for expected_status in ("model_missing", "http_503", "timeout", "unreachable"):
                assert (await provider.health()).status == expected_status

        await manager.flush()
    finally:
        manager.close()

    assert any(e.category is AlertCategory.TIMEOUT for e in manager.recent_alerts())


async def test_ollama_manager_helpers_setup_and_attach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HF quant derivation, env/Modelfile builders, setup, and non-owning attach."""
    for fname, expected_tag in (
        ("gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", "UD-IQ4_XS"),
        ("shieldgemma-2b.Q4_K_M.gguf", "Q4_K_M"),
        ("custom-Q8_0.gguf", "Q8_0"),
    ):
        assert derive_hf_quant_tag(fname) == expected_tag

    env = build_ollama_serve_env("http://127.0.0.1:22434", base_env={"PATH": "x"})
    expected_env = {
        "OLLAMA_HOST": "127.0.0.1:22434",
        "OLLAMA_FLASH_ATTENTION": "0",
        "OLLAMA_KV_CACHE_TYPE": "f16",
        "OLLAMA_MAX_LOADED_MODELS": "2",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_GPU_OVERHEAD": "1073741824",
    }
    assert {k: env[k] for k in expected_env} == expected_env

    assert build_ollama_modelfile("hf.co/r:Q4", num_ctx=EXPECTED_CTX) == (
        f"FROM hf.co/r:Q4\nPARAMETER num_ctx {EXPECTED_CTX}\nPARAMETER use_mmap true\n"
    )
    assert build_ollama_modelfile("hf.co/g:Q4", num_ctx=EXPECTED_GUARD_CTX, num_gpu=0) == (
        f"FROM hf.co/g:Q4\nPARAMETER num_ctx {EXPECTED_GUARD_CTX}\n"
        "PARAMETER use_mmap true\nPARAMETER num_gpu 0\n"
    )

    recorded_cmds: list[list[str]] = []

    def _fake_cli(_exe: Path, args: list[str], *, env: Mapping[str, str] | None = None) -> None:
        del env
        recorded_cmds.append(args)

    monkeypatch.setattr(ollama_manager, "_run_ollama_cli", _fake_cli)
    fake_exe, guard_cfg = tmp_path / "ollama.exe", GuardrailConfig(backend="ollama", enabled=True)
    full = await setup_ollama_models(GEMMA4_26B_A4B_PROFILE, guard_cfg, ollama_exe=fake_exe)
    assert list(full) == ["primary_model", "guardrail_model"]
    assert [c[0] for c in recorded_cmds] == ["pull", "create", "pull", "create"]

    recorded_cmds.clear()
    no_guard = await setup_ollama_models(
        GEMMA4_26B_A4B_PROFILE, guard_cfg, ollama_exe=fake_exe, skip_guardrail=True
    )
    assert list(no_guard) == ["primary_model"]
    assert [c[0] for c in recorded_cmds] == ["pull", "create"]

    sup = OllamaSupervisor(base_url="http://127.0.0.1:11434", log_dir=tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(return_value=httpx.Response(HTTP_OK))
        health = await sup.ensure_running(GEMMA4_26B_A4B_PROFILE)
        assert health.healthy is True
        assert health.model_loaded == GEMMA4_26B_A4B_PROFILE.name
        assert sup.active_profile() == GEMMA4_26B_A4B_PROFILE
    await sup.stop()
    assert sup.active_profile() is None


def test_cli_ollama_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify init/models/serve --backend ollama CLI flows and build_supervisors."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    cfg_path = tmp_path / "config.yaml"
    data_str = str(tmp_path / "data")
    init_args = ["init", "--backend", "ollama", "--config", str(cfg_path), "--data-dir", data_str]
    res_init = runner.invoke(cli_app, [*init_args, "--force"])
    assert res_init.exit_code == 0, res_init.output
    parsed_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert parsed_cfg["inference_backend"] == "ollama"
    assert parsed_cfg["guardrails"]["backend"] == "ollama"

    def _noop_cli(_exe: Path, _args: list[str], *, env: Mapping[str, str] | None = None) -> None:
        del env

    monkeypatch.setattr(ollama_manager, "_run_ollama_cli", _noop_cli)
    monkeypatch.setattr(ollama_manager, "resolve_ollama_executable", lambda: tmp_path / "ollama")
    res_setup = runner.invoke(
        cli_app, ["models", "setup", "--backend", "ollama", "--config", str(cfg_path)]
    )
    assert res_setup.exit_code == 0, res_setup.output
    assert "primary_model:" in res_setup.output

    sup, guard_sup = serve_cmd.build_supervisors(
        LaiSettings(data_dir=tmp_path / "data", inference_backend="ollama")
    )
    assert isinstance(sup, OllamaSupervisor)
    assert guard_sup is None

    events: list[str] = []

    class _FakeServer:
        def __init__(self, config: object) -> None:
            del config

        async def serve(self) -> None:
            events.append("uvicorn:serve")

    async def _noop_asgi(*_args: object) -> None:
        return None

    def _fake_app(_settings: LaiSettings) -> ASGIApp:
        return cast("ASGIApp", _noop_asgi)

    monkeypatch.setattr(serve_cmd, "_load_create_app", lambda: _fake_app)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    res_serve = runner.invoke(
        cli_app, ["serve", "--backend", "ollama", "--no-supervise", "--config", str(cfg_path)]
    )
    assert res_serve.exit_code == 0, res_serve.output
    assert events == ["uvicorn:serve"]


async def test_runtime_ollama_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_app_services wires OpenAICompatProvider and Ollama guardrail probe."""
    monkeypatch.chdir(tmp_path)
    services = await build_app_services(
        LaiSettings(
            data_dir=tmp_path / "svc_data",
            inference_backend="ollama",
            guardrails=GuardrailConfig(backend="llamacpp", enabled=True),
        )
    )
    try:
        assert isinstance(services.inference_provider, OpenAICompatProvider)
        assert services.guardrail_health_probe is not None
        ok_body = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        with respx.mock(assert_all_called=True) as mock:
            rt = mock.post("http://127.0.0.1:11434/v1/chat/completions").mock(
                return_value=httpx.Response(HTTP_OK, json=ok_body)
            )
            msgs = (InferenceMessage(role=MessageRole.USER, content="hi"),)
            await services.inference_provider.chat_complete(
                InferenceRequest(model="gemma4-26b-a4b", messages=msgs, enable_thinking=False)
            )
            await services.inference_provider.chat_complete(
                InferenceRequest(model="gemma4-26b-a4b", messages=msgs, enable_thinking=True)
            )
            s0 = json.loads(cast("httpx.Request", rt.calls[0].request).content)
            s1 = json.loads(cast("httpx.Request", rt.calls[1].request).content)
            assert s0["reasoning_effort"] == "none"
            assert "reasoning_effort" not in s1
    finally:
        await close_app_services(services)
