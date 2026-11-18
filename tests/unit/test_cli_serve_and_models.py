"""Unit tests for CLI models, serve supervision, __main__ entrypoint, and binary lookup."""

from __future__ import annotations

import importlib
import runpy
import sys
from typing import TYPE_CHECKING

import pytest
import uvicorn
import yaml
from fastapi import FastAPI
from typer.testing import CliRunner

from lai.cli import models_cmd, serve_cmd
from lai.cli.app import app as cli_app
from lai.config.profiles import GEMMA4_26B_A4B_PROFILE
from lai.config.settings import LaiSettings
from lai.domain.errors import InferenceBackendError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import BackendHealth, ModelProfile
from lai.guardrails import ShieldGemmaSupervisor
from lai.inference import LlamaCppSupervisor, ModelDownloader
from lai.ports.inference import InferenceSupervisorPort

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

EXPECTED_PORT = 8499


@pytest.fixture(autouse=True)
def _restore_sys_argv() -> Generator[None]:
    original = list(sys.argv)
    yield
    sys.argv[:] = original


class _RecordingDownloader:
    """Offline ModelDownloader collaborator recording all CLI-forwarded arguments."""

    def __init__(self, models_dir: Path, bin_dir: Path, *, fail_model: bool = False) -> None:
        self._models_dir = models_dir
        self._bin_dir = bin_dir
        self._fail_model = fail_model
        self.binary_calls: list[tuple[str | None, str | None, bool]] = []
        self.model_calls: list[tuple[str, bool, str | None]] = []

    async def download_llama_binaries(
        self,
        *,
        tag: str | None = None,
        cuda_version: str | None = None,
        force: bool = False,
    ) -> Path:
        self.binary_calls.append((tag, cuda_version, force))
        return self._bin_dir / "llama" / "llama-server.exe"

    async def download_model(
        self,
        profile_or_config: ModelProfile | GuardrailConfig,
        *,
        force: bool = False,
        token: str | None = None,
    ) -> Path:
        if self._fail_model:
            msg = "download failed"
            raise InferenceBackendError(msg)
        filename = str(profile_or_config.hf_filename)
        self.model_calls.append((filename, force, token))
        return self._models_dir / filename


class _RecordingInferenceSupervisor(InferenceSupervisorPort):
    """Typed InferenceSupervisorPort double recording started profiles and lifecycle order."""

    def __init__(self, events: list[str], *, fail_start: bool = False) -> None:
        self._events = events
        self._fail_start = fail_start
        self.started_profiles: list[ModelProfile] = []

    async def ensure_running(self, profile: ModelProfile) -> BackendHealth:
        self.started_profiles.append(profile)
        self._events.append("start:llama")
        if self._fail_start:
            msg = "llama startup error"
            raise InferenceBackendError(msg)
        return BackendHealth(healthy=True, status="ok", model_loaded=profile.name)

    async def stop(self) -> None:
        self._events.append("stop:llama")

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")

    def active_profile(self) -> ModelProfile | None:
        return self.started_profiles[-1] if self.started_profiles else None


class _RecordingGuardrailSupervisor(ShieldGemmaSupervisor):
    """Typed ShieldGemmaSupervisor subclass recording guardrail configs without spawning."""

    def __init__(self, tmp_path: Path, events: list[str]) -> None:
        super().__init__(
            executable_path=tmp_path / "llama-server.exe",
            model_path=tmp_path / "shieldgemma.gguf",
            log_dir=tmp_path / "logs",
        )
        self._events = events
        self.started_configs: list[GuardrailConfig] = []

    async def ensure_running(self, config: GuardrailConfig) -> BackendHealth:
        self.started_configs.append(config)
        self._events.append("start:shield")
        return BackendHealth(healthy=True, status="ok", model_loaded=config.ollama_model)

    async def stop(self) -> None:
        self._events.append("stop:shield")


def test_cli_models_setup_flags_list_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lai models setup forwards tag/cuda_version/force/token and handles failures."""
    cfg_path = tmp_path / "config.yaml"
    cfg_payload = {
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "bin_dir": str(tmp_path / "bin"),
    }
    cfg_path.write_text(yaml.safe_dump(cfg_payload), encoding="utf-8")

    real_dl = models_cmd.build_downloader(
        LaiSettings(data_dir=tmp_path / "data", models_dir=tmp_path / "m", bin_dir=tmp_path / "b")
    )
    assert isinstance(real_dl, ModelDownloader)

    captured: list[_RecordingDownloader] = []
    fail_next = False

    def _fake_build(settings: LaiSettings) -> _RecordingDownloader:
        stub = _RecordingDownloader(settings.models_dir, settings.bin_dir, fail_model=fail_next)
        captured.append(stub)
        return stub

    monkeypatch.setattr(models_cmd, "build_downloader", _fake_build)
    runner = CliRunner()

    res_default = runner.invoke(cli_app, ["models", "setup", "--config", str(cfg_path)])
    assert res_default.exit_code == 0, res_default.output
    assert captured[0].binary_calls == [(None, None, False)]
    assert captured[0].model_calls == [
        ("gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", False, None),
        ("shieldgemma-2b.Q4_K_M.gguf", False, None),
    ]
    assert "llama_server:" in res_default.output
    assert "primary_model:" in res_default.output
    assert "guardrail_model:" in res_default.output

    res_flags = runner.invoke(
        cli_app,
        [
            "models",
            "setup",
            "--config",
            str(cfg_path),
            "--tag",
            "b9897",
            "--cuda-version",
            "12.4",
            "--hf-token",
            "hf_tok_123",
            "--skip-guardrail",
            "--force",
        ],
    )
    assert res_flags.exit_code == 0, res_flags.output
    assert captured[1].binary_calls == [("b9897", "12.4", True)]
    assert captured[1].model_calls == [("gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", True, "hf_tok_123")]
    assert "guardrail_model:" not in res_flags.output

    res_skip_bin = runner.invoke(
        cli_app, ["models", "setup", "--config", str(cfg_path), "--skip-binaries"]
    )
    assert res_skip_bin.exit_code == 0, res_skip_bin.output
    assert captured[2].binary_calls == []
    assert len(captured[2].model_calls) == 2  # noqa: PLR2004

    fail_next = True
    res_fail = runner.invoke(cli_app, ["models", "setup", "--config", str(cfg_path)])
    assert res_fail.exit_code == 1
    assert "Model setup failed" in res_fail.output

    res_list = runner.invoke(cli_app, ["models", "list", "--config", str(cfg_path)])
    assert res_list.exit_code == 0
    assert "* gemma4-26b-a4b" in res_list.output


def test_cli_serve_supervisors_llamacpp_settings_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_supervisors(llamacpp), uvicorn.Config, LaiSettings mutations, and errors."""
    llama_dir = tmp_path / "bin" / "llama"
    llama_dir.mkdir(parents=True)
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    (llama_dir / exe_name).write_bytes(b"MZ")

    sup_on, guard_on = serve_cmd.build_supervisors(
        LaiSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            bin_dir=tmp_path / "bin",
            inference_backend="llamacpp",
            guardrails=GuardrailConfig(enabled=True),
        )
    )
    assert isinstance(sup_on, LlamaCppSupervisor)
    assert isinstance(guard_on, ShieldGemmaSupervisor)

    sup_off, guard_off = serve_cmd.build_supervisors(
        LaiSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            bin_dir=tmp_path / "bin",
            inference_backend="llamacpp",
            guardrails=GuardrailConfig(enabled=False),
        )
    )
    assert isinstance(sup_off, LlamaCppSupervisor)
    assert guard_off is None

    cfg_path = tmp_path / "config.yaml"
    cfg_payload = {
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "bin_dir": str(tmp_path / "bin"),
    }
    cfg_path.write_text(yaml.safe_dump(cfg_payload), encoding="utf-8")

    events: list[str] = []
    configs: list[uvicorn.Config] = []
    seen_settings: list[LaiSettings] = []
    fake_app = FastAPI()
    llama_rec = _RecordingInferenceSupervisor(events)
    shield_rec = _RecordingGuardrailSupervisor(tmp_path, events)

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            configs.append(config)

        async def serve(self) -> None:
            events.append("uvicorn:serve")

    def _fake_create_app(settings: LaiSettings) -> FastAPI:
        seen_settings.append(settings)
        return fake_app

    def _fake_sup_pair(
        _settings: LaiSettings,
    ) -> tuple[InferenceSupervisorPort, ShieldGemmaSupervisor | None]:
        return (llama_rec, shield_rec)

    monkeypatch.setattr(serve_cmd, "build_supervisors", _fake_sup_pair)
    monkeypatch.setattr(serve_cmd, "create_app", _fake_create_app)
    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)

    runner = CliRunner()
    res = runner.invoke(
        cli_app,
        [
            "serve",
            "--config",
            str(cfg_path),
            "--host",
            "127.0.0.2",
            "--port",
            str(EXPECTED_PORT),
            "--profile",
            "gemma4-26b-a4b",
            "--backend",
            "llamacpp",
        ],
    )
    assert res.exit_code == 0, res.output
    assert events == ["start:llama", "start:shield", "uvicorn:serve", "stop:shield", "stop:llama"]
    assert llama_rec.started_profiles == [GEMMA4_26B_A4B_PROFILE]
    assert seen_settings[0].host == "127.0.0.2"
    assert seen_settings[0].port == EXPECTED_PORT
    assert seen_settings[0].default_model_profile == "gemma4-26b-a4b"
    assert seen_settings[0].inference_backend == "llamacpp"
    assert shield_rec.started_configs == [seen_settings[0].guardrails]
    assert configs[0].app is fake_app
    assert configs[0].host == "127.0.0.2"
    assert configs[0].port == EXPECTED_PORT
    assert configs[0].log_config is None

    events.clear()
    failing = _RecordingInferenceSupervisor(events, fail_start=True)

    def _failing_sup_pair(
        _settings: LaiSettings,
    ) -> tuple[InferenceSupervisorPort, ShieldGemmaSupervisor | None]:
        return (failing, None)

    monkeypatch.setattr(serve_cmd, "build_supervisors", _failing_sup_pair)
    res_fail = runner.invoke(cli_app, ["serve", "--config", str(cfg_path)])
    assert res_fail.exit_code == 1
    assert "Server startup failed" in res_fail.output
    assert events == ["start:llama", "stop:llama"]


def test_cli_main_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Executing `python -m lai.cli` invokes Typer main() and exits cleanly on --help."""
    monkeypatch.setattr(sys, "argv", ["lai", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("lai.cli", run_name="__main__")
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Local AI platform CLI" in captured.out

    main_mod = importlib.import_module("lai.cli.__main__")
    assert callable(main_mod.main)
