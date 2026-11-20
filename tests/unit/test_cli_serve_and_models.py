"""Unit tests for CLI models, serve supervision, __main__ entrypoint, and binary lookup."""

from __future__ import annotations

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
EXPECTED_TWO_MODELS = 2


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
        return self._bin_dir / "llama" / "current" / "llama-server"

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


class _RecordingEmbedder:
    """Offline FastEmbed stub recording warmup queries."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2]


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
            executable_path=tmp_path / "llama-server",
            model_path=tmp_path / "shieldgemma.gguf",
            log_dir=tmp_path / "logs",
        )
        self._events = events
        self.started_configs: list[GuardrailConfig] = []

    async def ensure_running(self, config: GuardrailConfig) -> BackendHealth:
        self.started_configs.append(config)
        self._events.append("start:shield")
        return BackendHealth(healthy=True, status="ok", model_loaded=config.hf_filename)

    async def stop(self) -> None:
        self._events.append("stop:shield")


def _write_cli_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_payload = {
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "bin_dir": str(tmp_path / "bin"),
    }
    cfg_path.write_text(yaml.safe_dump(cfg_payload), encoding="utf-8")
    return cfg_path


def test_cli_models_build_downloader_and_list(tmp_path: Path) -> None:
    """Verify models_cmd.build_downloader and `lai models list` output."""
    cfg_path = _write_cli_config(tmp_path)
    real_dl = models_cmd.build_downloader(
        LaiSettings(data_dir=tmp_path / "data", models_dir=tmp_path / "m", bin_dir=tmp_path / "b")
    )
    assert isinstance(real_dl, ModelDownloader)

    res_list = CliRunner().invoke(cli_app, ["models", "list", "--config", str(cfg_path)])
    assert res_list.exit_code == 0
    assert "* gemma4-26b-a4b" in res_list.output


def test_cli_models_setup_default_flags_and_skip_binaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `lai models setup` forwards flags cleanly across default and skip modes."""
    cfg_path = _write_cli_config(tmp_path)
    captured: list[_RecordingDownloader] = []
    embedders: list[_RecordingEmbedder] = []

    def _fake_build(settings: LaiSettings) -> _RecordingDownloader:
        stub = _RecordingDownloader(settings.models_dir, settings.bin_dir)
        captured.append(stub)
        return stub

    def _fake_embedder(_settings: LaiSettings) -> _RecordingEmbedder:
        emb = _RecordingEmbedder()
        embedders.append(emb)
        return emb

    monkeypatch.setattr(models_cmd, "build_downloader", _fake_build)
    monkeypatch.setattr(models_cmd, "build_embedder", _fake_embedder)
    runner = CliRunner()

    res_default = runner.invoke(cli_app, ["models", "setup", "--config", str(cfg_path)])
    assert res_default.exit_code == 0, res_default.output
    assert captured[0].binary_calls == [(None, None, False)]
    assert captured[0].model_calls == [
        ("gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", False, None),
        ("shieldgemma-2b.Q4_K_M.gguf", False, None),
    ]
    assert embedders[0].queries == ["warmup"]
    for label in ("llama_server:", "primary_model:", "guardrail_model:", "embeddings_cache:"):
        assert label in res_default.output

    flags = [
        "--tag",
        "b12552",
        "--cuda-version",
        "12.8",
        "--hf-token",
        "hf_tok_123",
        "--skip-guardrail",
        "--skip-embeddings",
        "--force",
    ]
    res_flags = runner.invoke(cli_app, ["models", "setup", "--config", str(cfg_path), *flags])
    assert res_flags.exit_code == 0, res_flags.output
    assert captured[1].binary_calls == [("b12552", "12.8", True)]
    assert captured[1].model_calls == [("gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", True, "hf_tok_123")]
    assert "guardrail_model:" not in res_flags.output
    assert "embeddings_cache:" not in res_flags.output
    assert len(embedders) == 1

    res_skip_bin = runner.invoke(
        cli_app, ["models", "setup", "--config", str(cfg_path), "--skip-binaries"]
    )
    assert res_skip_bin.exit_code == 0, res_skip_bin.output
    assert captured[2].binary_calls == []
    assert len(captured[2].model_calls) == EXPECTED_TWO_MODELS
    assert len(embedders) == EXPECTED_TWO_MODELS


def test_cli_models_setup_failure_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `lai models setup` reports failure and exits 1 when downloads fail."""
    cfg_path = _write_cli_config(tmp_path)

    def _build_failing(settings: LaiSettings) -> _RecordingDownloader:
        return _RecordingDownloader(settings.models_dir, settings.bin_dir, fail_model=True)

    monkeypatch.setattr(models_cmd, "build_downloader", _build_failing)
    res_fail = CliRunner().invoke(cli_app, ["models", "setup", "--config", str(cfg_path)])
    assert res_fail.exit_code == 1
    assert "Model setup failed" in res_fail.output


def test_cli_serve_build_supervisors_llamacpp(tmp_path: Path) -> None:
    """Verify build_supervisors wires LlamaCppSupervisor and optional ShieldGemmaSupervisor."""
    llama_dir = tmp_path / "bin" / "llama" / "current"
    llama_dir.mkdir(parents=True)
    (llama_dir / "llama-server").write_bytes(b"ELF")

    sup_on, guard_on = serve_cmd.build_supervisors(
        LaiSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            bin_dir=tmp_path / "bin",
            guardrails=GuardrailConfig(enabled=True),
        )
    )
    sup_off, guard_off = serve_cmd.build_supervisors(
        LaiSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            bin_dir=tmp_path / "bin",
            guardrails=GuardrailConfig(enabled=False),
        )
    )
    assert (
        isinstance(sup_on, LlamaCppSupervisor),
        isinstance(guard_on, ShieldGemmaSupervisor),
    ) == (
        True,
        True,
    )
    assert (isinstance(sup_off, LlamaCppSupervisor), guard_off) == (True, None)


def test_cli_serve_runs_lifecycle_and_forwards_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `lai serve` starts/stops supervisors in order and forwards CLI overrides."""
    cfg_path = _write_cli_config(tmp_path)
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

    seen_config_paths: list[Path | None] = []

    def _fake_create_app(settings: LaiSettings, *, config_path: Path | None = None) -> FastAPI:
        seen_settings.append(settings)
        seen_config_paths.append(config_path)
        return fake_app

    def _fake_sup_pair(
        _settings: LaiSettings,
    ) -> tuple[InferenceSupervisorPort, ShieldGemmaSupervisor | None]:
        return (llama_rec, shield_rec)

    monkeypatch.setattr(serve_cmd, "build_supervisors", _fake_sup_pair)
    monkeypatch.setattr(serve_cmd, "create_app", _fake_create_app)
    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)

    serve_args = [
        "serve",
        "--config",
        str(cfg_path),
        "--host",
        "127.0.0.2",
        "--port",
        str(EXPECTED_PORT),
        "--profile",
        "gemma4-26b-a4b",
    ]
    res = CliRunner().invoke(cli_app, serve_args)
    assert res.exit_code == 0, res.output
    assert events == ["start:llama", "start:shield", "uvicorn:serve", "stop:shield", "stop:llama"]
    assert llama_rec.started_profiles == [GEMMA4_26B_A4B_PROFILE]
    assert (seen_settings[0].host, seen_settings[0].port) == ("127.0.0.2", EXPECTED_PORT)
    assert seen_settings[0].default_model_profile == "gemma4-26b-a4b"
    assert seen_config_paths == [cfg_path]
    assert shield_rec.started_configs == [seen_settings[0].guardrails]
    assert (configs[0].app, configs[0].host, configs[0].port, configs[0].log_config) == (
        fake_app,
        "127.0.0.2",
        EXPECTED_PORT,
        None,
    )


def test_cli_serve_startup_failure_stops_started_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `lai serve` stops started supervisors and exits 1 on startup failure."""
    cfg_path = _write_cli_config(tmp_path)
    events: list[str] = []
    failing = _RecordingInferenceSupervisor(events, fail_start=True)

    def _failing_sup_pair(
        _settings: LaiSettings,
    ) -> tuple[InferenceSupervisorPort, ShieldGemmaSupervisor | None]:
        return (failing, None)

    monkeypatch.setattr(serve_cmd, "build_supervisors", _failing_sup_pair)
    res_fail = CliRunner().invoke(cli_app, ["serve", "--config", str(cfg_path)])
    assert res_fail.exit_code == 1
    assert "Server startup failed" in res_fail.output
    assert events == ["start:llama", "stop:llama"]
