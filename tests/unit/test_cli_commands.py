"""Unit tests for all modular Typer CLI commands (init, keys, models, serve, tunnel)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import uvicorn
import yaml
from typer.testing import CliRunner

from lai.cli import models_cmd, serve_cmd
from lai.cli.app import app as cli_app
from lai.domain.inference import BackendHealth

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from starlette.types import ASGIApp

    from lai.config.settings import LaiSettings
    from lai.domain.guardrails import GuardrailConfig
    from lai.domain.inference import ModelProfile
    from lai.guardrails import ShieldGemmaSupervisor
    from lai.inference import LlamaCppSupervisor


class _StubDownloader:
    """Offline ModelDownloader stub recording requested artifacts."""

    def __init__(self, models_dir: Path, bin_dir: Path) -> None:
        self._models_dir = models_dir
        self._bin_dir = bin_dir
        self.calls: list[tuple[str, str]] = []

    async def download_llama_binaries(
        self,
        *,
        tag: str | None = None,
        cuda_version: str | None = None,
        force: bool = False,
    ) -> Path:
        del tag, cuda_version, force
        self.calls.append(("binaries", "llama-server.exe"))
        return self._bin_dir / "llama" / "llama-server.exe"

    async def download_model(
        self,
        profile_or_config: ModelProfile | GuardrailConfig,
        *,
        force: bool = False,
        token: str | None = None,
    ) -> Path:
        del force, token
        filename = str(profile_or_config.hf_filename)
        self.calls.append(("model", filename))
        return self._models_dir / filename


def test_cli_init_idempotent_and_force_overwrite(tmp_path: Path) -> None:
    """Verify lai init creates directories, DB, YAML config, and respects --force."""
    runner = CliRunner()
    cfg = tmp_path / "config.yaml"
    args = [
        "init",
        "--config",
        str(cfg),
        "--data-dir",
        str(tmp_path / "data"),
        "--models-dir",
        str(tmp_path / "models"),
        "--bin-dir",
        str(tmp_path / "bin"),
    ]
    res1 = runner.invoke(cli_app, args)
    assert res1.exit_code == 0, res1.output
    assert "Config (wrote):" in res1.output

    cfg.write_text("host: 127.0.0.2\n", encoding="utf-8")
    res2 = runner.invoke(cli_app, args)
    assert res2.exit_code == 0
    assert "Config (kept existing):" in res2.output
    assert "127.0.0.2" in cfg.read_text(encoding="utf-8")

    res3 = runner.invoke(cli_app, [*args, "--force"])
    assert res3.exit_code == 0
    assert "Config (wrote):" in res3.output
    assert "127.0.0.1" in cfg.read_text(encoding="utf-8")


def test_cli_keys_create_list_revoke_and_error_cases(tmp_path: Path) -> None:
    """Verify lai keys commands with explicit --db-path, --scope, and missing key errors."""
    runner = CliRunner()
    db_path = tmp_path / "keys.db"

    res_empty = runner.invoke(cli_app, ["keys", "list", "--db-path", str(db_path)])
    assert res_empty.exit_code == 0
    assert "No API keys found." in res_empty.output

    res_create = runner.invoke(
        cli_app,
        [
            "keys",
            "create",
            "--db-path",
            str(db_path),
            "--name",
            "ci-reader",
            "--scope",
            "projects:read",
            "--scope",
            "chat:read",
            "--expires-days",
            "7",
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    assert "projects:read,chat:read" in res_create.output
    assert "lai_sk_" in res_create.output

    res_list = runner.invoke(cli_app, ["keys", "list", "--db-path", str(db_path)])
    assert res_list.exit_code == 0
    line = res_list.output.strip().splitlines()[0]
    key_id = line.split("\t")[0]
    assert "\tactive\t" in line

    res_revoke = runner.invoke(cli_app, ["keys", "revoke", key_id, "--db-path", str(db_path)])
    assert res_revoke.exit_code == 0

    res_list_after = runner.invoke(cli_app, ["keys", "list", "--db-path", str(db_path)])
    assert "\trevoked\t" in res_list_after.output

    res_missing = runner.invoke(
        cli_app, ["keys", "revoke", "key_missing", "--db-path", str(db_path)]
    )
    assert res_missing.exit_code == 1


def test_cli_models_setup_and_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lai models setup downloads llama-server, Gemma 4, and ShieldGemma-2B."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "models_dir": str(tmp_path / "models"),
                "bin_dir": str(tmp_path / "bin"),
            }
        ),
        encoding="utf-8",
    )

    captured: list[_StubDownloader] = []

    def _fake_build(settings: LaiSettings) -> _StubDownloader:
        stub = _StubDownloader(settings.models_dir, settings.bin_dir)
        captured.append(stub)
        return stub

    monkeypatch.setattr(models_cmd, "build_downloader", _fake_build)
    runner = CliRunner()

    res_setup = runner.invoke(cli_app, ["models", "setup", "--config", str(cfg_path)])
    assert res_setup.exit_code == 0, res_setup.output
    assert len(captured) == 1
    assert captured[0].calls == [
        ("binaries", "llama-server.exe"),
        ("model", "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"),
        ("model", "shieldgemma-2b.Q4_K_M.gguf"),
    ]
    assert "llama_server:" in res_setup.output
    assert "primary_model:" in res_setup.output
    assert "guardrail_model:" in res_setup.output

    res_list = runner.invoke(cli_app, ["models", "list", "--config", str(cfg_path)])
    assert res_list.exit_code == 0
    assert "gemma4-26b-a4b" in res_list.output


def test_cli_serve_supervises_backends_and_uvicorn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lai serve starts and stops Gemma 4 + ShieldGemma supervisors around Uvicorn."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "models_dir": str(tmp_path / "models"),
                "bin_dir": str(tmp_path / "bin"),
            }
        ),
        encoding="utf-8",
    )

    events: list[str] = []

    class _FakeSupervisor:
        def __init__(self, label: str) -> None:
            self._label = label

        async def ensure_running(self, arg: object) -> BackendHealth:
            del arg
            events.append(f"start:{self._label}")
            return BackendHealth(healthy=True, status="ok")

        async def stop(self) -> None:
            events.append(f"stop:{self._label}")

    class _FakeServer:
        def __init__(self, config: object) -> None:
            del config

        async def serve(self) -> None:
            events.append("uvicorn:serve")

    def _fake_supervisors(
        _settings: LaiSettings,
    ) -> tuple[LlamaCppSupervisor, ShieldGemmaSupervisor | None]:
        return (
            cast("LlamaCppSupervisor", _FakeSupervisor("llama")),
            cast("ShieldGemmaSupervisor", _FakeSupervisor("shield")),
        )

    async def _noop_asgi(scope: object, receive: object, send: object) -> None:
        del scope, receive, send

    def _fake_create_app(settings: LaiSettings) -> ASGIApp:
        del settings
        return cast("ASGIApp", _noop_asgi)

    monkeypatch.setattr(serve_cmd, "build_supervisors", _fake_supervisors)
    monkeypatch.setattr(serve_cmd, "_load_create_app", lambda: _fake_create_app)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)

    runner = CliRunner()
    res = runner.invoke(
        cli_app,
        ["serve", "--config", str(cfg_path), "--host", "127.0.0.1", "--port", "8499"],
    )
    assert res.exit_code == 0, res.output
    assert events == [
        "start:llama",
        "start:shield",
        "uvicorn:serve",
        "stop:shield",
        "stop:llama",
    ]


def test_cli_tunnel_render_stdout_and_missing_config(tmp_path: Path) -> None:
    """Verify lai tunnel render-config stdout output and invalid config exit code."""
    runner = CliRunner()
    res = runner.invoke(
        cli_app,
        [
            "tunnel",
            "render-config",
            "--hostname",
            "chat.local",
            "--origin-url",
            "http://127.0.0.1:9000",
        ],
    )
    assert res.exit_code == 0, res.output
    parsed = yaml.safe_load(res.output)
    assert parsed["ingress"][0]["hostname"] == "chat.local"
    assert parsed["ingress"][0]["service"] == "http://127.0.0.1:9000"

    res_bad = runner.invoke(
        cli_app,
        [
            "tunnel",
            "render-config",
            "--hostname",
            "chat.local",
            "--config",
            str(tmp_path / "nope.yaml"),
        ],
    )
    assert res_bad.exit_code == 1
