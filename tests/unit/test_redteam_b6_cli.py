"""Red-team CLI proofs for B6(d)+(f)+(g) (concurrent serve, no root log files, refresh-binaries)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import pytest
import uvicorn
from typer.testing import CliRunner

from lai.cli import models_cmd, serve_cmd, upgrade_cmd
from lai.cli._upgrade_helpers import build_refresh_binaries_command, refresh_slot_binaries
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import LoggingConfig

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_b6_concurrent_supervisor_startup_in_serve_async(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6(d): serve_async starts primary llama-server and ShieldGemma concurrently."""
    barrier = asyncio.Barrier(2)
    stopped: list[str] = []

    class _ConcurrentSupervisor:
        def __init__(self, label: str) -> None:
            self._label = label

        async def ensure_running(self, cfg: object) -> None:
            del cfg
            await asyncio.wait_for(barrier.wait(), timeout=2.0)

        async def stop(self) -> None:
            stopped.append(self._label)

    class _ImmediateServer:
        def __init__(self, config: object) -> None:
            del config

        async def serve(self) -> None:
            return None

    def _build_sups(_settings: LaiSettings) -> tuple[_ConcurrentSupervisor, _ConcurrentSupervisor]:
        return (_ConcurrentSupervisor("gpu"), _ConcurrentSupervisor("shieldgemma"))

    def _fake_app(_settings: LaiSettings, config_path: Path | None = None) -> object:
        del config_path
        return object()

    monkeypatch.setattr(serve_cmd, "build_supervisors", _build_sups)
    monkeypatch.setattr(serve_cmd, "create_app", _fake_app)
    monkeypatch.setattr(uvicorn, "Server", _ImmediateServer)

    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        guardrails=GuardrailConfig(enabled=True),
    )
    await serve_cmd.serve_async(settings, supervise=True)
    assert set(stopped) == {"gpu", "shieldgemma"}


async def test_b6_upgrade_no_root_log_files_and_post_install_with_binaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6(f)+(g): upgrade observability disables file logs; --with-binaries runs new slot CLI."""
    log_dir = tmp_path / "logs"
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=log_dir, enable_file_logs=True),
    )
    mgr = upgrade_cmd._init_upgrade_observability(settings)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    try:
        logging.getLogger("lai.updates").error("upgrade_probe_event")
    finally:
        await upgrade_cmd._shutdown_upgrade_observability(mgr)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert not log_dir.exists() or not any(log_dir.iterdir())

    slot = tmp_path / "releases" / "v0.2.0"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("bin_dir: bin\n", encoding="utf-8")
    seen_cmd: list[list[str]] = []

    async def _runner(args: Sequence[str]) -> tuple[int, str, str]:
        seen_cmd.append(list(args))
        return 0, f"llama_server: {tmp_path / 'bin' / 'llama-server'}", ""

    out = await refresh_slot_binaries(
        slot, config_path=cfg_path, bin_dir=tmp_path / "bin", command_runner=_runner
    )
    assert out == tmp_path / "bin" / "llama-server"
    assert seen_cmd[0][-4:] == ["models", "refresh-binaries", "--config", str(cfg_path)]
    cmd_str = " ".join(
        build_refresh_binaries_command(slot, config_path=cfg_path, bin_dir=tmp_path / "bin")
    )
    assert str(slot) in cmd_str

    monkeypatch.delenv("LAI_CONFIG_FILE", raising=False)
    env_cmd = build_refresh_binaries_command(slot, config_path=None, bin_dir=tmp_path / "bin")
    assert "--config" not in env_cmd
    monkeypatch.setenv("LAI_CONFIG_FILE", str(cfg_path))
    env_cmd_set = build_refresh_binaries_command(slot, config_path=None, bin_dir=tmp_path / "bin")
    assert env_cmd_set[-2:] == ["--config", str(cfg_path)]


def test_b6_models_refresh_binaries_cli_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6(g): `lai models refresh-binaries` CLI command executes download_llama_binaries."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("bin_dir: bin\n", encoding="utf-8")

    class _FakeDl:
        async def download_llama_binaries(self, **_kw: object) -> Path:
            return tmp_path / "bin" / "llama-server"

    def _build_dl(_settings: LaiSettings) -> _FakeDl:
        return _FakeDl()

    monkeypatch.setattr(models_cmd, "build_downloader", _build_dl)
    res = CliRunner().invoke(cli_app, ["models", "refresh-binaries", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "llama_server:" in res.output


def test_b6_cli_init_honors_existing_fhs_config_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6: `lai init --config /etc/lai/config.yaml` uses YAML paths instead of relative ./data."""
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    fhs_root = tmp_path / "fhs"
    data_dir = fhs_root / "var_lib" / "data"
    models_dir = fhs_root / "var_lib" / "models"
    bin_dir = fhs_root / "var_lib" / "bin"
    log_dir = fhs_root / "var_log"
    backups_dir = fhs_root / "var_lib" / "backups"

    cfg_path = fhs_root / "etc" / "config.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        f"data_dir: {data_dir.as_posix()!r}\n"
        f"models_dir: {models_dir.as_posix()!r}\n"
        f"bin_dir: {bin_dir.as_posix()!r}\n"
        "logging:\n"
        f"  log_dir: {log_dir.as_posix()!r}\n"
        "updates:\n"
        f"  backups_dir: {backups_dir.as_posix()!r}\n",
        encoding="utf-8",
    )

    res = CliRunner().invoke(cli_app, ["init", "--config", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "Config (kept existing):" in res.output
    assert (data_dir / "lai.db").is_file()
    assert (data_dir / "blobs").is_dir()
    assert (data_dir / "lancedb").is_dir()
    assert models_dir.is_dir()
    assert bin_dir.is_dir()
    assert log_dir.is_dir()
    assert backups_dir.is_dir()
    assert not (cwd_dir / "data").exists()

    bad_cfg = fhs_root / "etc" / "bad.yaml"
    bad_cfg.write_text("data_dir: [unterminated\n", encoding="utf-8")
    res_bad = CliRunner().invoke(cli_app, ["init", "--config", str(bad_cfg)])
    assert res_bad.exit_code == 1
    assert "Configuration error" in res_bad.output
    assert not (cwd_dir / "data").exists()
