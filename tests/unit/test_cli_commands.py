"""Unit tests for Typer CLI commands (init, keys, tunnel, and shared helpers)."""

from __future__ import annotations

import asyncio
import shutil
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from typer.testing import CliRunner

from lai.cli._common import open_api_key_repo, resolve_llama_executable, resolve_log_dir
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.base import utc_now
from lai.domain.observability import LoggingConfig
from lai.security import generate_api_key

if TYPE_CHECKING:
    import pytest


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


async def _seed_expired_key(db_path: Path) -> None:
    expired = generate_api_key(
        name="stale",
        scopes=(ApiScope.PROJECTS_READ,),
        expires_at=utc_now() - timedelta(days=1),
    )
    async with open_api_key_repo(db_path) as repo:
        await repo.create(expired.record)


def test_cli_keys_create_list_expired_revoke_and_error_cases(tmp_path: Path) -> None:
    """Verify lai keys commands including --admin, expired keys, revoke, and missing errors."""
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

    res_default = runner.invoke(
        cli_app, ["keys", "create", "--db-path", str(db_path), "--name", "default-user"]
    )
    assert res_default.exit_code == 0, res_default.output
    assert "projects:write" in res_default.output

    res_admin = runner.invoke(
        cli_app, ["keys", "create", "--db-path", str(db_path), "--name", "root", "--admin"]
    )
    assert res_admin.exit_code == 0, res_admin.output
    assert "admin" in res_admin.output

    asyncio.run(_seed_expired_key(db_path))

    res_list = runner.invoke(cli_app, ["keys", "list", "--db-path", str(db_path)])
    assert res_list.exit_code == 0
    lines = res_list.output.strip().splitlines()
    reader_line = next(row for row in lines if "\tci-reader\t" in row)
    key_id = reader_line.split("\t")[0]
    assert "\tactive\t" in reader_line
    assert any("\tstale\t" in row and "\texpired\t" in row for row in lines)

    res_revoke = runner.invoke(cli_app, ["keys", "revoke", key_id, "--db-path", str(db_path)])
    assert res_revoke.exit_code == 0

    res_list_after = runner.invoke(cli_app, ["keys", "list", "--db-path", str(db_path)])
    assert "\trevoked\t" in res_list_after.output

    res_missing = runner.invoke(
        cli_app, ["keys", "revoke", "key_missing", "--db-path", str(db_path)]
    )
    assert res_missing.exit_code == 1


def test_cli_tunnel_render_stdout_file_and_missing_config(tmp_path: Path) -> None:
    """Verify lai tunnel render-config stdout, file output, config origin, and bad config."""
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

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("host: 127.0.0.9\nport: 8501\n", encoding="utf-8")
    out_file = tmp_path / "out" / "cloudflared.yaml"
    creds = tmp_path / "creds.json"
    res_file = runner.invoke(
        cli_app,
        [
            "tunnel",
            "render-config",
            "--hostname",
            "ai.local",
            "--tunnel",
            "t-1",
            "--credentials-file",
            str(creds),
            "--config",
            str(cfg_path),
            "--output",
            str(out_file),
        ],
    )
    assert res_file.exit_code == 0, res_file.output
    written = yaml.safe_load(out_file.read_text(encoding="utf-8"))
    assert written["tunnel"] == "t-1"
    assert written["credentials-file"] == str(creds)
    assert written["ingress"][0]["service"] == "http://127.0.0.9:8501"

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


def test_resolve_log_dir_custom_and_data_dir_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_log_dir respects explicit log_dir and derives from custom data_dir."""
    monkeypatch.chdir(tmp_path)
    default_derived = resolve_log_dir(LaiSettings(data_dir=tmp_path / "custom_data"))
    assert default_derived == tmp_path / "custom_data" / "logs"

    explicit = resolve_log_dir(
        LaiSettings(
            data_dir=tmp_path / "custom_data",
            logging=LoggingConfig(log_dir=tmp_path / "explicit_logs"),
        )
    )
    assert explicit == tmp_path / "explicit_logs"


def test_resolve_llama_executable_all_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover direct binary, nested rglob binary, PATH lookup, and missing fallback."""
    import sys  # noqa: PLC0415

    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

    bin_direct = tmp_path / "b1"
    direct_exe = bin_direct / "llama" / binary_name
    direct_exe.parent.mkdir(parents=True)
    direct_exe.write_bytes(b"MZ")
    assert resolve_llama_executable(bin_direct) == direct_exe

    bin_nested = tmp_path / "b2"
    nested_exe = bin_nested / "llama" / "nested" / binary_name
    nested_exe.parent.mkdir(parents=True)
    nested_exe.write_bytes(b"MZ")
    assert resolve_llama_executable(bin_nested) == nested_exe

    bin_path = tmp_path / "b3"
    (bin_path / "llama").mkdir(parents=True)

    def _which_llama(name: str) -> str | None:
        return "/usr/bin/llama-server" if name else None

    monkeypatch.setattr(shutil, "which", _which_llama)
    assert resolve_llama_executable(bin_path) == Path("/usr/bin/llama-server")

    bin_missing = tmp_path / "b4"

    def _which_none(_name: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", _which_none)
    assert resolve_llama_executable(bin_missing) == bin_missing / "llama" / binary_name
