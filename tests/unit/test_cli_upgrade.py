"""Unit tests for the `lai upgrade` CLI subcommands and FastEmbed setup failures."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml
from typer.testing import CliRunner

from lai.cli import models_cmd, upgrade_cmd
from lai.cli.app import app as cli_app
from lai.config.settings import load_settings
from lai.db import SqliteDatabase, SqliteRuntimeConfigRepository
from lai.domain.errors import LaiError
from lai.domain.updates import ReleaseArtifact, UpgradeCheckResult, UpgradeResult, UpgradeStatus
from lai.updates import UpgradeCoordinator

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import pytest

_GOOD_SHA = "a" * 64


class _StubCoordinator:
    """Offline UpgradeCoordinator double recording invocations and returning scripted results."""

    def __init__(
        self,
        *,
        check_res: UpgradeCheckResult | None = None,
        apply_res: UpgradeResult | None = None,
        rollback_res: UpgradeResult | None = None,
        error: LaiError | None = None,
    ) -> None:
        self._check_res = check_res
        self._apply_res = apply_res
        self._rollback_res = rollback_res
        self._error = error
        self.apply_calls: list[dict[str, object]] = []
        self.rollback_calls: list[dict[str, object]] = []

    async def check(self, current_version: str) -> UpgradeCheckResult:
        if self._error is not None:
            raise self._error
        return self._check_res or UpgradeCheckResult(
            current_version=current_version,
            latest_version="0.2.0",
            update_available=True,
            artifact=ReleaseArtifact(
                version="0.2.0",
                tag="v0.2.0",
                wheel_filename="lai-0.2.0-py3-none-any.whl",
                sha256=_GOOD_SHA,
            ),
        )

    async def apply(self, current_version: str, **kwargs: object) -> UpgradeResult:
        self.apply_calls.append({"current_version": current_version, **kwargs})
        if self._error is not None:
            raise self._error
        hook = kwargs.get("pre_activate_hook")
        res = self._apply_res or UpgradeResult(
            status=UpgradeStatus.UPGRADED,
            from_version=current_version,
            to_version="0.2.0",
            slot_path=Path("/opt/lai/releases/v0.2.0"),
            backup_path=Path("/var/lib/lai/backups/pre-v0.2.0.db"),
        )
        if callable(hook) and res.slot_path is not None:
            typed_hook = cast("Callable[[Path], Awaitable[Path | None]]", hook)
            await typed_hook(res.slot_path)
        return res

    async def rollback(self, current_version: str, **kwargs: object) -> UpgradeResult:
        self.rollback_calls.append({"current_version": current_version, **kwargs})
        if self._error is not None:
            raise self._error
        return self._rollback_res or UpgradeResult(
            status=UpgradeStatus.ROLLED_BACK,
            from_version=current_version,
            to_version="0.1.0",
            slot_path=Path("/opt/lai/releases/v0.1.0"),
            rolled_back=True,
        )


def _write_upgrade_cfg(tmp_path: Path, *, enabled: bool = False) -> Path:
    cfg_path = tmp_path / "config.yaml"
    payload = {
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "bin_dir": str(tmp_path / "bin"),
        "logging": {"log_dir": str(tmp_path / "logs"), "enable_file_logs": False},
        "updates": {
            "enabled": enabled,
            "releases_dir": str(tmp_path / "releases"),
            "current_symlink": str(tmp_path / "current"),
            "backups_dir": str(tmp_path / "backups"),
        },
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return cfg_path


async def _seed_updates_override(db_path: Path, *, enabled: bool) -> None:
    db = SqliteDatabase(db_path)
    await db.initialize()
    try:
        repo = SqliteRuntimeConfigRepository(db)
        await repo.upsert_section("updates", {"enabled": enabled})
    finally:
        await db.close()


def test_cli_upgrade_check_apply_if_enabled_sqlite_merge_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lai upgrade check/apply/rollback, SQLite override merging, and --with-binaries."""
    cfg_path = _write_upgrade_cfg(tmp_path, enabled=False)
    runner = CliRunner()

    real_coord = upgrade_cmd.build_upgrade_coordinator(load_settings(cfg_path))
    assert isinstance(real_coord, UpgradeCoordinator)

    stub = _StubCoordinator()
    binary_calls: list[Path] = []

    async def _fake_refresh_slot_binaries(
        slot_path: Path,
        *,
        config_path: Path | None,
        bin_dir: Path,
    ) -> Path:
        del config_path, bin_dir
        binary_calls.append(slot_path)
        return tmp_path / "bin" / "llama" / "current" / "llama-server"

    def _build_stub(*_args: object, **_kwargs: object) -> _StubCoordinator:
        return stub

    monkeypatch.setattr(upgrade_cmd, "build_upgrade_coordinator", _build_stub)
    monkeypatch.setattr(upgrade_cmd, "refresh_slot_binaries", _fake_refresh_slot_binaries)

    r_chk = runner.invoke(cli_app, ["upgrade", "check", "-c", str(cfg_path)])
    assert r_chk.exit_code == 0, r_chk.output
    assert "Update available: yes" in r_chk.output

    r_skip = runner.invoke(
        cli_app, ["upgrade", "apply", "-c", str(cfg_path), "--if-enabled", "--with-binaries"]
    )
    assert r_skip.exit_code == 0, r_skip.output
    assert "skipped_disabled" in r_skip.output
    assert binary_calls == []
    assert stub.apply_calls == []

    asyncio.run(_seed_updates_override(tmp_path / "data" / "lai.db", enabled=True))
    r_apply = runner.invoke(
        cli_app,
        [
            "upgrade",
            "apply",
            "-c",
            str(cfg_path),
            "--if-enabled",
            "--with-binaries",
            "--version",
            "0.2.0",
        ],
    )
    assert r_apply.exit_code == 0, r_apply.output
    assert "upgraded" in r_apply.output
    assert "llama_server:" in r_apply.output
    assert binary_calls == [Path("/opt/lai/releases/v0.2.0")]
    assert stub.apply_calls[0]["target_version"] == "0.2.0"

    r_bad_opt = runner.invoke(
        cli_app, ["upgrade", "apply", "-c", str(cfg_path), "--sha256", _GOOD_SHA]
    )
    assert r_bad_opt.exit_code == 2
    r_bad_req = runner.invoke(
        cli_app, ["upgrade", "apply", "-c", str(cfg_path), "--requirements", str(cfg_path)]
    )
    assert r_bad_req.exit_code == 2

    backup = tmp_path / "backups" / "pre.db"
    r_rb = runner.invoke(
        cli_app, ["upgrade", "rollback", "-c", str(cfg_path), "--backup", str(backup)]
    )
    assert r_rb.exit_code == 0, r_rb.output
    assert "rolled_back" in r_rb.output
    assert stub.rollback_calls[0]["backup_path"] == backup


def test_cli_upgrade_error_and_fastembed_failure_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lai upgrade and FastEmbed warmup failures exit cleanly with code 1."""
    cfg_path = _write_upgrade_cfg(tmp_path, enabled=True)
    runner = CliRunner()

    fail_stub = _StubCoordinator(
        apply_res=UpgradeResult(
            status=UpgradeStatus.ROLLED_BACK,
            from_version="0.1.0",
            to_version="0.2.0",
            rolled_back=True,
            error="health timeout",
        ),
        rollback_res=UpgradeResult(
            status=UpgradeStatus.FAILED,
            from_version="0.1.0",
            error="no previous slot",
        ),
    )

    def _build_fail(*_args: object, **_kwargs: object) -> _StubCoordinator:
        return fail_stub

    monkeypatch.setattr(upgrade_cmd, "build_upgrade_coordinator", _build_fail)
    assert runner.invoke(cli_app, ["upgrade", "apply", "-c", str(cfg_path)]).exit_code == 1
    assert runner.invoke(cli_app, ["upgrade", "rollback", "-c", str(cfg_path)]).exit_code == 1

    err_stub = _StubCoordinator(error=LaiError("network down", code="upgrade_source_unreachable"))

    def _build_err(*_args: object, **_kwargs: object) -> _StubCoordinator:
        return err_stub

    monkeypatch.setattr(upgrade_cmd, "build_upgrade_coordinator", _build_err)
    assert runner.invoke(cli_app, ["upgrade", "check", "-c", str(cfg_path)]).exit_code == 1
    assert runner.invoke(cli_app, ["upgrade", "apply", "-c", str(cfg_path)]).exit_code == 1
    assert runner.invoke(cli_app, ["upgrade", "rollback", "-c", str(cfg_path)]).exit_code == 1

    class _FailingEmbedder:
        async def embed_query(self, query: str) -> list[float]:
            del query
            msg = "offline onnx"
            raise RuntimeError(msg)

    class _OkDl:
        async def download_model(self, cfg: object, **_kw: object) -> Path:
            del cfg
            return tmp_path / "models" / "m.gguf"

    def _build_ok_dl(_settings: object) -> _OkDl:
        return _OkDl()

    def _build_fail_emb(_settings: object) -> _FailingEmbedder:
        return _FailingEmbedder()

    monkeypatch.setattr(models_cmd, "build_downloader", _build_ok_dl)
    monkeypatch.setattr(models_cmd, "build_embedder", _build_fail_emb)
    res_emb_fail = runner.invoke(
        cli_app,
        ["models", "setup", "-c", str(cfg_path), "--skip-binaries", "--skip-guardrail"],
    )
    assert res_emb_fail.exit_code == 1
    assert "embeddings_setup_failed" in res_emb_fail.output
