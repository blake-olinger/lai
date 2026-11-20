"""Red-team verification proofs for B6 (installer, checksums, hashed slots, and rollback)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory, AlertEvent, LoggingConfig
from lai.domain.updates import UpdateConfig, UpgradeStatus
from lai.observability import AlertManager, configure_observability
from lai.ports.observability import AlertSinkPort
from lai.ports.updates import DatabaseBackupPort, HealthProbePort, ServiceControllerPort
from lai.updates import LocalWheelSource, ReleaseSlotManager, UpgradeCoordinator

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WHL = "lai-0.2.0-py3-none-any.whl"
_WHL_BYTES = b"PK\x03\x04b6-proof-wheel"
_WHL_SHA = hashlib.sha256(_WHL_BYTES).hexdigest()
_REQ = "requirements-lock.txt"
_REQ_BYTES = (
    b"anyio==4.16.0 \\\n"
    b"    --hash=sha256:59bb03251907983d1603d27c3087ddb1e93299ea2163adcdba37899a4fb12ffb\n"
)
_REQ_SHA = hashlib.sha256(_REQ_BYTES).hexdigest()


class _MemorySink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


class _OrderedSlots(ReleaseSlotManager):
    def __init__(self, rel: Path, cur: Path, events: list[str]) -> None:
        super().__init__(rel, cur, command_runner=_ok_runner)
        self._events = events

    def activate_slot(self, slot_path: Path) -> Path | None:
        self._events.append(f"activate:{slot_path.name}")
        return super().activate_slot(slot_path)


class _OrderedService(ServiceControllerPort):
    def __init__(self, events: list[str], *, fail_stop: bool = False) -> None:
        self._events = events
        self._fail_stop = fail_stop

    async def stop(self, service_name: str) -> None:
        self._events.append(f"stop:{service_name}")
        if self._fail_stop:
            msg = "systemd stop failed"
            raise LaiError(msg, code="upgrade_service_stop_failed")

    async def restart(self, service_name: str) -> None:
        self._events.append(f"restart:{service_name}")


class _OrderedBackup(DatabaseBackupPort):
    def __init__(self, events: list[str], snap_path: Path) -> None:
        self._events = events
        self._snap_path = snap_path

    async def create_backup(self, db_path: Path, backups_dir: Path, label: str) -> Path:
        del db_path, backups_dir, label
        return self._snap_path

    async def restore_backup(self, backup_path: Path, db_path: Path) -> None:
        del backup_path, db_path
        self._events.append("restore_backup")


class _FailThenOkProbe(HealthProbePort):
    def __init__(self) -> None:
        self.calls = 0

    async def wait_healthy(
        self,
        url: str,
        *,
        expected_version: str | None,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> bool:
        del url, expected_version, timeout_seconds, poll_interval_seconds
        self.calls += 1
        return self.calls > 1


async def _ok_runner(args: Sequence[str]) -> tuple[int, str, str]:
    del args
    return 0, "", ""


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def test_b6_installer_systemd_unit_and_example_configs() -> None:
    """B6(a)+(d)+(f): Installer pins uv/Python 3.13/bootstrap; systemd unit & YAMLs hardened."""
    installer = (_REPO_ROOT / "deploy" / "systemd" / "install-ubuntu.sh").read_text(
        encoding="utf-8"
    )
    assert 'UV_VERSION="0.13.0"' in installer
    assert (
        'UV_SHA256_AMD64="1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154"'
        in installer
    )
    assert (
        'UV_SHA256_ARM64="3ccfb6af6e242433eb552f7d9676abd5412c6595c497e990d9c8cb7b5bd4d2c3"'
        in installer
    )
    assert 'PYTHON_VERSION="3.13"' in installer
    assert "umask 022" in installer
    assert "--no-bin" in installer
    assert "chmod -R a+rX /opt/lai/python" in installer
    assert "/opt/lai/python" in installer
    assert "/opt/lai/uv-cache" in installer
    assert "--require-hashes" in installer
    assert "--no-deps" in installer
    assert "mv -Tf" in installer
    release_yml = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert 'grep -q -- "--hash=sha256:" dist/requirements-lock.txt' in release_yml

    unit = (_REPO_ROOT / "deploy" / "systemd" / "lai-upgrade.service").read_text(encoding="utf-8")
    assert "Environment=UV_CACHE_DIR=/opt/lai/uv-cache" in unit
    assert "Environment=UV_PYTHON_INSTALL_DIR=/opt/lai/python" in unit
    assert "Environment=PATH=/usr/local/bin:" in unit
    assert "TimeoutStartSec=900s" in unit
    assert "ReadWritePaths=/opt/lai /var/lib/lai\n" in unit
    rw_line = next(line for line in unit.splitlines() if line.startswith("ReadWritePaths="))
    assert "/var/log/lai" not in rw_line

    assert UpdateConfig().health_timeout_seconds == 300.0
    for rel_path in (
        "config/config.example.yaml",
        "deploy/azure/config.azure-prod.example.yaml",
        "deploy/cloudflare/config.test.example.yaml",
    ):
        parsed = yaml.safe_load((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        assert float(parsed["updates"]["health_timeout_seconds"]) == 300.0, rel_path


async def test_b6_fail_closed_local_wheel_and_hashed_slot_install(tmp_path: Path) -> None:
    """B6(b)+(c): LocalWheelSource fails closed; slot install uses --require-hashes."""
    sink = _MemorySink()
    mgr = AlertManager(sinks=(sink,), cooldown_seconds=0.0)
    configure_observability(
        LoggingConfig(enable_file_logs=False, alert_cooldown_seconds=0.0, alert_on_errors=True),
        alert_manager=mgr,
    )
    wheel = tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    with pytest.raises(LaiError) as exc_info:
        await LocalWheelSource(wheel).get_latest()
    assert exc_info.value.code == "upgrade_checksum_missing"
    await mgr.flush()
    assert any(e.category is AlertCategory.SECURITY_ALERT for e in sink.events)

    cmds: list[list[str]] = []

    async def _runner(args: Sequence[str]) -> tuple[int, str, str]:
        cmds.append(list(args))
        return 0, "", ""

    slots = ReleaseSlotManager(tmp_path / "rel", tmp_path / "cur", command_runner=_runner)
    with pytest.raises(LaiError) as req_missing:
        await slots.install_slot("0.2.0", wheel)
    assert req_missing.value.code == "upgrade_requirements_missing"

    req = tmp_path / _REQ
    req.write_bytes(_REQ_BYTES)
    await slots.install_slot("0.2.0", wheel, requirements_path=req)
    assert cmds[0][:4] == ["uv", "venv", "--python", "3.13"]
    assert "--require-hashes" in cmds[1]
    assert "-r" in cmds[1]
    assert "--no-deps" in cmds[2]
    assert str(wheel) in cmds[2]


async def test_b6_rollback_order_stop_before_restore_and_abort_on_stop_failure(
    tmp_path: Path,
) -> None:
    """B6(e): _execute_rollback executes stop -> restore_backup -> activate_slot -> restart."""
    events: list[str] = []
    wheel = tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    (tmp_path / f"{_WHL}.sha256").write_text(f"{_WHL_SHA}\n", encoding="utf-8")
    req = tmp_path / _REQ
    req.write_bytes(_REQ_BYTES)
    (tmp_path / f"{_REQ}.sha256").write_text(f"{_REQ_SHA}\n", encoding="utf-8")
    db_path = tmp_path / "lai.db"
    db_path.write_bytes(b"db")
    snap_path = tmp_path / "backups" / "snap.db"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_bytes(b"snap")

    cfg = UpdateConfig(
        enabled=True,
        releases_dir=tmp_path / "releases",
        current_symlink=tmp_path / "current",
        backups_dir=tmp_path / "backups",
        auto_rollback=True,
    )
    slots = _OrderedSlots(cfg.releases_dir, cfg.current_symlink, events)
    v1 = await slots.install_slot("0.1.0", wheel, requirements_path=req)
    slots.activate_slot(v1)
    events.clear()

    coord = UpgradeCoordinator(
        config=cfg,
        source=LocalWheelSource(wheel),
        slots=slots,
        service=_OrderedService(events),
        health_probe=_FailThenOkProbe(),
        backup=_OrderedBackup(events, snap_path),
    )
    res = await coord.apply("0.1.0", db_path=db_path)
    assert res.status is UpgradeStatus.ROLLED_BACK
    assert events[2:] == [
        "stop:lai.service",
        "restore_backup",
        "activate:v0.1.0",
        "restart:lai.service",
    ]

    events.clear()
    coord_stop_fail = UpgradeCoordinator(
        config=cfg,
        source=LocalWheelSource(wheel),
        slots=slots,
        service=_OrderedService(events, fail_stop=True),
        health_probe=_FailThenOkProbe(),
        backup=_OrderedBackup(events, snap_path),
    )
    res_fail = await coord_stop_fail.apply("0.1.0", db_path=db_path)
    assert res_fail.status is UpgradeStatus.FAILED
    assert "restore_backup" not in events
