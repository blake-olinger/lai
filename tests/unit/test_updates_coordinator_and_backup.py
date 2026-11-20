"""Unit tests for WAL-safe SQLite backups/restores and UpgradeCoordinator orchestration."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lai.db.backup import SqliteDatabaseBackupAdapter, create_sqlite_backup, restore_sqlite_backup
from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory, AlertEvent, LoggingConfig
from lai.domain.updates import UpdateConfig, UpgradeStatus
from lai.observability import AlertManager, configure_observability
from lai.ports.observability import AlertSinkPort
from lai.ports.updates import (
    DatabaseBackupPort,
    HealthProbePort,
    ReleaseSourcePort,
    ServiceControllerPort,
    SlotInstallerPort,
)
from lai.updates import LocalWheelSource, ReleaseSlotManager, UpgradeCoordinator

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

_WHL = "lai-0.2.0-py3-none-any.whl"
_WHL_BYTES = b"PK\x03\x04valid-wheel-0.2.0"
_WHL_SHA = hashlib.sha256(_WHL_BYTES).hexdigest()


class _MemorySink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


class _StubService(ServiceControllerPort):
    def __init__(self, *, fail_calls: set[int] | None = None) -> None:
        self.restarts: list[str] = []
        self._fails = fail_calls or set()

    async def restart(self, service_name: str) -> None:
        self.restarts.append(service_name)
        if len(self.restarts) in self._fails:
            msg = f"restart failure #{len(self.restarts)}"
            raise LaiError(msg, code="upgrade_service_restart_failed", status_code=500)


class _StubProbe(HealthProbePort):
    def __init__(
        self, outcomes: list[bool | Exception], *, on_first: Callable[[], object] | None = None
    ) -> None:
        self._outcomes, self._on_first = list(outcomes), on_first
        self.calls: list[str | None] = []

    async def wait_healthy(
        self,
        url: str,
        *,
        expected_version: str | None,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> bool:
        del url, timeout_seconds, poll_interval_seconds
        self.calls.append(expected_version)
        if len(self.calls) == 1 and self._on_first is not None:
            self._on_first()
        nxt = self._outcomes.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FailingBackup(DatabaseBackupPort):
    def __init__(self, *, fail_create: bool = False, fail_restore: bool = False) -> None:
        self._fail_create, self._fail_restore = fail_create, fail_restore

    async def create_backup(self, db_path: Path, backups_dir: Path, label: str) -> Path:
        del backups_dir, label
        if self._fail_create:
            msg = "disk full"
            raise LaiError(msg, code="db_backup_failed", status_code=500)
        return db_path

    async def restore_backup(self, backup_path: Path, db_path: Path) -> None:
        del backup_path, db_path
        if self._fail_restore:
            msg = "restore io error"
            raise LaiError(msg, code="db_restore_failed", status_code=500)


class _FailingActivate(ReleaseSlotManager):
    def __init__(self, rel: Path, cur: Path, *, fail_calls: set[int]) -> None:
        super().__init__(rel, cur, command_runner=_ok_runner)
        self._fails, self._calls = fail_calls, 0

    def activate_slot(self, slot_path: Path) -> Path | None:
        self._calls += 1
        if self._calls in self._fails:
            msg = "symlink locked"
            raise LaiError(msg, code="upgrade_activation_failed", status_code=500)
        return super().activate_slot(slot_path)


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        h.flush()
        h.close()
    root.handlers.clear()


def _install_alerts() -> tuple[AlertManager, _MemorySink]:
    sink = _MemorySink()
    mgr = AlertManager(sinks=(sink,), cooldown_seconds=0.0)
    configure_observability(
        LoggingConfig(enable_file_logs=False, alert_cooldown_seconds=0.0, alert_on_errors=True),
        alert_manager=mgr,
    )
    return mgr, sink


def _wal_db(db_path: Path, rows: list[tuple[int, str]] | None = None) -> list[tuple[int, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        if rows is not None:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, val TEXT UNIQUE);"
            )
            conn.executemany("INSERT OR REPLACE INTO items (id, val) VALUES (?, ?);", rows)
            conn.commit()
        return [
            (int(r[0]), str(r[1])) for r in conn.execute("SELECT id, val FROM items ORDER BY id")
        ]
    finally:
        conn.close()


def _sidecars(db_path: Path, *, touch: bool) -> bool:
    wal, shm = Path(f"{db_path}-wal"), Path(f"{db_path}-shm")
    if touch:
        wal.write_bytes(b"wal")
        shm.write_bytes(b"shm")
    return wal.exists() or shm.exists()


async def _ok_runner(args: Sequence[str]) -> tuple[int, str, str]:
    del args
    return 0, "ok", ""


def _make_wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    (tmp_path / f"{_WHL}.sha256").write_text(f"{_WHL_SHA}\n", encoding="utf-8")
    return wheel


def _cfg(tmp_path: Path, *, enabled: bool = True, auto_rollback: bool = True) -> UpdateConfig:
    return UpdateConfig(
        enabled=enabled,
        releases_dir=tmp_path / "releases",
        current_symlink=tmp_path / "current",
        backups_dir=tmp_path / "backups",
        keep_releases=2,
        auto_rollback=auto_rollback,
    )


def _coord(  # noqa: PLR0913
    cfg: UpdateConfig,
    slots: SlotInstallerPort,
    src: ReleaseSourcePort,
    *,
    svc: ServiceControllerPort | None = None,
    probe: HealthProbePort | None = None,
    bak: DatabaseBackupPort | None = None,
) -> UpgradeCoordinator:
    return UpgradeCoordinator(
        config=cfg,
        source=src,
        slots=slots,
        service=svc or _StubService(),
        health_probe=probe or _StubProbe([]),
        backup=bak,
    )


async def test_sqlite_wal_online_backup_restore_and_guards(tmp_path: Path) -> None:
    """WAL online backup snapshots active frames, restores cleanly, and guards corrupt files."""
    db_path = tmp_path / "lai.db"
    _wal_db(db_path, [(1, "alpha"), (2, "beta")])
    writer = sqlite3.connect(str(db_path))
    try:
        writer.execute("PRAGMA journal_mode=WAL;")
        writer.execute("PRAGMA wal_autocheckpoint=0;")
        writer.execute("INSERT INTO items (id, val) VALUES (3, 'wal-frame');")
        writer.commit()
        snap = create_sqlite_backup(db_path, tmp_path / "backups" / "snap.db")
    finally:
        writer.close()
    assert snap.is_file()
    if os.name == "posix":
        assert (snap.stat().st_mode & 0o777) == 0o600
    expected_rows = [(1, "alpha"), (2, "beta"), (3, "wal-frame")]
    assert _wal_db(snap) == expected_rows
    _wal_db(db_path, [(1, "mutated"), (99, "bad")])
    assert _sidecars(db_path, touch=True)
    adapter = SqliteDatabaseBackupAdapter()
    fallback_snap = await adapter.create_backup(db_path, tmp_path / "backups", "!!!")
    labeled_snap = await adapter.create_backup(db_path, tmp_path / "backups", "pre v0.2.0")
    assert ("-backup-" in fallback_snap.name, "-pre_v0.2.0-" in labeled_snap.name) == (True, True)
    await adapter.restore_backup(snap, db_path)
    assert (_wal_db(db_path), _sidecars(db_path, touch=False)) == (expected_rows, False)
    mgr, sink = _install_alerts()
    corrupt = tmp_path / "backups" / "corrupt.db"
    corrupt.write_bytes(b"not-a-sqlite-db")
    corrupt_idx = tmp_path / "backups" / "corrupt_idx.db"
    corrupt_idx.write_bytes(snap.read_bytes().replace(b"alpha", b"XXXXX", 1))
    out_db = tmp_path / "backups" / "out.db"
    for fn, code in (
        (lambda: restore_sqlite_backup(corrupt, db_path), "db_restore_integrity_failed"),
        (lambda: restore_sqlite_backup(corrupt_idx, db_path), "db_restore_integrity_failed"),
        (lambda: create_sqlite_backup(corrupt, out_db), "db_backup_failed"),
        (lambda: restore_sqlite_backup(tmp_path / "nope.bak", db_path), "db_restore_failed"),
    ):
        with pytest.raises(LaiError) as err:
            fn()
        assert err.value.code == code
    assert not list((tmp_path / "backups").glob("*.tmp"))
    with pytest.raises(LaiError) as e_miss:
        await adapter.create_backup(tmp_path / "nope.db", tmp_path / "backups", "pre v0.2.0")
    assert (e_miss.value.code, _wal_db(db_path)) == ("db_backup_failed", expected_rows)
    await mgr.flush()
    assert any(e.category is AlertCategory.STORAGE_FAILURE for e in sink.events)


async def test_coordinator_check_guards_and_happy_path(tmp_path: Path) -> None:
    """Coordinator handles check, disabled skip, up-to-date, dry-run, happy-path, and pruning."""
    wheel, db_path = _make_wheel(tmp_path), tmp_path / "data" / "lai.db"
    db_path.parent.mkdir(parents=True)
    _wal_db(db_path, [(1, "prod")])

    dis_cfg = _cfg(tmp_path, enabled=False)
    slots = ReleaseSlotManager(
        dis_cfg.releases_dir, dis_cfg.current_symlink, command_runner=_ok_runner
    )
    src, st = LocalWheelSource(wheel), UpgradeStatus
    coord = _coord(dis_cfg, slots, src)
    chk = await coord.check("0.1.0")
    assert (chk.update_available, chk.latest_version) == (True, "0.2.0")
    assert not (await coord.check("0.2.0")).update_available
    got = (
        (await coord.apply("0.1.0", respect_enabled=True)).status,
        (await coord.apply("0.2.0")).status,
        (await coord.apply("0.1.0", target_version="0.2.0", dry_run=True)).status,
        (await coord.apply("0.2.0", dry_run=True, force=True)).status,
    )
    assert got == (st.SKIPPED_DISABLED, st.UP_TO_DATE, st.DRY_RUN, st.DRY_RUN)
    old_slot = await slots.install_slot("0.0.1", wheel)
    v1_slot = await slots.install_slot("0.1.0", wheel)
    slots.activate_slot(v1_slot)
    svc, probe = _StubService(), _StubProbe([True])
    res = await _coord(
        _cfg(tmp_path), slots, src, svc=svc, probe=probe, bak=SqliteDatabaseBackupAdapter()
    ).apply("0.1.0", target_version="0.2.0", db_path=db_path, respect_enabled=True)
    assert (res.status, svc.restarts, probe.calls) == (st.UPGRADED, ["lai.service"], ["0.2.0"])
    assert res.backup_path is not None
    assert res.backup_path.is_file()
    assert slots.current_slot() == (dis_cfg.releases_dir / "v0.2.0").resolve()
    assert (old_slot.exists(), v1_slot.is_dir()) == (False, True)


async def test_coordinator_failures_auto_and_manual_rollback(tmp_path: Path) -> None:
    """Coordinator aborts on pre-activation errors and handles auto and manual rollback."""
    mgr, sink = _install_alerts()
    wheel, cfg, st = _make_wheel(tmp_path), _cfg(tmp_path), UpgradeStatus
    src = LocalWheelSource(wheel)
    slots = ReleaseSlotManager(cfg.releases_dir, cfg.current_symlink, command_runner=_ok_runner)
    v1 = await slots.install_slot("0.1.0", wheel)
    slots.activate_slot(v1)
    db_path = tmp_path / "lai.db"
    _wal_db(db_path, [(1, "before")])

    async def fail_uv(args: Sequence[str]) -> tuple[int, str, str]:
        del args
        return 1, "", "install err"

    bad_slots = ReleaseSlotManager(cfg.releases_dir, cfg.current_symlink, command_runner=fail_uv)
    act_fail1 = _FailingActivate(cfg.releases_dir, cfg.current_symlink, fail_calls={1})
    for failing in (
        _coord(cfg, slots, LocalWheelSource(wheel, expected_sha256="f" * 64)),
        _coord(cfg, slots, src, bak=_FailingBackup(fail_create=True)),
        _coord(cfg, bad_slots, src),
        _coord(cfg, act_fail1, src),
    ):
        fail_res = await failing.apply("0.1.0", db_path=db_path)
        assert (fail_res.status, bool(fail_res.error)) == (st.FAILED, True)
    probe = _StubProbe([False, True], on_first=lambda: _wal_db(db_path, [(1, "broken-v2")]))
    rb = await _coord(cfg, slots, src, probe=probe, bak=SqliteDatabaseBackupAdapter()).apply(
        "0.1.0", db_path=db_path
    )
    assert (rb.status, rb.rolled_back, bool(rb.error)) == (st.ROLLED_BACK, True, True)
    assert (slots.current_slot(), _wal_db(db_path)) == (v1, [(1, "before")])
    r_rst = await _coord(
        cfg, slots, src, svc=_StubService(fail_calls={1}), probe=_StubProbe([True])
    ).apply("0.1.0")
    no_rb = await _coord(
        _cfg(tmp_path, auto_rollback=False), slots, src, probe=_StubProbe([OSError("sock")])
    ).apply("0.1.0")
    assert (r_rst.status, r_rst.rolled_back) == (st.ROLLED_BACK, True)
    assert (no_rb.status, no_rb.rolled_back) == (st.FAILED, False)
    slots.activate_slot(v1)
    act_fail2 = _FailingActivate(cfg.releases_dir, cfg.current_symlink, fail_calls={2})
    for rb_fail in (
        _coord(cfg, slots, src, probe=_StubProbe([False]), bak=_FailingBackup(fail_restore=True)),
        _coord(cfg, act_fail2, src, probe=_StubProbe([False])),
    ):
        assert (await rb_fail.apply("0.1.0", db_path=db_path)).status is st.FAILED
        slots.activate_slot(v1)
    v2 = (cfg.releases_dir / "v0.2.0").resolve()
    slots.activate_slot(v2)
    ok_rb = await _coord(cfg, slots, src, probe=_StubProbe([True])).rollback("0.2.0")
    slots.activate_slot(v2)
    bad_rb = await _coord(cfg, slots, src, probe=_StubProbe([False])).rollback("0.2.0")
    empty = ReleaseSlotManager(tmp_path / "e_rel", tmp_path / "e_cur", command_runner=_ok_runner)
    no_prev = await _coord(cfg, empty, src).rollback("0.1.0")
    assert (ok_rb.status, ok_rb.rolled_back, ok_rb.to_version) == (st.ROLLED_BACK, True, "0.1.0")
    assert (bad_rb.status, no_prev.status) == (st.FAILED, st.FAILED)
    await mgr.flush()
    ac = AlertCategory
    expected = {ac.SECURITY_ALERT, ac.STORAGE_FAILURE, ac.TIMEOUT, ac.APPLICATION_ERROR}
    assert expected.issubset({e.category for e in sink.events})
