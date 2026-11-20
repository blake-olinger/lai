"""Unit tests for WAL-safe online SQLite backup, integrity verification, and atomic restore."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lai.db.backup import SqliteDatabaseBackupAdapter, create_sqlite_backup, restore_sqlite_backup
from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory, AlertEvent, LoggingConfig
from lai.observability import AlertManager, configure_observability
from lai.ports.observability import AlertSinkPort

if TYPE_CHECKING:
    from collections.abc import Generator


class _MemorySink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


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

    sink = _MemorySink()
    mgr = AlertManager(sinks=(sink,), cooldown_seconds=0.0)
    configure_observability(
        LoggingConfig(enable_file_logs=False, alert_cooldown_seconds=0.0, alert_on_errors=True),
        alert_manager=mgr,
    )
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
