"""Online WAL-safe SQLite backup and integrity-verified atomic restore helpers."""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger
from lai.ports.updates import DatabaseBackupPort

if TYPE_CHECKING:
    import structlog

__all__ = [
    "SqliteDatabaseBackupAdapter",
    "create_sqlite_backup",
    "restore_sqlite_backup",
]

_BUSY_TIMEOUT_SECONDS = 30.0
_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_POSIX_FILE_MODE = 0o600


def _logger() -> structlog.stdlib.BoundLogger:
    return get_component_logger("db.backup")


def _chmod_private(path: Path) -> None:
    """Restrict database file permissions to owner read/write on POSIX hosts."""
    if os.name == "posix":  # pragma: no branch
        path.chmod(_POSIX_FILE_MODE)


def _preserve_owner(path: Path, reference: Path) -> None:
    """Preserve uid/gid of reference on POSIX so root-run restores keep service user ownership."""
    if sys.platform == "win32" or not reference.exists():  # pragma: no cover - POSIX guard
        return
    st = reference.stat()
    try:
        os.chown(path, st.st_uid, st.st_gid)
    except PermissionError:
        return


def _allocate_temp_file(parent: Path, prefix_name: str) -> Path:
    """Create and close an empty temporary file inside parent for atomic replacement."""
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{prefix_name}.",
        suffix=".tmp",
        dir=parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _log_storage_error(event: str, *, code: str, message: str, exc: Exception | None) -> LaiError:
    """Emit a STORAGE_FAILURE alert log record and build a corresponding LaiError."""
    _logger().error(
        event,
        error=message,
        exception_type=type(exc).__name__ if exc is not None else None,
        alert_category=AlertCategory.STORAGE_FAILURE.value,
    )
    return LaiError(message, code=code, status_code=500)


def _close_quietly(conn: sqlite3.Connection | None) -> None:
    """Close an open sqlite3 connection if present."""
    if conn is not None:
        conn.close()


def _verify_integrity(conn: sqlite3.Connection, backup_db: Path) -> None:
    """Verify PRAGMA integrity_check returns exactly a single 'ok' row."""
    rows = conn.execute("PRAGMA integrity_check;").fetchall()
    if len(rows) != 1 or str(rows[0][0]).lower() != "ok":
        detail = "; ".join(str(row[0]) for row in rows) or "empty integrity_check result"
        msg = f"SQLite backup integrity check failed for {backup_db}: {detail}"
        raise _log_storage_error(
            "sqlite_backup_integrity_failed",
            code="db_restore_integrity_failed",
            message=msg,
            exc=None,
        )


def _remove_wal_sidecars(target_db: Path) -> None:
    """Delete stale SQLite WAL and SHM sidecar files after atomic restore."""
    for suffix in ("-wal", "-shm"):
        Path(f"{target_db}{suffix}").unlink(missing_ok=True)


def create_sqlite_backup(src_db: Path, dest_db: Path) -> Path:
    """Create an online WAL-safe SQLite snapshot atomically at dest_db."""
    if not src_db.is_file():
        msg = f"Source SQLite database does not exist: {src_db}"
        raise _log_storage_error(
            "sqlite_backup_source_missing",
            code="db_backup_failed",
            message=msg,
            exc=None,
        )

    tmp_path = _allocate_temp_file(dest_db.parent, dest_db.name)
    src_conn: sqlite3.Connection | None = None
    dst_conn: sqlite3.Connection | None = None
    try:
        src_conn = sqlite3.connect(str(src_db), timeout=_BUSY_TIMEOUT_SECONDS)
        dst_conn = sqlite3.connect(str(tmp_path), timeout=_BUSY_TIMEOUT_SECONDS)
        src_conn.backup(dst_conn)
        _close_quietly(dst_conn)
        dst_conn = None
        _close_quietly(src_conn)
        src_conn = None
        _chmod_private(tmp_path)
        tmp_path.replace(dest_db)
        _chmod_private(dest_db)
    except (sqlite3.Error, OSError) as exc:
        _close_quietly(dst_conn)
        _close_quietly(src_conn)
        tmp_path.unlink(missing_ok=True)
        msg = f"SQLite online backup from {src_db} to {dest_db} failed: {exc}"
        raise _log_storage_error(
            "sqlite_backup_failed",
            code="db_backup_failed",
            message=msg,
            exc=exc,
        ) from exc
    return dest_db


def restore_sqlite_backup(backup_db: Path, target_db: Path) -> None:
    """Verify backup_db integrity and atomically restore it over target_db."""
    if not backup_db.is_file():
        msg = f"SQLite backup file does not exist: {backup_db}"
        raise _log_storage_error(
            "sqlite_restore_source_missing",
            code="db_restore_failed",
            message=msg,
            exc=None,
        )

    tmp_path: Path | None = None
    src_conn: sqlite3.Connection | None = None
    dst_conn: sqlite3.Connection | None = None
    try:
        src_conn = sqlite3.connect(str(backup_db), timeout=_BUSY_TIMEOUT_SECONDS)
        _verify_integrity(src_conn, backup_db)
        tmp_path = _allocate_temp_file(target_db.parent, target_db.name)
        dst_conn = sqlite3.connect(str(tmp_path), timeout=_BUSY_TIMEOUT_SECONDS)
        src_conn.backup(dst_conn)
        _close_quietly(dst_conn)
        dst_conn = None
        _close_quietly(src_conn)
        src_conn = None
        _chmod_private(tmp_path)
        _preserve_owner(tmp_path, target_db)
        tmp_path.replace(target_db)
        _chmod_private(target_db)
        _remove_wal_sidecars(target_db)
    except LaiError:
        _close_quietly(dst_conn)
        _close_quietly(src_conn)
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    except (sqlite3.Error, OSError) as exc:
        _close_quietly(dst_conn)
        _close_quietly(src_conn)
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        code = (
            "db_restore_integrity_failed"
            if isinstance(exc, sqlite3.DatabaseError)
            else "db_restore_failed"
        )
        msg = f"SQLite restore from {backup_db} to {target_db} failed: {exc}"
        raise _log_storage_error(
            "sqlite_restore_failed",
            code=code,
            message=msg,
            exc=exc,
        ) from exc


def _sanitize_label(label: str) -> str:
    """Normalize a backup label into a filesystem-safe filename segment."""
    cleaned = _LABEL_SAFE_RE.sub("_", label.strip()).strip("._-")
    return cleaned or "backup"


class SqliteDatabaseBackupAdapter(DatabaseBackupPort):
    """Async DatabaseBackupPort adapter delegating SQLite snapshot/restore to worker threads."""

    async def create_backup(self, db_path: Path, backups_dir: Path, label: str) -> Path:
        """Create a timestamped WAL-safe snapshot of db_path inside backups_dir."""
        safe_label = _sanitize_label(label)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        dest_db = backups_dir / f"{db_path.stem}-{safe_label}-{timestamp}.db"
        return await asyncio.to_thread(create_sqlite_backup, db_path, dest_db)

    async def restore_backup(self, backup_path: Path, db_path: Path) -> None:
        """Verify backup_path integrity and atomically replace db_path."""
        await asyncio.to_thread(restore_sqlite_backup, backup_path, db_path)
