"""Async SQLite connection lifecycle, WAL configuration, and write transaction manager."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from lai.db.schema import apply_schema
from lai.domain.errors import StorageFailureError
from lai.domain.observability import AlertCategory
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence


_POSIX_DIR_MODE = 0o700
_POSIX_FILE_MODE = 0o600


def _chmod_posix(path: Path, mode: int) -> None:
    """Apply POSIX file or directory permissions on a best-effort basis."""
    if os.name == "posix":
        with suppress(OSError):
            path.chmod(mode)


async def _safe_rollback(conn: aiosqlite.Connection) -> None:
    """Attempt SQLite ROLLBACK cleanly after a failed transaction."""
    try:
        await conn.execute("ROLLBACK;")
    except sqlite3.Error:
        return


def _log_and_raise_storage_error(operation: str, exc: Exception) -> StorageFailureError:
    """Log structured storage failure alert and construct StorageFailureError."""
    get_component_logger("db").error(
        "sqlite_operation_failed",
        operation=operation,
        error=str(exc),
        exception_type=type(exc).__name__,
        alert_category=AlertCategory.STORAGE_FAILURE.value,
    )
    msg = f"SQLite {operation} failed: {exc}"
    return StorageFailureError(msg)


class SqliteDatabase:
    """Manage a single async SQLite connection with WAL mode and serialized writes."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._connect_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    @property
    def db_path(self) -> Path | str:
        """Return the configured SQLite path or URI."""
        return self._db_path

    async def __aenter__(self) -> SqliteDatabase:
        """Open connection and apply schema when entering an async context."""
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the SQLite connection cleanly on context exit."""
        await self.close()

    async def connect(self) -> aiosqlite.Connection:
        """Open (or return) the configured aiosqlite connection with WAL & FK pragmas."""
        if self._conn is not None:
            return self._conn

        async with self._connect_lock:
            if self._conn is not None:
                return self._conn
            raw_path = str(self._db_path)
            is_local_file = raw_path != ":memory:" and not raw_path.startswith("file:")
            db_file = Path(raw_path) if is_local_file else None
            if db_file is not None:
                parent_existed = db_file.parent.exists()
                db_file.parent.mkdir(parents=True, exist_ok=True)
                if not parent_existed:
                    _chmod_posix(db_file.parent, _POSIX_DIR_MODE)
            try:
                conn = await aiosqlite.connect(self._db_path, isolation_level=None)
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA foreign_keys=ON;")
                await conn.execute("PRAGMA busy_timeout=5000;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
            except (sqlite3.Error, OSError) as exc:
                raise _log_and_raise_storage_error("connect", exc) from exc
            if db_file is not None and db_file.exists():
                _chmod_posix(db_file, _POSIX_FILE_MODE)
            self._conn = conn
            return conn

    async def initialize(self) -> None:
        """Ensure the database connection is open and schema tables/indexes exist."""
        async with self.write_transaction() as conn:
            await apply_schema(conn)

    async def close(self) -> None:
        """Close the active SQLite connection if open."""
        async with self._connect_lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

    @asynccontextmanager
    async def write_transaction(self) -> AsyncGenerator[aiosqlite.Connection]:
        """Acquire the serialized write lock and execute an atomic IMMEDIATE transaction."""
        conn = await self.connect()
        async with self._write_lock:
            try:
                await conn.execute("BEGIN IMMEDIATE;")
                yield conn
                await conn.execute("COMMIT;")
            except (sqlite3.Error, OSError) as exc:
                await _safe_rollback(conn)
                raise _log_and_raise_storage_error("write_transaction", exc) from exc
            except BaseException:
                await _safe_rollback(conn)
                raise

    async def execute_write(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a single write SQL statement inside a serialized transaction."""
        async with self.write_transaction() as conn:
            cursor = await conn.execute(sql, params)
            return cursor.rowcount

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> aiosqlite.Row | None:
        """Execute a read query and return at most one Row."""
        conn = await self.connect()
        try:
            async with conn.execute(sql, params) as cursor:
                return await cursor.fetchone()
        except (sqlite3.Error, OSError) as exc:
            raise _log_and_raise_storage_error("fetch_one", exc) from exc

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[aiosqlite.Row]:
        """Execute a read query and return all matching Rows."""
        conn = await self.connect()
        try:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return list(rows)
        except (sqlite3.Error, OSError) as exc:
            raise _log_and_raise_storage_error("fetch_all", exc) from exc
