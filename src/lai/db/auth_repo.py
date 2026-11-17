"""SQLite repository for hashed API key records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from lai.db._codecs import format_dt, format_opt_dt, parse_dt, parse_opt_dt
from lai.domain.auth import ApiKeyRecord, ApiScope
from lai.ports.repositories import ApiKeyRepositoryPort

if TYPE_CHECKING:
    from datetime import datetime

    import aiosqlite

    from lai.db.connection import SqliteDatabase

_SELECT_COLUMNS = (
    "id, name, key_prefix, key_hash, scopes_json, created_at, last_used_at, expires_at, revoked_at"
)


def _row_to_api_key(row: aiosqlite.Row) -> ApiKeyRecord:
    """Map an api_keys SQLite row to an ApiKeyRecord domain model."""
    raw_scopes = cast("list[str]", json.loads(str(row["scopes_json"])))
    return ApiKeyRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        key_prefix=str(row["key_prefix"]),
        key_hash=str(row["key_hash"]),
        scopes=tuple(ApiScope(item) for item in raw_scopes),
        created_at=parse_dt(str(row["created_at"])),
        last_used_at=parse_opt_dt(cast("str | None", row["last_used_at"])),
        expires_at=parse_opt_dt(cast("str | None", row["expires_at"])),
        revoked_at=parse_opt_dt(cast("str | None", row["revoked_at"])),
    )


class SqliteApiKeyRepository(ApiKeyRepositoryPort):
    """SQLite persistence adapter implementing ApiKeyRepositoryPort."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def create(self, record: ApiKeyRecord) -> ApiKeyRecord:
        """Persist a new API key record."""
        scopes_json = json.dumps([scope.value for scope in record.scopes])
        await self._db.execute_write(
            """
            INSERT INTO api_keys (
                id, name, key_prefix, key_hash, scopes_json,
                created_at, last_used_at, expires_at, revoked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.id,
                record.name,
                record.key_prefix,
                record.key_hash,
                scopes_json,
                format_dt(record.created_at),
                format_opt_dt(record.last_used_at),
                format_opt_dt(record.expires_at),
                format_opt_dt(record.revoked_at),
            ),
        )
        return record

    async def get_by_id(self, key_id: str) -> ApiKeyRecord | None:
        """Fetch an API key record by ID."""
        row = await self._db.fetch_one(
            f"SELECT {_SELECT_COLUMNS} FROM api_keys WHERE id = ?;",  # noqa: S608
            (key_id,),
        )
        return _row_to_api_key(row) if row is not None else None

    async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        """Fetch an API key record by its SHA-256 hex digest."""
        row = await self._db.fetch_one(
            f"SELECT {_SELECT_COLUMNS} FROM api_keys WHERE key_hash = ?;",  # noqa: S608
            (key_hash,),
        )
        return _row_to_api_key(row) if row is not None else None

    async def list_all(self) -> list[ApiKeyRecord]:
        """List all persisted API key records newest first."""
        rows = await self._db.fetch_all(
            f"SELECT {_SELECT_COLUMNS} FROM api_keys ORDER BY created_at DESC, rowid DESC;",  # noqa: S608
        )
        return [_row_to_api_key(row) for row in rows]

    async def touch_last_used(self, key_id: str, used_at: datetime) -> None:
        """Update the last_used_at timestamp for a key."""
        await self._db.execute_write(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?;",
            (format_dt(used_at), key_id),
        )

    async def revoke(self, key_id: str, revoked_at: datetime) -> bool:
        """Mark an API key as revoked, returning True if found."""
        updated = await self._db.execute_write(
            "UPDATE api_keys SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?;",
            (format_dt(revoked_at), key_id),
        )
        return updated > 0
