"""SQLite repository for registered external client applications."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, cast

from lai.db._codecs import format_dt, format_opt_dt, parse_dt, parse_opt_dt
from lai.domain.client_apps import ClientAppRecord
from lai.domain.errors import ConflictError
from lai.domain.tools import PermissionTier, ToolSpec
from lai.ports.repositories import ClientAppRepositoryPort

if TYPE_CHECKING:
    from datetime import datetime

    import aiosqlite

    from lai.db.connection import SqliteDatabase

_SELECT_COLUMNS = (
    "id, name, oauth_client_id, allowed_tiers_json, "
    "registered_tools_json, allowed_project_ids_json, "
    "created_at, last_used_at, expires_at, revoked_at"
)


def _row_to_client_app(row: aiosqlite.Row) -> ClientAppRecord:
    """Map a client_apps SQLite row to a ClientAppRecord domain model."""
    raw_tiers = cast("list[str]", json.loads(str(row["allowed_tiers_json"])))
    raw_tools = cast("list[object]", json.loads(str(row["registered_tools_json"])))
    raw_projects = cast("list[object]", json.loads(str(row["allowed_project_ids_json"])))
    return ClientAppRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        oauth_client_id=str(row["oauth_client_id"]),
        allowed_tiers=tuple(PermissionTier(item) for item in raw_tiers),
        registered_tools=tuple(ToolSpec.model_validate(item) for item in raw_tools),
        allowed_project_ids=tuple(str(item) for item in raw_projects),
        created_at=parse_dt(str(row["created_at"])),
        last_used_at=parse_opt_dt(cast("str | None", row["last_used_at"])),
        expires_at=parse_opt_dt(cast("str | None", row["expires_at"])),
        revoked_at=parse_opt_dt(cast("str | None", row["revoked_at"])),
    )


class SqliteClientAppRepository(ClientAppRepositoryPort):
    """SQLite persistence adapter implementing ClientAppRepositoryPort."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def create(self, record: ClientAppRecord) -> ClientAppRecord:
        """Persist a new client application record."""
        tiers_json = json.dumps([tier.value for tier in record.allowed_tiers])
        tools_json = json.dumps([spec.model_dump(mode="json") for spec in record.registered_tools])
        projects_json = json.dumps(list(record.allowed_project_ids))
        async with self._db.write_transaction() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO client_apps (
                        id, name, oauth_client_id, allowed_tiers_json,
                        registered_tools_json, allowed_project_ids_json,
                        created_at, last_used_at, expires_at, revoked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        record.id,
                        record.name,
                        record.oauth_client_id,
                        tiers_json,
                        tools_json,
                        projects_json,
                        format_dt(record.created_at),
                        format_opt_dt(record.last_used_at),
                        format_opt_dt(record.expires_at),
                        format_opt_dt(record.revoked_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                msg = (
                    "An active client app is already registered for "
                    f"oauth_client_id {record.oauth_client_id!r}."
                )
                raise ConflictError(
                    msg,
                    details={"oauth_client_id": record.oauth_client_id},
                ) from exc
        return record

    async def get_by_id(self, app_id: str) -> ClientAppRecord | None:
        """Fetch a client application record by ID."""
        row = await self._db.fetch_one(
            f"SELECT {_SELECT_COLUMNS} FROM client_apps WHERE id = ?;",  # noqa: S608
            (app_id,),
        )
        return _row_to_client_app(row) if row is not None else None

    async def get_by_oauth_client_id(self, oauth_client_id: str) -> ClientAppRecord | None:
        """Fetch the active (or latest revoked) client app bound to an OAuth client ID."""
        row = await self._db.fetch_one(
            f"SELECT {_SELECT_COLUMNS} FROM client_apps WHERE oauth_client_id = ? "  # noqa: S608
            "ORDER BY (revoked_at IS NOT NULL) ASC, created_at DESC, rowid DESC LIMIT 1;",
            (oauth_client_id,),
        )
        return _row_to_client_app(row) if row is not None else None

    async def list_all(self) -> list[ClientAppRecord]:
        """List all persisted client application records newest first."""
        rows = await self._db.fetch_all(
            f"SELECT {_SELECT_COLUMNS} FROM client_apps ORDER BY created_at DESC, rowid DESC;",  # noqa: S608
        )
        return [_row_to_client_app(row) for row in rows]

    async def touch_last_used(self, app_id: str, used_at: datetime) -> None:
        """Update the last_used_at timestamp for a client application."""
        await self._db.execute_write(
            "UPDATE client_apps SET last_used_at = ? WHERE id = ?;",
            (format_dt(used_at), app_id),
        )

    async def revoke(self, app_id: str, revoked_at: datetime) -> bool:
        """Mark a client application as revoked, returning True if found."""
        updated = await self._db.execute_write(
            "UPDATE client_apps SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?;",
            (format_dt(revoked_at), app_id),
        )
        return updated > 0
