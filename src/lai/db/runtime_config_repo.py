"""SQLite repository persisting per-section non-secret runtime configuration overrides."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from lai.domain.base import utc_now
from lai.ports.analytics import RuntimeConfigRepositoryPort

if TYPE_CHECKING:
    from lai.db.connection import SqliteDatabase


def _decode_json_object(raw: str) -> dict[str, object]:
    parsed: object = json.loads(raw)
    if not isinstance(parsed, dict):
        return {}
    mapping = cast("dict[object, object]", parsed)
    return {str(key): value for key, value in mapping.items()}


class SqliteRuntimeConfigRepository(RuntimeConfigRepositoryPort):
    """Stores per-section non-secret runtime overrides in SQLite so live edits survive restarts."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def get_section(self, section: str) -> dict[str, object] | None:
        """Return persisted override dictionary for section, or None if not set."""
        row = await self._db.fetch_one(
            "SELECT payload_json FROM runtime_settings WHERE section = ?;",
            (section,),
        )
        if row is None:
            return None
        return _decode_json_object(str(row["payload_json"]))

    async def get_all_sections(self) -> dict[str, dict[str, object]]:
        """Return all persisted non-secret runtime section overrides."""
        rows = await self._db.fetch_all(
            "SELECT section, payload_json FROM runtime_settings ORDER BY section ASC;"
        )
        return {str(row["section"]): _decode_json_object(str(row["payload_json"])) for row in rows}

    async def upsert_section(self, section: str, payload: dict[str, object]) -> None:
        """Insert or replace the non-secret runtime override payload for section."""
        serialized = json.dumps(payload, sort_keys=True)
        now_iso = utc_now().isoformat()
        await self._db.execute_write(
            """
            INSERT INTO runtime_settings (section, payload_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(section) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at;
            """,
            (section, serialized, now_iso),
        )

    async def delete_section(self, section: str) -> bool:
        """Delete any persisted override for section, returning True if removed."""
        deleted = await self._db.execute_write(
            "DELETE FROM runtime_settings WHERE section = ?;",
            (section,),
        )
        return deleted > 0
