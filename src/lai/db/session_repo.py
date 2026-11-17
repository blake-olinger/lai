"""SQLite repository for chat sessions and chronological messages."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from lai.db._codecs import clamp_limit, clamp_offset, format_dt, parse_dt
from lai.domain.chat import Citation, Message, MessageRole, Session, TokenUsage
from lai.ports.repositories import SessionRepositoryPort

if TYPE_CHECKING:
    import aiosqlite

    from lai.db.connection import SqliteDatabase

_SESSION_COLUMNS = "id, project_id, title, model_profile, system_prompt, created_at, updated_at"
_MESSAGE_COLUMNS = (
    "id, session_id, project_id, role, content, thinking, citations_json, "
    "tool_call_id, tool_name, usage_json, created_at"
)


def _row_to_session(row: aiosqlite.Row) -> Session:
    """Map a sessions SQLite row to a Session domain model."""
    return Session(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        title=str(row["title"]),
        model_profile=str(row["model_profile"]),
        system_prompt=cast("str | None", row["system_prompt"]),
        created_at=parse_dt(str(row["created_at"])),
        updated_at=parse_dt(str(row["updated_at"])),
    )


def _row_to_message(row: aiosqlite.Row) -> Message:
    """Map a messages SQLite row to a Message domain model."""
    raw_citations = cast("list[object]", json.loads(str(row["citations_json"])))
    citations = tuple(Citation.model_validate(item) for item in raw_citations)
    raw_usage = cast("str | None", row["usage_json"])
    usage = TokenUsage.model_validate_json(raw_usage) if raw_usage is not None else None
    return Message(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        project_id=str(row["project_id"]),
        role=MessageRole(str(row["role"])),
        content=str(row["content"]),
        thinking=cast("str | None", row["thinking"]),
        citations=citations,
        tool_call_id=cast("str | None", row["tool_call_id"]),
        tool_name=cast("str | None", row["tool_name"]),
        usage=usage,
        created_at=parse_dt(str(row["created_at"])),
    )


class SqliteSessionRepository(SessionRepositoryPort):
    """SQLite persistence adapter implementing SessionRepositoryPort."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def create_session(self, session: Session) -> Session:
        """Persist a new chat session."""
        await self._db.execute_write(
            """
            INSERT INTO sessions (
                id, project_id, title, model_profile, system_prompt, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                session.id,
                session.project_id,
                session.title,
                session.model_profile,
                session.system_prompt,
                format_dt(session.created_at),
                format_dt(session.updated_at),
            ),
        )
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Fetch a chat session by ID."""
        row = await self._db.fetch_one(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ?;",  # noqa: S608
            (session_id,),
        )
        return _row_to_session(row) if row is not None else None

    async def list_sessions(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        """List chat sessions belonging to a project."""
        rows = await self._db.fetch_all(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE project_id = ? "  # noqa: S608
            "ORDER BY updated_at DESC, created_at DESC, rowid DESC LIMIT ? OFFSET ?;",
            (project_id, clamp_limit(limit), clamp_offset(offset)),
        )
        return [_row_to_session(row) for row in rows]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and return True if it existed."""
        deleted = await self._db.execute_write(
            "DELETE FROM sessions WHERE id = ?;",
            (session_id,),
        )
        return deleted > 0

    async def add_message(self, message: Message) -> Message:
        """Append a message to a chat session and bump session updated_at."""
        citations_json = json.dumps(
            [citation.model_dump(mode="json") for citation in message.citations]
        )
        usage_json = message.usage.model_dump_json() if message.usage is not None else None
        created_iso = format_dt(message.created_at)

        async with self._db.write_transaction() as conn:
            await conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, project_id, role, content, thinking,
                    citations_json, tool_call_id, tool_name, usage_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    message.id,
                    message.session_id,
                    message.project_id,
                    message.role.value,
                    message.content,
                    message.thinking,
                    citations_json,
                    message.tool_call_id,
                    message.tool_name,
                    usage_json,
                    created_iso,
                ),
            )
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?;",
                (created_iso, message.session_id),
            )
        return message

    async def list_messages(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Message]:
        """List messages within a session in chronological order."""
        rows = await self._db.fetch_all(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE session_id = ? "  # noqa: S608
            "ORDER BY created_at ASC, rowid ASC LIMIT ? OFFSET ?;",
            (session_id, clamp_limit(limit), clamp_offset(offset)),
        )
        return [_row_to_message(row) for row in rows]
