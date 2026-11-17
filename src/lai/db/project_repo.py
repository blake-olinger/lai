"""SQLite repository for workspace projects."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lai.db._codecs import clamp_limit, clamp_offset, format_dt, parse_dt
from lai.domain.projects import Project
from lai.ports.repositories import ProjectRepositoryPort

if TYPE_CHECKING:
    import aiosqlite

    from lai.db.connection import SqliteDatabase

_SELECT_COLUMNS = (
    "id, name, description, workspace_root, default_model_profile, created_at, updated_at"
)


def _row_to_project(row: aiosqlite.Row) -> Project:
    """Map a projects SQLite row to a Project domain model."""
    return Project(
        id=str(row["id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        workspace_root=cast("str | None", row["workspace_root"]),
        default_model_profile=str(row["default_model_profile"]),
        created_at=parse_dt(str(row["created_at"])),
        updated_at=parse_dt(str(row["updated_at"])),
    )


class SqliteProjectRepository(ProjectRepositoryPort):
    """SQLite persistence adapter implementing ProjectRepositoryPort."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def create(self, project: Project) -> Project:
        """Persist a new project."""
        await self._db.execute_write(
            """
            INSERT INTO projects (
                id, name, description, workspace_root,
                default_model_profile, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                project.id,
                project.name,
                project.description,
                project.workspace_root,
                project.default_model_profile,
                format_dt(project.created_at),
                format_dt(project.updated_at),
            ),
        )
        return project

    async def get(self, project_id: str) -> Project | None:
        """Fetch a project by ID."""
        row = await self._db.fetch_one(
            f"SELECT {_SELECT_COLUMNS} FROM projects WHERE id = ?;",  # noqa: S608
            (project_id,),
        )
        return _row_to_project(row) if row is not None else None

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Project]:
        """List projects ordered by update and creation time descending."""
        rows = await self._db.fetch_all(
            f"SELECT {_SELECT_COLUMNS} FROM projects "  # noqa: S608
            "ORDER BY updated_at DESC, created_at DESC, rowid DESC LIMIT ? OFFSET ?;",
            (clamp_limit(limit), clamp_offset(offset)),
        )
        return [_row_to_project(row) for row in rows]

    async def delete(self, project_id: str) -> bool:
        """Delete a project, cleaning up FTS rows and cascading child records."""
        async with self._db.write_transaction() as conn:
            await conn.execute(
                "DELETE FROM chunks_fts WHERE project_id = ?;",
                (project_id,),
            )
            cursor = await conn.execute(
                "DELETE FROM projects WHERE id = ?;",
                (project_id,),
            )
            return cursor.rowcount > 0
