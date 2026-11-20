"""Idempotent SQLite schema DDL including FTS5 full-text chunk index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from lai.db._schema_analytics import ANALYTICS_AND_RUNTIME_STATEMENTS
from lai.domain.errors import LaiError

if TYPE_CHECKING:
    import aiosqlite

CURRENT_SCHEMA_VERSION: Final[int] = 1

_CLIENT_APPS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS client_apps (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        oauth_client_id TEXT NOT NULL,
        allowed_tiers_json TEXT NOT NULL,
        registered_tools_json TEXT NOT NULL DEFAULT '[]',
        allowed_project_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        last_used_at TEXT,
        expires_at TEXT,
        revoked_at TEXT
    );
    """

_CLIENT_APPS_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_client_apps_active
        ON client_apps(revoked_at, created_at DESC);
    """

_CLIENT_APPS_OAUTH_UNIQUE_INDEX_SQL = """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_client_apps_oauth_client_id_active
        ON client_apps(oauth_client_id)
        WHERE revoked_at IS NULL;
    """

_PROJECTS_OWNER_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_projects_owner
        ON projects(owner_principal_id, updated_at DESC, created_at DESC);
    """

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        key_prefix TEXT NOT NULL,
        key_hash TEXT NOT NULL UNIQUE,
        scopes_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_used_at TEXT,
        expires_at TEXT,
        revoked_at TEXT
    );
    """,
    _CLIENT_APPS_TABLE_SQL,
    _CLIENT_APPS_INDEX_SQL,
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        default_model_profile TEXT NOT NULL,
        owner_principal_id TEXT NOT NULL DEFAULT 'system',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        model_profile TEXT NOT NULL,
        system_prompt TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sessions_project
        ON sessions(project_id, updated_at DESC, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        thinking TEXT,
        citations_json TEXT NOT NULL DEFAULT '[]',
        tool_calls_json TEXT NOT NULL DEFAULT '[]',
        tool_call_id TEXT,
        tool_name TEXT,
        usage_json TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, created_at ASC);
    """,
    """
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        content_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        blob_path TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        log_summary_json TEXT,
        code_summary_json TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_files_project
        ON files(project_id, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS file_chunks (
        id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        token_count INTEGER NOT NULL,
        text TEXT NOT NULL,
        symbol_name TEXT,
        severity TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_file_chunks_file
        ON file_chunks(file_id, chunk_index ASC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_file_chunks_lines
        ON file_chunks(file_id, start_line, end_line);
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content,
        symbol_name,
        chunk_id UNINDEXED,
        file_id UNINDEXED,
        project_id UNINDEXED,
        severity UNINDEXED,
        tokenize='porter unicode61'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_approvals (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        tool_name TEXT NOT NULL,
        arguments_json TEXT NOT NULL,
        permission_tier TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL,
        requested_by TEXT NOT NULL,
        decided_by TEXT,
        decision_note TEXT,
        created_at TEXT NOT NULL,
        decided_at TEXT,
        execution_target TEXT NOT NULL DEFAULT 'server_attachment',
        client_app_id TEXT,
        tool_call_id TEXT,
        allowed_file_ids_json TEXT NOT NULL DEFAULT '[]',
        principal_scopes_json TEXT NOT NULL DEFAULT '[]',
        expires_at TEXT,
        execution_result_json TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_approvals_pending
        ON tool_approvals(project_id, status, created_at ASC);
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_audits (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        principal_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        permission_tier TEXT NOT NULL,
        policy_decision TEXT NOT NULL,
        arguments_json TEXT NOT NULL,
        success INTEGER NOT NULL,
        output_preview TEXT NOT NULL DEFAULT '',
        error TEXT,
        approval_id TEXT,
        duration_ms REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_audits_project
        ON tool_audits(project_id, created_at DESC);
    """,
    *ANALYTICS_AND_RUNTIME_STATEMENTS,
)


_PROJECTS_MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "owner_principal_id",
        "ALTER TABLE projects ADD COLUMN owner_principal_id TEXT NOT NULL DEFAULT 'system';",
    ),
)

_MESSAGES_MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "tool_calls_json",
        "ALTER TABLE messages ADD COLUMN tool_calls_json TEXT NOT NULL DEFAULT '[]';",
    ),
)

_TOOL_APPROVALS_MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "execution_target",
        "ALTER TABLE tool_approvals ADD COLUMN "
        "execution_target TEXT NOT NULL DEFAULT 'server_attachment';",
    ),
    ("client_app_id", "ALTER TABLE tool_approvals ADD COLUMN client_app_id TEXT;"),
    ("tool_call_id", "ALTER TABLE tool_approvals ADD COLUMN tool_call_id TEXT;"),
    (
        "allowed_file_ids_json",
        "ALTER TABLE tool_approvals ADD COLUMN allowed_file_ids_json TEXT NOT NULL DEFAULT '[]';",
    ),
    (
        "principal_scopes_json",
        "ALTER TABLE tool_approvals ADD COLUMN principal_scopes_json TEXT NOT NULL DEFAULT '[]';",
    ),
    ("expires_at", "ALTER TABLE tool_approvals ADD COLUMN expires_at TEXT;"),
    ("execution_result_json", "ALTER TABLE tool_approvals ADD COLUMN execution_result_json TEXT;"),
)


async def _migrate_table_columns(
    conn: aiosqlite.Connection,
    pragma_sql: str,
    migrations: tuple[tuple[str, str], ...],
) -> None:
    """Add any missing columns to an existing SQLite table idempotently."""
    cursor = await conn.execute(pragma_sql)
    rows = await cursor.fetchall()
    existing = {str(row[1]) for row in rows}
    for col_name, alter_sql in migrations:
        if col_name not in existing:
            await conn.execute(alter_sql)


async def _has_client_apps_table_unique_constraint(conn: aiosqlite.Connection) -> bool:
    """Return True if client_apps has an inline UNIQUE constraint (sqlite_autoindex)."""
    cursor = await conn.execute("PRAGMA index_list(client_apps);")
    rows = await cursor.fetchall()
    return any(str(row[3]) == "u" for row in rows)


async def _migrate_client_apps_table(conn: aiosqlite.Connection) -> None:
    """Migrate legacy secret-based or table-wide UNIQUE client_apps tables."""
    cursor = await conn.execute("PRAGMA table_info(client_apps);")
    rows = await cursor.fetchall()
    existing = {str(row[1]) for row in rows}
    if existing and "oauth_client_id" not in existing:
        await conn.execute("DROP TABLE IF EXISTS client_apps;")
        await conn.execute(_CLIENT_APPS_TABLE_SQL)
        await conn.execute(_CLIENT_APPS_INDEX_SQL)
    elif existing and await _has_client_apps_table_unique_constraint(conn):
        new_table_sql = _CLIENT_APPS_TABLE_SQL.replace(
            "IF NOT EXISTS client_apps", "client_apps__new", 1
        )
        await conn.execute(new_table_sql)
        await conn.execute(
            """
            INSERT INTO client_apps__new (
                id, name, oauth_client_id, allowed_tiers_json,
                registered_tools_json, allowed_project_ids_json,
                created_at, last_used_at, expires_at, revoked_at
            )
            SELECT
                id, name, oauth_client_id, allowed_tiers_json,
                registered_tools_json, allowed_project_ids_json,
                created_at, last_used_at, expires_at, revoked_at
            FROM client_apps;
            """
        )
        await conn.execute("DROP TABLE client_apps;")
        await conn.execute("ALTER TABLE client_apps__new RENAME TO client_apps;")
        await conn.execute(_CLIENT_APPS_INDEX_SQL)
    await conn.execute(_CLIENT_APPS_OAUTH_UNIQUE_INDEX_SQL)


async def _migrate_existing_tables(conn: aiosqlite.Connection) -> None:
    """Apply backward-compatible column migrations to pre-existing SQLite databases."""
    await _migrate_table_columns(conn, "PRAGMA table_info(projects);", _PROJECTS_MIGRATIONS)
    await conn.execute(_PROJECTS_OWNER_INDEX_SQL)
    await _migrate_table_columns(conn, "PRAGMA table_info(messages);", _MESSAGES_MIGRATIONS)
    await _migrate_table_columns(
        conn, "PRAGMA table_info(tool_approvals);", _TOOL_APPROVALS_MIGRATIONS
    )
    await _migrate_client_apps_table(conn)


async def apply_schema(conn: aiosqlite.Connection) -> None:
    """Verify PRAGMA user_version, execute idempotent DDL/migrations, and stamp version."""
    cursor = await conn.execute("PRAGMA user_version;")
    row = await cursor.fetchone()
    db_version = int(row[0]) if row is not None else 0
    if db_version > CURRENT_SCHEMA_VERSION:
        msg = (
            f"Database schema version {db_version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}."
        )
        raise LaiError(
            msg,
            code="schema_version_unsupported",
            status_code=500,
            details={
                "db_version": str(db_version),
                "supported_version": str(CURRENT_SCHEMA_VERSION),
            },
        )
    for statement in SCHEMA_STATEMENTS:
        await conn.execute(statement)
    await _migrate_existing_tables(conn)
    if db_version < CURRENT_SCHEMA_VERSION:
        await conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION};")
