"""Idempotent SQLite schema DDL including FTS5 full-text chunk index."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

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
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        workspace_root TEXT,
        default_model_profile TEXT NOT NULL,
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
        decided_at TEXT
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
)


async def apply_schema(conn: aiosqlite.Connection) -> None:
    """Execute all idempotent DDL statements on an open SQLite connection."""
    for statement in SCHEMA_STATEMENTS:
        await conn.execute(statement)
