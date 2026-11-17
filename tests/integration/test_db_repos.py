"""Integration tests for SQLite connection, WAL pragmas, FTS5 sanitizer, and repos."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from lai.db import (
    SqliteApiKeyRepository,
    SqliteDatabase,
    SqliteFileRepository,
    SqliteProjectRepository,
    SqliteSessionRepository,
    SqliteToolAuditRepository,
    sanitize_fts5_query,
)
from lai.domain import AlertCategory, ApiKeyRecord, ApiScope, LoggingConfig
from lai.domain.chat import Citation, Message, MessageRole, Session, TokenUsage
from lai.domain.errors import LaiError, NotFoundError, ValidationFailedError
from lai.domain.files import (
    CodeFileSummary,
    FileChunk,
    FileKind,
    FileRecord,
    FileStatus,
    LogFileSummary,
    LogSeverity,
)
from lai.domain.projects import Project
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
)
from lai.observability import configure_observability
from lai.ports.repositories import (
    ApiKeyRepositoryPort,
    FileRepositoryPort,
    ProjectRepositoryPort,
    SessionRepositoryPort,
    ToolAuditRepositoryPort,
)

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_BUSY_TIMEOUT_MS = 5000
EXPECTED_CONCURRENT_PROJECTS = 10
EXPECTED_TWO_PROJECTS = 2
EXPECTED_CITATION_LINE = 40
EXPECTED_TOTAL_TOKENS = 15
EXPECTED_THREE_CHUNKS = 3
EXPECTED_TWO_RANGE_CHUNKS = 2


async def test_sqlite_database_pragmas_idempotency_concurrency_and_failure_alerts(
    tmp_path: Path,
) -> None:
    """SqliteDatabase enables WAL, foreign keys, busy_timeout, serializes writes, and alerts."""
    db_path = tmp_path / "nested" / "lai.db"
    async with SqliteDatabase(db_path) as db:
        await db.initialize()
        conn = await db.connect()
        journal = await (await conn.execute("PRAGMA journal_mode;")).fetchone()
        fk = await (await conn.execute("PRAGMA foreign_keys;")).fetchone()
        busy = await (await conn.execute("PRAGMA busy_timeout;")).fetchone()
        assert journal is not None
        assert str(journal[0]).lower() == "wal"
        assert fk is not None
        assert int(fk[0]) == 1
        assert busy is not None
        assert int(busy[0]) == EXPECTED_BUSY_TIMEOUT_MS

        repo = SqliteProjectRepository(db)
        await asyncio.gather(
            *(
                repo.create(Project(name=f"proj-{idx}"))
                for idx in range(EXPECTED_CONCURRENT_PROJECTS)
            )
        )
        assert len(await repo.list_all(limit=20)) == EXPECTED_CONCURRENT_PROJECTS

        manager = configure_observability(LoggingConfig(enable_file_logs=False, level="INFO"))
        try:
            with pytest.raises(LaiError) as exc_info:
                await db.fetch_one("SELECT * FROM non_existent_table;")
            assert exc_info.value.code == "storage_failure"
            assert any(a.category is AlertCategory.STORAGE_FAILURE for a in manager.recent_alerts())
        finally:
            manager.close()
            logging.getLogger().handlers.clear()


async def test_api_key_repository_crud_touch_and_revoke(tmp_path: Path) -> None:
    """SqliteApiKeyRepository implements full ApiKeyRepositoryPort contract."""
    async with SqliteDatabase(tmp_path / "keys.db") as db:
        repo = SqliteApiKeyRepository(db)
        assert isinstance(repo, ApiKeyRepositoryPort)

        now = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
        rec = ApiKeyRecord(
            name="primary",
            key_prefix="lai_sk_abcdef1",
            key_hash="a" * 64,
            scopes=(ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE),
            created_at=now,
            expires_at=now + timedelta(days=30),
        )
        await repo.create(rec)
        assert await repo.get_by_id(rec.id) == rec
        assert await repo.get_by_hash(rec.key_hash) == rec
        assert await repo.list_all() == [rec]

        used_at = now + timedelta(minutes=5)
        await repo.touch_last_used(rec.id, used_at)
        touched = await repo.get_by_id(rec.id)
        assert touched is not None
        assert touched.last_used_at == used_at

        revoked_at = now + timedelta(minutes=10)
        assert await repo.revoke(rec.id, revoked_at) is True
        assert await repo.revoke("key_missing", revoked_at) is False
        revoked = await repo.get_by_id(rec.id)
        assert revoked is not None
        assert revoked.revoked_at == revoked_at


async def test_project_and_session_repositories_and_cascade_delete(tmp_path: Path) -> None:
    """Project and Session repos persist messages/citations and cascade deletes cleanly."""
    async with SqliteDatabase(tmp_path / "chat.db") as db:
        projects, sessions = SqliteProjectRepository(db), SqliteSessionRepository(db)
        assert isinstance(projects, ProjectRepositoryPort)
        assert isinstance(sessions, SessionRepositoryPort)

        p1 = await projects.create(Project(name="alpha", workspace_root="C:/repo/alpha"))
        p2 = await projects.create(Project(name="beta"))
        assert await projects.get(p1.id) == p1
        assert len(await projects.list_all()) == EXPECTED_TWO_PROJECTS
        s1 = await sessions.create_session(
            Session(project_id=p1.id, title="debug crash", system_prompt="Be concise")
        )
        assert await sessions.get_session(s1.id) == s1
        assert await sessions.list_sessions(p1.id) == [s1]
        t0 = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
        m1 = await sessions.add_message(
            Message(
                session_id=s1.id,
                project_id=p1.id,
                role=MessageRole.USER,
                content="Why 0x80070005?",
                created_at=t0,
            )
        )
        m2 = await sessions.add_message(
            Message(
                session_id=s1.id,
                project_id=p1.id,
                role=MessageRole.ASSISTANT,
                content="Check line 40.",
                thinking="Inspecting log...",
                citations=(
                    Citation(
                        file_id="fil_1", filename="app.log", start_line=EXPECTED_CITATION_LINE
                    ),
                ),
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=EXPECTED_TOTAL_TOKENS
                ),
                created_at=t0 + timedelta(seconds=1),
            )
        )
        listed = await sessions.list_messages(s1.id)
        assert [m.id for m in listed] == [m1.id, m2.id]
        assert listed[1].citations[0].start_line == EXPECTED_CITATION_LINE
        assert listed[1].usage is not None
        assert listed[1].usage.total_tokens == EXPECTED_TOTAL_TOKENS

        assert await projects.delete(p1.id) is True
        assert await projects.delete(p1.id) is False
        assert await sessions.get_session(s1.id) is None
        assert await sessions.list_messages(s1.id) == []
        assert [p.id for p in await projects.list_all()] == [p2.id]


async def test_file_repo_chunks_and_adversarial_fts5_queries(tmp_path: Path) -> None:
    """SqliteFileRepository persists summaries, filters chunks, and handles tricky FTS5 inputs."""
    async with SqliteDatabase(tmp_path / "files.db") as db:
        projects, files = SqliteProjectRepository(db), SqliteFileRepository(db)
        assert isinstance(files, FileRepositoryPort)
        proj = await projects.create(Project(name="logs"))
        other_proj = await projects.create(Project(name="other"))
        f1 = FileRecord(
            project_id=proj.id,
            filename="app.log",
            size_bytes=1024,
            sha256="b" * 64,
            blob_path="blobs/app.log",
            kind=FileKind.LOG,
            status=FileStatus.READY,
            token_count=120,
            chunk_count=3,
            log_summary=LogFileSummary(total_lines=150, error_count=2, warning_count=1),
        )
        await files.upsert_file(f1)
        f1_updated = f1.model_copy(
            update={"code_summary": CodeFileSummary(language="python", total_lines=150)}
        )
        await files.upsert_file(f1_updated)
        assert await files.get_file(f1.id) == f1_updated
        assert len(await files.list_files(proj.id)) == 1

        chunks = [
            FileChunk(
                file_id=f1.id,
                project_id=proj.id,
                chunk_index=0,
                start_line=1,
                end_line=50,
                token_count=40,
                text="Startup sequence completed cleanly",
                symbol_name="boot_service",
                severity=LogSeverity.INFO,
                metadata={"host": "win11"},
            ),
            FileChunk(
                file_id=f1.id,
                project_id=proj.id,
                chunk_index=1,
                start_line=51,
                end_line=100,
                token_count=45,
                text="FATAL error 0x80070005 in foo:bar-baz module C:\\logs\\app.log",
                symbol_name="handle_win32_error",
                severity=LogSeverity.ERROR,
            ),
            FileChunk(
                file_id=f1.id,
                project_id=proj.id,
                chunk_index=2,
                start_line=101,
                end_line=150,
                token_count=35,
                text='Quote test foo"bar and NOT found',
                severity=LogSeverity.WARNING,
            ),
        ]
        await files.replace_chunks(f1.id, chunks)
        assert len(await files.list_chunks(f1.id)) == EXPECTED_THREE_CHUNKS
        assert (
            len(await files.list_chunks(f1.id, start_line=40, end_line=60))
            == EXPECTED_TWO_RANGE_CHUNKS
        )
        with pytest.raises(ValidationFailedError):
            await files.replace_chunks("fil_wrong", chunks)

        assert sanitize_fts5_query('--- ::: "" \x00') == ""
        adversarial = (
            "0x80070005",
            "foo:bar-baz",
            "C:\\logs\\app.log",
            'foo"bar',
            "NOT",
            "AND OR NEAR",
            "content:secret",
            "---",
            "",
            "   ",
            "func(a, b) --flag * ^",
        )
        for raw_query in adversarial:
            assert isinstance(await files.search_fts(proj.id, raw_query), list)

        hex_hits = await files.search_fts(proj.id, "0x80070005")
        assert len(hex_hits) == 1
        assert hex_hits[0].chunk.id == chunks[1].id
        assert hex_hits[0].source == "fts"
        assert hex_hits[0].score > 0.0
        assert [h.chunk.id for h in await files.search_fts(proj.id, "foo:bar-baz")] == [
            chunks[1].id
        ]
        assert await files.search_fts(other_proj.id, "0x80070005") == []
        assert await files.search_fts(proj.id, "0x80070005", file_ids=[]) == []
        assert len(await files.search_fts(proj.id, "0x80070005", file_ids=[f1.id])) == 1
        assert await files.delete_file(f1.id) is True
        assert await files.search_fts(proj.id, "0x80070005") == []


async def test_tool_audit_repository_approvals_and_audits(tmp_path: Path) -> None:
    """SqliteToolAuditRepository persists approvals, transitions, and audit logs."""
    async with SqliteDatabase(tmp_path / "tools.db") as db:
        projects = SqliteProjectRepository(db)
        sessions = SqliteSessionRepository(db)
        tools = SqliteToolAuditRepository(db)
        assert isinstance(tools, ToolAuditRepositoryPort)

        proj = await projects.create(Project(name="tools-proj"))
        s1 = await sessions.create_session(Session(project_id=proj.id))
        s2 = await sessions.create_session(Session(project_id=proj.id))

        pending = ToolApprovalRecord(
            project_id=proj.id,
            session_id=s1.id,
            tool_name="run_shell",
            arguments_json='{"cmd": "dir"}',
            permission_tier=PermissionTier.EXEC,
            reason="Inspect workspace",
            requested_by="usr_1",
        )
        await tools.create_approval(pending)
        assert await tools.get_approval(pending.id) == pending
        assert await tools.list_pending_approvals(proj.id, session_id=s1.id) == [pending]
        assert await tools.list_pending_approvals(proj.id, session_id=s2.id) == []

        approved = pending.model_copy(
            update={
                "status": ApprovalStatus.APPROVED,
                "decided_by": "admin_1",
                "decision_note": "ok",
                "decided_at": datetime.now(UTC),
            }
        )
        await tools.update_approval(approved)
        assert await tools.get_approval(pending.id) == approved
        assert await tools.list_pending_approvals(proj.id) == []
        with pytest.raises(NotFoundError):
            await tools.update_approval(approved.model_copy(update={"id": "apr_missing"}))

        audit = ToolAuditRecord(
            project_id=proj.id,
            session_id=s1.id,
            principal_id="usr_1",
            tool_name="run_shell",
            permission_tier=PermissionTier.EXEC,
            policy_decision=PolicyDecision.ALLOW,
            arguments_json='{"cmd": "dir"}',
            success=True,
            output_preview="file1.txt",
            approval_id=pending.id,
            duration_ms=12.5,
        )
        await tools.record_audit(audit)
        assert await tools.list_audits(proj.id, session_id=s1.id) == [audit]
        assert await tools.list_audits(proj.id, session_id=s2.id) == []
