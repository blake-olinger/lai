"""Integration tests for SQLite connection, schema migrations, FTS5 sanitizer, and repos."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from lai import db as db_mod
from lai.domain import AlertCategory, ApiKeyRecord, ApiScope, LoggingConfig
from lai.domain import chat as dc
from lai.domain import files as df
from lai.domain import tools as dt
from lai.domain.errors import ConflictError, LaiError, NotFoundError, ValidationFailedError
from lai.domain.projects import Project
from lai.observability import configure_observability
from lai.ports import repositories as rp
from lai.security import build_client_app

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_BUSY_TIMEOUT_MS, EXPECTED_CONCURRENT, EXPECTED_TWO = 5000, 10, 2
EXPECTED_CITATION_LINE, EXPECTED_TOTAL_TOKENS = 40, 15
EXPECTED_THREE, EXPECTED_OWNER_BYTES = 3, 1500
_TS = "2026-01-01T00:00:00+00:00"
_LEGACY_DDL = (
    "CREATE TABLE projects(id TEXT PRIMARY KEY,name TEXT NOT NULL,description TEXT NOT NULL "
    "DEFAULT '',workspace_root TEXT,default_model_profile TEXT NOT NULL,created_at TEXT NOT NULL,"
    "updated_at TEXT NOT NULL);CREATE TABLE client_apps(id TEXT PRIMARY KEY,name TEXT NOT NULL,"
    "secret_prefix TEXT NOT NULL,secret_hash TEXT NOT NULL UNIQUE,allowed_tiers_json TEXT NOT NULL,"
    "registered_tools_json TEXT NOT NULL DEFAULT '[]',allowed_project_ids_json TEXT NOT NULL "
    "DEFAULT '[]',created_at TEXT NOT NULL,last_used_at TEXT,expires_at TEXT,revoked_at TEXT);"
    "CREATE TABLE messages(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,project_id TEXT NOT NULL,"
    "role TEXT NOT NULL,content TEXT NOT NULL,thinking TEXT,citations_json TEXT NOT NULL DEFAULT "
    "'[]',tool_call_id TEXT,tool_name TEXT,usage_json TEXT,created_at TEXT NOT NULL);"
    "CREATE TABLE tool_approvals(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,session_id TEXT NOT "
    "NULL,tool_name TEXT NOT NULL,arguments_json TEXT NOT NULL,permission_tier TEXT NOT NULL,"
    "reason TEXT NOT NULL,status TEXT NOT NULL,requested_by TEXT NOT NULL,decided_by TEXT,"
    "decision_note TEXT,created_at TEXT NOT NULL,decided_at TEXT);"
    f"INSERT INTO projects VALUES('prj_legacy','legacy','','/w','gemma4-26b-a4b','{_TS}','{_TS}');"
    f"INSERT INTO client_apps VALUES('app_old','old','lai_app_x','{'h' * 32}','[\"read\"]','[]',"
    f"'[]','{_TS}',NULL,NULL,NULL);INSERT INTO messages VALUES('msg_1','ses_1','prj_legacy','user',"
    f"'hi',NULL,'[]',NULL,NULL,NULL,'{_TS}');INSERT INTO tool_approvals VALUES('apr_1',"
    f"'prj_legacy','ses_1','t','{{}}','read','r','pending','u',NULL,NULL,'{_TS}',NULL);"
)


def _must[T](value: T | None) -> T:
    assert value is not None
    return value


async def test_sqlite_database_pragmas_concurrency_and_failure_alerts(tmp_path: Path) -> None:
    """SqliteDatabase enables WAL, foreign keys, busy_timeout, serializes writes, and alerts."""
    async with db_mod.SqliteDatabase(tmp_path / "nested" / "lai.db") as db:
        await db.initialize()
        conn = await db.connect()
        for sql, exp in (
            ("PRAGMA journal_mode;", "wal"),
            ("PRAGMA foreign_keys;", "1"),
            ("PRAGMA busy_timeout;", str(EXPECTED_BUSY_TIMEOUT_MS)),
        ):
            assert str(_must(await (await conn.execute(sql)).fetchone())[0]).lower() == exp
        repo = db_mod.SqliteProjectRepository(db)
        coros = [repo.create(Project(name=f"p-{i}")) for i in range(EXPECTED_CONCURRENT)]
        await asyncio.gather(*coros)
        assert len(await repo.list_all(limit=20)) == EXPECTED_CONCURRENT
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
    async with db_mod.SqliteDatabase(tmp_path / "keys.db") as db:
        repo = db_mod.SqliteApiKeyRepository(db)
        assert isinstance(repo, rp.ApiKeyRepositoryPort)
        now = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
        rec = ApiKeyRecord(
            name="primary",
            key_prefix="lai_sk_abcdef1",  # gitleaks:allow
            key_hash="a" * 64,
            scopes=(ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE),
            created_at=now,
            expires_at=now + timedelta(days=30),
        )
        await repo.create(rec)
        assert (await repo.get_by_id(rec.id), await repo.get_by_hash(rec.key_hash)) == (rec, rec)
        assert await repo.list_all() == [rec]
        t_use, t_rev = now + timedelta(minutes=5), now + timedelta(minutes=10)
        await repo.touch_last_used(rec.id, t_use)
        assert _must(await repo.get_by_id(rec.id)).last_used_at == t_use
        assert await repo.revoke(rec.id, t_rev) is True
        assert await repo.revoke("key_x", t_rev) is False
        assert _must(await repo.get_by_id(rec.id)).revoked_at == t_rev


async def test_project_and_session_repositories_owner_queries_and_cascade(tmp_path: Path) -> None:
    """Project and Session repos enforce owner scoping, persist messages, and cascade deletes."""
    async with db_mod.SqliteDatabase(tmp_path / "chat.db") as db:
        projects, sessions = db_mod.SqliteProjectRepository(db), db_mod.SqliteSessionRepository(db)
        assert isinstance(projects, rp.ProjectRepositoryPort)
        assert isinstance(sessions, rp.SessionRepositoryPort)
        t0 = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
        t1 = datetime(2026, 11, 17, 10, 0, 1, tzinfo=UTC)
        p1 = await projects.create(Project(name="a", owner_principal_id="u1", updated_at=t0))
        p2 = await projects.create(Project(name="b", owner_principal_id="u1", updated_at=t1))
        p3 = await projects.create(Project(name="g", owner_principal_id="u2"))
        assert (await projects.get(p1.id), len(await projects.list_all())) == (p1, EXPECTED_THREE)
        assert await projects.count_for_owner("u1") == EXPECTED_TWO
        assert await projects.count_for_owner("u9") == 0
        assert [p.id for p in await projects.list_for_owner("u1")] == [p2.id, p1.id]
        assert [p.id for p in await projects.list_for_owner("u1", limit=1, offset=1)] == [p1.id]
        assert await projects.list_for_owner("u2") == [p3]

        s1 = await sessions.create_session(dc.Session(project_id=p1.id, title="dbg"))
        assert await sessions.get_session(s1.id) == s1
        assert await sessions.list_sessions(p1.id) == [s1]
        tc = dc.InferenceToolCall(id="tc_1", name="read_log", arguments_json='{"line": 40}')
        cit = dc.Citation(file_id="fil_1", filename="app.log", start_line=40)
        usage = dc.TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        m1 = dc.Message(
            session_id=s1.id,
            project_id=p1.id,
            role=dc.MessageRole.USER,
            content="hi",
            created_at=t0,
        )
        m2 = dc.Message(
            session_id=s1.id,
            project_id=p1.id,
            role=dc.MessageRole.ASSISTANT,
            content="L40.",
            thinking="..",
            citations=(cit,),
            tool_calls=(tc,),
            usage=usage,
            created_at=t1,
        )
        await sessions.add_message(m1)
        await sessions.add_message(m2)
        listed = await sessions.list_messages(s1.id)
        assert [m.id for m in listed] == [m1.id, m2.id]
        assert listed[1].citations[0].start_line == EXPECTED_CITATION_LINE
        assert listed[1].tool_calls == (tc,)
        assert _must(listed[1].usage).total_tokens == EXPECTED_TOTAL_TOKENS
        assert (await projects.delete(p1.id), await projects.delete(p1.id)) == (True, False)
        assert await sessions.get_session(s1.id) is None
        assert await sessions.list_messages(s1.id) == []
        assert [p.id for p in await projects.list_all()] == [p3.id, p2.id]


async def test_file_repo_chunks_owner_bytes_and_adversarial_fts5(tmp_path: Path) -> None:
    """SqliteFileRepository persists summaries, sums owner bytes, and handles tricky FTS5."""
    async with db_mod.SqliteDatabase(tmp_path / "files.db") as db:
        projects, files = db_mod.SqliteProjectRepository(db), db_mod.SqliteFileRepository(db)
        assert isinstance(files, rp.FileRepositoryPort)
        proj = await projects.create(Project(name="logs", owner_principal_id="u1"))
        proj_b = await projects.create(Project(name="logs-b", owner_principal_id="u1"))
        other = await projects.create(Project(name="other", owner_principal_id="u2"))
        assert await files.sum_bytes_for_owner("u1") == 0

        ls = df.LogFileSummary(total_lines=150, error_count=2, warning_count=1)
        f1 = df.FileRecord(
            project_id=proj.id,
            filename="app.log",
            size_bytes=1024,
            sha256="b" * 64,
            blob_path="blobs/app.log",
            kind=df.FileKind.LOG,
            status=df.FileStatus.READY,
            log_summary=ls,
        )
        b_rec = f1.model_copy(update={"id": "fil_b", "project_id": proj_b.id, "size_bytes": 476})
        o_rec = f1.model_copy(update={"id": "fil_o", "project_id": other.id, "size_bytes": 9999})
        for rec in (f1, b_rec, o_rec):
            await files.upsert_file(rec)
        assert await files.sum_bytes_for_owner("u1") == EXPECTED_OWNER_BYTES
        cs = df.CodeFileSummary(language="py", total_lines=150)
        f1_up = f1.model_copy(update={"code_summary": cs})
        await files.upsert_file(f1_up)
        assert (await files.get_file(f1.id), len(await files.list_files(proj.id))) == (f1_up, 1)

        specs = (
            (1, 50, "Startup clean", "boot_service", df.LogSeverity.INFO),
            (51, 100, "FATAL 0x80070005 foo:bar-baz C:\\app.log", "err_fn", df.LogSeverity.ERROR),
            (101, 150, 'Quote test foo"bar and NOT found', None, df.LogSeverity.WARNING),
        )
        chunks = [
            df.FileChunk(
                file_id=f1.id,
                project_id=proj.id,
                chunk_index=idx,
                start_line=lo,
                end_line=hi,
                token_count=40,
                text=txt,
                symbol_name=sym,
                severity=sev,
                metadata={"host": "win11"} if idx == 0 else {},
            )
            for idx, (lo, hi, txt, sym, sev) in enumerate(specs)
        ]
        await files.replace_chunks(f1.id, chunks)
        assert len(await files.list_chunks(f1.id)) == EXPECTED_THREE
        assert len(await files.list_chunks(f1.id, start_line=40, end_line=60)) == EXPECTED_TWO
        with pytest.raises(ValidationFailedError):
            await files.replace_chunks("fil_wrong", chunks)

        assert db_mod.sanitize_fts5_query('--- ::: "" \x00') == ""
        adv_a = ("0x80070005", "foo:bar-baz", "C:\\logs\\app.log", 'foo"bar', "NOT", "AND OR NEAR")
        adv_b = ("content:secret", "---", "", "   ", "func(a, b) --flag * ^")
        for raw_query in adv_a + adv_b:
            assert isinstance(await files.search_fts(proj.id, raw_query), list)

        hex_hits = await files.search_fts(proj.id, "0x80070005")
        assert (len(hex_hits), hex_hits[0].score > 0.0) == (1, True)
        assert (hex_hits[0].chunk.id, hex_hits[0].source) == (chunks[1].id, "fts")
        foo_hits = await files.search_fts(proj.id, "foo:bar-baz")
        assert [h.chunk.id for h in foo_hits] == [chunks[1].id]
        assert await files.search_fts(other.id, "0x80070005") == []
        assert await files.search_fts(proj.id, "0x80070005", file_ids=[]) == []
        assert len(await files.search_fts(proj.id, "0x80070005", file_ids=[f1.id])) == 1
        assert await files.delete_file(f1.id) is True
        assert await files.sum_bytes_for_owner("u1") == EXPECTED_OWNER_BYTES - 1024
        assert await files.search_fts(proj.id, "0x80070005") == []


async def test_tool_audit_repository_approvals_cas_and_audits(tmp_path: Path) -> None:
    """SqliteToolAuditRepository persists new fields, atomic CAS transitions, and audits."""
    async with db_mod.SqliteDatabase(tmp_path / "tools.db") as db:
        projects, sessions = db_mod.SqliteProjectRepository(db), db_mod.SqliteSessionRepository(db)
        tools = db_mod.SqliteToolAuditRepository(db)
        assert isinstance(tools, rp.ToolAuditRepositoryPort)
        proj = await projects.create(Project(name="tools-proj"))
        s1 = await sessions.create_session(dc.Session(project_id=proj.id))
        s2 = await sessions.create_session(dc.Session(project_id=proj.id))
        base: dict[str, object] = {
            "project_id": proj.id,
            "session_id": s1.id,
            "tool_name": "run_shell",
            "arguments_json": '{"cmd": "dir"}',
            "permission_tier": dt.PermissionTier.EXEC,
        }
        t_exp = datetime(2026, 11, 18, 10, 0, tzinfo=UTC)
        pending = dt.ToolApprovalRecord.model_validate(
            {
                **base,
                "reason": "Inspect",
                "requested_by": "usr_1",
                "execution_target": dt.ToolExecutionTarget.CLIENT,
                "client_app_id": "app_1",
                "tool_call_id": "tc_99",
                "allowed_file_ids": ("fil_1",),
                "principal_scopes": (ApiScope.TOOLS_EXEC,),
                "expires_at": t_exp,
            }
        )
        await tools.create_approval(pending)
        assert await tools.get_approval(pending.id) == pending
        assert await tools.list_pending_approvals(proj.id, session_id=s1.id) == [pending]
        assert await tools.list_pending_approvals(proj.id, session_id=s2.id) == []

        t_dec = datetime.now(UTC)
        apr_upd = {
            "status": dt.ApprovalStatus.APPROVED,
            "decided_by": "a1",
            "decision_note": "ok",
            "decided_at": t_dec,
        }
        approved = pending.model_copy(update=apr_upd)
        fn, st = tools.transition_approval_status, dt.ApprovalStatus.PENDING
        cas = await asyncio.gather(
            *[fn(approved, expected_status=st) for _ in range(EXPECTED_CONCURRENT)]
        )
        assert (sum(1 for w in cas if w), await tools.get_approval(pending.id)) == (1, approved)
        assert await tools.list_pending_approvals(proj.id) == []

        res = dt.ToolResult(tool_name="run_shell", success=True, output="file1.txt")
        executed = approved.model_copy(
            update={"status": dt.ApprovalStatus.EXECUTED, "execution_result": res}
        )
        await tools.update_approval(executed)
        assert await tools.get_approval(pending.id) == executed
        with pytest.raises(NotFoundError):
            await tools.update_approval(executed.model_copy(update={"id": "apr_missing"}))

        audit = dt.ToolAuditRecord.model_validate(
            {
                **base,
                "principal_id": "usr_1",
                "policy_decision": dt.PolicyDecision.ALLOW,
                "success": True,
                "output_preview": "file1.txt",
                "approval_id": pending.id,
                "duration_ms": 12.5,
            }
        )
        await tools.record_audit(audit)
        assert await tools.list_audits(proj.id, session_id=s1.id) == [audit]
        assert await tools.list_audits(proj.id, session_id=s2.id) == []


async def test_client_app_repository_and_legacy_schema_migration(tmp_path: Path) -> None:
    """SqliteClientAppRepository CRUD works and apply_schema migrates legacy SQLite DBs."""
    legacy_path = tmp_path / "legacy.db"
    async with aiosqlite.connect(legacy_path) as raw:
        await raw.executescript(_LEGACY_DDL)
        await db_mod.apply_schema(raw)
        await db_mod.apply_schema(raw)
        await raw.commit()

    async with db_mod.SqliteDatabase(legacy_path) as db:
        leg_proj = _must(await db_mod.SqliteProjectRepository(db).get("prj_legacy"))
        assert leg_proj.owner_principal_id == "system"
        msgs = await db_mod.SqliteSessionRepository(db).list_messages("ses_1")
        assert (len(msgs), msgs[0].tool_calls) == (1, ())
        apr = _must(await db_mod.SqliteToolAuditRepository(db).get_approval("apr_1"))
        assert apr.execution_target is dt.ToolExecutionTarget.SERVER_ATTACHMENT
        assert (apr.allowed_file_ids, apr.principal_scopes, apr.execution_result) == ((), (), None)

        apps = db_mod.SqliteClientAppRepository(db)
        assert (isinstance(apps, rp.ClientAppRepositoryPort), await apps.list_all()) == (True, [])
        now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
        spec = dt.ToolSpec(name="run_ps", description="PS", permission_tier=dt.PermissionTier.EXEC)
        tiers, exp = (dt.PermissionTier.READ, dt.PermissionTier.EXEC), now + timedelta(days=7)
        rec = build_client_app("cli", "oauth-c1", tiers, (spec,), ("prj_1",), expires_at=exp)
        await apps.create(rec)
        assert (await apps.get_by_id(rec.id), await apps.list_all()) == (rec, [rec])
        assert await apps.get_by_oauth_client_id("oauth-c1") == rec
        assert await apps.get_by_oauth_client_id("missing-client") is None
        with pytest.raises(ConflictError):
            await apps.create(build_client_app("dup", "oauth-c1"))
        t_use, t_rev = now + timedelta(minutes=1), now + timedelta(minutes=2)
        await apps.touch_last_used(rec.id, t_use)
        assert _must(await apps.get_by_id(rec.id)).last_used_at == t_use
        assert (await apps.revoke(rec.id, t_rev), await apps.revoke("app_x", now)) == (True, False)
        assert _must(await apps.get_by_id(rec.id)).revoked_at == t_rev
