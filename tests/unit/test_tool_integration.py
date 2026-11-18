"""End-to-end unit tests wiring GuardedToolExecutor with SQLite repositories and LocalBlobStore."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from lai.db import (
    SqliteDatabase,
    SqliteFileRepository,
    SqliteProjectRepository,
    SqliteSessionRepository,
    SqliteToolAuditRepository,
)
from lai.domain.auth import ApiScope
from lai.domain.chat import Session
from lai.domain.errors import AuthorizationError
from lai.domain.files import FileChunk, FileKind, FileRecord, FileStatus
from lai.domain.projects import Project
from lai.domain.tools import PolicyDecision, ToolExecutionContext
from lai.ingestion.blob_store import LocalBlobStore
from lai.tools import GuardedToolExecutor, build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

EXPECTED_AUDITS = 4


async def _bytes_stream(payload: bytes) -> AsyncIterator[bytes]:
    yield payload


async def test_executor_invokes_all_phase1_plugins_and_audits_isolation(tmp_path: Path) -> None:
    """GuardedToolExecutor runs Phase 1 plugins against real SQLite + LocalBlobStore."""
    blob_store = LocalBlobStore(tmp_path / "blobs")
    async with SqliteDatabase(tmp_path / "lai.db") as db:
        projects = SqliteProjectRepository(db)
        sessions = SqliteSessionRepository(db)
        file_repo = SqliteFileRepository(db)
        audit_repo = SqliteToolAuditRepository(db)

        await projects.create(Project(id="prj_1", name="Project 1"))
        await projects.create(Project(id="prj_2", name="Project 2"))
        await sessions.create_session(Session(id="ses_1", project_id="prj_1", title="Session 1"))

        blob_1, size_1, sha_1 = await blob_store.write_stream(
            "prj_1", "fil_1", "app.py", _bytes_stream(b"print('hi')\n")
        )
        blob_2, size_2, sha_2 = await blob_store.write_stream(
            "prj_2", "fil_2", "secret.py", _bytes_stream(b"SECRET = 'x'\n")
        )
        await file_repo.upsert_file(
            FileRecord(
                id="fil_1",
                project_id="prj_1",
                filename="app.py",
                size_bytes=size_1,
                sha256=sha_1,
                blob_path=str(blob_1),
                kind=FileKind.CODE,
                status=FileStatus.READY,
                chunk_count=1,
            )
        )
        await file_repo.replace_chunks(
            "fil_1",
            [
                FileChunk(
                    id="chk_1",
                    file_id="fil_1",
                    project_id="prj_1",
                    chunk_index=0,
                    start_line=1,
                    end_line=1,
                    token_count=4,
                    text="print('hi')",
                )
            ],
        )
        await file_repo.upsert_file(
            FileRecord(
                id="fil_2",
                project_id="prj_2",
                filename="secret.py",
                size_bytes=size_2,
                sha256=sha_2,
                blob_path=str(blob_2),
                kind=FileKind.CODE,
                status=FileStatus.READY,
            )
        )

        registry = build_default_tool_registry(file_repo, blob_store)
        executor = GuardedToolExecutor(registry, audit_repo)
        ctx = ToolExecutionContext(
            project_id="prj_1",
            session_id="ses_1",
            principal_id="usr_1",
            principal_scopes=(ApiScope.TOOLS_SAFE,),
        )

        search_res = await executor.invoke("search_files", {"query": "hi"}, ctx)
        assert search_res.success is True
        assert json.loads(search_res.output)["results"][0]["chunk_id"] == "chk_1"

        read_res = await executor.invoke("read_file_lines", {"file_id": "fil_1"}, ctx)
        assert read_res.success is True
        assert json.loads(read_res.output)["content"] == "print('hi')"

        overview_res = await executor.invoke("get_file_overview", {"file_id": "fil_1"}, ctx)
        assert overview_res.success is True
        assert json.loads(overview_res.output)["filename"] == "app.py"

        with pytest.raises(AuthorizationError):
            await executor.invoke("get_file_overview", {"file_id": "fil_2"}, ctx)

        audits = await audit_repo.list_audits("prj_1", "ses_1")
        assert len(audits) == EXPECTED_AUDITS
        assert all(a.policy_decision is PolicyDecision.ALLOW for a in audits)
        assert [a.success for a in reversed(audits)] == [True, True, True, False]
        assert [a.tool_name for a in reversed(audits)] == [
            "search_files",
            "read_file_lines",
            "get_file_overview",
            "get_file_overview",
        ]
