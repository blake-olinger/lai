"""End-to-end tests wiring GuardedToolExecutor with Phase 1 read plugins."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lai.domain.auth import ApiScope
from lai.domain.errors import AuthorizationError
from lai.domain.files import FileChunk, FileKind, FileRecord, FileStatus, ScoredChunk
from lai.domain.tools import (
    ApprovalStatus,
    PolicyDecision,
    ToolApprovalRecord,
    ToolAuditRecord,
    ToolExecutionContext,
)
from lai.ports.repositories import FileRepositoryPort, ToolAuditRepositoryPort
from lai.ports.storage import BlobStorePort
from lai.tools import GuardedToolExecutor, build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

EXPECTED_AUDITS = 4


class _Repo(FileRepositoryPort):
    def __init__(self, files: dict[str, FileRecord]) -> None:
        self.files = files

    async def upsert_file(self, record: FileRecord) -> FileRecord:
        self.files[record.id] = record
        return record

    async def get_file(self, file_id: str) -> FileRecord | None:
        return self.files.get(file_id)

    async def list_files(
        self, project_id: str, limit: int = 200, offset: int = 0
    ) -> list[FileRecord]:
        del project_id, limit, offset
        return list(self.files.values())

    async def replace_chunks(self, file_id: str, chunks: Sequence[FileChunk]) -> None:
        del file_id, chunks

    async def list_chunks(
        self, file_id: str, start_line: int | None = None, end_line: int | None = None
    ) -> list[FileChunk]:
        del file_id, start_line, end_line
        return []

    async def search_fts(
        self, project_id: str, query: str, file_ids: Sequence[str] | None = None, limit: int = 20
    ) -> list[ScoredChunk]:
        del project_id, query, file_ids, limit
        return []

    async def delete_file(self, file_id: str) -> bool:
        return self.files.pop(file_id, None) is not None


class _Blobs(BlobStorePort):
    def __init__(self, blobs: dict[Path, str]) -> None:
        self.blobs = blobs

    async def write_stream(
        self, project_id: str, file_id: str, filename: str, chunks: AsyncIterator[bytes]
    ) -> tuple[Path, int, str]:
        del project_id, file_id, filename, chunks
        return Path("unused"), 0, "0" * 64

    async def read_text_range(
        self, blob_path: Path, start_line: int = 1, end_line: int | None = None
    ) -> str:
        del start_line, end_line
        return self.blobs[blob_path]

    async def read_bytes(self, blob_path: Path) -> bytes:
        del blob_path
        return b""

    async def delete_blob(self, blob_path: Path) -> bool:
        return self.blobs.pop(blob_path, None) is not None


class _Audits(ToolAuditRepositoryPort):
    def __init__(self) -> None:
        self.audits: list[ToolAuditRecord] = []

    async def create_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        return record

    async def get_approval(self, approval_id: str) -> ToolApprovalRecord | None:
        del approval_id
        return None

    async def list_pending_approvals(
        self, project_id: str, session_id: str | None = None
    ) -> list[ToolApprovalRecord]:
        del project_id, session_id
        return []

    async def update_approval(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        return record

    async def transition_approval_status(
        self, record: ToolApprovalRecord, *, expected_status: ApprovalStatus
    ) -> bool:
        del record, expected_status
        return True

    async def record_audit(self, record: ToolAuditRecord) -> ToolAuditRecord:
        self.audits.append(record)
        return record

    async def list_audits(
        self, project_id: str, session_id: str | None = None, limit: int = 100
    ) -> list[ToolAuditRecord]:
        del project_id, session_id, limit
        return list(self.audits)


async def test_executor_invokes_all_phase1_plugins_and_audits_isolation() -> None:
    """GuardedToolExecutor runs Phase 1 plugins under READ scopes and audits failures."""
    f_ok = FileRecord(
        id="fil_1",
        project_id="prj_1",
        filename="app.py",
        size_bytes=10,
        sha256="a" * 64,
        blob_path="blobs/app.py",
        kind=FileKind.CODE,
        status=FileStatus.READY,
    )
    f_other = f_ok.model_copy(update={"id": "fil_2", "project_id": "prj_2"})
    registry = build_default_tool_registry(
        _Repo({"fil_1": f_ok, "fil_2": f_other}),
        _Blobs({Path("blobs/app.py"): "print('hi')"}),
    )
    audits = _Audits()
    executor = GuardedToolExecutor(registry, audits)
    ctx = ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id="usr_1",
        principal_scopes=(ApiScope.TOOLS_SAFE,),
    )

    for tool_name, args in (
        ("search_files", {"query": "hi"}),
        ("read_file_lines", {"file_id": "fil_1"}),
        ("get_file_overview", {"file_id": "fil_1"}),
    ):
        result = await executor.invoke(tool_name, args, ctx)
        assert result.success is True

    with pytest.raises(AuthorizationError):
        await executor.invoke("get_file_overview", {"file_id": "fil_2"}, ctx)

    assert len(audits.audits) == EXPECTED_AUDITS
    assert all(a.policy_decision is PolicyDecision.ALLOW for a in audits.audits)
    assert [a.success for a in audits.audits] == [True, True, True, False]
