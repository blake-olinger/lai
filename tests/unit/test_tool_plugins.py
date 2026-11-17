"""Unit tests for Phase 1 read tool plugins and cross-project/file-scope isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lai.domain.auth import ApiScope
from lai.domain.errors import AuthorizationError, NotFoundError, ValidationFailedError
from lai.domain.files import (
    CodeFileSummary,
    ExceptionFingerprint,
    FileChunk,
    FileKind,
    FileRecord,
    FileStatus,
    LogFileSummary,
    LogSeverity,
    ScoredChunk,
)
from lai.domain.tools import PermissionTier, ToolExecutionContext
from lai.ports.repositories import FileRepositoryPort
from lai.ports.storage import BlobStorePort, EmbedderPort, VectorStorePort
from lai.ports.tools import ToolPluginPort
from lai.tools import (
    DEFAULT_MAX_LINES_PER_CALL,
    GetFileOverviewPlugin,
    ReadFileLinesPlugin,
    SearchFilesPlugin,
    build_default_tool_registry,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

EXPECTED_THREE = 3
CLAMPED_LINE_COUNT = 250
SHORT_SPAN_END = 15
EXPECTED_TWO_RESULTS = 2


class _FakeFileRepo(FileRepositoryPort):
    def __init__(self) -> None:
        self.files: dict[str, FileRecord] = {}
        self.fts_results: list[ScoredChunk] = []
        self.fts_calls: list[tuple[str, str, tuple[str, ...] | None, int]] = []

    async def upsert_file(self, record: FileRecord) -> FileRecord:
        self.files[record.id] = record
        return record

    async def get_file(self, file_id: str) -> FileRecord | None:
        return self.files.get(file_id)

    async def list_files(
        self, project_id: str, limit: int = 200, offset: int = 0
    ) -> list[FileRecord]:
        del limit, offset
        return [f for f in self.files.values() if f.project_id == project_id]

    async def replace_chunks(self, file_id: str, chunks: Sequence[FileChunk]) -> None:
        del file_id, chunks

    async def list_chunks(
        self, file_id: str, start_line: int | None = None, end_line: int | None = None
    ) -> list[FileChunk]:
        del file_id, start_line, end_line
        return []

    async def search_fts(
        self,
        project_id: str,
        query: str,
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        self.fts_calls.append((project_id, query, tuple(file_ids) if file_ids else None, limit))
        return list(self.fts_results)

    async def delete_file(self, file_id: str) -> bool:
        return self.files.pop(file_id, None) is not None


class _FakeBlobStore(BlobStorePort):
    def __init__(self) -> None:
        self.blobs: dict[Path, list[str]] = {}
        self.read_calls: list[tuple[Path, int, int | None]] = []

    async def write_stream(
        self, project_id: str, file_id: str, filename: str, chunks: AsyncIterator[bytes]
    ) -> tuple[Path, int, str]:
        del project_id, file_id, filename, chunks
        return Path("unused"), 0, "0" * 64

    async def read_text_range(
        self, blob_path: Path, start_line: int = 1, end_line: int | None = None
    ) -> str:
        self.read_calls.append((blob_path, start_line, end_line))
        if blob_path not in self.blobs:
            raise FileNotFoundError(str(blob_path))
        lines = self.blobs[blob_path]
        upper = len(lines) if end_line is None else min(end_line, len(lines))
        return "\n".join(lines[start_line - 1 : upper])

    async def read_bytes(self, blob_path: Path) -> bytes:
        del blob_path
        return b""

    async def delete_blob(self, blob_path: Path) -> bool:
        return self.blobs.pop(blob_path, None) is not None


class _FakeEmbedder(EmbedderPort):
    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return EXPECTED_THREE

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        del query
        return [0.1, 0.2, 0.3]


class _FakeVectorStore(VectorStorePort):
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, tuple[str, ...] | None, int]] = []

    async def upsert_chunks(
        self, project_id: str, chunks: Sequence[FileChunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        del project_id, chunks, vectors

    async def search_vectors(
        self,
        project_id: str,
        query_vector: Sequence[float],
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        del query_vector
        self.calls.append((project_id, tuple(file_ids) if file_ids else None, limit))
        return list(self.results)

    async def delete_file_vectors(self, project_id: str, file_id: str) -> None:
        del project_id, file_id


def _file(file_id: str, project_id: str, filename: str, kind: FileKind) -> FileRecord:
    return FileRecord(
        id=file_id,
        project_id=project_id,
        filename=filename,
        size_bytes=128,
        sha256="a" * 64,
        blob_path=f"blobs/{project_id}/{file_id}.txt",
        kind=kind,
        status=FileStatus.READY,
    )


def _chunk(
    chunk_id: str, file_id: str, project_id: str, text: str, start_line: int = 1
) -> FileChunk:
    return FileChunk(
        id=chunk_id,
        file_id=file_id,
        project_id=project_id,
        chunk_index=0,
        start_line=start_line,
        end_line=start_line + 9,
        token_count=20,
        text=text,
    )


def _ctx(project_id: str = "prj_1", allowed_file_ids: tuple[str, ...] = ()) -> ToolExecutionContext:
    return ToolExecutionContext(
        project_id=project_id,
        session_id="ses_1",
        principal_id="usr_1",
        principal_scopes=(ApiScope.TOOLS_SAFE,),
        allowed_file_ids=allowed_file_ids,
    )


async def test_search_files_fts_hybrid_and_scoping() -> None:
    """SearchFilesPlugin enforces project/file scoping and fuses FTS + vector results."""
    repo = _FakeFileRepo()
    for rec in (
        _file("fil_1", "prj_1", "app.log", FileKind.LOG),
        _file("fil_2", "prj_1", "other.log", FileKind.LOG),
        _file("fil_3", "prj_2", "secret.log", FileKind.LOG),
    ):
        await repo.upsert_file(rec)

    c1 = _chunk("chk_1", "fil_1", "prj_1", "ERROR timeout", 1)
    c2 = _chunk("chk_2", "fil_1", "prj_1", "WARN retry", 11)
    repo.fts_results = [
        ScoredChunk(chunk=c1, score=0.9, source="fts"),
        ScoredChunk(chunk=_chunk("chk_3", "fil_2", "prj_1", "other"), score=0.8, source="fts"),
        ScoredChunk(chunk=_chunk("chk_4", "fil_3", "prj_2", "cross"), score=0.7, source="fts"),
    ]
    vec = _FakeVectorStore(
        [
            ScoredChunk(chunk=c1, score=0.95, source="vector"),
            ScoredChunk(chunk=c2, score=0.6, source="vector"),
        ]
    )

    fts_only = SearchFilesPlugin(repo)
    assert isinstance(fts_only, ToolPluginPort)
    fts_res = await fts_only.execute({"query": "timeout", "limit": 10}, _ctx("prj_1", ("fil_1",)))
    fts_data = json.loads(fts_res.output)
    assert repo.fts_calls[-1] == ("prj_1", "timeout", ("fil_1",), 10)
    assert (fts_data["count"], fts_data["results"][0]["filename"]) == (1, "app.log")

    hybrid = SearchFilesPlugin(repo, embedder=_FakeEmbedder(), vector_store=vec)
    hyb_res = await hybrid.execute({"query": "timeout"}, _ctx("prj_1", ("fil_1",)))
    hyb_data = json.loads(hyb_res.output)
    assert vec.calls[-1] == ("prj_1", ("fil_1",), 20)
    assert hyb_data["count"] == EXPECTED_TWO_RESULTS
    assert hyb_data["results"][0]["chunk_id"] == "chk_1"
    assert hyb_data["results"][0]["source"] == "hybrid"
    assert hyb_data["results"][1]["source"] == "vector"

    with pytest.raises(AuthorizationError):
        await hybrid.execute({"query": "timeout", "file_ids": ["fil_2"]}, _ctx("prj_1", ("fil_1",)))
    for bad_args in ({"query": "   "}, {"query": "ok", "limit": 0}):
        with pytest.raises(ValidationFailedError):
            await hybrid.execute(bad_args, _ctx("prj_1"))


async def test_read_file_lines_clamping_and_isolation() -> None:
    """ReadFileLinesPlugin clamps spans to 250 lines and blocks cross-project/scope reads."""
    repo, blob = _FakeFileRepo(), _FakeBlobStore()
    f1 = _file("fil_1", "prj_1", "big.log", FileKind.LOG)
    f_other = _file("fil_2", "prj_2", "secret.py", FileKind.CODE)
    f_missing = _file("fil_3", "prj_1", "gone.log", FileKind.LOG)
    for rec in (f1, f_other, f_missing):
        await repo.upsert_file(rec)
    blob.blobs[Path(f1.blob_path)] = [f"line {i}" for i in range(1, 401)]
    blob.blobs[Path(f_other.blob_path)] = ["secret"]

    plugin = ReadFileLinesPlugin(repo, blob)
    clamped = await plugin.execute(
        {"file_id": "fil_1", "start_line": 1, "end_line": 500}, _ctx("prj_1", ("fil_1",))
    )
    assert clamped.truncated is True
    assert blob.read_calls[-1] == (Path(f1.blob_path), 1, DEFAULT_MAX_LINES_PER_CALL)
    clamped_data = json.loads(clamped.output)
    assert clamped_data["end_line"] == CLAMPED_LINE_COUNT
    assert len(clamped_data["content"].splitlines()) == CLAMPED_LINE_COUNT

    short = await plugin.execute(
        {"file_id": "fil_1", "start_line": 5, "end_line": SHORT_SPAN_END}, _ctx("prj_1")
    )
    assert short.truncated is False
    assert json.loads(short.output)["content"].splitlines()[0] == "line 5"

    for bad_args, ctx in (
        ({"file_id": "fil_2"}, _ctx("prj_1")),
        ({"file_id": "fil_1"}, _ctx("prj_1", ("fil_other",))),
    ):
        with pytest.raises(AuthorizationError):
            await plugin.execute(bad_args, ctx)
    for missing_id in ("fil_none", "fil_3"):
        with pytest.raises(NotFoundError):
            await plugin.execute({"file_id": missing_id}, _ctx("prj_1"))
    for bad_span in (
        {"file_id": "fil_1", "start_line": 0},
        {"file_id": "fil_1", "start_line": 10, "end_line": 5},
    ):
        with pytest.raises(ValidationFailedError):
            await plugin.execute(bad_span, _ctx("prj_1"))


async def test_get_file_overview_and_default_registry() -> None:
    """GetFileOverviewPlugin returns summaries and default registry exposes READ tools."""
    repo, blob = _FakeFileRepo(), _FakeBlobStore()
    fp = ExceptionFingerprint(
        fingerprint="fp1",
        exception_type="TimeoutError",
        message_sample="timed out",
        count=3,
        first_line=12,
        last_line=99,
    )
    log_rec = _file("fil_log", "prj_1", "srv.log", FileKind.LOG).model_copy(
        update={
            "log_summary": LogFileSummary(
                total_lines=100,
                severity_counts={LogSeverity.ERROR: 3},
                error_count=3,
                top_exceptions=(fp,),
            )
        }
    )
    code_rec = _file("fil_code", "prj_1", "main.py", FileKind.CODE).model_copy(
        update={
            "code_summary": CodeFileSummary(
                language="python", total_lines=40, symbols=("main",), imports=("json",)
            )
        }
    )
    for rec in (log_rec, code_rec, _file("fil_other", "prj_2", "other.py", FileKind.CODE)):
        await repo.upsert_file(rec)

    overview = GetFileOverviewPlugin(repo)
    log_out = json.loads((await overview.execute({"file_id": "fil_log"}, _ctx("prj_1"))).output)
    assert (log_out["log_summary"]["error_count"], log_out["code_summary"]) == (
        EXPECTED_THREE,
        None,
    )
    code_out = json.loads((await overview.execute({"file_id": "fil_code"}, _ctx("prj_1"))).output)
    assert (code_out["code_summary"]["language"], code_out["code_summary"]["symbols"]) == (
        "python",
        ["main"],
    )

    for bad_args, ctx in (
        ({"file_id": "fil_other"}, _ctx("prj_1")),
        ({"file_id": "fil_log"}, _ctx("prj_1", ("fil_code",))),
    ):
        with pytest.raises(AuthorizationError):
            await overview.execute(bad_args, ctx)
    with pytest.raises(NotFoundError):
        await overview.execute({"file_id": "fil_missing"}, _ctx("prj_1"))

    registry = build_default_tool_registry(repo, blob)
    specs = registry.list_specs()
    assert {s.name for s in specs} == {"search_files", "read_file_lines", "get_file_overview"}
    assert all(
        s.permission_tier is PermissionTier.READ and not s.requires_approval_by_default
        for s in specs
    )
