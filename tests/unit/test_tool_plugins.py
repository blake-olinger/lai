"""Unit tests for Phase 1 read tool plugins and cross-project/file-scope isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lai.domain.auth import ApiScope
from lai.domain.errors import AuthorizationError, NotFoundError, ValidationFailedError
from lai.domain.files import (
    FileChunk,
    FileKind,
    FileRecord,
    FileStatus,
    ScoredChunk,
)
from lai.domain.tools import ToolExecutionContext
from lai.ingestion.embedder import DeterministicHashEmbedder
from lai.ports.repositories import FileRepositoryPort
from lai.ports.storage import BlobStorePort, VectorStorePort
from lai.ports.tools import ToolPluginPort
from lai.tools import (
    DEFAULT_MAX_LINES_PER_CALL,
    ReadFileLinesPlugin,
    SearchFilesPlugin,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

EXPECTED_THREE = 3
CLAMPED_LINE_COUNT = 250
SHORT_SPAN_END = 15
EXPECTED_TWO_RESULTS = 2
DEFAULT_VECTOR_LIMIT = 20


class _RecordingFileRepo(FileRepositoryPort):
    def __init__(self) -> None:
        self.files: dict[str, FileRecord] = {}
        self.chunks_by_file: dict[str, list[FileChunk]] = {}
        self.fts_results: list[ScoredChunk] = []
        self.fts_calls: list[tuple[str, str, tuple[str, ...] | None, int]] = []
        self.list_calls: list[tuple[str, int, int]] = []

    async def upsert_file(self, record: FileRecord) -> FileRecord:
        self.files[record.id] = record
        return record

    async def get_file(self, file_id: str) -> FileRecord | None:
        return self.files.get(file_id)

    async def list_files(
        self, project_id: str, limit: int = 200, offset: int = 0
    ) -> list[FileRecord]:
        self.list_calls.append((project_id, limit, offset))
        return [f for f in self.files.values() if f.project_id == project_id][
            offset : offset + limit
        ]

    async def replace_chunks(self, file_id: str, chunks: Sequence[FileChunk]) -> None:
        self.chunks_by_file[file_id] = list(chunks)

    async def list_chunks(
        self,
        file_id: str,
        start_line: int | None = None,
        end_line: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[FileChunk]:
        matched = [
            c
            for c in self.chunks_by_file.get(file_id, [])
            if (start_line is None or c.end_line >= start_line)
            and (end_line is None or c.start_line <= end_line)
        ]
        return matched[offset : offset + limit] if limit is not None else matched[offset:]

    async def search_fts(
        self, project_id: str, query: str, file_ids: Sequence[str] | None = None, limit: int = 20
    ) -> list[ScoredChunk]:
        self.fts_calls.append((project_id, query, tuple(file_ids) if file_ids else None, limit))
        return list(self.fts_results)

    async def delete_file(self, file_id: str) -> bool:
        self.chunks_by_file.pop(file_id, None)
        return self.files.pop(file_id, None) is not None

    async def sum_bytes_for_owner(self, owner_principal_id: str) -> int:
        return sum(f.size_bytes for f in self.files.values() if owner_principal_id)


class _RecordingBlobStore(BlobStorePort):
    def __init__(self) -> None:
        self.blobs: dict[Path, list[str]] = {}
        self.read_calls: list[tuple[Path, int, int | None]] = []

    async def write_stream(
        self, project_id: str, file_id: str, filename: str, chunks: AsyncIterator[bytes]
    ) -> tuple[Path, int, str]:
        data = b"".join([chunk async for chunk in chunks])
        path = Path("blobs") / project_id / f"{file_id}-{filename}"
        self.blobs[path] = data.decode("utf-8").splitlines()
        return path, len(data), "a" * 64

    async def read_text_range(
        self,
        blob_path: Path,
        start_line: int = 1,
        end_line: int | None = None,
        *,
        max_chars: int | None = None,
    ) -> str:
        self.read_calls.append((blob_path, start_line, end_line))
        if blob_path not in self.blobs:
            raise FileNotFoundError(str(blob_path))
        lines = self.blobs[blob_path]
        upper = len(lines) if end_line is None else min(end_line, len(lines))
        text = "\n".join(lines[start_line - 1 : upper])
        return text[:max_chars] if max_chars is not None else text

    async def read_bytes(self, blob_path: Path) -> bytes:
        if blob_path not in self.blobs:
            raise FileNotFoundError(str(blob_path))
        return "\n".join(self.blobs[blob_path]).encode("utf-8")

    async def delete_blob(self, blob_path: Path) -> bool:
        return self.blobs.pop(blob_path, None) is not None


class _RecordingVectorStore(VectorStorePort):
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[float], tuple[str, ...] | None, int]] = []
        self.upserts: list[tuple[str, int, int]] = []
        self.deletes: list[tuple[str, str]] = []

    async def upsert_chunks(
        self, project_id: str, chunks: Sequence[FileChunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        self.upserts.append((project_id, len(chunks), len(vectors)))

    async def search_vectors(
        self,
        project_id: str,
        query_vector: Sequence[float],
        file_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[ScoredChunk]:
        self.calls.append(
            (project_id, list(query_vector), tuple(file_ids) if file_ids else None, limit)
        )
        return list(self.results)

    async def delete_file_vectors(self, project_id: str, file_id: str) -> None:
        self.deletes.append((project_id, file_id))


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
    repo = _RecordingFileRepo()
    for rec in (
        _file("fil_1", "prj_1", "app.log", FileKind.LOG),
        _file("fil_2", "prj_1", "other.log", FileKind.LOG),
        _file("fil_3", "prj_2", "secret.log", FileKind.LOG),
    ):
        await repo.upsert_file(rec)

    c1 = _chunk("chk_1", "fil_1", "prj_1", "ERROR timeout", 1)
    c2 = _chunk("chk_2", "fil_1", "prj_1", "WARN retry", 11)
    spoofed_cross = _chunk("chk_spoof", "fil_3", "prj_1", "spoofed project chunk", 1)
    repo.fts_results = [
        ScoredChunk(chunk=c1, score=0.9, source="fts"),
        ScoredChunk(chunk=_chunk("chk_3", "fil_2", "prj_1", "other"), score=0.8, source="fts"),
        ScoredChunk(chunk=_chunk("chk_4", "fil_3", "prj_2", "cross"), score=0.7, source="fts"),
        ScoredChunk(chunk=spoofed_cross, score=0.6, source="fts"),
    ]
    embedder = DeterministicHashEmbedder(dimensions=EXPECTED_THREE)
    vec = _RecordingVectorStore(
        [
            ScoredChunk(chunk=c1, score=0.95, source="vector"),
            ScoredChunk(chunk=c2, score=0.6, source="vector"),
        ]
    )

    fts_only = SearchFilesPlugin(repo)
    assert isinstance(fts_only, ToolPluginPort)
    fts_res = await fts_only.execute(
        {"query": "timeout", "file_ids": ["fil_1"], "limit": 10}, _ctx("prj_1")
    )
    fts_data = json.loads(fts_res.output)
    assert repo.fts_calls[-1] == ("prj_1", "timeout", ("fil_1",), 10)
    assert (fts_data["count"], fts_data["results"][0]["filename"]) == (1, "app.log")

    # Verify spoofed project_id chunk matching fil_3 (owned by prj_2) is dropped when unrestricted
    unrestricted = await fts_only.execute({"query": "timeout"}, _ctx("prj_1"))
    assert all(r["file_id"] != "fil_3" for r in json.loads(unrestricted.output)["results"])

    hybrid = SearchFilesPlugin(repo, embedder=embedder, vector_store=vec)
    hyb_res = await hybrid.execute({"query": "timeout"}, _ctx("prj_1", ("fil_1",)))
    hyb_data = json.loads(hyb_res.output)
    expected_vec = await embedder.embed_query("timeout")
    assert vec.calls[-1] == ("prj_1", expected_vec, ("fil_1",), DEFAULT_VECTOR_LIMIT)
    first = hyb_data["results"][0]
    assert hyb_data["count"] == EXPECTED_TWO_RESULTS
    assert (first["chunk_id"], first["source"]) == ("chk_1", "hybrid")
    assert hyb_data["results"][1]["source"] == "vector"

    with pytest.raises(AuthorizationError):
        await hybrid.execute({"query": "timeout", "file_ids": ["fil_2"]}, _ctx("prj_1", ("fil_1",)))
    bad_arg_cases: tuple[dict[str, object], ...] = (
        {"query": "   "},
        {"query": "ok", "limit": 0},
        {"query": "ok", "file_ids": "fil_1"},
        {"query": "ok", "file_ids": ["   "]},
    )
    for bad_args in bad_arg_cases:
        with pytest.raises(ValidationFailedError):
            await hybrid.execute(bad_args, _ctx("prj_1"))


async def test_read_file_lines_clamping_and_isolation() -> None:
    """ReadFileLinesPlugin clamps spans to 250 lines and blocks cross-project/scope reads."""
    repo, blob = _RecordingFileRepo(), _RecordingBlobStore()
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
    assert blob.read_calls[-1] == (Path(f1.blob_path), 5, SHORT_SPAN_END)
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


RecordingFileRepo = _RecordingFileRepo
RecordingBlobStore = _RecordingBlobStore
make_file_record = _file
make_tool_ctx = _ctx
