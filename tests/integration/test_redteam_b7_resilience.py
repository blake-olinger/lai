"""Red-team integration proofs for B7 (upload worker thread, READY gating, bounds, 413)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lai.config.profiles import get_model_profile
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import ContextMode, TokenUsage, TurnInput
from lai.domain.errors import ConflictError, ValidationFailedError
from lai.domain.files import FileKind, FileRecord, FileStatus
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
)
from lai.domain.observability import LoggingConfig
from lai.domain.tools import ToolExecutionContext
from lai.ingestion import LocalBlobStore
from lai.ingestion import pipeline as ingestion_pipeline
from lai.orchestrator._direct import DirectDeps, run_direct_stream
from lai.orchestrator._helpers import resolve_target_files
from lai.orchestrator.aggregator import MapReduceAggregator
from lai.orchestrator.classifier import classify_context_mode
from lai.ports.inference import InferenceProviderPort
from lai.scheduler import InferenceSpooler
from lai.tools.plugins.read_lines import DEFAULT_MAX_CHARS_PER_CALL, ReadFileLinesPlugin
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADMIN = tuple(ApiScope)


class _SimpleInference(InferenceProviderPort):
    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="ok")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        yield InferenceChunk(content_delta="ok", finish_reason="stop", usage=TokenUsage())

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _CountingBlob(LocalBlobStore):
    def __init__(self, blobs_dir: Path) -> None:
        super().__init__(blobs_dir)
        self.reads: list[Path] = []

    async def read_text_range(
        self,
        blob_path: Path,
        start_line: int = 1,
        end_line: int | None = None,
        *,
        max_chars: int | None = None,
    ) -> str:
        self.reads.append(blob_path)
        return await super().read_text_range(
            blob_path, start_line=start_line, end_line=end_line, max_chars=max_chars
        )


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_b7_upload_worker_thread_pending_exclusion_direct_cap_and_read_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B7(a)+(b)+(c)+(e)+(h): Off-loop indexing, READY gating, direct cap, 32KiB read_lines."""
    nginx = (_REPO_ROOT / "deploy" / "azure" / "nginx-lai.conf.example").read_text(encoding="utf-8")
    assert "client_max_body_size 2M;" in nginx
    assert "location /v1/files" in nginx
    assert "client_max_body_size 250M;" in nginx

    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    async with open_services_harness(settings, IntegrationSpec(use_hash_embedder=True)) as h:
        orig_decode = ingestion_pipeline._decode_and_chunk  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        def _slow_decode(record: FileRecord, raw_bytes: bytes, target: int) -> object:
            time.sleep(0.15)
            return orig_decode(record, raw_bytes, target)

        monkeypatch.setattr(ingestion_pipeline, "_decode_and_chunk", _slow_decode)
        headers = await mint_auth_headers(h.svc, "admin", _ADMIN)
        prj = await create_project(h.client, headers, "p-b7")

        ticks = 0

        async def _probe() -> None:
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.02)
                ticks += 1

        up_task = asyncio.create_task(
            h.client.post(
                "/v1/files",
                headers=headers,
                data={"project_id": prj, "wait": "true"},
                files={"file": ("app.log", b"INFO line 1\nERROR line 2\n" * 50, "text/plain")},
            )
        )
        await _probe()
        up_resp = await up_task
        assert up_resp.status_code == HTTPStatus.CREATED
        assert ticks == 5

        pending = FileRecord(
            id="fil_pending_1",
            project_id=prj,
            filename="big.log",
            content_type="text/plain",
            kind=FileKind.LOG,
            size_bytes=50_000_000,
            sha256="b" * 64,
            blob_path="blobs/missing",
            status=FileStatus.PENDING,
            token_count=0,
        )
        await h.svc.file_repo.upsert_file(pending)
        fallback = await resolve_target_files(h.svc.file_repo, prj, ())
        assert all(f.id != pending.id for f in fallback)
        with pytest.raises(ConflictError) as exc_409:
            await resolve_target_files(h.svc.file_repo, prj, (pending.id,))
        assert exc_409.value.code == "file_not_ready"

        direct_blob = _CountingBlob(tmp_path / "blobs")
        huge_ready = pending.model_copy(
            update={"id": "fil_huge", "status": FileStatus.READY, "token_count": 200_000}
        )
        profile = get_model_profile(settings.default_model_profile)
        principal = AuthPrincipal(principal_id="u1", auth_method=AuthMethod.API_KEY, scopes=_ADMIN)
        with pytest.raises(ValidationFailedError):
            async for _ in run_direct_stream(
                DirectDeps(
                    file_repo=h.svc.file_repo,
                    blob_store=direct_blob,
                    inference_provider=_SimpleInference(),
                    spooler=InferenceSpooler(),
                ),
                turn_input=TurnInput(project_id=prj, session_id="s1", user_message="read it"),
                principal=principal,
                profile=profile,
                target_files=(huge_ready,),
                existing_messages=(),
                session_system_prompt=None,
            ):
                pass
        assert direct_blob.reads == []

        async def _five_mb() -> AsyncGenerator[bytes]:
            yield b"x" * (5 * 1024 * 1024)

        min_blob, min_size, min_sha = await h.svc.blob_store.write_stream(
            prj, "fil_min", "minified.js", _five_mb()
        )
        min_rec = FileRecord(
            id="fil_min",
            project_id=prj,
            filename="minified.js",
            content_type="application/javascript",
            kind=FileKind.CODE,
            size_bytes=min_size,
            sha256=min_sha,
            blob_path=str(min_blob),
            status=FileStatus.READY,
            token_count=500_000,
        )
        await h.svc.file_repo.upsert_file(min_rec)
        plugin = ReadFileLinesPlugin(h.svc.file_repo, h.svc.blob_store)
        res = await plugin.execute(
            {"file_id": min_rec.id, "start_line": 1, "end_line": 5},
            ToolExecutionContext(
                principal_id="u1", project_id=prj, session_id="s1", request_id="r1"
            ),
        )
        out = json.loads(res.output)
        assert out["truncated"] is True
        assert len(out["content"]) <= DEFAULT_MAX_CHARS_PER_CALL


async def test_b7_map_reduce_caps_chunk_pagination_and_json_body_limit(tmp_path: Path) -> None:
    """B7(d)+(f)+(g): Map-Reduce window cap, classifier fallback, chunk pagination, 413 cap."""
    f1 = FileRecord(
        id="f1",
        project_id="p1",
        filename="a.log",
        content_type="text/plain",
        kind=FileKind.LOG,
        size_bytes=100,
        sha256="a" * 64,
        blob_path="k1",
        status=FileStatus.READY,
        token_count=500_000,
    )
    f2 = f1.model_copy(update={"id": "f2", "filename": "b.log", "token_count": 50_000})
    assert (
        classify_context_mode(
            ContextMode.AUTO,
            "summarize all errors",
            (f1,),
            direct_budget_tokens=12_000,
            max_map_reduce_tokens=100_000,
        )
        is ContextMode.RETRIEVAL
    )
    assert (
        classify_context_mode(
            ContextMode.AUTO,
            "give me an overview",
            (f2, f2.model_copy(update={"id": "f3"})),
            direct_budget_tokens=12_000,
            explicit_files=False,
        )
        is ContextMode.RETRIEVAL
    )

    blob = _CountingBlob(tmp_path / "blobs")

    async def _big_lines() -> AsyncGenerator[bytes]:
        yield ("line\n" * 4000).encode()

    p1, _, _ = await blob.write_stream("p1", "f1", "a.log", _big_lines())
    p2, _, _ = await blob.write_stream("p1", "f2", "b.log", _big_lines())

    agg = MapReduceAggregator(
        blob_store=blob,
        inference_provider=_SimpleInference(),
        spooler=InferenceSpooler(),
        window_tokens=100,
        overlap_tokens=10,
        max_windows=2,
    )
    principal = AuthPrincipal(principal_id="u1", auth_method=AuthMethod.API_KEY, scopes=_ADMIN)
    with pytest.raises(ValidationFailedError):
        async for _ in agg.run_stream(
            turn_input=TurnInput(project_id="p1", session_id="s1", user_message="audit all"),
            principal=principal,
            model_profile=get_model_profile("gemma4-26b-a4b"),
            files=(
                f1.model_copy(update={"token_count": 300, "blob_path": str(p1)}),
                f2.model_copy(update={"token_count": 300, "blob_path": str(p2)}),
            ),
        ):
            pass
    assert p1 in blob.reads
    assert p2 not in blob.reads

    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
        max_json_body_bytes=4096,
        max_upload_bytes=65_536,
    )
    async with open_services_harness(settings, IntegrationSpec(use_hash_embedder=True)) as h:
        oversized = b'{"model":"x","messages":[{"role":"user","content":"' + (b"A" * 5000) + b'"}]}'
        r_cl = await h.client.post(
            "/v1/chat/completions", content=oversized, headers={"Content-Type": "application/json"}
        )
        assert r_cl.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

        async def _chunked() -> AsyncGenerator[bytes]:
            yield oversized[:2500]
            yield oversized[2500:]

        r_st = await h.client.post(
            "/v1/chat/completions", content=_chunked(), headers={"Content-Type": "application/json"}
        )
        assert r_st.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

        up_chunks = 0

        async def _chunked_upload() -> AsyncGenerator[bytes]:
            nonlocal up_chunks
            yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
            for _ in range(15):
                up_chunks += 1
                yield b"A" * 100_000
            yield b"\r\n--b--\r\n"

        r_up = await h.client.post(
            "/v1/files",
            content=_chunked_upload(),
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
        assert r_up.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert up_chunks < 15

        headers = await mint_auth_headers(h.svc, "admin", _ADMIN)
        prj = await create_project(h.client, headers, "p-chunks")
        big_txt = "\n".join(f"line {i} " + ("word " * 80) for i in range(40)).encode()
        up = await h.client.post(
            "/v1/files",
            headers=headers,
            data={"project_id": prj, "wait": "true"},
            files={"file": ("multi.txt", big_txt, "text/plain")},
        )
        fid = up.json()["id"]
        all_chunks = (await h.client.get(f"/v1/files/{fid}/chunks", headers=headers)).json()[
            "items"
        ]
        assert len(all_chunks) >= 3
        page = (
            await h.client.get(f"/v1/files/{fid}/chunks?limit=2&offset=1", headers=headers)
        ).json()
        assert page["count"] == 2
        assert [c["id"] for c in page["items"]] == [all_chunks[1]["id"], all_chunks[2]["id"]]
        offset_only = await h.svc.file_repo.list_chunks(fid, limit=None, offset=1)
        assert [c.id for c in offset_only] == [c["id"] for c in all_chunks[1:]]
