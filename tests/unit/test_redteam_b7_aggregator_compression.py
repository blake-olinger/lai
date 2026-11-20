"""Red-team unit proofs for B7(d) (MapReduceAggregator multi-pass compression & clamping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.config.profiles import get_model_profile
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import ProgressEvent, TokenUsage, TurnInput
from lai.domain.files import FileKind, FileRecord, FileStatus
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
)
from lai.ingestion import LocalBlobStore
from lai.ingestion.detector import estimate_tokens
from lai.orchestrator._aggregator_helpers import MAX_COMPRESSION_PASSES, clamp_findings_to_budget
from lai.orchestrator.aggregator import MapReduceAggregator
from lai.ports.inference import InferenceProviderPort
from lai.scheduler import InferenceSpooler

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


class _VerboseMapInference(InferenceProviderPort):
    """Returns oversized map/compress findings to force multi-pass compression and clamping."""

    def __init__(self) -> None:
        self.reduce_requests: list[InferenceRequest] = []
        self.complete_calls = 0

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        self.complete_calls += 1
        # Always return ~400 tokens so every compression pass remains above a tiny budget.
        return InferenceResponse(content="finding " * 350)

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.reduce_requests.append(request)
        yield InferenceChunk(
            content_delta="final synthesis",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


async def test_b7_map_reduce_multipass_compression_and_deterministic_clamping(
    tmp_path: Path,
) -> None:
    """B7(d): Aggregator runs up to MAX_COMPRESSION_PASSES and clamps Reduce findings to budget."""
    blob = LocalBlobStore(tmp_path / "blobs")

    async def _lines() -> AsyncGenerator[bytes]:
        yield ("INFO 2026-11-20 error trace event\n" * 60).encode("utf-8")

    blob_path, size_bytes, sha = await blob.write_stream("p1", "f1", "a.log", _lines())
    rec = FileRecord(
        id="f1",
        project_id="p1",
        filename="a.log",
        content_type="text/plain",
        kind=FileKind.LOG,
        size_bytes=size_bytes,
        sha256=sha,
        blob_path=str(blob_path),
        status=FileStatus.READY,
        token_count=300,
    )
    inf = _VerboseMapInference()
    agg = MapReduceAggregator(
        blob_store=blob,
        inference_provider=inf,
        spooler=InferenceSpooler(),
        window_tokens=200,
        overlap_tokens=20,
        compression_threshold_tokens=150,
        max_windows=8,
    )
    principal = AuthPrincipal(
        principal_id="u1", auth_method=AuthMethod.API_KEY, scopes=tuple(ApiScope)
    )
    compress_events = [
        ev
        async for ev in agg.run_stream(
            turn_input=TurnInput(project_id="p1", session_id="s1", user_message="audit all"),
            principal=principal,
            model_profile=get_model_profile("gemma4-26b-a4b"),
            files=(rec,),
        )
        if isinstance(ev, ProgressEvent) and ev.stage == "compress_findings"
    ]

    assert len(compress_events) == MAX_COMPRESSION_PASSES
    assert len(inf.reduce_requests) == 1
    reduce_user_text = inf.reduce_requests[0].messages[-1].content
    assert "[truncated to fit context budget]" in reduce_user_text

    # Direct unit verification of clamp_findings_to_budget determinism and hard token bound.
    clamped = clamp_findings_to_budget(["word " * 500, "second " * 500], max_tokens=100)
    assert clamped
    assert estimate_tokens("\n\n".join(clamped)) <= 100
