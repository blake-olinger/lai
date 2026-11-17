"""Edge-case integration tests for RRF retrieval, approvals, loop cap, and interleaving."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from lai.config.settings import LaiSettings
from lai.db import (
    SqliteDatabase,
    SqliteFileRepository,
    SqliteProjectRepository,
    SqliteSessionRepository,
    SqliteToolAuditRepository,
)
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import (
    ApprovalRequiredEvent,
    ContextMode,
    ProgressEvent,
    Session,
    TokenUsage,
    TurnCompletedEvent,
    TurnInput,
)
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.projects import Project
from lai.domain.scheduler import JobPriority
from lai.domain.tools import PermissionTier, ToolExecutionContext, ToolResult, ToolSpec
from lai.guardrails import AllowAllGuardrailProvider, CompositeGuardrailPipeline
from lai.ingestion import (
    DeterministicHashEmbedder,
    LanceDbVectorStore,
    LocalBlobStore,
    LocalIngestionPipeline,
)
from lai.observability import PromptAuditLogger
from lai.orchestrator import TurnOrchestrator
from lai.orchestrator._retrieval import MAX_TOOL_ITERATIONS
from lai.ports.inference import InferenceProviderPort
from lai.ports.tools import ToolPluginPort
from lai.scheduler import InferenceSpooler, SessionGate
from lai.tools import GuardedToolExecutor, build_default_tool_registry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
    from pathlib import Path

    from lai.domain.files import FileRecord

EXPECTED_LOOP_LEASES = MAX_TOOL_ITERATIONS + 1
LARGE_FINDING_REPEATS = 5500


class _ApprovalPlugin(ToolPluginPort):
    def __init__(self) -> None:
        self._spec = ToolSpec(
            name="deploy_patch",
            description="Requires human approval",
            permission_tier=PermissionTier.READ,
            parameters_schema={"type": "object", "properties": {"target": {"type": "string"}}},
            requires_approval_by_default=True,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(tool_name=self._spec.name, success=True, output="ok")


class _AsyncInference(InferenceProviderPort):
    def __init__(
        self,
        *,
        complete_fn: Callable[[InferenceRequest], Any],
        stream_fn: Callable[[InferenceRequest], Sequence[InferenceChunk]],
    ) -> None:
        self._complete_fn, self._stream_fn = complete_fn, stream_fn

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        res = self._complete_fn(request)
        if asyncio.iscoroutine(res):
            return await res  # type: ignore[no-any-return]
        return res  # type: ignore[no-any-return]

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        for chunk in self._stream_fn(request):
            await asyncio.sleep(0)
            yield chunk

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


async def _ingest_oversized(
    tmp_path: Path,
    db: SqliteDatabase,
    filename: str,
    text: str,
) -> tuple[
    FileRecord, SqliteFileRepository, LocalBlobStore, LanceDbVectorStore, DeterministicHashEmbedder
]:
    blobs = LocalBlobStore(tmp_path / "blobs")
    file_repo = SqliteFileRepository(db)
    embedder = DeterministicHashEmbedder()
    vec_store = LanceDbVectorStore(tmp_path / "lancedb")
    pipe = LocalIngestionPipeline(
        blob_store=blobs, file_repo=file_repo, embedder=embedder, vector_store=vec_store
    )

    async def _gen() -> AsyncIterator[bytes]:
        yield text.encode("utf-8")

    rec = await pipe.ingest_upload("prj_1", filename, "text/plain", _gen())
    oversized = rec.model_copy(update={"token_count": 5000})
    await file_repo.upsert_file(oversized)
    return oversized, file_repo, blobs, vec_store, embedder


async def test_retrieval_rrf_approval_and_bounded_tool_loop(tmp_path: Path) -> None:
    """Retrieval fuses LanceDB hits, pauses on ApprovalRequiredEvent, and caps at 4 loops."""
    db = SqliteDatabase(tmp_path / "lai.db")
    await db.initialize()
    audit_log = PromptAuditLogger(tmp_path / "logs")
    try:
        await SqliteProjectRepository(db).create(Project(id="prj_1", name="P1"))
        session_repo = SqliteSessionRepository(db)
        await session_repo.create_session(Session(id="ses_1", project_id="prj_1", title="S1"))
        rec, file_repo, blobs, vec_store, embedder = await _ingest_oversized(
            tmp_path, db, "svc.py", "def run():\n    return 42\n"
        )
        reg = build_default_tool_registry(file_repo, blobs)
        reg.register(_ApprovalPlugin())
        exe = GuardedToolExecutor(reg, SqliteToolAuditRepository(db))
        pipeline = CompositeGuardrailPipeline(
            GuardrailConfig(enabled=True),
            providers=[AllowAllGuardrailProvider()],
            audit_logger=audit_log,
        )
        settings = LaiSettings(data_dir=tmp_path, direct_file_budget_tokens=256)
        principal = AuthPrincipal(
            principal_id="usr_1",
            auth_method=AuthMethod.API_KEY,
            scopes=(ApiScope.CHAT_WRITE, ApiScope.FILES_READ, ApiScope.TOOLS_SAFE),
        )

        orch_appr = TurnOrchestrator(
            settings,
            session_repo,
            file_repo,
            blobs,
            _AsyncInference(
                complete_fn=lambda _r: InferenceResponse(content=""),
                stream_fn=lambda _r: [
                    InferenceChunk(
                        tool_calls=(
                            InferenceToolCall(
                                id="tc_a", name="deploy_patch", arguments_json='{"target": "prod"}'
                            ),
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            ),
            InferenceSpooler(),
            SessionGate(),
            pipeline,
            tool_registry=reg,
            tool_executor=exe,
            embedder=embedder,
            vector_store=vec_store,
        )
        appr_events = [
            ev
            async for ev in orch_appr.run_turn_stream(
                TurnInput(session_id="ses_1", project_id="prj_1", user_message="Explain svc.py"),
                principal,
            )
        ]
        appr_ev = next(ev for ev in appr_events if isinstance(ev, ApprovalRequiredEvent))
        done_ev = next(ev for ev in appr_events if isinstance(ev, TurnCompletedEvent))
        assert appr_ev.tool_name == "deploy_patch"
        assert done_ev.output.pending_approval_id == appr_ev.approval_id

        spooler_loop = InferenceSpooler()
        orch_loop = TurnOrchestrator(
            settings,
            session_repo,
            file_repo,
            blobs,
            _AsyncInference(
                complete_fn=lambda _r: InferenceResponse(content=""),
                stream_fn=lambda req: (
                    [InferenceChunk(content_delta="Loop capped answer.")]
                    if not req.tools
                    else [
                        InferenceChunk(
                            tool_calls=(
                                InferenceToolCall(
                                    id="tc_l",
                                    name="read_file_lines",
                                    arguments_json=json.dumps({"file_id": rec.id, "start_line": 1}),
                                ),
                            ),
                            finish_reason="tool_calls",
                        )
                    ]
                ),
            ),
            spooler_loop,
            SessionGate(),
            pipeline,
            tool_registry=reg,
            tool_executor=exe,
        )
        loop_out = await orch_loop.run_turn(
            TurnInput(session_id="ses_1", project_id="prj_1", user_message="Inspect svc.py"),
            principal,
        )
        assert loop_out.message.content == "Loop capped answer."
        assert spooler_loop.snapshot().total_completed == EXPECTED_LOOP_LEASES
    finally:
        audit_log.close()
        await db.close()


async def test_map_reduce_compression_and_interactive_interleaving(tmp_path: Path) -> None:
    """Oversized >10k findings trigger compression, and INTERACTIVE interleaves between windows."""
    db = SqliteDatabase(tmp_path / "lai.db")
    await db.initialize()
    audit_log = PromptAuditLogger(tmp_path / "logs")
    try:
        await SqliteProjectRepository(db).create(Project(id="prj_1", name="P1"))
        session_repo = SqliteSessionRepository(db)
        await session_repo.create_session(Session(id="ses_1", project_id="prj_1", title="S1"))
        lines = [f"2026-11-17T10:00:{i:02d}Z ERROR worker_{i} timeout" for i in range(1, 35)]
        _, file_repo, blobs, _, _ = await _ingest_oversized(
            tmp_path, db, "big.log", "\n".join(lines)
        )

        first_started, allow_finish = asyncio.Event(), asyncio.Event()
        order: list[JobPriority] = []

        async def _complete(req: InferenceRequest) -> InferenceResponse:
            order.append(JobPriority.MAP_REDUCE)
            if "Findings batch" in req.messages[-1].content:
                return InferenceResponse(
                    content="Compressed [big.log:L1-L34].",
                    usage=TokenUsage(prompt_tokens=80, completion_tokens=20, total_tokens=100),
                )
            if not first_started.is_set():
                first_started.set()
                await allow_finish.wait()
            return InferenceResponse(
                content="timeout error " * LARGE_FINDING_REPEATS,
                usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
            )

        def _stream(req: InferenceRequest) -> list[InferenceChunk]:
            if req.messages[-1].content.endswith("Ping"):
                order.append(JobPriority.INTERACTIVE)
                return [InferenceChunk(content_delta="Pong")]
            order.append(JobPriority.MAP_REDUCE)
            return [InferenceChunk(content_delta="Done [big.log:L1-L34].")]

        settings = LaiSettings(
            data_dir=tmp_path,
            direct_file_budget_tokens=256,
            map_window_tokens=260,
            map_window_overlap_tokens=30,
        )
        pipeline = CompositeGuardrailPipeline(
            GuardrailConfig(enabled=True),
            providers=[AllowAllGuardrailProvider()],
            audit_logger=audit_log,
        )
        principal = AuthPrincipal(
            principal_id="usr_1",
            auth_method=AuthMethod.API_KEY,
            scopes=(ApiScope.CHAT_WRITE, ApiScope.FILES_READ),
        )
        orch = TurnOrchestrator(
            settings,
            session_repo,
            file_repo,
            blobs,
            _AsyncInference(complete_fn=_complete, stream_fn=_stream),
            InferenceSpooler(),
            SessionGate(),
            pipeline,
        )

        mr_task = asyncio.create_task(
            _collect(
                orch,
                TurnInput(
                    session_id="ses_1",
                    project_id="prj_1",
                    user_message="Summarize all errors across the log",
                ),
                principal,
            )
        )
        await first_started.wait()
        ping_task = asyncio.create_task(
            orch.run_turn(
                TurnInput(
                    session_id="ses_ping",
                    project_id="prj_1",
                    user_message="Ping",
                    context_mode=ContextMode.DIRECT,
                ),
                principal,
            )
        )
        await asyncio.sleep(0.02)
        allow_finish.set()
        ping_out = await ping_task
        mr_events = await mr_task

        assert ping_out.message.content == "Pong"
        assert "compress_findings" in [
            ev.stage for ev in mr_events if isinstance(ev, ProgressEvent)
        ]
        assert JobPriority.INTERACTIVE in order
        assert order.index(JobPriority.INTERACTIVE) < len(order) - 1
    finally:
        audit_log.close()
        await db.close()


async def _collect(
    orch: TurnOrchestrator, turn_input: TurnInput, principal: AuthPrincipal
) -> list[object]:
    return [ev async for ev in orch.run_turn_stream(turn_input, principal)]
