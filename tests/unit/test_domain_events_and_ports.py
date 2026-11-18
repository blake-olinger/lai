"""Unit tests for auth, files, orchestrator events, scheduler models, and ports."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from lai import db, guardrails, inference, ingestion, orchestrator, scheduler, security, tools
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiKeyRecord, ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import (
    ApprovalRequiredEvent,
    ClientToolCallEvent,
    ContentDeltaEvent,
    ContextMode,
    InferenceToolCall,
    KeepaliveEvent,
    Message,
    MessageRole,
    OrchestratorEvent,
    ProgressEvent,
    ThinkingDeltaEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnInput,
    TurnOutput,
)
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
from lai.domain.guardrails import GuardrailConfig
from lai.domain.scheduler import JobPriority, LeaseRequest, SpoolerSnapshot
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec
from lai.ports import AuthenticatorPort, AuthProviderPort, EdgeVerifierPort
from lai.ports.inference import InferenceProviderPort, InferenceSupervisorPort
from lai.ports.orchestrator import OrchestratorPort
from lai.ports.repositories import (
    ApiKeyRepositoryPort,
    ClientAppRepositoryPort,
    FileRepositoryPort,
    ProjectRepositoryPort,
    SessionRepositoryPort,
    ToolAuditRepositoryPort,
)
from lai.ports.scheduler import InferenceSpoolerPort, SessionGatePort
from lai.ports.storage import (
    BlobStorePort,
    EmbedderPort,
    IngestionPipelinePort,
    VectorStorePort,
)
from lai.ports.tools import (
    ToolExecutorPort,
    ToolPluginPort,
    ToolPolicyPort,
    ToolRegistryPort,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MAX_QUEUED_JOBS = 64


def test_auth_models_scope_and_expiry_checks() -> None:
    """ApiKeyRecord checks revocation/expiry and AuthPrincipal checks scopes."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    active_key = ApiKeyRecord(
        name="dev",
        key_prefix="lai_dev_",
        key_hash="a" * 64,
        scopes=(ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE),
        created_at=now,
        expires_at=now + timedelta(days=1),
    )
    assert active_key.is_active(now=now) is True
    assert active_key.is_active(now=now + timedelta(days=2)) is False
    assert active_key.model_copy(update={"revoked_at": now}).is_active(now=now) is False

    reader = AuthPrincipal(
        principal_id="usr_1",
        auth_method=AuthMethod.API_KEY,
        scopes=(ApiScope.PROJECTS_READ,),
    )
    assert reader.has_scope(ApiScope.PROJECTS_READ) is True
    assert reader.has_scope(ApiScope.ADMIN) is False

    admin = AuthPrincipal(
        principal_id="usr_admin",
        auth_method=AuthMethod.CLOUDFLARE_ACCESS,
        scopes=(ApiScope.ADMIN,),
        email="ops@example.com",
    )
    assert admin.has_scope(ApiScope.TOOLS_APPROVE) is True


def test_orchestrator_event_discriminated_union() -> None:
    """OrchestratorEvent validates all seven event variants cleanly."""
    adapter: TypeAdapter[OrchestratorEvent] = TypeAdapter(OrchestratorEvent)
    client_ev = ClientToolCallEvent(
        approval_id="apr_1",
        tool_call_id="tc_1",
        tool_name="local_exec",
        arguments_json='{"cmd":"ls"}',
        requires_approval=True,
    )
    turn_output = TurnOutput(
        session_id="ses_1",
        message=Message(
            session_id="ses_1",
            project_id="prj_1",
            role=MessageRole.ASSISTANT,
            content="done",
            tool_calls=(InferenceToolCall(id="tc_1", name="local_exec", arguments_json="{}"),),
        ),
        context_mode_used=ContextMode.DIRECT,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        pending_client_tool_call=client_ev,
    )
    events: list[OrchestratorEvent] = [
        KeepaliveEvent(),
        ProgressEvent(stage="map", message="chunk 1/2", completed_steps=1, total_steps=2),
        ThinkingDeltaEvent(delta="hmm"),
        ContentDeltaEvent(delta="hi"),
        ApprovalRequiredEvent(
            approval_id="apr_1",
            tool_name="write_file",
            arguments_preview="{}",
            reason="write tier",
        ),
        client_ev,
        TurnCompletedEvent(output=turn_output),
    ]
    for event in events:
        assert adapter.validate_python(event.model_dump()) == event

    with pytest.raises(ValidationError):
        adapter.validate_python({"event_type": "unknown_event"})


def test_file_and_scheduler_domain_models() -> None:
    """File, summary, chunk, tool, and scheduler models validate defaults cleanly."""
    log_summary = LogFileSummary(
        total_lines=100,
        severity_counts={LogSeverity.ERROR: 2},
        error_count=2,
        top_exceptions=(
            ExceptionFingerprint(
                fingerprint="fp_1",
                exception_type="ValueError",
                message_sample="bad input",
                count=2,
                first_line=10,
                last_line=90,
            ),
        ),
    )
    code_summary = CodeFileSummary(language="python", total_lines=50, symbols=("main",))
    file_rec = FileRecord(
        project_id="prj_1",
        filename="app.log",
        size_bytes=512,
        sha256="b" * 64,
        blob_path="data/blobs/app.log",
        kind=FileKind.LOG,
        status=FileStatus.READY,
        log_summary=log_summary,
        code_summary=code_summary,
    )
    chunk = FileChunk(
        file_id=file_rec.id,
        project_id="prj_1",
        chunk_index=0,
        start_line=1,
        end_line=25,
        token_count=120,
        text="ERROR ValueError",
        severity=LogSeverity.ERROR,
    )
    assert ScoredChunk(chunk=chunk, score=0.95, source="hybrid").chunk.file_id == file_rec.id
    spec = ToolSpec(
        name="read_file",
        description="Read workspace file",
        permission_tier=PermissionTier.READ,
        parameters_schema={"type": "object"},
    )
    assert spec.requires_approval_by_default is False
    assert spec.execution_target is ToolExecutionTarget.SERVER_ATTACHMENT
    assert LeaseRequest(session_id="ses_1", project_id="prj_1").priority == JobPriority.INTERACTIVE
    assert SpoolerSnapshot().max_queued_jobs == DEFAULT_MAX_QUEUED_JOBS
    turn_in = TurnInput(
        session_id="ses_1",
        project_id="prj_1",
        user_message="Hi",
        client_tools=(spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT}),),
        client_app_id="app_1",
        client_allowed_tiers=(PermissionTier.READ,),
    )
    assert turn_in.context_mode == ContextMode.AUTO
    assert turn_in.client_tools[0].execution_target is ToolExecutionTarget.CLIENT


class _NonConformingAdapterStub:
    """Minimal stub lacking every runtime-checkable port method."""


async def test_ports_are_runtime_checkable_protocols(tmp_path: Path) -> None:
    """All concrete infrastructure and orchestrator adapters satisfy their ports."""
    async with (
        db.SqliteDatabase(tmp_path / "ports.db") as sqlite_db,
        httpx.AsyncClient() as http_client,
    ):
        api_key_repo = db.SqliteApiKeyRepository(sqlite_db)
        client_app_repo = db.SqliteClientAppRepository(sqlite_db)
        project_repo = db.SqliteProjectRepository(sqlite_db)
        session_repo = db.SqliteSessionRepository(sqlite_db)
        file_repo = db.SqliteFileRepository(sqlite_db)
        tool_audit_repo = db.SqliteToolAuditRepository(sqlite_db)

        api_key_auth = security.ApiKeyAuthProvider(api_key_repo)
        oidc_auth = security.OidcJwtAuthProvider(
            security.OidcProviderConfig(), http_client=http_client
        )
        edge_verifier = security.CloudflareEdgeVerifier(
            security.CloudflareEdgeConfig(), http_client=http_client
        )
        authenticator = security.CompositeAuthenticator(
            providers=(api_key_auth, oidc_auth), edge_verifier=edge_verifier
        )

        blob_store = ingestion.LocalBlobStore(tmp_path / "blobs")
        fastembed = ingestion.FastEmbedCpuAdapter(cache_dir=tmp_path / "fastembed")
        hash_embedder = ingestion.DeterministicHashEmbedder()
        vector_store = ingestion.LanceDbVectorStore(
            tmp_path / "lancedb", dimensions=fastembed.dimensions
        )
        pipeline = ingestion.LocalIngestionPipeline(
            blob_store=blob_store,
            file_repo=file_repo,
            embedder=fastembed,
            vector_store=vector_store,
        )

        llama_provider = inference.LlamaCppProvider(client=http_client)
        openai_provider = inference.OpenAICompatProvider(
            "http://127.0.0.1:11434", client=http_client
        )
        llama_supervisor = inference.LlamaCppSupervisor(
            executable_path=tmp_path / "llama-server",
            models_dir=tmp_path / "models",
            log_dir=tmp_path / "logs",
        )
        ollama_supervisor = inference.OllamaSupervisor(log_dir=tmp_path / "logs")

        search_plugin = tools.SearchFilesPlugin(
            file_repo, embedder=fastembed, vector_store=vector_store
        )
        read_plugin = tools.ReadFileLinesPlugin(file_repo, blob_store)
        overview_plugin = tools.GetFileOverviewPlugin(file_repo)
        registry = tools.InMemoryToolRegistry((search_plugin, read_plugin, overview_plugin))
        policy = tools.DefaultToolPolicy()
        executor = tools.GuardedToolExecutor(registry, tool_audit_repo, policy)
        session_gate = scheduler.SessionGate()
        spooler = scheduler.InferenceSpooler()

        with guardrails.CompositeGuardrailPipeline(
            GuardrailConfig(enabled=False),
            providers=(guardrails.AllowAllGuardrailProvider(),),
            log_dir=tmp_path / "audit",
        ) as guardrail_pipeline:
            turn_orchestrator = orchestrator.TurnOrchestrator(
                settings=LaiSettings(),
                session_repo=session_repo,
                file_repo=file_repo,
                blob_store=blob_store,
                inference_provider=llama_provider,
                spooler=spooler,
                session_gate=session_gate,
                guardrail_pipeline=guardrail_pipeline,
                tool_registry=registry,
                tool_executor=executor,
                embedder=fastembed,
                vector_store=vector_store,
                client_app_repo=client_app_repo,
            )

            checks: tuple[tuple[object, type[object]], ...] = (
                (api_key_auth, AuthProviderPort),
                (oidc_auth, AuthProviderPort),
                (edge_verifier, EdgeVerifierPort),
                (authenticator, AuthenticatorPort),
                (api_key_repo, ApiKeyRepositoryPort),
                (client_app_repo, ClientAppRepositoryPort),
                (project_repo, ProjectRepositoryPort),
                (session_repo, SessionRepositoryPort),
                (file_repo, FileRepositoryPort),
                (tool_audit_repo, ToolAuditRepositoryPort),
                (blob_store, BlobStorePort),
                (fastembed, EmbedderPort),
                (hash_embedder, EmbedderPort),
                (vector_store, VectorStorePort),
                (pipeline, IngestionPipelinePort),
                (llama_provider, InferenceProviderPort),
                (openai_provider, InferenceProviderPort),
                (llama_supervisor, InferenceSupervisorPort),
                (ollama_supervisor, InferenceSupervisorPort),
                (search_plugin, ToolPluginPort),
                (read_plugin, ToolPluginPort),
                (overview_plugin, ToolPluginPort),
                (registry, ToolRegistryPort),
                (policy, ToolPolicyPort),
                (executor, ToolExecutorPort),
                (session_gate, SessionGatePort),
                (spooler, InferenceSpoolerPort),
                (turn_orchestrator, OrchestratorPort),
            )
            stub = _NonConformingAdapterStub()
            for adapter_instance, proto in checks:
                assert isinstance(adapter_instance, proto)
                assert not isinstance(stub, proto)
