"""Unit tests for auth, files, orchestrator events, scheduler models, and ports."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from lai.domain.auth import ApiKeyRecord, ApiScope, AuthPrincipal
from lai.domain.chat import (
    ApprovalRequiredEvent,
    ContentDeltaEvent,
    ContextMode,
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
from lai.domain.scheduler import JobPriority, LeaseRequest, SpoolerSnapshot
from lai.domain.tools import PermissionTier, ToolSpec
from lai.ports.inference import InferenceProviderPort, InferenceSupervisorPort
from lai.ports.orchestrator import OrchestratorPort
from lai.ports.repositories import (
    ApiKeyRepositoryPort,
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
        auth_method="api_key",
        scopes=(ApiScope.PROJECTS_READ,),
    )
    assert reader.has_scope(ApiScope.PROJECTS_READ) is True
    assert reader.has_scope(ApiScope.ADMIN) is False

    admin = AuthPrincipal(
        principal_id="usr_admin",
        auth_method="cloudflare_access",
        scopes=(ApiScope.ADMIN,),
        email="ops@example.com",
    )
    assert admin.has_scope(ApiScope.TOOLS_APPROVE) is True


def test_orchestrator_event_discriminated_union() -> None:
    """OrchestratorEvent validates all six event variants cleanly."""
    adapter: TypeAdapter[OrchestratorEvent] = TypeAdapter(OrchestratorEvent)
    turn_output = TurnOutput(
        session_id="ses_1",
        message=Message(
            session_id="ses_1",
            project_id="prj_1",
            role=MessageRole.ASSISTANT,
            content="done",
        ),
        context_mode_used=ContextMode.DIRECT,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
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
    assert LeaseRequest(session_id="ses_1", project_id="prj_1").priority == JobPriority.INTERACTIVE
    assert SpoolerSnapshot().max_queued_jobs == DEFAULT_MAX_QUEUED_JOBS
    assert (
        TurnInput(session_id="ses_1", project_id="prj_1", user_message="Hi").context_mode
        == ContextMode.AUTO
    )


def test_ports_are_runtime_checkable_protocols() -> None:
    """All infrastructure and orchestrator ports expose runtime-checkable protocols."""
    protocols: tuple[type[object], ...] = (
        ApiKeyRepositoryPort,
        ProjectRepositoryPort,
        SessionRepositoryPort,
        FileRepositoryPort,
        ToolAuditRepositoryPort,
        BlobStorePort,
        EmbedderPort,
        VectorStorePort,
        IngestionPipelinePort,
        InferenceProviderPort,
        InferenceSupervisorPort,
        ToolPluginPort,
        ToolRegistryPort,
        ToolPolicyPort,
        ToolExecutorPort,
        SessionGatePort,
        InferenceSpoolerPort,
        OrchestratorPort,
    )
    dummy = object()
    for proto in protocols:
        assert not isinstance(dummy, proto)
