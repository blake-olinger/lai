"""End-to-end integration tests for guarded EXEC tool approval, execution, audit, and replay."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import ContextMode, TokenUsage
from lai.domain.errors import AuthorizationError, ToolApprovalRequiredError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.observability import LoggingConfig
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    PolicyDecision,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.ports.tools import ToolPluginPort
from lai.security import generate_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator
    from pathlib import Path

    from lai.api.deps import AppServices

_EXPECTED_EXEC_ONCE = 1


class _PowershellPreviewPlugin(ToolPluginPort):
    """Mock Phase 2 EXEC plugin requiring explicit human approval."""

    def __init__(self) -> None:
        self.executed_commands: list[str] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="powershell_preview",
            description="Preview a read-only PowerShell command.",
            permission_tier=PermissionTier.EXEC,
            parameters_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            requires_approval_by_default=True,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        cmd = str(arguments.get("command", ""))
        self.executed_commands.append(cmd)
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            output=f"Executed preview: {cmd}",
        )


class _ToolCallingInferenceProvider(InferenceProviderPort):
    """Fake inference provider emitting a powershell_preview tool call on first iteration."""

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="unused")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        has_tool_result = any(msg.role.value == "tool" for msg in request.messages)
        if not has_tool_result:
            tc = InferenceToolCall(
                id="tc_ps_1",
                name="powershell_preview",
                arguments_json=json.dumps({"command": "Get-Process"}),
            )
            yield InferenceChunk(tool_calls=(tc,), finish_reason="tool_calls")
            return
        yield InferenceChunk(
            content_delta="Tool denied or finished.",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@dataclass(slots=True)
class _ToolEnv:
    svc: AppServices
    client: httpx.AsyncClient
    plugin: _PowershellPreviewPlugin
    project_id: str
    session_id: str
    keys: dict[str, str]


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.fixture
async def tool_env(tmp_path: Path) -> AsyncIterator[_ToolEnv]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    svc = await build_app_services(settings)
    plugin = _PowershellPreviewPlugin()
    svc.tool_registry.register(plugin)
    inf = _ToolCallingInferenceProvider()
    svc.inference_provider = inf
    svc.orchestrator = TurnOrchestrator(
        settings=settings,
        session_repo=svc.session_repo,
        file_repo=svc.file_repo,
        blob_store=svc.blob_store,
        inference_provider=inf,
        spooler=svc.spooler,
        session_gate=svc.session_gate,
        guardrail_pipeline=svc.guardrail_pipeline,
        tool_registry=svc.tool_registry,
        tool_executor=svc.tool_executor,
        embedder=None,
        vector_store=None,
    )

    keys: dict[str, str] = {}
    for name, scopes in (
        (
            "chat_only",
            (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
        ),
        (
            "exec_user",
            (ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE, ApiScope.TOOLS_EXEC),
        ),
        (
            "approver",
            (ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE),
        ),
    ):
        created = generate_api_key(name, scopes)
        await svc.api_key_repo.create(created.record)
        keys[name] = created.plaintext_key

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            chat_hdrs = {"Authorization": f"Bearer {keys['chat_only']}"}
            pid = (
                await client.post("/v1/projects", headers=chat_hdrs, json={"name": "tool-proj"})
            ).json()["id"]
            sid = (
                await client.post("/v1/sessions", headers=chat_hdrs, json={"project_id": pid})
            ).json()["id"]
            yield _ToolEnv(
                svc=svc,
                client=client,
                plugin=plugin,
                project_id=pid,
                session_id=sid,
                keys=keys,
            )
    finally:
        await close_app_services(svc)


async def test_exec_tool_policy_approval_execution_audit_and_replay_prevention(
    tool_env: _ToolEnv,
) -> None:
    """Verify DENY without TOOLS_EXEC, approval gate with TOOLS_EXEC, execution, and replay."""
    client = tool_env.client
    svc = tool_env.svc
    plugin = tool_env.plugin
    pid = tool_env.project_id
    sid = tool_env.session_id

    h_chat = {"Authorization": f"Bearer {tool_env.keys['chat_only']}"}
    h_exec = {"Authorization": f"Bearer {tool_env.keys['exec_user']}"}
    h_approve = {"Authorization": f"Bearer {tool_env.keys['approver']}"}

    # 1. Key lacking TOOLS_EXEC is DENIED; audit records policy_denied; plugin never executes
    r_deny = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=h_chat,
        json={"content": "run ps", "context_mode": ContextMode.RETRIEVAL.value, "stream": False},
    )
    assert r_deny.status_code == HTTPStatus.OK
    assert plugin.executed_commands == []
    audits_after_deny = await svc.tool_audit_repo.list_audits(pid, sid)
    assert any(
        a.tool_name == "powershell_preview"
        and a.policy_decision is PolicyDecision.DENY
        and not a.success
        and a.error == "policy_denied"
        for a in audits_after_deny
    )

    # 2. Key with TOOLS_EXEC emits SSE ApprovalRequiredEvent and creates PENDING approval
    r_sse = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=h_exec,
        json={"content": "run ps", "context_mode": ContextMode.RETRIEVAL.value, "stream": True},
    )
    assert r_sse.status_code == HTTPStatus.OK
    assert "event: approval_required" in r_sse.text
    approval_frames = [
        json.loads(line.removeprefix("data: "))
        for line in r_sse.text.splitlines()
        if line.startswith("data: ") and '"event_type":"approval_required"' in line
    ]
    assert len(approval_frames) == 1
    approval_id = str(approval_frames[0]["approval_id"])
    assert approval_frames[0]["tool_name"] == "powershell_preview"
    assert plugin.executed_commands == []

    # 3. Verify GET approval shows PENDING and exec_user cannot resolve without TOOLS_APPROVE
    r_get = await client.get(f"/v1/tool-approvals/{approval_id}", headers=h_approve)
    assert r_get.status_code == HTTPStatus.OK
    assert r_get.json()["status"] == ApprovalStatus.PENDING.value

    r_forbidden = await client.post(
        f"/v1/tool-approvals/{approval_id}/resolve",
        headers=h_exec,
        json={"approved": True},
    )
    assert r_forbidden.status_code == HTTPStatus.FORBIDDEN

    # 4. Approving with TOOLS_APPROVE executes plugin once and records immutable audit records
    r_resolve = await client.post(
        f"/v1/tool-approvals/{approval_id}/resolve",
        headers=h_approve,
        json={"approved": True, "note": "approved by reviewer"},
    )
    assert r_resolve.status_code == HTTPStatus.OK
    resolved_body = r_resolve.json()
    assert resolved_body["status"] == ApprovalStatus.EXECUTED.value
    assert resolved_body["decision_note"] == "approved by reviewer"
    assert plugin.executed_commands == ["Get-Process"]

    audits = await svc.tool_audit_repo.list_audits(pid, sid)
    assert any(
        a.approval_id == approval_id
        and a.policy_decision is PolicyDecision.ALLOW
        and a.success
        and "Executed preview: Get-Process" in a.output_preview
        for a in audits
    )

    # 5. Replay prevention: re-resolving HTTP route returns 422; re-invoking executor raises 403
    r_replay = await client.post(
        f"/v1/tool-approvals/{approval_id}/resolve",
        headers=h_approve,
        json={"approved": True},
    )
    assert r_replay.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert len(plugin.executed_commands) == _EXPECTED_EXEC_ONCE

    replay_ctx = ToolExecutionContext(
        project_id=pid,
        session_id=sid,
        principal_id="exec-replay",
        principal_scopes=(ApiScope.TOOLS_EXEC,),
        approved_approval_id=approval_id,
    )
    with pytest.raises(AuthorizationError):
        await svc.tool_executor.invoke("powershell_preview", {"command": "Get-Process"}, replay_ctx)
    assert len(plugin.executed_commands) == _EXPECTED_EXEC_ONCE

    # 6. Verify ToolApprovalRequiredError maps to HTTP 409 Conflict envelope and denial path
    with pytest.raises(ToolApprovalRequiredError) as conflict_exc:
        await svc.tool_executor.invoke(
            "powershell_preview",
            {"command": "Get-Date"},
            ToolExecutionContext(
                project_id=pid,
                session_id=sid,
                principal_id="exec-deny",
                principal_scopes=(ApiScope.TOOLS_EXEC,),
            ),
        )
    second_id = conflict_exc.value.approval_id
    assert conflict_exc.value.status_code == HTTPStatus.CONFLICT

    r_deny_resolve = await client.post(
        f"/v1/tool-approvals/{second_id}/resolve",
        headers=h_approve,
        json={"approved": False, "note": "rejected"},
    )
    assert r_deny_resolve.status_code == HTTPStatus.OK
    assert r_deny_resolve.json()["status"] == ApprovalStatus.DENIED.value
    assert len(plugin.executed_commands) == _EXPECTED_EXEC_ONCE
