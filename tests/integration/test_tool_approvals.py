"""Integration tests for host-tier ban, gated READ confinement, CAS races, and CLIENT tools."""

from __future__ import annotations

import asyncio
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
from lai.domain.chat import ContextMode
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
    ToolExecutionTarget,
    ToolResult,
    ToolSpec,
)
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.ports.tools import ToolPluginPort
from lai.security import generate_api_key, generate_client_app
from lai.tools import GuardedToolExecutor

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator
    from pathlib import Path

    from lai.api.deps import AppServices

_CAS_CONCURRENCY = 10


class _HostWritePlugin(ToolPluginPort):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="host_write",
            description="Forbidden host write plugin",
            permission_tier=PermissionTier.WRITE,
            execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        self.calls += 1
        return ToolResult(tool_name=self.spec.name, success=True, output="written")


class _GatedReadPlugin(ToolPluginPort):
    def __init__(self) -> None:
        self.seen_contexts: list[ToolExecutionContext] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="gated_read",
            description="Gated server attachment read plugin",
            permission_tier=PermissionTier.READ,
            execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
            requires_approval_by_default=True,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments
        self.seen_contexts.append(context)
        return ToolResult(tool_name=self.spec.name, success=True, output="read_ok")


class _ClientToolInferenceProvider(InferenceProviderPort):
    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="unused")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        tc = InferenceToolCall(
            id="tc_ps_1",
            name="powershell_preview",
            arguments_json=json.dumps({"command": "Get-Process"}),
        )
        yield InferenceChunk(tool_calls=(tc,), finish_reason="tool_calls")

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@dataclass(slots=True)
class _ToolEnv:
    svc: AppServices
    client: httpx.AsyncClient
    host_plugin: _HostWritePlugin
    read_plugin: _GatedReadPlugin
    project_id: str
    session_id: str
    keys: dict[str, str]
    app_id: str
    app_secret: str


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
    host_plugin, read_plugin = _HostWritePlugin(), _GatedReadPlugin()
    svc.tool_registry.register(host_plugin)
    svc.tool_registry.register(read_plugin)
    inf = _ClientToolInferenceProvider()
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
        client_app_repo=svc.client_app_repo,
    )
    keys: dict[str, str] = {}
    for name, scopes in (
        ("chat_only", (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE)),
        ("exec_user", (ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE, ApiScope.TOOLS_EXEC)),
        ("approver", (ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE)),
    ):
        created = generate_api_key(name, scopes)
        await svc.api_key_repo.create(created.record)
        keys[name] = created.plaintext_key

    ps_spec = ToolSpec(
        name="powershell_preview",
        description="Preview a local PowerShell command.",
        permission_tier=PermissionTier.EXEC,
        parameters_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        requires_approval_by_default=True,
        execution_target=ToolExecutionTarget.CLIENT,
    )
    created_app = generate_client_app(
        "ps-helper",
        allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
        registered_tools=(ps_spec,),
    )
    await svc.client_app_repo.create(created_app.record)

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            hdrs = {"Authorization": f"Bearer {keys['chat_only']}"}
            pid = (await client.post("/v1/projects", headers=hdrs, json={"name": "tp"})).json()[
                "id"
            ]
            sid = (
                await client.post("/v1/sessions", headers=hdrs, json={"project_id": pid})
            ).json()["id"]
            yield _ToolEnv(
                svc,
                client,
                host_plugin,
                read_plugin,
                pid,
                sid,
                keys,
                created_app.record.id,
                created_app.plaintext_secret,
            )
    finally:
        await close_app_services(svc)


async def test_server_host_ban_gated_read_restart_confinement_and_resolve_cas(
    tool_env: _ToolEnv,
) -> None:
    """Verify server host-tier ban, restart confinement restore, and 10-way /resolve CAS."""
    svc, client = tool_env.svc, tool_env.client
    pid, sid = tool_env.project_id, tool_env.session_id
    h_approve = {"Authorization": f"Bearer {tool_env.keys['approver']}"}

    with pytest.raises(AuthorizationError):
        await svc.tool_executor.invoke(
            "host_write",
            {},
            ToolExecutionContext(
                project_id=pid,
                session_id=sid,
                principal_id="adm",
                principal_scopes=(ApiScope.ADMIN, ApiScope.TOOLS_EXEC),
            ),
        )
    assert tool_env.host_plugin.calls == 0
    audits = await svc.tool_audit_repo.list_audits(pid, sid)
    assert any(
        a.tool_name == "host_write" and a.error == "server_host_tool_forbidden" for a in audits
    )

    with pytest.raises(ToolApprovalRequiredError) as conflict_exc:
        await svc.tool_executor.invoke(
            "gated_read",
            {"file_id": "fil_scoped"},
            ToolExecutionContext(
                project_id=pid,
                session_id=sid,
                principal_id="reader",
                principal_scopes=(ApiScope.TOOLS_SAFE,),
                allowed_file_ids=("fil_scoped",),
            ),
        )
    assert conflict_exc.value.status_code == HTTPStatus.CONFLICT
    aid = conflict_exc.value.approval_id

    # Fresh executor simulates restart/eviction of in-memory _pending_contexts
    svc.tool_executor = GuardedToolExecutor(svc.tool_registry, svc.tool_audit_repo)
    responses = await asyncio.gather(
        *[
            client.post(
                f"/v1/tool-approvals/{aid}/resolve", headers=h_approve, json={"approved": True}
            )
            for _ in range(_CAS_CONCURRENCY)
        ]
    )
    expected_codes = [HTTPStatus.OK] + [HTTPStatus.UNPROCESSABLE_ENTITY] * (_CAS_CONCURRENCY - 1)
    assert sorted(r.status_code for r in responses) == expected_codes
    assert len(tool_env.read_plugin.seen_contexts) == 1
    assert tool_env.read_plugin.seen_contexts[0].allowed_file_ids == ("fil_scoped",)
    assert tool_env.read_plugin.seen_contexts[0].principal_scopes == (ApiScope.TOOLS_SAFE,)


async def test_client_exec_tool_policy_approval_cas_and_denial(tool_env: _ToolEnv) -> None:
    """Verify CLIENT EXEC DENY without TOOLS_EXEC, SSE dispatch, /resolve CAS, and denial."""
    client, svc = tool_env.client, tool_env.svc
    pid, sid = tool_env.project_id, tool_env.session_id
    cli_hdrs = {"X-LAI-Client-Id": tool_env.app_id, "X-LAI-Client-Secret": tool_env.app_secret}
    h_chat = {"Authorization": f"Bearer {tool_env.keys['chat_only']}", **cli_hdrs}
    h_exec = {"Authorization": f"Bearer {tool_env.keys['exec_user']}", **cli_hdrs}
    h_approve = {"Authorization": f"Bearer {tool_env.keys['approver']}"}

    r_deny = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=h_chat,
        json={"content": "run ps", "context_mode": ContextMode.RETRIEVAL.value, "stream": False},
    )
    assert r_deny.status_code == HTTPStatus.OK
    audits_deny = await svc.tool_audit_repo.list_audits(pid, sid)
    assert any(
        a.tool_name == "powershell_preview"
        and a.policy_decision is PolicyDecision.DENY
        and a.error == "policy_denied"
        for a in audits_deny
    )

    r_sse = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=h_exec,
        json={"content": "run ps", "context_mode": ContextMode.RETRIEVAL.value, "stream": True},
    )
    assert r_sse.status_code == HTTPStatus.OK
    assert "event: approval_required" in r_sse.text
    assert "event: client_tool_call" in r_sse.text
    evs = [json.loads(ln[6:]) for ln in r_sse.text.splitlines() if ln.startswith("data: ")]
    frames = [e for e in evs if e.get("event_type") == "client_tool_call"]
    assert len(frames) == 1
    approval_id = str(frames[0]["approval_id"])
    assert frames[0]["requires_approval"] is True

    r_get = await client.get(f"/v1/tool-approvals/{approval_id}", headers=h_approve)
    assert r_get.json()["status"] == ApprovalStatus.PENDING.value
    assert r_get.json()["execution_target"] == ToolExecutionTarget.CLIENT.value
    r_forbid = await client.post(
        f"/v1/tool-approvals/{approval_id}/resolve", headers=h_exec, json={"approved": True}
    )
    assert r_forbid.status_code == HTTPStatus.FORBIDDEN

    resolve_resps = await asyncio.gather(
        *[
            client.post(
                f"/v1/tool-approvals/{approval_id}/resolve",
                headers=h_approve,
                json={"approved": True, "note": "approved by reviewer"},
            )
            for _ in range(_CAS_CONCURRENCY)
        ]
    )
    expected_codes = [HTTPStatus.OK] + [HTTPStatus.UNPROCESSABLE_ENTITY] * (_CAS_CONCURRENCY - 1)
    assert sorted(r.status_code for r in resolve_resps) == expected_codes
    winner = next(r.json() for r in resolve_resps if r.status_code == HTTPStatus.OK)
    assert winner["status"] == ApprovalStatus.APPROVED.value
    assert winner["expires_at"] is not None

    r_second = await client.post(
        f"/v1/sessions/{sid}/messages",
        headers=h_exec,
        json={"content": "run ps 2", "context_mode": ContextMode.RETRIEVAL.value, "stream": False},
    )
    second_id = r_second.json()["pending_approval_id"]
    r_reject = await client.post(
        f"/v1/tool-approvals/{second_id}/resolve",
        headers=h_approve,
        json={"approved": False, "note": "rejected"},
    )
    assert (r_reject.status_code, r_reject.json()["status"]) == (
        HTTPStatus.OK,
        ApprovalStatus.DENIED.value,
    )
