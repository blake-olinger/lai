"""Integration tests for Wave 2B endpoints, _client_auth, _resume, and _client_results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus as S
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import MessageRole, TokenUsage
from lai.domain.errors import AuthorizationError, NotFoundError
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
    ToolApprovalRecord,
    ToolExecutionContext,
    ToolExecutionTarget,
    ToolResult,
    ToolSpec,
)
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.ports.tools import ToolPluginPort
from lai.security import build_client_app, generate_api_key
from lai.tools import GuardedToolExecutor
from tests.integration._helpers import OidcTestAuthority, configure_test_oidc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator
    from pathlib import Path

    from lai.api.deps import AppServices


class _ServerEchoPlugin(ToolPluginPort):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="srv_echo",
            description="Echo value",
            permission_tier=PermissionTier.READ,
            execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(tool_name="srv_echo", success=True, output="srv-ok")


class _InterToolInference(InferenceProviderPort):
    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []
        self.emit_server_tool_on_resume = False

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="unused")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.requests.append(request)
        if not request.messages:
            yield InferenceChunk(content_delta="empty", finish_reason="stop")
            return
        last = request.messages[-1]
        if last.role is MessageRole.TOOL:
            if self.emit_server_tool_on_resume and last.name == "adhoc_read":
                tc = InferenceToolCall(id="tc_srv_1", name="srv_echo", arguments_json="{}")
                yield InferenceChunk(tool_calls=(tc,), finish_reason="tool_calls")
                return
            yield InferenceChunk(
                content_delta=f"after-tool:{last.content}",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
            )
            return

        if any(t.name == "adhoc_read" for t in request.tools):
            tc = InferenceToolCall(id="tc_adhoc", name="adhoc_read", arguments_json="{}")
            yield InferenceChunk(tool_calls=(tc,), finish_reason="tool_calls")
            return

        yield InferenceChunk(
            content_delta="plain",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@dataclass(slots=True)
class _EdgeEnv:
    svc: AppServices
    client: httpx.AsyncClient
    inf: _InterToolInference
    oidc: OidcTestAuthority
    admin_hdrs: dict[str, str]
    writer_hdrs: dict[str, str]
    approver_hdrs: dict[str, str]
    project_id: str
    session_id: str


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _mk_orch(svc: AppServices, *, with_app_repo: bool = True) -> TurnOrchestrator:
    return TurnOrchestrator(
        settings=svc.settings,
        session_repo=svc.session_repo,
        file_repo=svc.file_repo,
        blob_store=svc.blob_store,
        inference_provider=svc.inference_provider,
        spooler=svc.spooler,
        session_gate=svc.session_gate,
        guardrail_pipeline=svc.guardrail_pipeline,
        tool_registry=svc.tool_registry,
        tool_executor=svc.tool_executor,
        client_app_repo=svc.client_app_repo if with_app_repo else None,
    )


@pytest.fixture
async def edge_env(tmp_path: Path) -> AsyncIterator[_EdgeEnv]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    svc = await build_app_services(settings)
    svc.tool_registry.register(_ServerEchoPlugin())
    svc.tool_executor = GuardedToolExecutor(svc.tool_registry, svc.tool_audit_repo)
    inf = _InterToolInference()
    svc.inference_provider = inf
    svc.orchestrator = _mk_orch(svc)
    adm_k = generate_api_key("admin", (ApiScope.ADMIN,))
    apr_k = generate_api_key("approver", (ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE))
    await svc.api_key_repo.create(adm_k.record)
    await svc.api_key_repo.create(apr_k.record)
    oidc = configure_test_oidc(
        svc,
        audiences=("turn-coerce-cid", "shadow-cid", "esc-cid", "unreg-cid"),
        default_scopes=(ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
    )
    adm_h = {"Authorization": f"Bearer {adm_k.plaintext_key}"}
    apr_h = {"Authorization": f"Bearer {apr_k.plaintext_key}"}
    wr_h = oidc.mint_headers(sub="edge-writer", aud="turn-coerce-cid")

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as cli:
            pid = (await cli.post("/v1/projects", headers=wr_h, json={"name": "edge"})).json()["id"]
            sid = (await cli.post("/v1/sessions", headers=wr_h, json={"project_id": pid})).json()[
                "id"
            ]
            yield _EdgeEnv(svc, cli, inf, oidc, adm_h, wr_h, apr_h, pid, sid)
    finally:
        await close_app_services(svc)


def _sp(name: str, tier: PermissionTier = PermissionTier.READ) -> ToolSpec:
    return ToolSpec(name=name, description=name, permission_tier=tier)


async def test_asgi_endpoint_4xx_and_client_auth_edge_branches(edge_env: _EdgeEnv) -> None:
    """Verify missing ASGI positive/4xx endpoints and _client_auth validation paths."""
    c, pid, sid = edge_env.client, edge_env.project_id, edge_env.session_id
    adm, wr, apr = edge_env.admin_hdrs, edge_env.writer_hdrs, edge_env.approver_hdrs

    for path in ("/v1/models", "/v1/tools", "/v1/tool-approvals/apr_x", "/v1/client-apps"):
        assert (await c.get(path)).status_code == S.UNAUTHORIZED
    assert (await c.get("/v1/models", headers=apr)).status_code == S.FORBIDDEN
    assert (await c.get("/v1/tools", headers=apr)).status_code == S.OK
    k_chat = generate_api_key("chat", (ApiScope.CHAT_READ,))
    await edge_env.svc.api_key_repo.create(k_chat.record)
    ch = {"Authorization": f"Bearer {k_chat.plaintext_key}"}
    assert (await c.get("/v1/tools", headers=ch)).status_code == S.FORBIDDEN
    assert (await c.get("/v1/tool-approvals/apr_missing", headers=ch)).status_code == S.FORBIDDEN
    assert (await c.get("/v1/tool-approvals/apr_missing", headers=apr)).status_code == S.NOT_FOUND
    assert (await c.get("/v1/client-apps", headers=wr)).status_code == S.FORBIDDEN
    assert (await c.get(f"/v1/projects/{pid}", headers=adm)).status_code == S.OK
    assert (await c.delete("/v1/projects/prj_nope", headers=adm)).status_code == S.NOT_FOUND

    created_app = build_client_app(
        "turn-coerce", oauth_client_id="turn-coerce-cid", allowed_tiers=(PermissionTier.READ,)
    )
    await edge_env.svc.client_app_repo.create(created_app)
    adhoc = {
        "name": "adhoc_read",
        "description": "Adhoc",
        "permission_tier": "read",
        "execution_target": "server_attachment",
    }
    dup_j = {"content": "x", "client_tools": [adhoc, adhoc]}
    assert (
        await c.post(f"/v1/sessions/{sid}/messages", headers=wr, json=dup_j)
    ).status_code == S.UNPROCESSABLE_CONTENT

    sh = build_client_app(
        "sh",
        oauth_client_id="shadow-cid",
        allowed_tiers=(PermissionTier.READ,),
        registered_tools=(_sp("search_files"),),
    )
    esc = build_client_app(
        "esc",
        oauth_client_id="esc-cid",
        allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
        registered_tools=(_sp("bad_exec", PermissionTier.EXEC),),
    )
    esc_rec = esc.model_copy(update={"allowed_tiers": (PermissionTier.READ,)})
    for app_rec, cid, hdr_code in (
        (sh, "shadow-cid", S.UNPROCESSABLE_CONTENT),
        (esc_rec, "esc-cid", S.FORBIDDEN),
    ):
        await edge_env.svc.client_app_repo.create(app_rec)
        h_bad = edge_env.oidc.mint_headers(sub="edge-writer", aud=cid)
        assert (
            await c.post(f"/v1/sessions/{sid}/messages", headers=h_bad, json={"content": "x"})
        ).status_code == hdr_code

    r_ok = await c.post(
        f"/v1/sessions/{sid}/messages",
        headers=wr,
        json={"content": "go", "client_tools": [adhoc]},
    )
    assert r_ok.status_code == S.OK
    aid = r_ok.json()["pending_approval_id"]
    unreg_h = edge_env.oidc.mint_headers(sub="edge-writer", aud="unreg-cid")
    assert (
        await c.post(f"/v1/tool-approvals/{aid}/result", headers=unreg_h, json={"output": "x"})
    ).status_code == S.FORBIDDEN

    edge_env.inf.emit_server_tool_on_resume = True
    r_fail = await c.post(
        f"/v1/tool-approvals/{aid}/result",
        headers=wr,
        json={"output": "stderr line", "success": False, "error": "exit 1"},
    )
    assert (r_fail.status_code, r_fail.json()["message"]["content"]) == (S.OK, "after-tool:srv-ok")


async def test_resume_direct_fallback_errors_and_client_tool_resolution(edge_env: _EdgeEnv) -> None:
    """Verify non-streaming resume_client_tool, _append_resume_messages fallback, and 404/403."""
    svc = edge_env.svc
    sid2 = (
        await edge_env.client.post(
            "/v1/sessions", headers=edge_env.writer_hdrs, json={"project_id": edge_env.project_id}
        )
    ).json()["id"]
    principal = AuthPrincipal(
        principal_id="u1",
        auth_method=AuthMethod.API_KEY,
        scopes=(ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
    )
    apr = ToolApprovalRecord(
        project_id=edge_env.project_id,
        session_id=sid2,
        tool_name="adhoc_read",
        arguments_json="{}",
        permission_tier=PermissionTier.READ,
        execution_target=ToolExecutionTarget.CLIENT,
        reason="needed",
        status=ApprovalStatus.APPROVED,
        requested_by="u1",
        tool_call_id="tc_fallback",
    )
    out = await svc.orchestrator.resume_client_tool(
        apr, ToolResult(tool_name="adhoc_read", success=False, output=""), principal
    )
    assert '"error": "tool execution failed"' in out.message.content
    msgs = await svc.session_repo.list_messages(sid2)
    assert [m.role for m in msgs] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]

    res_ok = ToolResult(tool_name="adhoc_read", success=True, output="ok")
    with pytest.raises(NotFoundError):
        await svc.orchestrator.resume_client_tool(
            apr.model_copy(update={"session_id": "ses_missing"}), res_ok, principal
        )
    with pytest.raises(AuthorizationError):
        await svc.orchestrator.resume_client_tool(
            apr.model_copy(update={"project_id": "prj_other"}), res_ok, principal
        )

    no_repo_orch = _mk_orch(svc, with_app_repo=False)
    await no_repo_orch.resume_client_tool(
        apr.model_copy(update={"client_app_id": "app_c", "tool_call_id": "tc_nr"}),
        res_ok,
        principal,
    )
    base_srv = ["search_files", "read_file_lines", "get_file_overview", "srv_echo"]
    assert [t.name for t in edge_env.inf.requests[-1].tools] == base_srv

    exp = build_client_app(
        "e",
        "exp-cid",
        expires_at=datetime.now(UTC) - timedelta(seconds=5),
        registered_tools=(_sp("srv_echo"),),
    )
    prj = build_client_app("p", "prj-cid", (PermissionTier.READ,), (_sp("srv_echo"),), ("prj_x",))
    flt = build_client_app("f", "flt-cid", (PermissionTier.READ,), (_sp("srv_echo"), _sp("ok_cli")))
    for app_rec, exp_names in ((exp, base_srv), (prj, base_srv), (flt, [*base_srv, "ok_cli"])):
        await svc.client_app_repo.create(app_rec)
        upd = {"client_app_id": app_rec.id, "tool_call_id": app_rec.id}
        await svc.orchestrator.resume_client_tool(apr.model_copy(update=upd), res_ok, principal)
        assert [t.name for t in edge_env.inf.requests[-1].tools] == exp_names
