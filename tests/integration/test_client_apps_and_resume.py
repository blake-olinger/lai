"""Integration tests for /v1/client-apps, turn client auth, CAS /result, and APPROVAL_RESUME."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus as S
from typing import TYPE_CHECKING

import httpx
import pytest

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.observability import LoggingConfig
from lai.domain.scheduler import JobPriority
from lai.domain.tools import ApprovalStatus, PermissionTier, ToolSpec
from lai.orchestrator import TurnOrchestrator
from lai.ports.inference import InferenceProviderPort
from lai.security import build_client_app, generate_api_key
from tests.integration._helpers import (
    OidcTestAuthority,
    configure_test_oidc,
    create_project,
    create_session,
    register_client_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Generator
    from pathlib import Path

    from lai.api.deps import AppServices
    from lai.ports.scheduler import InferenceSpoolerPort

_CAS_WORKERS = 10


class _ResumeTrackingInference(InferenceProviderPort):
    """Inference stub recording spooler lease priorities and supporting chained tool calls."""

    def __init__(self, spooler: InferenceSpoolerPort) -> None:
        self.spooler = spooler
        self.requests: list[InferenceRequest] = []
        self.priorities: list[JobPriority | None] = []
        self.chain_read_after_exec = False

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="unused")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.requests.append(request)
        self.priorities.append(self.spooler.snapshot().active_priority)
        tool_msgs = [m for m in request.messages if m.role.value == "tool"]
        if not tool_msgs:
            if any(t.name == "run_local" for t in request.tools):
                tc = InferenceToolCall(
                    id="tc_exec_1", name="run_local", arguments_json='{"cmd":"ls"}'
                )
                yield InferenceChunk(tool_calls=(tc,), finish_reason="tool_calls")
                return
            yield InferenceChunk(
                content_delta="no-client-tools",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )
            return

        if self.chain_read_after_exec and len(tool_msgs) == 1:
            tc2 = InferenceToolCall(
                id="tc_read_2", name="read_local", arguments_json='{"path":"b"}'
            )
            yield InferenceChunk(tool_calls=(tc2,), finish_reason="tool_calls")
            return

        yield InferenceChunk(
            content_delta=f"Resumed answer with: {tool_msgs[-1].content}",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=15, completion_tokens=6, total_tokens=21),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@dataclass(slots=True)
class _Env:
    svc: AppServices
    client: httpx.AsyncClient
    inf: _ResumeTrackingInference
    oidc: OidcTestAuthority
    admin_hdrs: dict[str, str]
    user_hdrs: dict[str, str]
    no_oidc_hdrs: dict[str, str]
    project_id: str
    other_project_id: str
    session_id: str


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[_Env]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    svc = await build_app_services(settings)
    inf = _ResumeTrackingInference(svc.spooler)
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
    admin_key = generate_api_key("adm", (ApiScope.ADMIN,))
    no_oidc_key = generate_api_key(
        "usr-no-oidc",
        (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.CHAT_READ, ApiScope.TOOLS_EXEC),
    )
    await svc.api_key_repo.create(admin_key.record)
    await svc.api_key_repo.create(no_oidc_key.record)
    oidc = configure_test_oidc(
        svc,
        audiences=("cli-client", "desktop-client", "exp-client", "unreg-client"),
        default_scopes=(
            ApiScope.PROJECTS_READ,
            ApiScope.PROJECTS_WRITE,
            ApiScope.CHAT_READ,
            ApiScope.CHAT_WRITE,
            ApiScope.TOOLS_SAFE,
            ApiScope.TOOLS_EXEC,
        ),
    )
    admin_hdrs = {"Authorization": f"Bearer {admin_key.plaintext_key}"}
    no_oidc_hdrs = {"Authorization": f"Bearer {no_oidc_key.plaintext_key}"}
    user_hdrs = oidc.mint_headers(sub="usr-1", aud="cli-client")

    app = create_app(settings, services=svc)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            p1 = await create_project(client, user_hdrs, "p1")
            p2 = await create_project(client, user_hdrs, "p2")
            s1 = await create_session(client, user_hdrs, p1)
            yield _Env(svc, client, inf, oidc, admin_hdrs, user_hdrs, no_oidc_hdrs, p1, p2, s1)
    finally:
        await close_app_services(svc)


async def test_client_apps_crud_stripping_tier_shadow_and_project_scope(env: _Env) -> None:
    """Verify /v1/client-apps CRUD, fail-closed OIDC client binding, tier/shadow 403/422."""
    c, adm, usr, pid = env.client, env.admin_hdrs, env.user_hdrs, env.project_id
    run_tool = {
        "name": "run_local",
        "description": "Run",
        "permission_tier": "exec",
        "execution_target": "client",
    }
    read_tool = {
        "name": "read_local",
        "description": "Read",
        "permission_tier": "read",
        "execution_target": "client",
    }
    create_payload = {
        "name": "cli-app",
        "oauth_client_id": "cli-client",
        "allowed_tiers": ["read"],
        "registered_tools": [read_tool],
        "allowed_project_ids": [pid],
    }
    assert (
        await c.post("/v1/client-apps", headers=usr, json=create_payload)
    ).status_code == S.FORBIDDEN
    r_cr = await c.post("/v1/client-apps", headers=adm, json=create_payload)
    assert r_cr.status_code == S.CREATED
    assert "secret_hash" not in r_cr.text
    assert "client_secret" not in r_cr.text
    app_id = r_cr.json()["id"]
    assert r_cr.json()["oauth_client_id"] == "cli-client"
    r_ls = await c.get("/v1/client-apps", headers=adm)
    assert (r_ls.status_code, r_ls.json()["count"]) == (S.OK, 1)
    assert "secret_hash" not in r_ls.text

    s_url = f"/v1/sessions/{env.session_id}/messages"
    oai_msgs = [{"role": "user", "content": "hi"}]
    r_api_fail = await c.post(
        "/v1/chat/completions",
        headers=env.no_oidc_hdrs,
        json={"messages": oai_msgs, "client_tools": [run_tool]},
    )
    assert r_api_fail.status_code == S.UNAUTHORIZED
    r_api_plain = await c.post(
        "/v1/chat/completions", headers=env.no_oidc_hdrs, json={"messages": oai_msgs}
    )
    assert (r_api_plain.status_code, r_api_plain.json()["choices"][0]["message"]["content"]) == (
        S.OK,
        "no-client-tools",
    )
    unreg_h = env.oidc.mint_headers(sub="usr-1", aud="unreg-client")
    r_unreg = await c.post(
        s_url, headers=unreg_h, json={"content": "hi", "client_tools": [run_tool]}
    )
    assert r_unreg.status_code == S.FORBIDDEN
    r_plain = await c.post(s_url, headers=unreg_h, json={"content": "hi"})
    assert (r_plain.status_code, r_plain.json()["message"]["content"]) == (S.OK, "no-client-tools")
    assert all(t.name != "run_local" for t in env.inf.requests[-1].tools)

    shadow_t = {"name": "search_files", "description": "x", "permission_tier": "read"}
    assert (
        await c.post(s_url, headers=usr, json={"content": "hi", "client_tools": [run_tool]})
    ).status_code == S.FORBIDDEN
    assert (
        await c.post(s_url, headers=usr, json={"content": "hi", "client_tools": [shadow_t]})
    ).status_code == S.UNPROCESSABLE_CONTENT
    r_prj = await c.post(
        "/v1/chat/completions",
        headers={**usr, "X-LAI-Project-Id": env.other_project_id},
        json={"messages": oai_msgs, "client_tools": [read_tool]},
    )
    assert r_prj.status_code == S.FORBIDDEN

    assert (await c.delete(f"/v1/client-apps/{app_id}", headers=adm)).status_code == S.OK
    r_rev = await c.post(
        "/v1/chat/completions",
        headers=usr,
        json={"project_id": pid, "messages": oai_msgs, "client_tools": [read_tool]},
    )
    assert r_rev.status_code == S.FORBIDDEN

    exp = build_client_app(
        "exp",
        oauth_client_id="exp-client",
        expires_at=datetime.now(UTC) - timedelta(seconds=10),
    )
    await env.svc.client_app_repo.create(exp)
    exp_h = env.oidc.mint_headers(sub="usr-1", aud="exp-client")
    r_exp = await c.post(s_url, headers=exp_h, json={"content": "hi", "client_tools": [read_tool]})
    assert r_exp.status_code == S.FORBIDDEN


async def test_pending_resolve_10way_result_cas_approval_resume_and_chaining(env: _Env) -> None:
    """Verify PENDING -> /resolve -> 10x /result CAS -> APPROVAL_RESUME + chaining + history."""
    c, adm, sid = env.client, env.admin_hdrs, env.session_id
    env.inf.chain_read_after_exec = True
    exec_spec = ToolSpec(name="run_local", description="Run", permission_tier=PermissionTier.EXEC)
    read_spec = ToolSpec(name="read_local", description="Read", permission_tier=PermissionTier.READ)
    await register_client_app(
        env.svc,
        "desktop",
        oauth_client_id="desktop-client",
        allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
        registered_tools=(exec_spec, read_spec),
        allowed_project_ids=(env.project_id,),
    )
    cli_hdrs = env.oidc.mint_headers(sub="usr-1", aud="desktop-client")

    r_turn = await c.post(
        f"/v1/sessions/{sid}/messages", headers=cli_hdrs, json={"content": "Inspect workspace"}
    )
    assert r_turn.status_code == S.OK
    aid_1 = r_turn.json()["pending_approval_id"]
    assert r_turn.json()["pending_client_tool_call"]["requires_approval"] is True
    assert env.inf.priorities[0] is JobPriority.INTERACTIVE

    r_res = await c.post(
        f"/v1/tool-approvals/{aid_1}/resolve", headers=adm, json={"approved": True}
    )
    assert (r_res.status_code, r_res.json()["status"]) == (S.OK, ApprovalStatus.APPROVED.value)

    cas_resps = await asyncio.gather(
        *[
            c.post(
                f"/v1/tool-approvals/{aid_1}/result",
                headers=cli_hdrs,
                json={"output": "exec-ok", "success": True, "duration_ms": 12.5},
            )
            for _ in range(_CAS_WORKERS)
        ]
    )
    expected_codes = [S.OK] + [S.UNPROCESSABLE_ENTITY] * (_CAS_WORKERS - 1)
    assert sorted(r.status_code for r in cas_resps) == expected_codes
    assert env.inf.priorities[1] is JobPriority.APPROVAL_RESUME

    ok_resume = next(r.json() for r in cas_resps if r.status_code == S.OK)
    aid_2 = ok_resume["pending_approval_id"]
    assert ok_resume["pending_client_tool_call"]["tool_name"] == "read_local"
    assert ok_resume["pending_client_tool_call"]["requires_approval"] is False

    r_res2 = await c.post(
        f"/v1/tool-approvals/{aid_2}/result",
        headers=cli_hdrs,
        json={"output": "chained-read-ok", "success": True},
    )
    assert r_res2.status_code == S.OK
    assert r_res2.json()["message"]["content"] == "Resumed answer with: chained-read-ok"
    assert env.inf.priorities[2] is JobPriority.APPROVAL_RESUME

    hist = (await c.get(f"/v1/sessions/{sid}", headers=cli_hdrs)).json()["messages"]
    assert [(m["role"], m["tool_call_id"], len(m["tool_calls"])) for m in hist] == [
        ("user", None, 0),
        ("assistant", None, 1),
        ("tool", "tc_exec_1", 0),
        ("assistant", None, 1),
        ("tool", "tc_read_2", 0),
        ("assistant", None, 0),
    ]
