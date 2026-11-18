"""Integration tests for CLIENT EXEC tool policy denial, SSE dispatch, CAS, and rejection."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import ContextMode
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
    ToolExecutionTarget,
    ToolSpec,
)
from lai.ports.inference import InferenceProviderPort
from tests.integration._helpers import (
    IntegrationHarness,
    IntegrationSpec,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
    register_client_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    import httpx

_CAS_CONCURRENCY = 10


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
class _ClientToolEnv:
    harness: IntegrationHarness
    project_id: str
    session_id: str
    h_chat: dict[str, str]
    h_exec: dict[str, str]
    h_approve: dict[str, str]

    @property
    def client(self) -> httpx.AsyncClient:
        return self.harness.client


@asynccontextmanager
async def _open_client_tool_env(tmp_path: Path) -> AsyncGenerator[_ClientToolEnv]:
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    spec = IntegrationSpec(
        inference_provider=_ClientToolInferenceProvider(),
        use_hash_embedder=True,
    )
    async with open_services_harness(settings, spec) as harness:
        h_chat_base = await mint_auth_headers(
            harness.svc,
            "chat_only",
            (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
        )
        h_exec_base = await mint_auth_headers(
            harness.svc,
            "exec_user",
            (ApiScope.PROJECTS_READ, ApiScope.CHAT_WRITE, ApiScope.TOOLS_EXEC),
        )
        h_approve = await mint_auth_headers(
            harness.svc, "approver", (ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE)
        )
        ps_spec = ToolSpec(
            name="powershell_preview",
            description="Preview a local PowerShell command.",
            permission_tier=PermissionTier.EXEC,
            parameters_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            requires_approval_by_default=True,
            execution_target=ToolExecutionTarget.CLIENT,
        )
        app_id, app_secret = await register_client_app(
            harness.svc,
            "ps-helper",
            allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
            registered_tools=(ps_spec,),
        )
        cli_hdrs = {"X-LAI-Client-Id": app_id, "X-LAI-Client-Secret": app_secret}
        pid = await create_project(harness.client, h_chat_base, "tp")
        sid = await create_session(harness.client, h_chat_base, pid)
        yield _ClientToolEnv(
            harness=harness,
            project_id=pid,
            session_id=sid,
            h_chat={**h_chat_base, **cli_hdrs},
            h_exec={**h_exec_base, **cli_hdrs},
            h_approve=h_approve,
        )


async def test_client_exec_tool_denied_without_tools_exec_scope(tmp_path: Path) -> None:
    """Caller without TOOLS_EXEC hits PolicyDecision.DENY and writes policy_denied audit."""
    async with _open_client_tool_env(tmp_path) as env:
        r_deny = await env.client.post(
            f"/v1/sessions/{env.session_id}/messages",
            headers=env.h_chat,
            json={
                "content": "run ps",
                "context_mode": ContextMode.RETRIEVAL.value,
                "stream": False,
            },
        )
        assert r_deny.status_code == HTTPStatus.OK
        audits = await env.harness.svc.tool_audit_repo.list_audits(env.project_id, env.session_id)
        assert any(
            a.tool_name == "powershell_preview"
            and a.policy_decision is PolicyDecision.DENY
            and a.error == "policy_denied"
            for a in audits
        )


async def test_client_exec_tool_sse_dispatch_and_10way_resolve_cas(tmp_path: Path) -> None:
    """CLIENT EXEC emits SSE approval events, forbids non-approver, and enforces CAS."""
    async with _open_client_tool_env(tmp_path) as env:
        r_sse = await env.client.post(
            f"/v1/sessions/{env.session_id}/messages",
            headers=env.h_exec,
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

        r_get = await env.client.get(f"/v1/tool-approvals/{approval_id}", headers=env.h_approve)
        assert r_get.json()["status"] == ApprovalStatus.PENDING.value
        assert r_get.json()["execution_target"] == ToolExecutionTarget.CLIENT.value
        r_forbid = await env.client.post(
            f"/v1/tool-approvals/{approval_id}/resolve",
            headers=env.h_exec,
            json={"approved": True},
        )
        assert r_forbid.status_code == HTTPStatus.FORBIDDEN

        resolve_resps = await asyncio.gather(
            *[
                env.client.post(
                    f"/v1/tool-approvals/{approval_id}/resolve",
                    headers=env.h_approve,
                    json={"approved": True, "note": "approved by reviewer"},
                )
                for _ in range(_CAS_CONCURRENCY)
            ]
        )
        expected_codes = [HTTPStatus.OK] + [HTTPStatus.UNPROCESSABLE_ENTITY] * (
            _CAS_CONCURRENCY - 1
        )
        assert sorted(r.status_code for r in resolve_resps) == expected_codes
        winner = next(r.json() for r in resolve_resps if r.status_code == HTTPStatus.OK)
        assert winner["status"] == ApprovalStatus.APPROVED.value
        assert winner["expires_at"] is not None


async def test_client_exec_tool_denial_transitions_to_denied(tmp_path: Path) -> None:
    """Resolving a pending CLIENT approval with approved=False sets DENIED status."""
    async with _open_client_tool_env(tmp_path) as env:
        r_turn = await env.client.post(
            f"/v1/sessions/{env.session_id}/messages",
            headers=env.h_exec,
            json={
                "content": "run ps 2",
                "context_mode": ContextMode.RETRIEVAL.value,
                "stream": False,
            },
        )
        approval_id = r_turn.json()["pending_approval_id"]
        r_reject = await env.client.post(
            f"/v1/tool-approvals/{approval_id}/resolve",
            headers=env.h_approve,
            json={"approved": False, "note": "rejected"},
        )
        assert (r_reject.status_code, r_reject.json()["status"]) == (
            HTTPStatus.OK,
            ApprovalStatus.DENIED.value,
        )
