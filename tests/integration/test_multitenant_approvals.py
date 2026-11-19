"""Integration tests for cross-principal tool approval isolation under shared OIDC client."""

from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus as S
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from lai import db as db_mod
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.errors import AuthorizationError, ConflictError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    ToolExecutionTarget,
    ToolResult,
    ToolSpec,
)
from lai.ports.inference import InferenceProviderPort
from lai.security import build_client_app
from tests.integration._helpers import (
    IntegrationSpec,
    configure_test_oidc,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
    register_client_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


class _ClientToolPingInference(InferenceProviderPort):
    """Inference double emitting one CLIENT tool call then completing after resume."""

    def __init__(self, *, emit_tool_call: bool = True) -> None:
        self._emit_tool_call = emit_tool_call

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="unused")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        if self._emit_tool_call and (
            not request.messages or request.messages[-1].role.value != "tool"
        ):
            tc = InferenceToolCall(id="tc_ping", name="client_ping", arguments_json="{}")
            yield InferenceChunk(tool_calls=(tc,), finish_reason="tool_calls")
            return
        suffix = request.messages[-1].content if self._emit_tool_call else "ok"
        yield InferenceChunk(
            content_delta=f"ping-done:{suffix}",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


async def test_shared_oidc_client_cross_principal_approval_hijack_and_admin_access(
    tmp_path: Path,
) -> None:
    """Two principals sharing oauth_client_id cannot hijack approvals; ADMIN retains access."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    spec = IntegrationSpec(inference_provider=_ClientToolPingInference(), use_hash_embedder=True)
    async with open_services_harness(settings, spec) as h:
        oidc = configure_test_oidc(
            h.svc,
            audiences=("shared-spa",),
            default_scopes=(
                ApiScope.PROJECTS_WRITE,
                ApiScope.CHAT_WRITE,
                ApiScope.TOOLS_SAFE,
                ApiScope.TOOLS_APPROVE,
            ),
        )
        ping_tool = ToolSpec(
            name="client_ping",
            description="Ping",
            permission_tier=PermissionTier.READ,
            execution_target=ToolExecutionTarget.CLIENT,
            requires_approval_by_default=True,
        )
        app_rec = await register_client_app(
            h.svc,
            "shared-spa-app",
            oauth_client_id="shared-spa",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(ping_tool,),
        )
        h_alice = oidc.mint_headers(sub="alice", email="alice@acme.test", aud="shared-spa")
        h_bob = oidc.mint_headers(sub="bob", email="bob@acme.test", aud="shared-spa")
        h_adm = await mint_auth_headers(h.svc, "adm", (ApiScope.ADMIN,))

        p_a = await create_project(h.client, h_alice, "alice-spa")
        s_a = await create_session(h.client, h_alice, p_a)
        r_turn = await h.client.post(
            f"/v1/sessions/{s_a}/messages", headers=h_alice, json={"content": "ping"}
        )
        aid = r_turn.json()["pending_approval_id"]

        assert (
            await h.client.get(f"/v1/tool-approvals/{aid}", headers=h_bob)
        ).status_code == S.NOT_FOUND
        assert (
            await h.client.post(
                f"/v1/tool-approvals/{aid}/resolve", headers=h_bob, json={"approved": True}
            )
        ).status_code == S.NOT_FOUND
        assert (
            await h.client.post(
                f"/v1/tool-approvals/{aid}/result", headers=h_bob, json={"output": "hijack"}
            )
        ).status_code == S.NOT_FOUND

        assert (await h.client.get(f"/v1/tool-approvals/{aid}", headers=h_adm)).status_code == S.OK
        r_res = await h.client.post(
            f"/v1/tool-approvals/{aid}/resolve", headers=h_adm, json={"approved": True}
        )
        assert r_res.json()["status"] == ApprovalStatus.APPROVED.value

        with pytest.raises(AuthorizationError):
            await h.svc.tool_executor.submit_client_result(
                aid,
                app_rec.id,
                ToolResult(tool_name="client_ping", success=True, output="spoof"),
                principal_id="oidc:bob",
                is_admin=False,
            )

        r_ok = await h.client.post(
            f"/v1/tool-approvals/{aid}/result", headers=h_alice, json={"output": "pong"}
        )
        assert r_ok.status_code == S.OK
        assert r_ok.json()["message"]["content"] == "ping-done:pong"


async def test_client_app_conflict_revoke_reregister_and_unique_migration(tmp_path: Path) -> None:
    """Active duplicate oauth_client_id returns 409; revoke permits re-register; DB migrates."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    async with open_services_harness(settings) as h:
        h_adm = await mint_auth_headers(h.svc, "adm", (ApiScope.ADMIN,))
        payload = {"name": "web-spa", "oauth_client_id": "cid-spa", "allowed_tiers": ["read"]}
        r1 = await h.client.post("/v1/client-apps", headers=h_adm, json=payload)
        assert r1.status_code == S.CREATED
        id1 = str(r1.json()["id"])

        r_dup = await h.client.post("/v1/client-apps", headers=h_adm, json=payload)
        assert (r_dup.status_code, r_dup.json()["error"]["code"]) == (S.CONFLICT, "conflict")
        alerts = h.svc.alert_manager.recent_alerts()
        assert not any(a.category is AlertCategory.STORAGE_FAILURE for a in alerts)

        assert (await h.client.delete(f"/v1/client-apps/{id1}", headers=h_adm)).status_code == S.OK
        r2 = await h.client.post("/v1/client-apps", headers=h_adm, json=payload)
        assert r2.status_code == S.CREATED
        id2 = str(r2.json()["id"])
        assert id2 != id1
        active = await h.svc.client_app_repo.get_by_oauth_client_id("cid-spa")
        assert active is not None
        assert (active.id, active.is_active()) == (id2, True)
        r_dup2 = await h.client.post("/v1/client-apps", headers=h_adm, json=payload)
        assert r_dup2.status_code == S.CONFLICT
        assert (await h.client.delete(f"/v1/client-apps/{id2}", headers=h_adm)).status_code == S.OK
        revoked_rec = await h.svc.client_app_repo.get_by_oauth_client_id("cid-spa")
        assert revoked_rec is not None
        assert (revoked_rec.id, revoked_rec.is_active()) == (id2, False)

    mig_path = tmp_path / "inline_unique.db"
    ts = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    async with aiosqlite.connect(mig_path) as raw:
        await raw.executescript(
            "CREATE TABLE client_apps (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "oauth_client_id TEXT NOT NULL UNIQUE, allowed_tiers_json TEXT NOT NULL, "
            "registered_tools_json TEXT NOT NULL DEFAULT '[]', "
            "allowed_project_ids_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, "
            "last_used_at TEXT, expires_at TEXT, revoked_at TEXT);"
            f"INSERT INTO client_apps VALUES('app_old_act','a','cid-active','[\"read\"]',"
            f"'[]','[]','{ts}',NULL,NULL,NULL);"
            f"INSERT INTO client_apps VALUES('app_old_rev','r','cid-revoked','[\"read\"]',"
            f"'[]','[]','{ts}',NULL,NULL,'{ts}');"
        )
        await raw.commit()

    async with db_mod.SqliteDatabase(mig_path) as db:
        await db.initialize()
        repo = db_mod.SqliteClientAppRepository(db)
        assert {rec.id for rec in await repo.list_all()} == {"app_old_act", "app_old_rev"}
        re_reg = await repo.create(build_client_app("re-reg", "cid-revoked"))
        fetched = await repo.get_by_oauth_client_id("cid-revoked")
        assert fetched is not None
        assert fetched.id == re_reg.id
        with pytest.raises(ConflictError):
            await repo.create(build_client_app("dup-active", "cid-active"))


async def test_session_message_attached_file_ids_tenancy_and_project_scope(tmp_path: Path) -> None:
    """Session messages mask cross-tenant attached_file_ids as 404 and wrong-project as 403."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )
    spec = IntegrationSpec(
        inference_provider=_ClientToolPingInference(emit_tool_call=False),
        use_hash_embedder=True,
    )
    async with open_services_harness(settings, spec) as h:
        scopes = (
            ApiScope.PROJECTS_WRITE,
            ApiScope.FILES_WRITE,
            ApiScope.CHAT_WRITE,
            ApiScope.TOOLS_SAFE,
        )
        h_alice = await mint_auth_headers(h.svc, "alice", scopes)
        h_bob = await mint_auth_headers(h.svc, "bob", scopes)
        p_a1 = await create_project(h.client, h_alice, "alice-1")
        p_a2 = await create_project(h.client, h_alice, "alice-2")
        p_b = await create_project(h.client, h_bob, "bob-1")
        s_a1 = await create_session(h.client, h_alice, p_a1)

        f_a1 = str(
            (
                await h.client.post(
                    "/v1/files?wait=true",
                    headers=h_alice,
                    data={"project_id": p_a1},
                    files={"file": ("a1.txt", b"alice1\n", "text/plain")},
                )
            ).json()["id"]
        )
        f_a2 = str(
            (
                await h.client.post(
                    "/v1/files?wait=true",
                    headers=h_alice,
                    data={"project_id": p_a2},
                    files={"file": ("a2.txt", b"alice2\n", "text/plain")},
                )
            ).json()["id"]
        )
        f_b = str(
            (
                await h.client.post(
                    "/v1/files?wait=true",
                    headers=h_bob,
                    data={"project_id": p_b},
                    files={"file": ("b.txt", b"bob\n", "text/plain")},
                )
            ).json()["id"]
        )

        for bad_id in (f_b, "fil_missing"):
            r_404 = await h.client.post(
                f"/v1/sessions/{s_a1}/messages",
                headers=h_alice,
                json={"content": "check", "attached_file_ids": [bad_id]},
            )
            assert (r_404.status_code, r_404.json()["error"]["code"]) == (
                S.NOT_FOUND,
                "not_found",
            )

        r_403 = await h.client.post(
            f"/v1/sessions/{s_a1}/messages",
            headers=h_alice,
            json={"content": "check", "attached_file_ids": [f_a2]},
        )
        assert (r_403.status_code, r_403.json()["error"]["code"]) == (
            S.FORBIDDEN,
            "authorization_denied",
        )

        r_ok = await h.client.post(
            f"/v1/sessions/{s_a1}/messages",
            headers=h_alice,
            json={"content": "check", "attached_file_ids": [f_a1]},
        )
        assert r_ok.status_code == S.OK
