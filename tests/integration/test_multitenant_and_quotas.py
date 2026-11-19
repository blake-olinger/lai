"""Integration tests for multi-tenant 404 isolation, per-principal quotas, and admin RBAC."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus as S
from typing import TYPE_CHECKING

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import BackendHealth, InferenceChunk, InferenceRequest, InferenceResponse
from lai.domain.observability import LoggingConfig
from lai.ports.inference import InferenceProviderPort
from lai.ports.scheduler import InferenceSpoolerPort
from lai.scheduler import InferenceSpooler
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from lai.domain.scheduler import LeaseRequest, SpoolerSnapshot


class _TenantTestInference(InferenceProviderPort):
    """Inference double supporting optional hold gates for queue quota tests."""

    def __init__(self, *, hold_first_call: bool = False) -> None:
        self.hold_first_call = hold_first_call
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._held_once = False

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="ok")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        if self.hold_first_call and not self._held_once:
            self._held_once = True
            self.entered.set()
            await self.release.wait()
        yield InferenceChunk(
            content_delta="tenant-ok",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _RecordingSpooler(InferenceSpoolerPort):
    """Spooler wrapper recording LeaseRequest metadata without altering scheduling."""

    def __init__(self) -> None:
        self._inner = InferenceSpooler()
        self.requests: list[LeaseRequest] = []

    @asynccontextmanager
    async def lease(self, request: LeaseRequest) -> AsyncGenerator[None]:
        self.requests.append(request)
        async with self._inner.lease(request):
            yield

    def snapshot(self) -> SpoolerSnapshot:
        return self._inner.snapshot()


_THREE_PROJECTS = 3


def _tenant_settings(
    tmp_path: Path,
    *,
    max_projects_per_principal: int = 10,
    max_storage_bytes_per_principal: int = 1_073_741_824,
    max_queued_jobs_per_principal: int = 4,
) -> LaiSettings:
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
        max_projects_per_principal=max_projects_per_principal,
        max_storage_bytes_per_principal=max_storage_bytes_per_principal,
        max_queued_jobs_per_principal=max_queued_jobs_per_principal,
    )


async def test_cross_tenant_404_isolation_across_projects_sessions_files_and_openai(
    tmp_path: Path,
) -> None:
    """Non-admins receive 404 across another principal's projects, sessions, files, and OpenAI."""
    inf = _TenantTestInference()
    spooler = _RecordingSpooler()
    spec = IntegrationSpec(inference_provider=inf, spooler=spooler, use_hash_embedder=True)
    async with open_services_harness(_tenant_settings(tmp_path), spec) as h:
        scopes_a = (
            ApiScope.PROJECTS_WRITE,
            ApiScope.FILES_WRITE,
            ApiScope.CHAT_WRITE,
            ApiScope.TOOLS_SAFE,
        )
        h_a = await mint_auth_headers(h.svc, "alice", scopes_a)
        h_b = await mint_auth_headers(h.svc, "bob", scopes_a)
        h_adm = await mint_auth_headers(h.svc, "admin", (ApiScope.ADMIN,))
        h_ro = await mint_auth_headers(h.svc, "reader", (ApiScope.CHAT_READ,))

        assert (await h.client.get("/v1/system/status", headers=h_ro)).status_code == S.FORBIDDEN
        assert (await h.client.get("/v1/system/status", headers=h_adm)).status_code == S.OK

        p_a = await create_project(h.client, h_a, "alice-proj")
        p_a2 = await create_project(h.client, h_a, "alice-proj-2")
        p_b = await create_project(h.client, h_b, "bob-proj")
        items_a = (await h.client.get("/v1/projects", headers=h_a)).json()["items"]
        items_b = (await h.client.get("/v1/projects", headers=h_b)).json()["items"]
        assert {p["id"] for p in items_a} == {p_a, p_a2}
        assert len(items_a) == len({p_a, p_a2})
        assert [p["id"] for p in items_b] == [p_b]
        assert (await h.client.get("/v1/projects", headers=h_adm)).json()[
            "count"
        ] == _THREE_PROJECTS
        assert (await h.client.get(f"/v1/projects/{p_a}", headers=h_b)).status_code == S.NOT_FOUND
        assert (
            await h.client.delete(f"/v1/projects/{p_a}", headers=h_b)
        ).status_code == S.NOT_FOUND
        assert (await h.client.get(f"/v1/projects/{p_a}", headers=h_adm)).status_code == S.OK

        s_a = await create_session(h.client, h_a, p_a)
        assert (
            await h.client.post("/v1/sessions", headers=h_b, json={"project_id": p_a})
        ).status_code == S.NOT_FOUND
        assert (
            await h.client.get(f"/v1/sessions?project_id={p_a}", headers=h_b)
        ).status_code == S.NOT_FOUND
        assert (await h.client.get(f"/v1/sessions/{s_a}", headers=h_b)).status_code == S.NOT_FOUND
        assert (
            await h.client.delete(f"/v1/sessions/{s_a}", headers=h_b)
        ).status_code == S.NOT_FOUND
        assert (
            await h.client.post(f"/v1/sessions/{s_a}/messages", headers=h_b, json={"content": "x"})
        ).status_code == S.NOT_FOUND
        assert (await h.client.get(f"/v1/sessions/{s_a}", headers=h_adm)).status_code == S.OK

        f_a = (
            await h.client.post(
                "/v1/files?wait=true",
                headers=h_a,
                data={"project_id": p_a},
                files={"file": ("a.txt", b"hello alice\n", "text/plain")},
            )
        ).json()["id"]
        assert (
            await h.client.post(
                "/v1/files?wait=true",
                headers=h_b,
                data={"project_id": p_a},
                files={"file": ("b.txt", b"no", "text/plain")},
            )
        ).status_code == S.NOT_FOUND
        assert (
            await h.client.post(
                f"/v1/files/stream?project_id={p_a}&wait=true",
                headers={**h_b, "X-LAI-Filename": "b.txt"},
                content=b"no",
            )
        ).status_code == S.NOT_FOUND
        for method, url in (
            ("GET", f"/v1/files?project_id={p_a}"),
            ("GET", f"/v1/files/{f_a}"),
            ("GET", f"/v1/files/{f_a}/chunks"),
            ("DELETE", f"/v1/files/{f_a}"),
        ):
            assert (await h.client.request(method, url, headers=h_b)).status_code == S.NOT_FOUND
        assert (await h.client.get(f"/v1/files/{f_a}", headers=h_adm)).status_code == S.OK

        oai_msgs = [{"role": "user", "content": "hi"}]
        for payload in (
            {"project_id": p_a, "messages": oai_msgs},
            {"session_id": s_a, "messages": oai_msgs},
            {"project_id": p_b, "file_ids": [f_a], "messages": oai_msgs},
        ):
            assert (
                await h.client.post("/v1/chat/completions", headers=h_b, json=payload)
            ).status_code == S.NOT_FOUND

        r_no_proj = await h.client.post(
            "/v1/chat/completions", headers=h_a, json={"file_ids": [f_a], "messages": oai_msgs}
        )
        assert r_no_proj.status_code == S.UNPROCESSABLE_CONTENT
        r_wrong_proj = await h.client.post(
            "/v1/chat/completions",
            headers=h_a,
            json={"project_id": p_a2, "file_ids": [f_a], "messages": oai_msgs},
        )
        assert r_wrong_proj.status_code == S.FORBIDDEN

        assert (
            await h.client.post("/v1/chat/completions", headers=h_a, json={"messages": oai_msgs})
        ).status_code == S.OK
        assert (
            await h.client.post("/v1/chat/completions", headers=h_b, json={"messages": oai_msgs})
        ).status_code == S.OK
        lease_a, lease_b = spooler.requests[-2], spooler.requests[-1]
        assert lease_a.project_id == f"default:{lease_a.principal_id}"
        assert lease_b.project_id == f"default:{lease_b.principal_id}"
        assert lease_a.project_id != lease_b.project_id


async def test_project_and_storage_quotas_enforce_limits_and_exempt_admins(tmp_path: Path) -> None:
    """Verify max_projects_per_principal (403) and max_storage_bytes_per_principal (413)."""
    settings = _tenant_settings(
        tmp_path, max_projects_per_principal=2, max_storage_bytes_per_principal=16
    )
    async with open_services_harness(settings, IntegrationSpec(use_hash_embedder=True)) as h:
        h_user = await mint_auth_headers(
            h.svc, "quota-user", (ApiScope.PROJECTS_WRITE, ApiScope.FILES_WRITE)
        )
        h_adm = await mint_auth_headers(h.svc, "quota-admin", (ApiScope.ADMIN,))

        p1 = await create_project(h.client, h_user, "q1")
        _ = await create_project(h.client, h_user, "q2")
        r_q3 = await h.client.post("/v1/projects", headers=h_user, json={"name": "q3"})
        assert (r_q3.status_code, r_q3.json()["error"]["code"]) == (S.FORBIDDEN, "quota_exceeded")
        assert r_q3.json()["error"]["details"]["max_projects_per_principal"] == "2"

        for i in range(3):
            assert (
                await h.client.post("/v1/projects", headers=h_adm, json={"name": f"adm-{i}"})
            ).status_code == S.CREATED

        r_up1 = await h.client.post(
            "/v1/files?wait=true",
            headers=h_user,
            data={"project_id": p1},
            files={"file": ("ten.txt", b"0123456789", "text/plain")},
        )
        assert r_up1.status_code == S.CREATED

        r_mid = await h.client.post(
            "/v1/files?wait=true",
            headers=h_user,
            data={"project_id": p1},
            files={"file": ("over.txt", b"abcdefghij", "text/plain")},
        )
        assert r_mid.status_code == S.REQUEST_ENTITY_TOO_LARGE
        listed_files = (await h.client.get(f"/v1/files?project_id={p1}", headers=h_user)).json()
        assert listed_files["count"] == 1

        r_exact = await h.client.post(
            f"/v1/files/stream?project_id={p1}&wait=true",
            headers={**h_user, "X-LAI-Filename": "six.txt"},
            content=b"123456",
        )
        assert r_exact.status_code == S.CREATED

        r_pre = await h.client.post(
            f"/v1/files/stream?project_id={p1}&wait=true",
            headers={**h_user, "X-LAI-Filename": "one.txt"},
            content=b"x",
        )
        assert r_pre.status_code == S.REQUEST_ENTITY_TOO_LARGE

        r_adm_up = await h.client.post(
            "/v1/files?wait=true",
            headers=h_adm,
            data={"project_id": p1},
            files={"file": ("big.txt", b"x" * 64, "text/plain")},
        )
        assert r_adm_up.status_code == S.CREATED


async def test_max_queued_jobs_per_principal_503_retry_after_and_admin_bypass(
    tmp_path: Path,
) -> None:
    """Verify max_queued_jobs_per_principal returns 503 + Retry-After while admins bypass."""
    inf = _TenantTestInference(hold_first_call=True)
    settings = _tenant_settings(tmp_path, max_queued_jobs_per_principal=1)
    async with open_services_harness(settings, IntegrationSpec(inference_provider=inf)) as h:
        h_u1 = await mint_auth_headers(h.svc, "u1", (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE))
        h_u2 = await mint_auth_headers(h.svc, "u2", (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE))
        h_adm = await mint_auth_headers(h.svc, "adm", (ApiScope.ADMIN,))

        p1 = await create_project(h.client, h_u1, "qp1")
        s1_a = await create_session(h.client, h_u1, p1)
        s1_b = await create_session(h.client, h_u1, p1)
        s1_c = await create_session(h.client, h_u1, p1)

        t_hold = asyncio.create_task(
            h.client.post(f"/v1/sessions/{s1_a}/messages", headers=h_u1, json={"content": "hold"})
        )
        await asyncio.wait_for(inf.entered.wait(), timeout=5.0)

        t_q1 = asyncio.create_task(
            h.client.post(f"/v1/sessions/{s1_b}/messages", headers=h_u1, json={"content": "q1"})
        )
        await asyncio.sleep(0.03)

        r_503 = await h.client.post(
            f"/v1/sessions/{s1_c}/messages", headers=h_u1, json={"content": "q2"}
        )
        assert (r_503.status_code, r_503.json()["error"]["code"]) == (
            S.SERVICE_UNAVAILABLE,
            "queue_full",
        )
        assert int(r_503.headers["Retry-After"]) >= 1

        oai_msg = {"messages": [{"role": "user", "content": "ok"}]}
        t_u2 = asyncio.create_task(
            h.client.post("/v1/chat/completions", headers=h_u2, json=oai_msg)
        )
        t_adm1 = asyncio.create_task(
            h.client.post("/v1/chat/completions", headers=h_adm, json=oai_msg)
        )
        t_adm2 = asyncio.create_task(
            h.client.post("/v1/chat/completions", headers=h_adm, json=oai_msg)
        )
        await asyncio.sleep(0.03)
        inf.release.set()
        done = await asyncio.wait_for(
            asyncio.gather(t_hold, t_q1, t_u2, t_adm1, t_adm2), timeout=5.0
        )
        assert all(resp.status_code == S.OK for resp in done)
