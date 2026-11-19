"""Unit boundary and mutation-killing tests for lai.api._tenancy and lai.api._client_auth."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from lai.api._client_auth import (
    require_authenticated_client_app,
    resolve_turn_client_context,
)
from lai.api._tenancy import (
    enforce_and_wrap_storage_quota,
    enforce_project_quota,
    is_admin_principal,
    require_accessible_approval,
    require_accessible_file,
    require_accessible_project,
    require_accessible_session,
    require_project_files,
)
from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope, AuthMethod, AuthPrincipal
from lai.domain.chat import Session
from lai.domain.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationFailedError,
)
from lai.domain.files import FileKind, FileRecord, FileStatus
from lai.domain.projects import Project
from lai.domain.tools import (
    ApprovalStatus,
    PermissionTier,
    ToolApprovalRecord,
    ToolExecutionTarget,
    ToolSpec,
)
from lai.security import build_client_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from lai.api.deps import AppServices
    from lai.domain.client_apps import ClientAppRecord


def _principal(pid: str, *scopes: ApiScope, oauth_client_id: str | None = None) -> AuthPrincipal:
    return AuthPrincipal(
        principal_id=pid,
        auth_method=AuthMethod.OIDC if oauth_client_id else AuthMethod.API_KEY,
        scopes=scopes,
        oauth_client_id=oauth_client_id,
    )


async def test_tenancy_boundaries_and_quota_thresholds() -> None:
    """Kill comparison/detail mutants across _tenancy.py ownership and quota boundaries."""
    alice = _principal("alice", ApiScope.PROJECTS_WRITE)
    bob = _principal("bob", ApiScope.PROJECTS_WRITE)
    adm = _principal("root", ApiScope.ADMIN)
    assert is_admin_principal(adm) is True
    assert is_admin_principal(alice) is False

    prj_a = Project(id="prj_a", name="A", owner_principal_id="alice")
    ses_a = Session(id="ses_a", project_id="prj_a")
    fil_a = FileRecord(
        id="fil_a",
        project_id="prj_a",
        filename="a.txt",
        size_bytes=5,
        sha256="0" * 64,
        blob_path="a.txt",
        kind=FileKind.TEXT,
        status=FileStatus.READY,
    )
    apr_a = ToolApprovalRecord(
        id="apr_a",
        project_id="prj_a",
        session_id="ses_a",
        tool_name="t",
        arguments_json="{}",
        permission_tier=PermissionTier.READ,
        execution_target=ToolExecutionTarget.CLIENT,
        reason="r",
        status=ApprovalStatus.PENDING,
        requested_by="alice",
    )

    state = {"proj_count": 1, "used_bytes": 9}

    class _RepoStub:
        async def get(self, pid: str) -> Project | None:
            return prj_a if pid == "prj_a" else None

        async def count_for_owner(self, owner: str) -> int:
            assert owner == "alice"
            return state["proj_count"]

        async def get_session(self, sid: str) -> Session | None:
            return ses_a if sid == "ses_a" else None

        async def get_file(self, fid: str) -> FileRecord | None:
            return fil_a if fid == "fil_a" else None

        async def sum_bytes_for_owner(self, owner: str) -> int:
            assert owner == "alice"
            return state["used_bytes"]

        async def get_approval(self, aid: str) -> ToolApprovalRecord | None:
            return apr_a if aid == "apr_a" else None

    repo = _RepoStub()
    svc = cast(
        "AppServices",
        SimpleNamespace(
            settings=LaiSettings(max_projects_per_principal=2, max_storage_bytes_per_principal=10),
            project_repo=repo,
            session_repo=repo,
            file_repo=repo,
            tool_audit_repo=repo,
        ),
    )

    assert (await require_accessible_project(svc, "prj_a", alice)).id == "prj_a"
    assert (await require_accessible_project(svc, "prj_a", adm)).id == "prj_a"
    assert (await require_accessible_session(svc, "ses_a", alice)).id == "ses_a"
    assert (await require_accessible_file(svc, "fil_a", alice)).id == "fil_a"
    assert (await require_accessible_approval(svc, "apr_a", alice)).id == "apr_a"

    for missing_id, key, result_fn in (
        ("prj_missing", "project_id", require_accessible_project),
        ("ses_missing", "session_id", require_accessible_session),
        ("fil_missing", "file_id", require_accessible_file),
        ("apr_missing", "approval_id", require_accessible_approval),
    ):
        with pytest.raises(NotFoundError) as exc_m:
            await result_fn(svc, missing_id, alice)
        assert exc_m.value.details == {key: missing_id}

    for owned_id, key, result_fn in (
        ("prj_a", "project_id", require_accessible_project),
        ("ses_a", "session_id", require_accessible_session),
        ("fil_a", "file_id", require_accessible_file),
        ("apr_a", "approval_id", require_accessible_approval),
    ):
        with pytest.raises(NotFoundError) as exc_o:
            await result_fn(svc, owned_id, bob)
        assert exc_o.value.details == {key: owned_id}

    await enforce_project_quota(svc, alice)
    state["proj_count"] = 2
    with pytest.raises(AuthorizationError) as exc_q:
        await enforce_project_quota(svc, alice)
    assert exc_q.value.code == "quota_exceeded"
    assert exc_q.value.details == {"max_projects_per_principal": "2", "current_projects": "2"}
    await enforce_project_quota(svc, adm)

    async def _one_chunk(data: bytes) -> AsyncIterator[bytes]:
        yield data

    raw_stream = _one_chunk(b"1234567890")
    assert (await enforce_and_wrap_storage_quota(svc, adm, raw_stream)) is raw_stream

    exact_stream = await enforce_and_wrap_storage_quota(svc, alice, _one_chunk(b"x"))
    assert [c async for c in exact_stream] == [b"x"]

    over_stream = await enforce_and_wrap_storage_quota(svc, alice, _one_chunk(b"xy"))
    with pytest.raises(PayloadTooLargeError) as exc_mid:
        _ = [c async for c in over_stream]
    assert exc_mid.value.details == {"max_storage_bytes_per_principal": "10"}

    state["used_bytes"] = 10
    with pytest.raises(PayloadTooLargeError) as exc_pre:
        await enforce_and_wrap_storage_quota(svc, alice, _one_chunk(b""))
    assert exc_pre.value.details == {"max_storage_bytes_per_principal": "10"}


async def test_client_auth_empty_whitespace_and_merge_boundaries() -> None:
    """Verify _client_auth blank oauth_client_id, allowed_project_ids, and tool merging."""
    tool_srv = ToolSpec(
        name="search_files",
        description="srv",
        permission_tier=PermissionTier.READ,
        execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
    )
    tool_cli = ToolSpec(
        name="cli_tool",
        description="cli",
        permission_tier=PermissionTier.READ,
        execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
    )
    app = build_client_app(
        "spa",
        oauth_client_id="cid-1",
        allowed_tiers=(PermissionTier.READ,),
        registered_tools=(tool_cli,),
        allowed_project_ids=(),
    )
    touched: list[str] = []

    class _AppRepo:
        async def get_by_oauth_client_id(self, cid: str) -> ClientAppRecord | None:
            return app if cid == "cid-1" else None

        async def touch_last_used(self, app_id: str, _ts: datetime) -> None:
            touched.append(app_id)

    svc = cast(
        "AppServices",
        SimpleNamespace(
            client_app_repo=_AppRepo(),
            tool_registry=SimpleNamespace(list_specs=lambda: [tool_srv]),
        ),
    )

    blank_p = _principal("u1", ApiScope.CHAT_WRITE, oauth_client_id="   ")
    empty_ctx = await resolve_turn_client_context(
        svc, principal=blank_p, project_id="prj_1", requested_tools=()
    )
    assert (empty_ctx.client_app_id, empty_ctx.client_tools) == (None, ())
    with pytest.raises(AuthenticationError):
        await resolve_turn_client_context(
            svc, principal=blank_p, project_id="prj_1", requested_tools=(tool_cli,)
        )
    with pytest.raises(AuthenticationError):
        await require_authenticated_client_app(svc, blank_p)

    ok_p = _principal("u1", ApiScope.CHAT_WRITE, oauth_client_id=" cid-1 ")
    ctx = await resolve_turn_client_context(
        svc, principal=ok_p, project_id="any_project", requested_tools=()
    )
    assert ctx.client_app_id == app.id
    assert ctx.client_tools[0].execution_target is ToolExecutionTarget.CLIENT
    assert await require_authenticated_client_app(svc, ok_p) == app
    assert touched == [app.id, app.id]

    with pytest.raises(ValidationFailedError, match="Duplicate"):
        await resolve_turn_client_context(
            svc, principal=ok_p, project_id="prj_1", requested_tools=(tool_cli, tool_cli)
        )
    with pytest.raises(ValidationFailedError, match="shadow"):
        await resolve_turn_client_context(
            svc, principal=ok_p, project_id="prj_1", requested_tools=(tool_srv,)
        )

    revoked = app.model_copy(update={"revoked_at": datetime.now(UTC)})

    class _RevokedRepo:
        async def get_by_oauth_client_id(self, cid: str) -> ClientAppRecord | None:
            del cid
            return revoked

    svc_rev = cast("AppServices", SimpleNamespace(client_app_repo=_RevokedRepo()))
    with pytest.raises(AuthorizationError):
        await require_authenticated_client_app(svc_rev, ok_p)


async def test_require_project_files_boundaries() -> None:
    """require_project_files deduplicates IDs, enforces 404 tenancy, and enforces 403 project."""
    alice = _principal("alice", ApiScope.CHAT_WRITE)
    bob = _principal("bob", ApiScope.CHAT_WRITE)
    prj_a = Project(id="prj_a", name="A", owner_principal_id="alice")
    prj_a2 = Project(id="prj_a2", name="A2", owner_principal_id="alice")
    fil_a = FileRecord(
        id="fil_a",
        project_id="prj_a",
        filename="a.txt",
        size_bytes=5,
        sha256="0" * 64,
        blob_path="a.txt",
        kind=FileKind.TEXT,
        status=FileStatus.READY,
    )
    fil_a2 = fil_a.model_copy(update={"id": "fil_a2", "project_id": "prj_a2"})
    file_lookups: list[str] = []

    class _Repo:
        async def get(self, pid: str) -> Project | None:
            return {"prj_a": prj_a, "prj_a2": prj_a2}.get(pid)

        async def get_file(self, fid: str) -> FileRecord | None:
            file_lookups.append(fid)
            return {"fil_a": fil_a, "fil_a2": fil_a2}.get(fid)

    repo = _Repo()
    svc = cast("AppServices", SimpleNamespace(project_repo=repo, file_repo=repo))

    await require_project_files(svc, "prj_a", (), alice)
    assert file_lookups == []
    await require_project_files(svc, "prj_a", ("fil_a", "fil_a"), alice)
    assert file_lookups == ["fil_a"]
    with pytest.raises(NotFoundError) as exc_m:
        await require_project_files(svc, "prj_a", ("fil_missing",), alice)
    assert exc_m.value.details == {"file_id": "fil_missing"}
    with pytest.raises(NotFoundError) as exc_o:
        await require_project_files(svc, "prj_a", ("fil_a",), bob)
    assert exc_o.value.details == {"file_id": "fil_a"}
    with pytest.raises(AuthorizationError) as exc_p:
        await require_project_files(svc, "prj_a", ("fil_a2",), alice)
    assert exc_p.value.details == {"file_id": "fil_a2", "project_id": "prj_a"}
