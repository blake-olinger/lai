"""Integration tests for server host-tier ban and restart-safe gated READ approval CAS."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import pytest

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.errors import AuthorizationError, ToolApprovalRequiredError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import LoggingConfig
from lai.domain.tools import (
    PermissionTier,
    ToolExecutionContext,
    ToolExecutionTarget,
    ToolResult,
    ToolSpec,
)
from lai.ports.tools import ToolPluginPort
from tests.integration._helpers import (
    IntegrationSpec,
    create_project,
    create_session,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from tests.integration._helpers import IntegrationHarness

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


def _tool_settings(tmp_path: Path) -> LaiSettings:
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )


async def test_server_host_write_plugin_is_forbidden_and_audited(
    integration_settings: LaiSettings,
    services_harness_factory: Callable[
        [LaiSettings, IntegrationSpec | None],
        AbstractAsyncContextManager[IntegrationHarness],
    ],
) -> None:
    """Server execution of non-READ attachment tools raises AuthorizationError and audits."""
    host_plugin = _HostWritePlugin()
    spec = IntegrationSpec(tools=(host_plugin,))
    async with services_harness_factory(integration_settings, spec) as harness:
        headers = await mint_auth_headers(
            harness.svc,
            "chat_only",
            (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
        )
        pid = await create_project(harness.client, headers, "tp")
        sid = await create_session(harness.client, headers, pid)
        ctx = ToolExecutionContext(
            project_id=pid,
            session_id=sid,
            principal_id="adm",
            principal_scopes=(ApiScope.ADMIN, ApiScope.TOOLS_EXEC),
        )
        with pytest.raises(AuthorizationError):
            await harness.svc.tool_executor.invoke("host_write", {}, ctx)

        assert host_plugin.calls == 0
        audits = await harness.svc.tool_audit_repo.list_audits(pid, sid)
        assert any(
            a.tool_name == "host_write" and a.error == "server_host_tool_forbidden" for a in audits
        )


async def test_gated_read_restores_confinement_across_restart_and_enforces_cas(
    tmp_path: Path,
) -> None:
    """Pending approval persists across full restart, restores confinement, and enforces CAS."""
    settings = _tool_settings(tmp_path)

    async with open_services_harness(settings, IntegrationSpec(tools=(_GatedReadPlugin(),))) as h1:
        h_chat = await mint_auth_headers(
            h1.svc,
            "chat_only",
            (ApiScope.PROJECTS_WRITE, ApiScope.CHAT_WRITE, ApiScope.TOOLS_SAFE),
        )
        pid = await create_project(h1.client, h_chat, "tp")
        sid = await create_session(h1.client, h_chat, pid)
        ctx = ToolExecutionContext(
            project_id=pid,
            session_id=sid,
            principal_id="reader",
            principal_scopes=(ApiScope.TOOLS_SAFE,),
            allowed_file_ids=("fil_scoped",),
        )
        with pytest.raises(ToolApprovalRequiredError) as exc_info:
            await h1.svc.tool_executor.invoke("gated_read", {"file_id": "fil_scoped"}, ctx)
        assert exc_info.value.status_code == HTTPStatus.CONFLICT
        approval_id = exc_info.value.approval_id

    # Fresh AppServices on the same data_dir verifies SQLite confinement restore after restart.
    read_plugin_after_restart = _GatedReadPlugin()
    spec_after = IntegrationSpec(tools=(read_plugin_after_restart,))
    async with open_services_harness(settings, spec_after) as h2:
        h_approve = await mint_auth_headers(
            h2.svc, "approver", (ApiScope.TOOLS_SAFE, ApiScope.TOOLS_APPROVE)
        )
        responses = await asyncio.gather(
            *[
                h2.client.post(
                    f"/v1/tool-approvals/{approval_id}/resolve",
                    headers=h_approve,
                    json={"approved": True},
                )
                for _ in range(_CAS_CONCURRENCY)
            ]
        )

    expected_codes = [HTTPStatus.OK] + [HTTPStatus.UNPROCESSABLE_ENTITY] * (_CAS_CONCURRENCY - 1)
    assert sorted(r.status_code for r in responses) == expected_codes
    assert len(read_plugin_after_restart.seen_contexts) == 1
    restored = read_plugin_after_restart.seen_contexts[0]
    assert restored.allowed_file_ids == ("fil_scoped",)
    assert restored.principal_scopes == (ApiScope.TOOLS_SAFE,)
