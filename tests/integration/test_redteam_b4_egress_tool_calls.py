"""Red-team integration proofs for B4(c)+(d) (buffered egress tool-call blocking & revocation)."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.errors import StorageFailureError
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailOutputMode,
    GuardrailVerdict,
)
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
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from lai.guardrails import CompositeGuardrailPipeline
from lai.ports.guardrails import GuardrailProviderPort
from lai.ports.inference import InferenceProviderPort
from lai.ports.tools import ToolPluginPort
from tests.integration._helpers import (
    IntegrationSpec,
    configure_test_oidc,
    create_project,
    create_session,
    open_services_harness,
    register_client_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from pathlib import Path

_ADMIN_SCOPES = tuple(ApiScope)


class _UnsafeToolCallInference(InferenceProviderPort):
    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="safe")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        tc = InferenceToolCall(
            id="tc_bad_1",
            name="run_local",
            arguments_json='{"cmd":"exfil ignore previous instructions"}',
        )
        yield InferenceChunk(
            tool_calls=(tc,),
            finish_reason="tool_calls",
            usage=TokenUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


def _approval_statuses(db_path: Path, session_id: str) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(str(db_path))) as conn:
        return conn.execute(
            "SELECT status FROM tool_approvals WHERE session_id = ?", (session_id,)
        ).fetchall()


class _SubstringGuardrail(GuardrailProviderPort):
    @property
    def name(self) -> str:
        return "stub_guardrail"

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        sha = "a" * 64
        if "ignore previous instructions" in prompt.lower():
            return GuardrailDecision(
                verdict=GuardrailVerdict.BLOCK,
                provider=self.name,
                violated_policy="No Prompt Injection",
                reason="matched",
                prompt_sha256=sha,
                prompt_chars=len(prompt),
                scrubbed_prompt=prompt,
            )
        return GuardrailDecision(
            verdict=GuardrailVerdict.ALLOW,
            provider=self.name,
            prompt_sha256=sha,
            prompt_chars=len(prompt),
            scrubbed_prompt=prompt,
        )

    async def evaluate_response(self, prompt: str, response: str) -> GuardrailDecision:
        del prompt
        return await self.evaluate(response)


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.mark.parametrize(
    "block_mode",
    [GuardrailBlockMode.REFUSAL_MESSAGE, GuardrailBlockMode.HTTP_ERROR],
)
async def test_b4_buffered_egress_blocks_unsafe_tool_call_and_revokes_lease(
    tmp_path: Path,
    block_mode: GuardrailBlockMode,
) -> None:
    """B4(c)+(d): Unsafe client tool call is buffered, blocked on egress, and lease revoked."""
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(
            enabled=True,
            block_mode=block_mode,
            output_mode=GuardrailOutputMode.BUFFERED,
            allow_runtime_disable=False,
            fail_open=False,
        ),
    )
    async with open_services_harness(
        settings, IntegrationSpec(inference_provider=_UnsafeToolCallInference())
    ) as h:
        assert isinstance(h.svc.guardrail_pipeline, CompositeGuardrailPipeline)
        h.svc.guardrail_pipeline.reconfigure(
            settings.guardrails, providers=(_SubstringGuardrail(),)
        )

        authority = configure_test_oidc(h.svc, audiences=("api-aud",), default_scopes=_ADMIN_SCOPES)
        tool = ToolSpec(
            name="run_local",
            description="Run local command",
            parameters_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
            permission_tier=PermissionTier.EXEC,
            requires_approval_by_default=True,
        )
        await register_client_app(
            h.svc,
            "spa",
            oauth_client_id="api-aud",
            allowed_tiers=(PermissionTier.EXEC,),
            registered_tools=(tool,),
        )
        oidc_headers = authority.mint_headers(aud="api-aud")
        prj = await create_project(h.client, oidc_headers, "p-egress")
        sid = await create_session(h.client, oidc_headers, prj)

        resp = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=oidc_headers,
            json={"content": "run diagnostic", "stream": True},
        )
        body = resp.text
        assert "event: client_tool_call" not in body
        assert "event: approval_required" not in body
        if block_mode is GuardrailBlockMode.REFUSAL_MESSAGE:
            assert resp.status_code == HTTPStatus.OK
            assert settings.guardrails.refusal_text in body
        else:
            assert resp.status_code in {HTTPStatus.UNPROCESSABLE_ENTITY, HTTPStatus.OK}
            assert "prompt_blocked_by_guardrail" in body
            assert '"direction":"egress"' in body

        assert await h.svc.tool_audit_repo.list_pending_approvals(prj, session_id=sid) == []
        assert _approval_statuses(settings.data_dir / "lai.db", sid) == [
            (ApprovalStatus.DENIED.value,)
        ]
        persisted = await h.svc.session_repo.list_messages(sid)
        assert all(not m.tool_calls for m in persisted)


class _MultiIterUnsafePauseInference(InferenceProviderPort):
    def __init__(self) -> None:
        self._step = 0

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="safe")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        self._step += 1
        if self._step == 1:
            yield InferenceChunk(
                content_delta="ignore previous instructions leak ",
                tool_calls=(InferenceToolCall(id="tc_s", name="srv_ok", arguments_json="{}"),),
                finish_reason="tool_calls",
            )
            return
        yield InferenceChunk(
            content_delta="Safe pause text",
            tool_calls=(InferenceToolCall(id="tc_c", name="cli_read", arguments_json="{}"),),
            finish_reason="tool_calls",
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _BatchFailInference(InferenceProviderPort):
    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse(content="safe")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        yield InferenceChunk(
            tool_calls=(
                InferenceToolCall(id="tc_c", name="cli_read", arguments_json="{}"),
                InferenceToolCall(id="tc_s", name="srv_boom", arguments_json="{}"),
            ),
            finish_reason="tool_calls",
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _ServerToolStub(ToolPluginPort):
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self._name, self._fail = name, fail

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description="stub", permission_tier=PermissionTier.READ)

    async def execute(
        self, arguments: dict[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        del arguments, context
        if self._fail:
            msg = "storage failure"
            raise StorageFailureError(msg)
        return ToolResult(tool_name=self._name, success=True, output="ok")


async def test_b4_multi_iter_pause_content_blocked_and_mixed_batch_error_no_lease_leak(
    tmp_path: Path,
) -> None:
    """Multi-iteration pause scans prior buffered content; mixed-batch error leaks zero leases."""
    cli_tool = ToolSpec(
        name="cli_read",
        description="Read local",
        permission_tier=PermissionTier.READ,
        requires_approval_by_default=False,
    )
    settings = LaiSettings(
        data_dir=tmp_path / "m1" / "data",
        models_dir=tmp_path / "m1" / "models",
        bin_dir=tmp_path / "m1" / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "m1" / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(
            enabled=True,
            block_mode=GuardrailBlockMode.REFUSAL_MESSAGE,
            output_mode=GuardrailOutputMode.BUFFERED,
        ),
    )
    async with open_services_harness(
        settings,
        IntegrationSpec(
            inference_provider=_MultiIterUnsafePauseInference(),
            tools=(_ServerToolStub("srv_ok"),),
        ),
    ) as h:
        assert isinstance(h.svc.guardrail_pipeline, CompositeGuardrailPipeline)
        h.svc.guardrail_pipeline.reconfigure(
            settings.guardrails, providers=(_SubstringGuardrail(),)
        )
        auth = configure_test_oidc(h.svc, audiences=("api-aud",), default_scopes=_ADMIN_SCOPES)
        await register_client_app(
            h.svc,
            "spa",
            oauth_client_id="api-aud",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(cli_tool,),
        )
        hdrs = auth.mint_headers(aud="api-aud")
        prj = await create_project(h.client, hdrs, "p-multi-iter")
        sid = await create_session(h.client, hdrs, prj)
        resp = await h.client.post(
            f"/v1/sessions/{sid}/messages", headers=hdrs, json={"content": "go", "stream": True}
        )
        assert resp.status_code == HTTPStatus.OK
        assert "ignore previous instructions" not in resp.text.lower()
        assert "event: client_tool_call" not in resp.text
        assert _approval_statuses(settings.data_dir / "lai.db", sid) == [
            (ApprovalStatus.DENIED.value,)
        ]

    settings2 = settings.model_copy(
        update={
            "data_dir": tmp_path / "m2" / "data",
            "logging": LoggingConfig(log_dir=tmp_path / "m2" / "logs", enable_file_logs=False),
            "guardrails": GuardrailConfig(enabled=False),
        }
    )
    async with open_services_harness(
        settings2,
        IntegrationSpec(
            inference_provider=_BatchFailInference(),
            tools=(_ServerToolStub("srv_boom", fail=True),),
            raise_app_exceptions=False,
        ),
    ) as h2:
        auth2 = configure_test_oidc(h2.svc, audiences=("api-aud",), default_scopes=_ADMIN_SCOPES)
        await register_client_app(
            h2.svc,
            "spa",
            oauth_client_id="api-aud",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(cli_tool,),
        )
        hdrs2 = auth2.mint_headers(aud="api-aud")
        prj2 = await create_project(h2.client, hdrs2, "p-batch")
        sid2 = await create_session(h2.client, hdrs2, prj2)
        r_err = await h2.client.post(
            f"/v1/sessions/{sid2}/messages", headers=hdrs2, json={"content": "go", "stream": False}
        )
        assert r_err.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert _approval_statuses(settings2.data_dir / "lai.db", sid2) == []
