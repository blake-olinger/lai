"""Red-team integration proofs for B8 (stateless client-tools FK & post-header keepalive)."""

from __future__ import annotations

import asyncio
import json
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import TokenUsage
from lai.domain.errors import GuardrailBlockedError, InferenceBackendError, StorageFailureError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.observability import AlertCategory, AlertEvent, LoggingConfig
from lai.domain.tools import (
    PermissionTier,
    ToolApprovalRecord,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from lai.observability import AlertManager, configure_observability
from lai.orchestrator._retrieval_search import invoke_single_tool
from lai.ports.inference import InferenceProviderPort
from lai.ports.observability import AlertSinkPort
from lai.ports.tools import ToolExecutorPort
from tests.integration._helpers import (
    IntegrationSpec,
    configure_test_oidc,
    mint_auth_headers,
    open_services_harness,
    register_client_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Mapping
    from pathlib import Path

_ADMIN = tuple(ApiScope)


class _MemorySink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


class _SlowFailingInference(InferenceProviderPort):
    def __init__(self, *, delay_s: float = 0.0, exc: Exception | None = None) -> None:
        self.delay_s = delay_s
        self.exc = exc
        self.calls = 0

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        del request
        self.calls += 1
        return InferenceResponse(content="ok")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        del request
        self.calls += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.exc is not None:
            raise self.exc
        yield InferenceChunk(
            content_delta="stateless ok",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


class _BrokenExecutor(ToolExecutorPort):
    async def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
        *,
        client_tools: Mapping[str, ToolSpec] | None = None,
    ) -> ToolResult:
        del tool_name, arguments, context, client_tools
        msg = "SQLite FK failure"
        raise StorageFailureError(msg)

    async def approve_and_execute(
        self,
        approval_id: str,
        decided_by: str,
        approved: bool,
        note: str | None = None,
    ) -> ToolApprovalRecord:
        del approval_id, decided_by, approved, note
        msg = "not used"
        raise NotImplementedError(msg)

    async def submit_client_result(
        self,
        approval_id: str,
        client_app_id: str,
        result: ToolResult,
        *,
        principal_id: str,
        is_admin: bool = False,
    ) -> ToolResult:
        del approval_id, client_app_id, result, principal_id, is_admin
        msg = "not used"
        raise NotImplementedError(msg)

    async def revoke_approval(self, approval_id: str, *, reason: str) -> bool:
        del approval_id, reason
        return False


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


async def test_b8_stateless_client_tools_storage_error_and_post_header_keepalive(
    tmp_path: Path,
) -> None:
    """B8(a)+(b)+(c): Stateless client_tools 422, StorageFailureError, keepalive JSON error."""
    with pytest.raises(StorageFailureError):
        await invoke_single_tool(
            _BrokenExecutor(),
            InferenceToolCall(id="tc1", name="t", arguments_json="{}"),
            ToolExecutionContext(
                principal_id="u1", project_id="p1", session_id="s1", request_id="r1"
            ),
        )

    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
        json_keepalive_threshold_seconds=0.05,
        keepalive_interval_seconds=0.02,
    )
    inf = _SlowFailingInference()
    async with open_services_harness(settings, IntegrationSpec(inference_provider=inf)) as h:
        authority = configure_test_oidc(h.svc, audiences=("spa-app",), default_scopes=_ADMIN)
        tool = ToolSpec(
            name="ping_client",
            description="Ping",
            parameters_schema={"type": "object", "properties": {}},
            permission_tier=PermissionTier.READ,
        )
        await register_client_app(
            h.svc,
            "spa",
            oauth_client_id="spa-app",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(tool,),
        )
        oidc_h = authority.mint_headers(aud="spa-app")

        r_422 = await h.client.post(
            "/v1/chat/completions",
            headers=oidc_h,
            json={
                "model": settings.default_model_profile,
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
                "client_tools": [tool.model_dump(mode="json")],
            },
        )
        assert r_422.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert inf.calls == 0

        r_ok = await h.client.post(
            "/v1/chat/completions",
            headers=oidc_h,
            json={
                "model": settings.default_model_profile,
                "stream": False,
                "messages": [{"role": "user", "content": "hi stateless"}],
            },
        )
        assert r_ok.status_code == HTTPStatus.OK, r_ok.text

    sink = _MemorySink()
    mgr = AlertManager(sinks=(sink,), cooldown_seconds=0.0)
    configure_observability(
        LoggingConfig(enable_file_logs=False, alert_cooldown_seconds=0.0, alert_on_errors=True),
        alert_manager=mgr,
    )
    for exc, expected_code in (
        (
            GuardrailBlockedError(
                "blocked", violated_policy="No Hate Speech", details={"direction": "egress"}
            ),
            "prompt_blocked_by_guardrail",
        ),
        (InferenceBackendError("gpu offline"), "inference_backend_error"),
    ):
        slow_inf = _SlowFailingInference(delay_s=0.12, exc=exc)
        async with open_services_harness(
            settings,
            IntegrationSpec(inference_provider=slow_inf, raise_app_exceptions=True),
        ) as h:
            headers = await mint_auth_headers(h.svc, "admin", _ADMIN)
            resp = await h.client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": settings.default_model_profile,
                    "stream": False,
                    "messages": [{"role": "user", "content": "slow request"}],
                },
            )
            assert resp.content.startswith(b"\n")
            parsed = json.loads(resp.content.decode("utf-8"))
            assert parsed["error"]["code"] == expected_code
    await mgr.flush()
    assert not any(e.category is AlertCategory.FATAL_EXCEPTION for e in sink.events)
