"""Red-team integration proofs for B9(a)+(b)+(c)+(e) (200-msg amnesia & multi-tool batches)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

import pytest

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import Message, MessageRole, TokenUsage
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.observability import LoggingConfig
from lai.domain.tools import PermissionTier, ToolResult, ToolSpec
from lai.inference._wire import serialize_chat_request
from lai.ports.inference import InferenceProviderPort
from lai.ports.tools import ToolPluginPort
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
    from collections.abc import AsyncGenerator, Generator
    from pathlib import Path

    from lai.domain.tools import ToolExecutionContext

_ADMIN = tuple(ApiScope)


def assert_gemma4_jinja_tool_invariants(wire_messages: list[dict[str, object]]) -> None:
    """Assert Gemma 4's exact GGUF chat_template invariant: zero unclosed <|tool_response>."""
    idx = 0
    total = len(wire_messages)
    while idx < total:
        msg = wire_messages[idx]
        role = msg["role"]
        assert role != "tool", f"Orphaned tool message at index {idx}: {msg}"
        raw_tcs = msg.get("tool_calls")
        if role == "assistant" and isinstance(raw_tcs, list) and raw_tcs:
            expected_ids = [
                str(cast("dict[str, object]", tc)["id"]) for tc in cast("list[object]", raw_tcs)
            ]
            cursor = idx + 1
            seen_ids: list[str] = []
            while cursor < total and wire_messages[cursor]["role"] == "tool":
                seen_ids.append(str(wire_messages[cursor].get("tool_call_id", "")))
                cursor += 1
            assert seen_ids == expected_ids, (
                f"Assistant tool_calls {expected_ids} followed by {seen_ids} at idx {idx}"
            )
            idx = cursor
            continue
        idx += 1


class _EchoServerTool(ToolPluginPort):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="srv_ping",
            description="Server ping",
            parameters_schema={"type": "object", "properties": {}},
            permission_tier=PermissionTier.READ,
        )

    async def execute(
        self, arguments: dict[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        del arguments, context
        return ToolResult(tool_name="srv_ping", success=True, output='{"pong": true}')


class _ScriptedInference(InferenceProviderPort):
    def __init__(self, script: list[tuple[InferenceToolCall, ...] | str]) -> None:
        self._script = list(script)
        self.requests: list[InferenceRequest] = []

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(content="complete")

    async def chat_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceChunk]:
        self.requests.append(request)
        nxt = self._script.pop(0) if self._script else "final answer"
        if isinstance(nxt, tuple):
            yield InferenceChunk(tool_calls=nxt, finish_reason="tool_calls")
            return
        yield InferenceChunk(
            content_delta=nxt,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, status="ok")


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _settings(tmp_path: Path) -> LaiSettings:
    return LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
    )


async def test_b9_250_message_recent_window_and_tie_break_order(tmp_path: Path) -> None:
    """B9(a): 250-message session with identical timestamps returns #51-#250 and feeds #250."""
    inf = _ScriptedInference(["saw latest"])
    async with open_services_harness(
        _settings(tmp_path), IntegrationSpec(inference_provider=inf)
    ) as h:
        headers = await mint_auth_headers(h.svc, "admin", _ADMIN)
        prj = await create_project(h.client, headers, "p-250")
        sid = await create_session(h.client, headers, prj)
        same_ts = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
        for idx in range(1, 251):
            role = MessageRole.USER if idx % 2 == 1 else MessageRole.ASSISTANT
            await h.svc.session_repo.add_message(
                Message(
                    project_id=prj,
                    session_id=sid,
                    role=role,
                    content=f"msg-{idx:03d}",
                    created_at=same_ts,
                )
            )
        recent = await h.svc.session_repo.list_recent_messages(sid, limit=200)
        assert len(recent) == 200
        assert [m.content for m in recent[:3]] == ["msg-051", "msg-052", "msg-053"]
        assert [m.content for m in recent[-3:]] == ["msg-248", "msg-249", "msg-250"]

        r = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "turn 251", "stream": False},
        )
        assert r.status_code == HTTPStatus.OK
        hist = [m.content for m in inf.requests[0].messages]
        assert "msg-250" in hist
        assert "msg-001" not in hist


async def test_b9_mixed_server_client_and_sibling_client_deferral_and_resume(
    tmp_path: Path,
) -> None:
    """B9(b)+(c)+(e): [server, client] and [client_1, client_2] persist cleanly and resume."""
    tc_srv = InferenceToolCall(id="tc_srv", name="srv_ping", arguments_json="{}")
    tc_c1 = InferenceToolCall(id="tc_c1", name="cli_one", arguments_json='{"x":1}')
    tc_c2 = InferenceToolCall(id="tc_c2", name="cli_two", arguments_json='{"y":2}')
    inf = _ScriptedInference([(tc_srv, tc_c1), "resumed ok", (tc_c1, tc_c2), "both ok"])

    async with open_services_harness(
        _settings(tmp_path),
        IntegrationSpec(inference_provider=inf, tools=(_EchoServerTool(),)),
    ) as h:
        authority = configure_test_oidc(h.svc, audiences=("spa",), default_scopes=_ADMIN)
        t1 = ToolSpec(
            name="cli_one",
            description="c1",
            parameters_schema={"type": "object", "properties": {}},
            permission_tier=PermissionTier.READ,
            requires_approval_by_default=False,
        )
        t2 = t1.model_copy(update={"name": "cli_two", "description": "c2"})
        await register_client_app(
            h.svc,
            "spa",
            oauth_client_id="spa",
            allowed_tiers=(PermissionTier.READ,),
            registered_tools=(t1, t2),
        )
        headers = authority.mint_headers(aud="spa")
        prj = await create_project(h.client, headers, "p-multi")
        sid = await create_session(h.client, headers, prj)

        r1 = await h.client.post(
            f"/v1/sessions/{sid}/messages", headers=headers, json={"content": "go", "stream": False}
        )
        assert r1.status_code == HTTPStatus.OK
        msgs_paused = await h.svc.session_repo.list_recent_messages(sid)
        assert [m.role for m in msgs_paused] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ]
        assert [tc.id for tc in msgs_paused[1].tool_calls] == ["tc_srv", "tc_c1"]
        assert msgs_paused[2].tool_call_id == "tc_srv"

        apr_id = r1.json()["pending_approval_id"]
        r_res = await h.client.post(
            f"/v1/tool-approvals/{apr_id}/result",
            headers=headers,
            json={"success": True, "output": '{"ok":1}', "stream": False},
        )
        assert r_res.status_code == HTTPStatus.OK
        wire = cast(
            "list[dict[str, object]]",
            serialize_chat_request(inf.requests[1], stream=False)["messages"],
        )
        assert_gemma4_jinja_tool_invariants(wire)

        r2 = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "two", "stream": False},
        )
        assert r2.status_code == HTTPStatus.OK
        tail = await h.svc.session_repo.list_recent_messages(sid)
        assert tail[-1].role is MessageRole.TOOL
        assert tail[-1].tool_call_id == "tc_c2"
        assert "deferred_one_client_tool_per_turn" in tail[-1].content
        apr2 = r2.json()["pending_approval_id"]
        r_res2 = await h.client.post(
            f"/v1/tool-approvals/{apr2}/result",
            headers=headers,
            json={"success": True, "output": '{"ok":2}', "stream": False},
        )
        assert r_res2.status_code == HTTPStatus.OK
        wire2 = cast(
            "list[dict[str, object]]",
            serialize_chat_request(inf.requests[3], stream=False)["messages"],
        )
        assert_gemma4_jinja_tool_invariants(wire2)
