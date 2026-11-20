"""Red-team integration proofs for B9(d)+(e) (denied/expired/blocked/abandoned/trim history)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

import pytest

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.chat import Message, MessageRole, TokenUsage
from lai.domain.guardrails import GuardrailConfig, GuardrailDecision, GuardrailVerdict
from lai.domain.inference import (
    BackendHealth,
    InferenceChunk,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceToolCall,
)
from lai.domain.observability import LoggingConfig
from lai.domain.tools import ApprovalStatus, PermissionTier, ToolSpec
from lai.guardrails import CompositeGuardrailPipeline
from lai.inference._wire import serialize_chat_request
from lai.orchestrator._history import repair_tool_call_history, trim_history_to_budget
from lai.ports.guardrails import GuardrailProviderPort
from lai.ports.inference import InferenceProviderPort
from tests.integration._helpers import (
    IntegrationSpec,
    configure_test_oidc,
    create_project,
    create_session,
    open_services_harness,
    register_client_app,
)
from tests.integration.test_redteam_b9_history_and_tools import assert_gemma4_jinja_tool_invariants

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from pathlib import Path

_ADMIN = tuple(ApiScope)


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


class _BlockNeedleGuardrail(GuardrailProviderPort):
    def __init__(self, needle: str = "SECRET_BLOCK_MARKER") -> None:
        self._needle = needle

    @property
    def name(self) -> str:
        return "block_needle"

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        sha = "d" * 64
        if self._needle in prompt:
            return GuardrailDecision(
                verdict=GuardrailVerdict.BLOCK,
                provider=self.name,
                violated_policy="No Prompt Injection",
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


async def test_b9_denied_expired_blocked_abandoned_and_atomic_trim(tmp_path: Path) -> None:
    """B9(d)+(e): Denied, expired, guardrail-blocked, abandoned, and trimmed histories valid."""
    tc1 = InferenceToolCall(id="tc_deny", name="cli_exec", arguments_json="{}")
    tc2 = InferenceToolCall(id="tc_exp", name="cli_exec", arguments_json="{}")
    tc3 = InferenceToolCall(id="tc_blk", name="cli_exec", arguments_json="{}")
    inf = _ScriptedInference([(tc1,), "after deny", (tc2,), "after exp", (tc3,), "after blk"])
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=tmp_path / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=True),
    )

    async with open_services_harness(settings, IntegrationSpec(inference_provider=inf)) as h:
        assert isinstance(h.svc.guardrail_pipeline, CompositeGuardrailPipeline)
        h.svc.guardrail_pipeline.reconfigure(
            settings.guardrails, providers=(_BlockNeedleGuardrail(),)
        )
        authority = configure_test_oidc(h.svc, audiences=("spa",), default_scopes=_ADMIN)
        exec_tool = ToolSpec(
            name="cli_exec",
            description="exec",
            parameters_schema={"type": "object", "properties": {}},
            permission_tier=PermissionTier.EXEC,
            requires_approval_by_default=True,
        )
        await register_client_app(
            h.svc,
            "spa",
            oauth_client_id="spa",
            allowed_tiers=(PermissionTier.EXEC,),
            registered_tools=(exec_tool,),
        )
        headers = authority.mint_headers(aud="spa")
        prj = await create_project(h.client, headers, "p-term")
        sid = await create_session(h.client, headers, prj)

        r_d = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "d", "stream": False},
        )
        apr_d = r_d.json()["pending_approval_id"]
        res_deny = await h.client.post(
            f"/v1/tool-approvals/{apr_d}/resolve", headers=headers, json={"approved": False}
        )
        assert res_deny.status_code == HTTPStatus.OK
        await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "next1", "stream": False},
        )
        w1 = cast(
            "list[dict[str, object]]",
            serialize_chat_request(inf.requests[1], stream=False)["messages"],
        )
        assert_gemma4_jinja_tool_invariants(w1)

        r_e = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "e", "stream": False},
        )
        apr_e = r_e.json()["pending_approval_id"]
        rec = await h.svc.tool_audit_repo.get_approval(apr_e)
        assert rec is not None
        await h.svc.tool_audit_repo.update_approval(
            rec.model_copy(
                update={
                    "status": ApprovalStatus.APPROVED,
                    "expires_at": datetime.now(UTC) - timedelta(seconds=5),
                }
            )
        )
        res_exp = await h.client.post(
            f"/v1/tool-approvals/{apr_e}/result",
            headers=headers,
            json={"success": True, "output": "late", "stream": False},
        )
        assert res_exp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "next2", "stream": False},
        )
        w2 = cast(
            "list[dict[str, object]]",
            serialize_chat_request(inf.requests[3], stream=False)["messages"],
        )
        assert_gemma4_jinja_tool_invariants(w2)

        r_b = await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "b", "stream": False},
        )
        apr_b = r_b.json()["pending_approval_id"]
        await h.client.post(
            f"/v1/tool-approvals/{apr_b}/resolve", headers=headers, json={"approved": True}
        )
        res_blk = await h.client.post(
            f"/v1/tool-approvals/{apr_b}/result",
            headers=headers,
            json={"success": True, "output": "SECRET_BLOCK_MARKER", "stream": False},
        )
        assert res_blk.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert res_blk.json()["error"]["details"]["violated_policy"] == "No Prompt Injection"
        await h.client.post(
            f"/v1/sessions/{sid}/messages",
            headers=headers,
            json={"content": "next3", "stream": False},
        )
        w3 = cast(
            "list[dict[str, object]]",
            serialize_chat_request(inf.requests[5], stream=False)["messages"],
        )
        assert_gemma4_jinja_tool_invariants(w3)

    abandoned = (
        InferenceMessage(role=MessageRole.USER, content="hi"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="", tool_calls=(tc1,)),
        InferenceMessage(role=MessageRole.USER, content="never mind"),
    )
    repaired = repair_tool_call_history(abandoned)
    wire_rep = cast(
        "list[dict[str, object]]",
        serialize_chat_request(InferenceRequest(model="m", messages=repaired), stream=False)[
            "messages"
        ],
    )
    assert_gemma4_jinja_tool_invariants(wire_rep)

    msgs = [
        Message(project_id="p", session_id="s", role=MessageRole.USER, content="x " * 100),
        Message(
            project_id="p",
            session_id="s",
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(tc1,),
        ),
        Message(
            project_id="p",
            session_id="s",
            role=MessageRole.TOOL,
            content="ok",
            tool_name="cli_exec",
            tool_call_id="tc_deny",
        ),
        Message(project_id="p", session_id="s", role=MessageRole.USER, content="latest"),
    ]
    trimmed = trim_history_to_budget(msgs, max_history_tokens=40)
    wire_trim = cast(
        "list[dict[str, object]]",
        serialize_chat_request(InferenceRequest(model="m", messages=trimmed), stream=False)[
            "messages"
        ],
    )
    assert_gemma4_jinja_tool_invariants(wire_trim)
