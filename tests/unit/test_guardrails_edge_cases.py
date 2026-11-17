"""Additional edge-case unit tests for guardrail providers and CompositeGuardrailPipeline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from lai.domain import GuardrailConfig, GuardrailVerdict, SafetyPrinciple
from lai.domain.errors import InferenceBackendError
from lai.guardrails import (
    CompositeGuardrailPipeline,
    RegexPolicyGuardrailProvider,
    RegexPolicyRule,
    ShieldGemmaCpuProvider,
)
from lai.observability import bind_request_id

if TYPE_CHECKING:
    from pathlib import Path

HTTP_OK = 200


async def test_shieldgemma_injected_client_invalid_json_and_health_failures() -> None:
    """ShieldGemmaCpuProvider handles injected clients, malformed JSON, and transport errors."""
    cfg = GuardrailConfig(host="127.0.0.1", port=18090, timeout_seconds=1.0)
    principle = SafetyPrinciple(name="P", description="Desc")
    completion_url = "http://127.0.0.1:18090/completion"
    health_url = "http://127.0.0.1:18090/health"

    with respx.mock(assert_all_called=True) as mock:
        mock.post(completion_url).mock(
            side_effect=[
                httpx.Response(HTTP_OK, text="not-json"),
                httpx.Response(HTTP_OK, json={"wrong_field": 123}),
                httpx.Response(HTTP_OK, json={"content": "No"}),
            ],
        )
        mock.get(health_url).mock(
            side_effect=[
                httpx.ReadTimeout("slow"),
                httpx.ConnectError("refused"),
            ],
        )
        async with httpx.AsyncClient() as shared_client:
            provider = ShieldGemmaCpuProvider(cfg, principles=(principle,), client=shared_client)
            with pytest.raises(InferenceBackendError, match="missing string 'content'"):
                await provider.evaluate("hello")
            with pytest.raises(InferenceBackendError, match="missing string 'content'"):
                await provider.evaluate("hello")
            assert (await provider.evaluate("hello")).verdict is GuardrailVerdict.ALLOW
            assert (await provider.health()).status == "timeout"
            assert (await provider.health()).status == "unreachable"


async def test_regex_rule_objects_and_pipeline_unconfigured_and_register(tmp_path: Path) -> None:
    """RegexPolicyRule instances, register_provider, and empty-provider modes behave cleanly."""
    regex = RegexPolicyGuardrailProvider(
        rules=(
            RegexPolicyRule(
                policy_name="CustomPolicy",
                pattern=re.compile(r"forbidden_token"),
            ),
            ("TuplePolicy", r"banned_phrase"),
        ),
    )
    assert (await regex.evaluate("contains forbidden_token here")).violated_policy == "CustomPolicy"
    assert (await regex.evaluate("contains banned_phrase here")).violated_policy == "TuplePolicy"

    with CompositeGuardrailPipeline(
        GuardrailConfig(fail_open=False),
        providers=(),
        log_dir=tmp_path,
    ) as closed_empty:
        with bind_request_id("req_ctx"):
            res_closed = await closed_empty.check_prompt("hello")
        assert res_closed.verdict is GuardrailVerdict.BLOCK
        assert res_closed.violated_policy == "guardrail_unconfigured"

    with CompositeGuardrailPipeline(
        GuardrailConfig(fail_open=True),
        providers=(),
        log_dir=tmp_path,
    ) as open_empty:
        assert (await open_empty.check_prompt("hello")).verdict is GuardrailVerdict.ALLOW
        open_empty.register_provider(regex)
        assert (await open_empty.check_prompt("banned_phrase")).verdict is GuardrailVerdict.BLOCK

    rows = [
        json.loads(line)
        for line in (tmp_path / "prompts_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["request_id"] == "req_ctx"
    assert [row["verdict"] for row in rows] == ["block", "allow", "block"]
