"""Unit tests for ShieldGemma prompts, providers, and CompositeGuardrailPipeline."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

from lai.domain import (
    AlertCategory,
    AlertEvent,
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailVerdict,
    LoggingConfig,
    SafetyPrinciple,
)
from lai.domain.analytics import AnalyticsReport, GuardrailUsageEvent, TokenUsageRecord, UsageEvent
from lai.domain.errors import InferenceBackendError
from lai.domain.guardrails import GuardrailDirection, GuardrailOutputMode
from lai.guardrails import (
    DEFAULT_SAFETY_PRINCIPLES,
    OUTPUT_SAFETY_PRINCIPLES,
    CompositeGuardrailPipeline,
    RegexPolicyGuardrailProvider,
    ShieldGemmaCpuProvider,
    build_shieldgemma_prompt,
    build_shieldgemma_response_prompt,
    compute_yes_probability,
    parse_shieldgemma_verdict,
)
from lai.observability import AlertManager, configure_observability
from lai.ports import AlertSinkPort, GuardrailPipelinePort, GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime
    from pathlib import Path

EXPECTED_DEFAULT_PRINCIPLES = 5
EXPECTED_OUTPUT_PRINCIPLES = 4
EXPECTED_SHORT_CIRCUIT_CALLS = 2
HTTP_OK = 200
HTTP_ERROR = 500
HTTP_UNAVAILABLE = 503


class _MemorySink(AlertSinkPort):
    """In-memory alert sink recording emitted AlertEvents."""

    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


class _SpyAnalyticsRepo:
    """In-memory UsageAnalyticsPort stub capturing recorded guardrail efficacy events."""

    def __init__(self, *, fail_on_write: bool = False) -> None:
        self.guardrail_events: list[GuardrailUsageEvent] = []
        self._fail_on_write = fail_on_write

    async def record_request(self, event: UsageEvent) -> None:
        del event

    async def record_tokens(self, record: TokenUsageRecord) -> None:
        del record

    async def record_guardrail(self, event: GuardrailUsageEvent) -> None:
        if self._fail_on_write:
            msg = "sqlite busy"
            raise RuntimeError(msg)
        self.guardrail_events.append(event)

    async def get_report(
        self, window_hours: int = 24, *, now: datetime | None = None
    ) -> AnalyticsReport:
        del now
        return AnalyticsReport(
            window_hours=window_hours,
            total_requests=0,
            error_count=0,
            error_rate=0.0,
            unique_principals=0,
            unique_client_apps=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
        )

    async def prune_before(self, cutoff: datetime) -> int:
        del cutoff
        return 0


@pytest.fixture(autouse=True)
def _reset_root_logging() -> Generator[None]:
    """Ensure root logger handlers created in tests are closed cleanly."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def test_principles_templates_probabilities_and_verdict_parsing() -> None:
    """ShieldGemma ingress/egress templates, probability parsing, and verdicts behave accurately."""
    names = [p.name for p in DEFAULT_SAFETY_PRINCIPLES]
    assert len(DEFAULT_SAFETY_PRINCIPLES) == EXPECTED_DEFAULT_PRINCIPLES
    assert len(OUTPUT_SAFETY_PRINCIPLES) == EXPECTED_OUTPUT_PRINCIPLES
    assert "No Prompt Injection" in names
    assert "No Prompt Injection" not in [p.name for p in OUTPUT_SAFETY_PRINCIPLES]

    injection_policy = next(p for p in DEFAULT_SAFETY_PRINCIPLES if p.name == "No Prompt Injection")
    assert "kill pid" in injection_policy.description.lower()

    prompt = build_shieldgemma_prompt("How do I kill pid 1234?", injection_policy)
    assert "<start_of_turn>user" in prompt
    assert "Human Question: How do I kill pid 1234?" in prompt
    assert "Your answer must start with 'Yes' or 'No'." in prompt

    resp_prompt = build_shieldgemma_response_prompt(
        "How do I build a bomb?",
        "I cannot help with explosives.",
        OUTPUT_SAFETY_PRINCIPLES[3],
    )
    assert "Human Question: How do I build a bomb?" in resp_prompt
    assert "Chatbot Response: I cannot help with explosives." in resp_prompt
    assert "Does the chatbot response violate the above principle?" in resp_prompt

    assert parse_shieldgemma_verdict("Yes") is True
    assert parse_shieldgemma_verdict("  yes, violates policy\n") is True
    assert parse_shieldgemma_verdict("No") is False
    assert parse_shieldgemma_verdict("  no\nSafe.") is False
    for ambiguous in ("", "   ", "Maybe", "Yesterday", "Nothing"):
        with pytest.raises(ValueError, match="Ambiguous ShieldGemma verdict"):
            parse_shieldgemma_verdict(ambiguous)

    llama_probs = [
        {
            "content": "Yes",
            "probs": [{"tok_str": "Yes", "prob": 0.91}, {"tok_str": "No", "prob": 0.09}],
        }
    ]
    assert compute_yes_probability(llama_probs) == pytest.approx(0.91)
    assert compute_yes_probability(None) is None
    assert compute_yes_probability([]) is None


def _prob_resp(content: str, p_yes: float, p_no: float) -> httpx.Response:
    return httpx.Response(
        HTTP_OK,
        json={
            "content": content,
            "completion_probabilities": [
                {"probs": [{"tok_str": "Yes", "prob": p_yes}, {"tok_str": "No", "prob": p_no}]}
            ],
        },
    )


async def test_shieldgemma_provider_ingress_egress_and_n_probs_calibration() -> None:
    """ShieldGemmaCpuProvider sends n_probs=5, calibrates thresholds, and evaluates responses."""
    cfg = GuardrailConfig(
        host="127.0.0.1", port=18082, timeout_seconds=2.0, violation_threshold=0.8
    )
    provider = ShieldGemmaCpuProvider(cfg)
    assert isinstance(provider, GuardrailProviderPort)
    assert provider.name == "shieldgemma-2b-cpu"

    url = "http://127.0.0.1:18082/completion"
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(url).mock(return_value=_prob_resp("No", 0.1, 0.9))
        decision = await provider.evaluate("kill pid 42 after OOM crash traceback")
        assert (decision.verdict, decision.direction) == (
            GuardrailVerdict.ALLOW,
            GuardrailDirection.INGRESS,
        )
        assert decision.violation_score == pytest.approx(0.1)
        assert route.call_count == EXPECTED_DEFAULT_PRINCIPLES
        sent_body = json.loads(cast("httpx.Request", route.calls[0].request).content.decode())
        assert (sent_body["n_predict"], sent_body["n_probs"], sent_body["temperature"]) == (
            4,
            5,
            0.0,
        )
        assert sent_body["stop"] == ["\n", "<end_of_turn>"]

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(url).mock(
            side_effect=[_prob_resp("No", 0.05, 0.95), _prob_resp("Yes", 0.92, 0.08)]
        )
        egress = await provider.evaluate_response(
            "help me", "harassing reply with Bearer lai_sk_secret999"
        )
        assert (egress.verdict, egress.direction, egress.violated_policy) == (
            GuardrailVerdict.BLOCK,
            GuardrailDirection.EGRESS,
            "No Harassment",
        )
        assert egress.violation_score == pytest.approx(0.92)
        assert "lai_sk_secret999" not in egress.scrubbed_prompt
        assert route.call_count == EXPECTED_SHORT_CIRCUIT_CALLS


async def test_shieldgemma_provider_timeout_alert_errors_and_health(tmp_path: Path) -> None:
    """ShieldGemmaCpuProvider emits TIMEOUT alerts, rejects invalid output, and checks health."""
    sink = _MemorySink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0, log_dir=tmp_path)
    configure_observability(LoggingConfig(log_dir=tmp_path, enable_file_logs=True), manager)
    try:
        cfg = GuardrailConfig(host="127.0.0.1", port=18082, timeout_seconds=0.5)
        provider = ShieldGemmaCpuProvider(
            cfg,
            principles=(SafetyPrinciple(name="TestPolicy", description="Test desc"),),
        )
        completion_url = "http://127.0.0.1:18082/completion"
        health_url = "http://127.0.0.1:18082/health"

        with respx.mock(assert_all_called=True) as mock:
            mock.post(completion_url).mock(side_effect=httpx.ReadTimeout("read timed out"))
            with pytest.raises(InferenceBackendError, match="timed out"):
                await provider.evaluate("hello")

        with respx.mock(assert_all_called=True) as mock:
            mock.post(completion_url).mock(
                return_value=httpx.Response(HTTP_OK, json={"content": "Maybe"}),
            )
            with pytest.raises(InferenceBackendError, match="Ambiguous"):
                await provider.evaluate("hello")

        with respx.mock(assert_all_called=True) as mock:
            mock.get(health_url).mock(
                return_value=httpx.Response(HTTP_OK, json={"status": "ok", "slots_idle": 2}),
            )
            healthy = await provider.health()
            assert (healthy.healthy, healthy.slots_idle) == (True, 2)

        with respx.mock(assert_all_called=True) as mock:
            mock.get(health_url).mock(
                return_value=httpx.Response(HTTP_UNAVAILABLE, json={"error": "loading"}),
            )
            loading = await provider.health()
            assert (loading.healthy, loading.status) == (False, "http_503")

        await manager.flush()
    finally:
        manager.close()

    timeout_alerts = [e for e in manager.recent_alerts() if e.category is AlertCategory.TIMEOUT]
    assert len(timeout_alerts) == 1
    assert timeout_alerts[0].component == "guardrails"


async def test_pipeline_audits_all_directions_analytics_and_output_modes(tmp_path: Path) -> None:
    """Pipeline audits ingress/egress/tool_output, emits SECURITY_ALERT, and records analytics."""
    sink = _MemorySink()
    manager = AlertManager(sinks=(sink,), cooldown_seconds=0.0, log_dir=tmp_path / "alerts")
    configure_observability(
        LoggingConfig(log_dir=tmp_path / "alerts", enable_file_logs=True), manager
    )
    spy_analytics = _SpyAnalyticsRepo()
    cfg = GuardrailConfig(
        fail_open=False,
        block_mode=GuardrailBlockMode.HTTP_ERROR,
        output_mode=GuardrailOutputMode.AUDIT_ONLY,
    )
    regex = RegexPolicyGuardrailProvider(rules=(("BadOutput", r"UNSAFE_TEXT"),))
    try:
        with CompositeGuardrailPipeline(
            cfg, providers=[regex], analytics_repo=spy_analytics, log_dir=tmp_path
        ) as pipeline:
            assert isinstance(pipeline, GuardrailPipelinePort)
            res_in = await pipeline.check_prompt("safe prompt token=mysecret99", request_id="req_1")
            assert (res_in.verdict, res_in.direction) == (
                GuardrailVerdict.ALLOW,
                GuardrailDirection.INGRESS,
            )
            res_tool = await pipeline.check_prompt(
                "safe tool output", direction=GuardrailDirection.TOOL_OUTPUT, request_id="req_2"
            )
            assert res_tool.direction is GuardrailDirection.TOOL_OUTPUT

            # AUDIT_ONLY evaluates, audits, and emits SECURITY_ALERT without raising.
            res_eg = await pipeline.check_response(
                "q", "UNSAFE_TEXT email=dev@example.com", request_id="req_3"
            )
            assert (res_eg.verdict, res_eg.direction, res_eg.violated_policy) == (
                GuardrailVerdict.BLOCK,
                GuardrailDirection.EGRESS,
                "BadOutput",
            )

        # OFF mode returns passthrough ALLOW; failing analytics never breaks requests.
        failing_analytics = _SpyAnalyticsRepo(fail_on_write=True)
        cfg_off = GuardrailConfig(output_mode=GuardrailOutputMode.OFF)
        with CompositeGuardrailPipeline(
            cfg_off, providers=[regex], analytics_repo=failing_analytics, log_dir=tmp_path
        ) as off_pipe:
            res_off = await off_pipe.check_response("q", "UNSAFE_TEXT")
            assert (res_off.verdict, res_off.provider) == (GuardrailVerdict.ALLOW, "allow-all")
        await manager.flush()
    finally:
        manager.close()

    sec_alerts = [e for e in manager.recent_alerts() if e.category is AlertCategory.SECURITY_ALERT]
    assert len(sec_alerts) == 1
    assert sec_alerts[0].component == "guardrails"

    audit_text = (tmp_path / "prompts_audit.jsonl").read_text(encoding="utf-8")
    assert "mysecret99" not in audit_text
    assert "dev@example.com" not in audit_text
    rows = [json.loads(line) for line in audit_text.splitlines()]
    assert [row["direction"] for row in rows] == ["ingress", "tool_output", "egress", "egress"]
    assert [ev.direction for ev in spy_analytics.guardrail_events] == [
        GuardrailDirection.INGRESS,
        GuardrailDirection.TOOL_OUTPUT,
        GuardrailDirection.EGRESS,
    ]
