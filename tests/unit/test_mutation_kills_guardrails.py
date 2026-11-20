"""Mutation-killing tests for ShieldGemma CPU calibration, egress, and guardrail pipeline."""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

from lai.domain import (
    GuardrailConfig,
    GuardrailDecision,
    GuardrailVerdict,
    LoggingConfig,
    PromptAuditEntry,
    SafetyPrinciple,
)
from lai.domain.analytics import AnalyticsReport, GuardrailUsageEvent, TokenUsageRecord, UsageEvent
from lai.domain.errors import InferenceBackendError
from lai.domain.guardrails import GuardrailDirection, GuardrailOutputMode
from lai.guardrails import (
    AllowAllGuardrailProvider,
    CompositeGuardrailPipeline,
    ShieldGemmaCpuProvider,
    compute_yes_probability,
)
from lai.observability import PromptAuditLogger, configure_observability
from lai.ports import GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from datetime import datetime
    from pathlib import Path

_HTTP_OK = 200
_HTTP_UNAVAILABLE = 503
_EXPECTED_LATENCY_MS = 50.0
_SLOTS_TOTAL_TWO = 2
_THRESHOLD_EXACT = 0.8
_COMPLEMENT_PROB = 0.2
_SAFE_YES_PROB = 0.1
_SAFE_NO_PROB = 0.9
_AUDIT_SHA = "a" * 64


class _BrokenProvider(GuardrailProviderPort):
    """Test guardrail provider that always raises to verify fail-open and fail-closed logging."""

    @property
    def name(self) -> str:
        return "broken-provider"

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        del prompt
        msg = "provider down"
        raise InferenceBackendError(msg)

    async def evaluate_response(self, prompt: str, response: str) -> GuardrailDecision:
        del prompt, response
        msg = "provider down"
        raise InferenceBackendError(msg)


class _RecordingAnalytics:
    """Minimal UsageAnalyticsPort stub capturing GuardrailUsageEvent records."""

    def __init__(self) -> None:
        self.events: list[GuardrailUsageEvent] = []

    async def record_request(self, event: UsageEvent) -> None:
        del event

    async def record_tokens(self, record: TokenUsageRecord) -> None:
        del record

    async def record_guardrail(self, event: GuardrailUsageEvent) -> None:
        self.events.append(event)

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
    """Ensure root logger handlers created in tests are flushed and closed."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def _seq_handler(
    clock: list[float],
    outcomes: list[httpx.Response | Exception],
) -> Callable[[httpx.Request], httpx.Response]:
    """Return a single respx callback cycling outcomes and advancing clock by 50ms per call."""
    it = iter(outcomes)

    def _handler(_request: httpx.Request) -> httpx.Response:
        clock[0] += 0.05
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    return _handler


def _probs_payload(p_yes: float, p_no: float) -> list[dict[str, object]]:
    return [
        {
            "content": "Yes" if p_yes >= p_no else "No",
            "probs": [
                {"tok_str": "Yes", "prob": p_yes},
                {"tok_str": "No", "prob": p_no},
            ],
        },
    ]


def _logprobs_payload(p_yes: float, p_no: float) -> list[dict[str, object]]:
    return [
        {
            "top_logprobs": [
                {"token": "Yes", "logprob": math.log(p_yes)},
                {"token": "No", "logprob": math.log(p_no)},
            ],
        },
    ]


def test_compute_yes_probability_handles_probs_top_logprobs_and_malformed_inputs() -> None:
    """compute_yes_probability normalizes probs & top_logprobs and ignores malformed entries."""
    assert compute_yes_probability(_probs_payload(_THRESHOLD_EXACT, _COMPLEMENT_PROB)) == (
        pytest.approx(_THRESHOLD_EXACT)
    )
    assert compute_yes_probability(_logprobs_payload(_THRESHOLD_EXACT, _COMPLEMENT_PROB)) == (
        pytest.approx(_THRESHOLD_EXACT)
    )
    empty_list: list[object] = []
    bad_inputs: tuple[object, ...] = (
        "not-a-list",
        empty_list,
        ["not-a-dict"],
        [{"other": empty_list}],
        [{"probs": [{"tok_str": "Yes", "prob": True}, {"tok_str": "No", "prob": -0.5}]}],
        [{"probs": [{"tok_str": "Yes", "prob": float("nan")}, {"tok_str": "Other", "prob": 1.0}]}],
        [{"top_logprobs": [{"token": "Yes", "logprob": True}, {"token": "No", "logprob": 1e9}]}],
        [{"probs": [{"tok_str": "Yes", "prob": 0.0}, {"tok_str": "No", "prob": 0.0}]}],
    )
    for bad in bad_inputs:
        assert compute_yes_probability(bad) is None


def _ok_probs(payload: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(_HTTP_OK, json={"content": "Yes", "completion_probabilities": payload})


async def test_shieldgemma_cpu_threshold_boundary_egress_latency_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert n_probs >= threshold boundary, egress evaluation, latency_ms, and health faults."""
    clock = [10.0]
    monkeypatch.setattr("lai.guardrails.shieldgemma.time.perf_counter", lambda: clock[0])
    cfg = GuardrailConfig(
        host="127.0.0.1", port=18101, timeout_seconds=1.0, violation_threshold=_THRESHOLD_EXACT
    )
    principle = SafetyPrinciple(name="P", description="Test CPU policy")
    completion_url = "http://127.0.0.1:18101/completion"
    health_url = "http://127.0.0.1:18101/health"
    provider = ShieldGemmaCpuProvider(cfg, principles=(principle,), output_principles=(principle,))

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(completion_url).mock(
            side_effect=_seq_handler(
                clock,
                [
                    _ok_probs(_probs_payload(_THRESHOLD_EXACT, _COMPLEMENT_PROB)),
                    _ok_probs(_logprobs_payload(_SAFE_YES_PROB, _SAFE_NO_PROB)),
                    _ok_probs(_probs_payload(0.95, 0.05)),
                ],
            ),
        )
        blocked = await provider.evaluate("threshold exact prompt")
        allowed = await provider.evaluate("below threshold despite Yes text")
        egress_blocked = await provider.evaluate_response("user prompt", "bad chatbot output")

        assert blocked.verdict is GuardrailVerdict.BLOCK
        assert blocked.violation_score == pytest.approx(_THRESHOLD_EXACT)
        assert blocked.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)
        assert allowed.verdict is GuardrailVerdict.ALLOW
        assert allowed.violation_score == pytest.approx(_SAFE_YES_PROB)
        assert allowed.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)
        assert (egress_blocked.verdict, egress_blocked.direction) == (
            GuardrailVerdict.BLOCK,
            GuardrailDirection.EGRESS,
        )
        assert egress_blocked.violation_score == pytest.approx(0.95)
        assert egress_blocked.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

        sent = json.loads(cast("httpx.Request", route.calls[0].request).content.decode("utf-8"))
        assert sent["n_probs"] == 5

    with respx.mock(assert_all_called=True) as mock:
        mock.get(health_url).mock(
            side_effect=_seq_handler(
                clock,
                [
                    httpx.ReadTimeout("timeout"),
                    httpx.ConnectError("refused"),
                    httpx.Response(_HTTP_UNAVAILABLE, text=""),
                    httpx.Response(
                        _HTTP_OK,
                        json={
                            "status": "ok",
                            "slots_idle": 0,
                            "slots_processing": _SLOTS_TOTAL_TWO,
                        },
                    ),
                    httpx.Response(
                        _HTTP_OK,
                        json={"status": "ok", "slots_idle": -1, "slots_total": 1},
                    ),
                ],
            ),
        )
        h_timeout = await provider.health()
        h_unreach = await provider.health()
        h_http = await provider.health()
        h_zero_idle = await provider.health()
        h_neg_idle = await provider.health()

        for fault in (h_timeout, h_unreach, h_http):
            assert fault.healthy is False
            assert fault.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

        assert (h_zero_idle.healthy, h_zero_idle.slots_idle, h_zero_idle.slots_total) == (
            True,
            0,
            _SLOTS_TOTAL_TWO,
        )
        assert (h_neg_idle.healthy, h_neg_idle.slots_idle, h_neg_idle.slots_total) == (True, 0, 1)


async def test_pipeline_audit_ownership_fail_closed_analytics_and_log_dedup(tmp_path: Path) -> None:
    """Pipeline preserves audit_logger ownership, tracks fail_closed analytics, and dedups logs."""
    audit_dir = tmp_path / "audits"
    ext_logger = PromptAuditLogger(audit_dir)
    analytics = _RecordingAnalytics()
    try:
        with CompositeGuardrailPipeline(
            GuardrailConfig(enabled=True, fail_open=False),
            providers=[AllowAllGuardrailProvider()],
            audit_logger=ext_logger,
            analytics_repo=analytics,
        ) as owned_false_pipeline:
            res = await owned_false_pipeline.check_prompt("safe query")
            assert res.verdict is GuardrailVerdict.ALLOW
            assert analytics.events[-1].fail_closed is False

        ext_logger.log_entry(
            PromptAuditEntry(
                verdict=GuardrailVerdict.ALLOW,
                provider="external",
                prompt_sha256=_AUDIT_SHA,
                prompt_chars=5,
                scrubbed_prompt="after",
            ),
        )
    finally:
        ext_logger.close()

    audit_lines = (audit_dir / "prompts_audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == _SLOTS_TOTAL_TWO

    log_dir = tmp_path / "logs"
    manager = configure_observability(
        LoggingConfig(log_dir=log_dir, level="INFO", json_console=False, enable_file_logs=True),
    )
    try:
        for cfg, provs, sub in (
            (GuardrailConfig(enabled=False), (), "a1"),
            (GuardrailConfig(enabled=True, fail_open=True), (), "a2"),
            (GuardrailConfig(enabled=True, fail_open=False), (), "a3"),
            (GuardrailConfig(enabled=True, fail_open=True), (_BrokenProvider(),), "a4"),
            (GuardrailConfig(enabled=True, fail_open=False), (_BrokenProvider(),), "a5"),
        ):
            with CompositeGuardrailPipeline(
                cfg, providers=provs, analytics_repo=analytics, log_dir=tmp_path / sub
            ) as p:
                await p.check_prompt("hello guardrail")
        with CompositeGuardrailPipeline(
            GuardrailConfig(
                enabled=True, fail_open=False, output_mode=GuardrailOutputMode.BUFFERED
            ),
            providers=(_BrokenProvider(),),
            analytics_repo=analytics,
            log_dir=tmp_path / "a6",
        ) as p_eg:
            eg = await p_eg.check_response("q", "model reply")
            assert (eg.verdict, eg.direction) == (GuardrailVerdict.BLOCK, GuardrailDirection.EGRESS)
    finally:
        manager.close()
        for handler in list(logging.getLogger().handlers):
            handler.flush()
            handler.close()
        logging.getLogger().handlers.clear()

    events = [
        json.loads(line)["event"]
        for line in (log_dir / "lai.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events.count("guardrail_ruling") == 3
    assert events.count("guardrail_unconfigured_fail_closed") == 1
    assert events.count("guardrail_fail_closed") == 2
    assert sum(1 for ev in analytics.events if ev.fail_closed) == 3
