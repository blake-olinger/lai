"""Mutation-killing tests for Ollama, ShieldGemma CPU, and guardrail pipeline behavior."""

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
from lai.domain.errors import InferenceBackendError
from lai.guardrails import (
    AllowAllGuardrailProvider,
    CompositeGuardrailPipeline,
    OllamaShieldGemmaProvider,
    ShieldGemmaCpuProvider,
)
from lai.observability import PromptAuditLogger, configure_observability
from lai.ports import GuardrailProviderPort

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
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


def _logprobs_payload(p_yes: float, p_no: float) -> list[dict[str, object]]:
    return [
        {
            "top_logprobs": [
                {"token": "Yes", "logprob": math.log(p_yes)},
                {"token": "No", "logprob": math.log(p_no)},
            ],
        },
    ]


async def test_ollama_guardrail_payload_threshold_latency_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert Ollama generate payload booleans, >= threshold, and health latency/status."""
    clock = [10.0]
    monkeypatch.setattr("lai.guardrails.ollama_provider.time.perf_counter", lambda: clock[0])
    cfg = GuardrailConfig(
        backend="ollama",
        ollama_base_url="http://127.0.0.1:18100",
        ollama_model="lai-shieldgemma-2b:latest",
        violation_threshold=_THRESHOLD_EXACT,
    )
    principle = SafetyPrinciple(name="P", description="Test safety policy")
    gen_url = "http://127.0.0.1:18100/api/generate"
    tags_url = "http://127.0.0.1:18100/api/tags"
    provider = OllamaShieldGemmaProvider(cfg, principles=(principle,))

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(gen_url).mock(
            side_effect=_seq_handler(
                clock,
                [
                    httpx.Response(
                        _HTTP_OK,
                        json={
                            "response": "Yes",
                            "logprobs": _logprobs_payload(_THRESHOLD_EXACT, _COMPLEMENT_PROB),
                        },
                    ),
                    httpx.Response(
                        _HTTP_OK,
                        json={
                            "response": "No",
                            "logprobs": _logprobs_payload(_SAFE_YES_PROB, _SAFE_NO_PROB),
                        },
                    ),
                ],
            ),
        )
        blocked = await provider.evaluate("threshold edge prompt")
        allowed = await provider.evaluate("clean prompt")

        assert blocked.verdict is GuardrailVerdict.BLOCK
        assert blocked.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)
        assert allowed.verdict is GuardrailVerdict.ALLOW
        assert allowed.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

        sent = json.loads(cast("httpx.Request", route.calls[0].request).content.decode("utf-8"))
        assert sent["raw"] is True
        assert sent["stream"] is False
        assert sent["logprobs"] is True
        assert sent["options"]["use_mmap"] is True

    with respx.mock(assert_all_called=True) as mock:
        mock.get(tags_url).mock(
            side_effect=_seq_handler(
                clock,
                [
                    httpx.ReadTimeout("slow tags"),
                    httpx.ConnectError("tags down"),
                    httpx.Response(_HTTP_UNAVAILABLE, text=""),
                    httpx.Response(_HTTP_OK, json={"models": [{"name": "other:latest"}]}),
                    httpx.Response(
                        _HTTP_OK,
                        json={
                            "models": [
                                "ignored-str",
                                {"model": "lai-shieldgemma-2b"},
                            ],
                        },
                    ),
                ],
            ),
        )
        h_timeout = await provider.health()
        h_unreach = await provider.health()
        h_http = await provider.health()
        h_missing = await provider.health()
        h_prefix = await provider.health()

        assert h_timeout.healthy is False
        assert h_timeout.status == "timeout"
        assert h_timeout.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

        assert h_unreach.healthy is False
        assert h_unreach.status == "unreachable"
        assert h_unreach.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

        assert h_http.healthy is False
        assert h_http.status == "http_503"
        assert h_http.error == "HTTP 503"
        assert h_http.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

        assert h_missing.healthy is False
        assert h_missing.status == "model_missing"

        assert h_prefix.healthy is True
        assert h_prefix.status == "ok"


async def test_shieldgemma_cpu_latency_slot_math_and_unhealthy_booleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert ShieldGemmaCpuProvider latency_ms, 0-slot totals, and healthy=False on faults."""
    clock = [10.0]
    monkeypatch.setattr("lai.guardrails.shieldgemma.time.perf_counter", lambda: clock[0])
    cfg = GuardrailConfig(host="127.0.0.1", port=18101, timeout_seconds=1.0)
    principle = SafetyPrinciple(name="P", description="Test CPU policy")
    completion_url = "http://127.0.0.1:18101/completion"
    health_url = "http://127.0.0.1:18101/health"
    provider = ShieldGemmaCpuProvider(cfg, principles=(principle,))

    with respx.mock(assert_all_called=True) as mock:
        mock.post(completion_url).mock(
            side_effect=_seq_handler(
                clock,
                [
                    httpx.Response(_HTTP_OK, json={"content": "Yes"}),
                    httpx.Response(_HTTP_OK, json={"content": "No"}),
                ],
            ),
        )
        blocked = await provider.evaluate("bad input")
        allowed = await provider.evaluate("good input")
        assert blocked.verdict is GuardrailVerdict.BLOCK
        assert blocked.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)
        assert allowed.verdict is GuardrailVerdict.ALLOW
        assert allowed.latency_ms == pytest.approx(_EXPECTED_LATENCY_MS)

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

        assert h_zero_idle.healthy is True
        assert h_zero_idle.slots_idle == 0
        assert h_zero_idle.slots_total == _SLOTS_TOTAL_TWO

        assert h_neg_idle.healthy is True
        assert h_neg_idle.slots_idle == 0
        assert h_neg_idle.slots_total == 1


async def test_pipeline_audit_logger_ownership_and_ruling_log_dedup(tmp_path: Path) -> None:
    """Pipeline preserves caller audit_logger ownership and deduplicates fail-closed logs."""
    audit_dir = tmp_path / "audits"
    ext_logger = PromptAuditLogger(audit_dir)
    try:
        with CompositeGuardrailPipeline(
            GuardrailConfig(enabled=True, fail_open=False),
            providers=[AllowAllGuardrailProvider()],
            audit_logger=ext_logger,
        ) as owned_false_pipeline:
            res = await owned_false_pipeline.check_prompt("safe query")
            assert res.verdict is GuardrailVerdict.ALLOW

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
            with CompositeGuardrailPipeline(cfg, providers=provs, log_dir=tmp_path / sub) as p:
                await p.check_prompt("hello guardrail")
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
    assert events.count("guardrail_ruling") == 3  # noqa: PLR2004
    assert events.count("guardrail_unconfigured_fail_closed") == 1
    assert events.count("guardrail_fail_closed") == 1
