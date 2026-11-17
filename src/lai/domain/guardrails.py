"""Pre-flight safety guardrail contracts, audit entries, and configuration models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from lai.domain.base import StrictModel, utc_now


class GuardrailVerdict(StrEnum):
    """Decision returned by a pre-flight prompt guardrail evaluation."""

    ALLOW = "allow"
    BLOCK = "block"


class GuardrailBlockMode(StrEnum):
    """Response mode when a prompt is blocked by the guardrail policy."""

    REFUSAL_MESSAGE = "refusal_message"
    HTTP_ERROR = "http_error"


class SafetyPrinciple(StrictModel):
    """Named safety principle evaluated by the guardrail prompt template."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GuardrailDecision(StrictModel):
    """Structured output produced by evaluating a user prompt against guardrails."""

    verdict: GuardrailVerdict
    provider: str = Field(min_length=1)
    violated_policy: str | None = None
    reason: str | None = None
    raw_output: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    prompt_chars: int = Field(ge=0)
    scrubbed_prompt: str


class PromptAuditEntry(StrictModel):
    """Privacy-preserving 24h rotating prompt guardrail audit log entry."""

    timestamp: datetime = Field(default_factory=utc_now)
    request_id: str | None = None
    session_id: str | None = None
    project_id: str | None = None
    principal_id: str | None = None
    verdict: GuardrailVerdict
    provider: str = Field(min_length=1)
    violated_policy: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    prompt_chars: int = Field(ge=0)
    scrubbed_prompt: str


class GuardrailConfig(StrictModel):
    """Configuration for pre-flight ShieldGemma-2B guardrail evaluation."""

    enabled: bool = True
    backend: Literal["llamacpp", "ollama"] = "llamacpp"
    block_mode: GuardrailBlockMode = GuardrailBlockMode.REFUSAL_MESSAGE
    refusal_text: str = (
        "I cannot fulfill this request because the prompt was flagged "
        "by the safety guardrail policy."
    )
    host: str = "127.0.0.1"
    port: int = Field(default=8082, ge=1, le=65535)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "lai-shieldgemma-2b"
    hf_repo: str = "QuantFactory/shieldgemma-2b-GGUF"
    fallback_hf_repo: str = "mradermacher/shieldgemma-2b-GGUF"
    hf_filename: str = "shieldgemma-2b.Q4_K_M.gguf"
    cpu_threads: int = Field(default=12, ge=1)
    ctx_size: int = Field(default=4096, ge=512)
    parallel_slots: int = Field(default=2, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    violation_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    fail_open: bool = False
    max_logged_prompt_chars: int = Field(default=2000, ge=64)
