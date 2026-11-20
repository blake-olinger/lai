"""Pre-flight and egress safety guardrail contracts, audit entries, and configuration models."""

from __future__ import annotations

import os
import re
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from lai.domain.base import StrictModel, utc_now
from lai.domain.errors import ValidationFailedError

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_CPU_FALLBACK = 4
_MAX_DEFAULT_CPU_THREADS = 12
_MIN_TOKENS_PER_SLOT = 512
MIN_LOCKED_TIMEOUT_SECONDS = 5.0


def _default_cpu_threads() -> int:
    """Return a bounded default CPU thread count for local ShieldGemma inference."""
    return min(os.cpu_count() or _DEFAULT_CPU_FALLBACK, _MAX_DEFAULT_CPU_THREADS)


class GuardrailVerdict(StrEnum):
    """Decision returned by a guardrail safety evaluation."""

    ALLOW = "allow"
    BLOCK = "block"


class GuardrailBlockMode(StrEnum):
    """Response mode when a prompt or output is blocked by the guardrail policy."""

    REFUSAL_MESSAGE = "refusal_message"
    HTTP_ERROR = "http_error"


class GuardrailDirection(StrEnum):
    """Direction of text inspected by the guardrail pipeline."""

    INGRESS = "ingress"
    EGRESS = "egress"
    TOOL_OUTPUT = "tool_output"


class GuardrailOutputMode(StrEnum):
    """Enforcement mode for ShieldGemma model response egress scanning."""

    BUFFERED = "buffered"
    AUDIT_ONLY = "audit_only"
    OFF = "off"


class SafetyPrinciple(StrictModel):
    """Named safety principle evaluated by the guardrail prompt template."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GuardrailDecision(StrictModel):
    """Structured output produced by evaluating text against safety guardrails."""

    verdict: GuardrailVerdict
    provider: str = Field(min_length=1)
    direction: GuardrailDirection = GuardrailDirection.INGRESS
    violated_policy: str | None = None
    violation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    raw_output: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    prompt_chars: int = Field(ge=0)
    scrubbed_prompt: str


class PromptAuditEntry(StrictModel):
    """Privacy-preserving 24h rotating guardrail audit log entry."""

    timestamp: datetime = Field(default_factory=utc_now)
    request_id: str | None = None
    session_id: str | None = None
    project_id: str | None = None
    principal_id: str | None = None
    direction: GuardrailDirection = GuardrailDirection.INGRESS
    verdict: GuardrailVerdict
    provider: str = Field(min_length=1)
    violated_policy: str | None = None
    violation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    prompt_chars: int = Field(ge=0)
    scrubbed_prompt: str


_DEFAULT_SHIELDGEMMA_HF_REVISION = "8755ece4b20dd84d7564131e385d9c205d03621a"
_DEFAULT_SHIELDGEMMA_SHA256 = "47b0c3f4ec0bf93659ab2fc92cf2041374ef78bf1cb5b8c790421f463e7b7979"


class GuardrailConfig(StrictModel):
    """Configuration for ShieldGemma-2B ingress and egress guardrail evaluation."""

    enabled: bool = True
    block_mode: GuardrailBlockMode = GuardrailBlockMode.REFUSAL_MESSAGE
    output_mode: GuardrailOutputMode = GuardrailOutputMode.BUFFERED
    refusal_text: str = (
        "I cannot fulfill this request because the prompt was flagged "
        "by the safety guardrail policy."
    )
    host: str = "127.0.0.1"
    port: int = Field(default=8082, ge=1, le=65535)
    hf_repo: str = "QuantFactory/shieldgemma-2b-GGUF"
    fallback_hf_repo: str = "mradermacher/shieldgemma-2b-GGUF"
    hf_filename: str = "shieldgemma-2b.Q4_K_M.gguf"
    hf_revision: str | None = _DEFAULT_SHIELDGEMMA_HF_REVISION
    sha256: str | None = _DEFAULT_SHIELDGEMMA_SHA256
    cpu_threads: int = Field(default_factory=_default_cpu_threads, ge=1)
    ctx_size: int = Field(default=8192, ge=512)
    parallel_slots: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    violation_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    fail_open: bool = False
    allow_runtime_disable: bool = False
    max_violation_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_logged_prompt_chars: int = Field(default=2000, ge=64)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not _SHA256_HEX_RE.fullmatch(normalized):
            msg = "sha256 must be a 64-character lowercase hexadecimal digest."
            raise ValueError(msg)
        return normalized

    @field_validator("hf_revision")
    @classmethod
    def _validate_hf_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            msg = "hf_revision must not be empty when set."
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def _validate_threshold_bounds(self) -> Self:
        if self.violation_threshold > self.max_violation_threshold:
            msg = (
                f"violation_threshold ({self.violation_threshold}) must not exceed "
                f"max_violation_threshold ({self.max_violation_threshold})."
            )
            raise ValueError(msg)
        if self.ctx_size // self.parallel_slots < _MIN_TOKENS_PER_SLOT:
            msg = (
                f"Per-slot context size ({self.ctx_size} // {self.parallel_slots}) "
                f"must be at least {_MIN_TOKENS_PER_SLOT} tokens."
            )
            raise ValueError(msg)
        return self


_OUTPUT_MODE_STRICTNESS: dict[GuardrailOutputMode, int] = {
    GuardrailOutputMode.OFF: 0,
    GuardrailOutputMode.AUDIT_ONLY: 1,
    GuardrailOutputMode.BUFFERED: 2,
}


def _validate_locked_runtime_fields(
    boot_config: GuardrailConfig,
    candidate: GuardrailConfig,
) -> None:
    """Enforce non-weakening checks while allow_runtime_disable is False."""
    if candidate.allow_runtime_disable:
        msg = "Cannot enable allow_runtime_disable at runtime; update YAML config and restart."
        raise ValidationFailedError(msg, details={"field": "allow_runtime_disable"})

    if candidate.max_violation_threshold > boot_config.max_violation_threshold:
        msg = "Cannot raise max_violation_threshold at runtime; update YAML config and restart."
        raise ValidationFailedError(msg, details={"field": "max_violation_threshold"})

    if boot_config.enabled and not candidate.enabled:
        msg = "Guardrails cannot be disabled at runtime while allow_runtime_disable is False."
        raise ValidationFailedError(msg, details={"field": "enabled"})

    if not boot_config.fail_open and candidate.fail_open:
        msg = (
            "Guardrail fail_open cannot be enabled at runtime while allow_runtime_disable is False."
        )
        raise ValidationFailedError(msg, details={"field": "fail_open"})

    if (
        _OUTPUT_MODE_STRICTNESS[candidate.output_mode]
        < _OUTPUT_MODE_STRICTNESS[boot_config.output_mode]
    ):
        msg = (
            f"Guardrail output_mode cannot be downgraded from {boot_config.output_mode.value!r} "
            f"to {candidate.output_mode.value!r} at runtime while allow_runtime_disable is False."
        )
        raise ValidationFailedError(msg, details={"field": "output_mode"})

    if (
        candidate.timeout_seconds < MIN_LOCKED_TIMEOUT_SECONDS
        and candidate.timeout_seconds < boot_config.timeout_seconds
    ):
        msg = (
            f"Guardrail timeout_seconds ({candidate.timeout_seconds}) cannot be lowered "
            f"below {MIN_LOCKED_TIMEOUT_SECONDS}s while allow_runtime_disable is False."
        )
        raise ValidationFailedError(
            msg,
            details={
                "field": "timeout_seconds",
                "min_timeout_seconds": str(MIN_LOCKED_TIMEOUT_SECONDS),
            },
        )


def validate_guardrail_runtime_update(
    boot_config: GuardrailConfig,
    candidate: GuardrailConfig,
) -> None:
    """Enforce boot-only guardrail locks against runtime policy hot-swap mutations."""
    if candidate.violation_threshold > boot_config.max_violation_threshold:
        msg = (
            f"violation_threshold ({candidate.violation_threshold}) exceeds boot-locked "
            f"max_violation_threshold ({boot_config.max_violation_threshold})."
        )
        raise ValidationFailedError(
            msg,
            details={
                "field": "violation_threshold",
                "max_violation_threshold": str(boot_config.max_violation_threshold),
            },
        )

    if boot_config.allow_runtime_disable:
        return

    _validate_locked_runtime_fields(boot_config, candidate)
