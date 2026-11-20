"""Inference provider, model profile, integrity, manifest, and backend health domain models."""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, field_validator

from lai.domain.base import StrictModel
from lai.domain.chat import InferenceToolCall, MessageRole, TokenUsage

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

__all__ = [
    "BackendHealth",
    "InferenceChunk",
    "InferenceMessage",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceToolCall",
    "InferenceToolDef",
    "LlamaCppBinarySpec",
    "ModelIntegrityConfig",
    "ModelProfile",
]


def _normalize_sha256(value: str) -> str:
    """Validate and normalize a 64-character lowercase hexadecimal SHA-256 digest."""
    normalized = value.strip().lower()
    if _SHA256_HEX_RE.fullmatch(normalized) is None:
        msg = "SHA-256 digest must be a 64-character hexadecimal string."
        raise ValueError(msg)
    return normalized


class ModelProfile(StrictModel):
    """Hardware-tuned llama-server model and runtime configuration."""

    name: str = Field(min_length=1)
    hf_repo: str = Field(min_length=1)
    fallback_hf_repo: str | None = None
    hf_filename: str = Field(min_length=1)
    hf_revision: str | None = None
    sha256: str | None = None
    ctx_size: int = Field(default=24576, ge=512)
    flash_attn: bool = True
    kv_cache_type_k: str = "q8_0"
    kv_cache_type_v: str = "q8_0"
    parallel_slots: int = Field(default=1, ge=1)
    n_gpu_layers: int = Field(default=999, ge=0)
    cpu_moe_layers: int = Field(default=3, ge=0)
    batch_size: int = Field(default=2048, ge=1)
    ubatch_size: int = Field(default=512, ge=1)
    jinja: bool = True
    reasoning_format: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1)

    @field_validator("hf_revision")
    @classmethod
    def _validate_hf_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            msg = "hf_revision must not be empty when specified."
            raise ValueError(msg)
        return cleaned

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_sha256(value)


class ModelIntegrityConfig(StrictModel):
    """Supply-chain integrity and allowlist settings for local GGUF model artifacts."""

    strict_allowlist: bool = True
    verify_sha256_on_startup: bool = True
    approved_model_sha256s: tuple[str, ...] = ()
    allowed_hf_repos: tuple[str, ...] = ()

    @field_validator("approved_model_sha256s")
    @classmethod
    def _validate_approved_sha256s(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalize_sha256(item) for item in value)

    @field_validator("allowed_hf_repos")
    @classmethod
    def _validate_allowed_hf_repos(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for repo in value:
            item = repo.strip()
            if not item:
                msg = "allowed_hf_repos entries must not be empty."
                raise ValueError(msg)
            cleaned.append(item)
        return tuple(cleaned)


class LlamaCppBinarySpec(StrictModel):
    """Pinned Linux x64 llama.cpp CUDA binary and cudart release archive metadata."""

    tag: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    asset_name: str = Field(min_length=1)
    sha256: str
    cudart_asset_name: str = Field(min_length=1)
    cudart_sha256: str

    @field_validator("sha256", "cudart_sha256")
    @classmethod
    def _validate_digests(cls, value: str) -> str:
        return _normalize_sha256(value)


class InferenceMessage(StrictModel):
    """Single prompt or response turn serialized for the inference backend."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[InferenceToolCall, ...] = ()


class InferenceToolDef(StrictModel):
    """JSON Schema tool definition passed to the inference backend."""

    name: str = Field(min_length=1)
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(StrictModel):
    """Normalized inference request sent to llama.cpp or another provider."""

    model: str = Field(min_length=1)
    messages: tuple[InferenceMessage, ...]
    tools: tuple[InferenceToolDef, ...] = ()
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1)
    stream: bool = True
    enable_thinking: bool = True


class InferenceChunk(StrictModel):
    """Single streaming chunk returned by an inference provider."""

    content_delta: str = ""
    thinking_delta: str = ""
    tool_calls: tuple[InferenceToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class InferenceResponse(StrictModel):
    """Aggregated non-streaming response returned by an inference provider."""

    content: str
    thinking: str = ""
    tool_calls: tuple[InferenceToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class BackendHealth(StrictModel):
    """Health snapshot reported by the supervised llama-server process."""

    healthy: bool
    status: str = Field(min_length=1)
    model_loaded: str | None = None
    slots_total: int = Field(default=0, ge=0)
    slots_idle: int = Field(default=0, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    error: str | None = None
