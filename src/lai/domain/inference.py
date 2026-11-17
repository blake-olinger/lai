"""Inference provider, model profile, and backend health domain models."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.chat import MessageRole, TokenUsage


class ModelProfile(StrictModel):
    """Hardware-tuned llama-server model and runtime configuration."""

    name: str = Field(min_length=1)
    hf_repo: str = Field(min_length=1)
    fallback_hf_repo: str | None = None
    hf_filename: str = Field(min_length=1)
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


class InferenceToolCall(StrictModel):
    """Tool invocation emitted by the inference model."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


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
