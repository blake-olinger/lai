"""OpenAI-compatible models and chat completions request/response DTOs."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from lai.domain.base import StrictModel
from lai.domain.chat import ClientToolCallEvent, ContextMode
from lai.domain.tools import ToolSpec


class OpenAIModelInfo(StrictModel):
    """Single model descriptor returned by GET /v1/models."""

    id: str = Field(min_length=1)
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class OpenAIModelListResponse(StrictModel):
    """OpenAI-compatible model listing envelope."""

    object: Literal["list"] = "list"
    data: list[OpenAIModelInfo]


class OpenAIChatMessageInput(BaseModel):
    """Incoming OpenAI message supporting text strings or multimodal text parts."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    """Drop-in OpenAI /v1/chat/completions payload with optional LAI routing fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    model: str | None = Field(default=None, min_length=1)
    messages: list[OpenAIChatMessageInput] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    project_id: str | None = None
    file_ids: list[str] | None = None
    session_id: str | None = None
    context_mode: ContextMode | None = None
    enable_tools: bool = True
    enable_thinking: bool = True
    client_tools: list[ToolSpec] = Field(default_factory=list)


class OpenAIToolFunction(StrictModel):
    """Function name and JSON-encoded arguments inside an OpenAI tool call."""

    name: str = Field(min_length=1)
    arguments: str


class OpenAIToolCall(StrictModel):
    """Non-streaming OpenAI assistant tool_calls item."""

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIDeltaToolCall(StrictModel):
    """Streaming OpenAI delta.tool_calls item."""

    index: int = 0
    id: str | None = None
    type: Literal["function"] | None = None
    function: OpenAIToolFunction | None = None


class OpenAIChatChoiceMessage(StrictModel):
    """Assistant message object returned in a non-streaming completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str
    reasoning_content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChatChoice(StrictModel):
    """Single completion choice entry."""

    index: int = 0
    message: OpenAIChatChoiceMessage
    finish_reason: str | None = "stop"


class OpenAIUsage(StrictModel):
    """Token usage breakdown compatible with OpenAI responses."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class OpenAIChatCompletionResponse(StrictModel):
    """Non-streaming OpenAI chat completion response envelope."""

    id: str = Field(min_length=1)
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str = Field(min_length=1)
    choices: list[OpenAIChatChoice]
    usage: OpenAIUsage
    lai_client_tool_call: ClientToolCallEvent | None = None


class OpenAIDeltaPayload(StrictModel):
    """Incremental SSE delta object inside a chat.completion.chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[OpenAIDeltaToolCall] | None = None


class OpenAIStreamChoice(StrictModel):
    """Single streaming chunk choice entry."""

    index: int = 0
    delta: OpenAIDeltaPayload
    finish_reason: str | None = None


class OpenAIChatCompletionChunk(StrictModel):
    """Streaming SSE chunk matching OpenAI chat.completion.chunk schema."""

    id: str = Field(min_length=1)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str = Field(min_length=1)
    choices: list[OpenAIStreamChoice]
    usage: OpenAIUsage | None = None
    lai_client_tool_call: ClientToolCallEvent | None = None
