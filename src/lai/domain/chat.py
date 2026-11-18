"""Chat session, message, turn, and orchestrator streaming event models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_message_id, new_session_id
from lai.domain.tools import PermissionTier, ToolSpec


class MessageRole(StrEnum):
    """Conversation message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContextMode(StrEnum):
    """Strategy used by the orchestrator to feed attached files into context."""

    AUTO = "auto"
    DIRECT = "direct"
    MAP_REDUCE = "map_reduce"
    RETRIEVAL = "retrieval"


class Citation(StrictModel):
    """Line-accurate provenance pointer into a project file."""

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    chunk_id: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    excerpt: str | None = None


class TokenUsage(StrictModel):
    """Token accounting reported by an inference turn."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class InferenceToolCall(StrictModel):
    """Tool invocation emitted by the inference model."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


class Message(StrictModel):
    """Persisted message within a chat session."""

    id: str = Field(default_factory=new_message_id)
    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    role: MessageRole
    content: str
    thinking: str | None = None
    citations: tuple[Citation, ...] = ()
    tool_calls: tuple[InferenceToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    usage: TokenUsage | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Session(StrictModel):
    """Conversation session scoped to a single project."""

    id: str = Field(default_factory=new_session_id)
    project_id: str = Field(min_length=1)
    title: str = Field(default="New Session", min_length=1, max_length=200)
    model_profile: str = "gemma4-26b-a4b"
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TurnInput(StrictModel):
    """Validated user turn request passed to the orchestrator."""

    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    attached_file_ids: tuple[str, ...] = ()
    context_mode: ContextMode = ContextMode.AUTO
    enable_tools: bool = True
    enable_thinking: bool = True
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model_profile: str | None = None
    system_prompt: str | None = None
    prior_messages: tuple[Message, ...] = ()
    extra_guardrail_prompts: tuple[str, ...] = ()
    client_tools: tuple[ToolSpec, ...] = ()
    client_app_id: str | None = None
    client_allowed_tiers: tuple[PermissionTier, ...] = ()


class ClientToolCallEvent(StrictModel):
    """Event emitted when a client-executed tool invocation pauses for caller execution."""

    event_type: Literal["client_tool_call"] = "client_tool_call"
    approval_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments_json: str
    requires_approval: bool
    expires_at: datetime | None = None


class TurnOutput(StrictModel):
    """Final result of a completed orchestrator turn."""

    session_id: str = Field(min_length=1)
    message: Message
    context_mode_used: ContextMode
    citations: tuple[Citation, ...] = ()
    usage: TokenUsage
    pending_approval_id: str | None = None
    pending_client_tool_call: ClientToolCallEvent | None = None
    guardrail_blocked: bool = False
    guardrail_policy: str | None = None


class KeepaliveEvent(StrictModel):
    """Heartbeat emitted while waiting on long-running inference or map-reduce."""

    event_type: Literal["keepalive"] = "keepalive"
    timestamp: datetime = Field(default_factory=utc_now)


class ProgressEvent(StrictModel):
    """Progress update emitted during multi-stage context processing or tools."""

    event_type: Literal["progress"] = "progress"
    stage: str = Field(min_length=1)
    message: str
    completed_steps: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=0)


class ThinkingDeltaEvent(StrictModel):
    """Incremental reasoning token delta emitted by the model."""

    event_type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ContentDeltaEvent(StrictModel):
    """Incremental response token delta emitted by the model."""

    event_type: Literal["content_delta"] = "content_delta"
    delta: str


class ApprovalRequiredEvent(StrictModel):
    """Event emitted when a tool invocation pauses for explicit user approval."""

    event_type: Literal["approval_required"] = "approval_required"
    approval_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments_preview: str
    reason: str


class TurnCompletedEvent(StrictModel):
    """Terminal event delivering the persisted TurnOutput."""

    event_type: Literal["turn_completed"] = "turn_completed"
    output: TurnOutput


OrchestratorEvent = Annotated[
    KeepaliveEvent
    | ProgressEvent
    | ThinkingDeltaEvent
    | ContentDeltaEvent
    | ApprovalRequiredEvent
    | ClientToolCallEvent
    | TurnCompletedEvent,
    Field(discriminator="event_type"),
]
