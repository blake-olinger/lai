"""Chat session and turn message API schemas."""

from __future__ import annotations

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.chat import ContextMode, Message, Session
from lai.domain.tools import ToolSpec


class SessionCreateRequest(StrictModel):
    """Payload for creating a new chat session within a project."""

    project_id: str = Field(min_length=1)
    title: str = Field(default="New Session", min_length=1, max_length=200)
    model_profile: str | None = None
    system_prompt: str | None = None


class SessionListResponse(StrictModel):
    """Paginated list of chat sessions for a project."""

    items: list[Session]
    count: int = Field(ge=0)


class SessionDetailResponse(StrictModel):
    """Session metadata paired with chronological message history."""

    session: Session
    messages: list[Message]


class SessionMessageRequest(StrictModel):
    """User turn payload posted to a chat session."""

    content: str = Field(min_length=1)
    attached_file_ids: tuple[str, ...] = ()
    context_mode: ContextMode = ContextMode.AUTO
    stream: bool = False
    enable_tools: bool = True
    enable_thinking: bool = True
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    client_tools: tuple[ToolSpec, ...] = ()
