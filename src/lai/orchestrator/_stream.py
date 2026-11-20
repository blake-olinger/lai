"""Shared streaming inference lease helper and normalized turn outcome model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lai.domain.chat import (
    Citation,
    ContentDeltaEvent,
    OrchestratorEvent,
    ThinkingDeltaEvent,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from lai.domain.chat import ClientToolCallEvent
    from lai.domain.inference import InferenceMessage, InferenceRequest, InferenceToolCall
    from lai.domain.scheduler import LeaseRequest
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.scheduler import InferenceSpoolerPort


@dataclass(slots=True)
class TurnOutcome:
    """Internal normalized completion state produced by any routing strategy."""

    content: str
    thinking: str
    citations: tuple[Citation, ...]
    usage: TokenUsage
    pending_approval_id: str | None = None
    pending_client_tool_call: ClientToolCallEvent | None = None
    pending_tool_call: InferenceToolCall | None = None
    assistant_tool_calls: tuple[InferenceToolCall, ...] = ()
    intermediate_messages: tuple[InferenceMessage, ...] = ()
    post_assistant_tool_messages: tuple[InferenceMessage, ...] = ()


@dataclass(slots=True)
class StreamedInferenceAccumulator:
    """Mutable collector populated while streaming a leased InferenceRequest."""

    content_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    tool_calls: list[InferenceToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def content(self) -> str:
        """Return concatenated content deltas."""
        return "".join(self.content_parts)

    @property
    def thinking(self) -> str:
        """Return concatenated thinking deltas."""
        return "".join(self.thinking_parts)


async def stream_leased_inference(
    *,
    spooler: InferenceSpoolerPort,
    inference_provider: InferenceProviderPort,
    lease_request: LeaseRequest,
    inference_request: InferenceRequest,
    acc: StreamedInferenceAccumulator,
) -> AsyncGenerator[OrchestratorEvent]:
    """Acquire an exclusive spooler lease and yield Thinking/Content delta events into acc."""
    async with spooler.lease(lease_request):
        async for chunk in inference_provider.chat_stream(inference_request):
            if chunk.thinking_delta:
                acc.thinking_parts.append(chunk.thinking_delta)
                yield ThinkingDeltaEvent(delta=chunk.thinking_delta)
            if chunk.content_delta:
                acc.content_parts.append(chunk.content_delta)
                yield ContentDeltaEvent(delta=chunk.content_delta)
            if chunk.tool_calls:
                acc.tool_calls.extend(chunk.tool_calls)
            if chunk.usage is not None:
                acc.usage = chunk.usage
