"""Local inference provider and process supervisor protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from lai.domain.inference import (
        BackendHealth,
        InferenceChunk,
        InferenceRequest,
        InferenceResponse,
        ModelProfile,
    )


@runtime_checkable
class InferenceProviderPort(Protocol):
    """Chat completion and streaming contract against a local inference server."""

    async def chat_complete(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a non-streaming inference request and return full response."""
        ...

    def chat_stream(self, request: InferenceRequest) -> AsyncIterator[InferenceChunk]:
        """Stream incremental inference chunks from the backend."""
        ...

    async def health(self) -> BackendHealth:
        """Query current readiness and slot status of the inference backend."""
        ...


@runtime_checkable
class InferenceSupervisorPort(Protocol):
    """Lifecycle management contract for the supervised llama-server process."""

    async def ensure_running(self, profile: ModelProfile) -> BackendHealth:
        """Start or verify llama-server for the requested model profile."""
        ...

    async def stop(self) -> None:
        """Gracefully terminate the supervised inference process."""
        ...

    async def health(self) -> BackendHealth:
        """Return process and endpoint health status."""
        ...

    def active_profile(self) -> ModelProfile | None:
        """Return the currently loaded model profile, if any."""
        ...
