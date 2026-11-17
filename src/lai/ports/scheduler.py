"""Session serialization and priority inference spooler protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from lai.domain.scheduler import LeaseRequest, SpoolerSnapshot


@runtime_checkable
class SessionGatePort(Protocol):
    """Per-session mutual exclusion gate preserving message ordering."""

    def acquire(self, session_id: str) -> AbstractAsyncContextManager[None]:
        """Acquire exclusive access for turns targeting session_id."""
        ...


@runtime_checkable
class InferenceSpoolerPort(Protocol):
    """Priority-ordered single-slot admission controller for inference calls."""

    def lease(self, request: LeaseRequest) -> AbstractAsyncContextManager[None]:
        """Acquire the active inference slot according to job priority."""
        ...

    def snapshot(self) -> SpoolerSnapshot:
        """Return a point-in-time snapshot of active and queued spooler jobs."""
        ...
