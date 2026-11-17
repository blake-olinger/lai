"""Observability alert sink and alert manager protocols (PEP 544)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lai.domain.observability import AlertEvent


@runtime_checkable
class AlertSinkPort(Protocol):
    """Asynchronous delivery target for emitted operational alerts."""

    async def send_alert(self, event: AlertEvent) -> None:
        """Deliver an alert event to an external or local sink."""
        ...


@runtime_checkable
class AlertManagerPort(Protocol):
    """Alert deduplication, ring buffering, persistence, and sink fan-out."""

    def emit(self, event: AlertEvent) -> None:
        """Accept an alert event, apply cooldown deduplication, and dispatch."""
        ...

    async def flush(self) -> None:
        """Await completion of all pending asynchronous sink deliveries."""
        ...

    def recent_alerts(self, limit: int = 50) -> list[AlertEvent]:
        """Return up to limit most recently emitted alert events."""
        ...
