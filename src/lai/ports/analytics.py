"""Hexagonal ports for endpoint usage analytics and persisted runtime config overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from lai.domain.analytics import AnalyticsReport, TokenUsageRecord, UsageEvent


@runtime_checkable
class UsageAnalyticsPort(Protocol):
    """Storage-agnostic contract for recording and querying endpoint usage analytics."""

    async def record_request(self, event: UsageEvent) -> None:
        """Record a completed HTTP request into the current time rollup bucket."""
        ...

    async def record_tokens(self, record: TokenUsageRecord) -> None:
        """Record inference token totals for a principal and optional OAuth client app."""
        ...

    async def get_report(
        self,
        window_hours: int = 24,
        *,
        now: datetime | None = None,
    ) -> AnalyticsReport:
        """Aggregate usage metrics across the trailing window_hours."""
        ...

    async def prune_before(self, cutoff: datetime) -> int:
        """Delete rollup buckets older than cutoff and return total rows deleted."""
        ...


@runtime_checkable
class RuntimeConfigRepositoryPort(Protocol):
    """Per-section persistence for non-secret runtime hot-swap overrides."""

    async def get_section(self, section: str) -> dict[str, object] | None:
        """Return persisted non-secret override mapping for section, or None."""
        ...

    async def get_all_sections(self) -> dict[str, dict[str, object]]:
        """Return all persisted non-secret runtime override mappings keyed by section."""
        ...

    async def upsert_section(self, section: str, payload: dict[str, object]) -> None:
        """Persist non-secret runtime override mapping for section."""
        ...

    async def delete_section(self, section: str) -> bool:
        """Delete any persisted override for section, returning True if it existed."""
        ...
