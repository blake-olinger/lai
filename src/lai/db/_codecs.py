"""Internal serialization and pagination helpers shared across SQLite repositories."""

from __future__ import annotations

from datetime import UTC, datetime


def format_dt(value: datetime) -> str:
    """Serialize a datetime as a normalized UTC ISO-8601 string."""
    aware = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.isoformat()


def format_opt_dt(value: datetime | None) -> str | None:
    """Serialize an optional datetime as UTC ISO-8601 or None."""
    return format_dt(value) if value is not None else None


def parse_dt(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_opt_dt(raw: str | None) -> datetime | None:
    """Parse an optional ISO-8601 timestamp or return None."""
    return parse_dt(raw) if raw is not None else None


def clamp_limit(limit: int) -> int:
    """Clamp pagination limit to a non-negative integer."""
    return max(limit, 0)


def clamp_offset(offset: int) -> int:
    """Clamp pagination offset to a non-negative integer."""
    return max(offset, 0)
