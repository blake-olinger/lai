"""Base Pydantic models and UTC timestamp helpers for the domain layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Immutable domain model forbidding unknown fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=False,
    )


class MutableStateModel(BaseModel):
    """Mutable domain state model forbidding unknown fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_default=True,
    )
