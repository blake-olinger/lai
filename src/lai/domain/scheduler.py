"""Scheduler priority and inference spooler telemetry domain models."""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_job_id


class JobPriority(IntEnum):
    """Priority order for inference spooler scheduling (lower runs first)."""

    INTERACTIVE = 10
    APPROVAL_RESUME = 20
    MAP_REDUCE = 30
    BACKGROUND_INDEX = 40


class SpoolerSnapshot(StrictModel):
    """Point-in-time status of the single-slot inference spooler."""

    active_job_id: str | None = None
    active_session_id: str | None = None
    active_priority: JobPriority | None = None
    queued_count: int = Field(default=0, ge=0)
    max_queued_jobs: int = Field(default=64, ge=1)
    queued_by_priority: dict[str, int] = Field(default_factory=dict)
    total_completed: int = Field(default=0, ge=0)
    total_rejected: int = Field(default=0, ge=0)


class LeaseRequest(StrictModel):
    """Work item requesting an exclusive lease on the inference backend."""

    job_id: str = Field(default_factory=new_job_id)
    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    priority: JobPriority = JobPriority.INTERACTIVE
    enqueued_at: datetime = Field(default_factory=utc_now)
