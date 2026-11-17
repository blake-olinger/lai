"""Shared API schema DTOs for error envelopes, health, status, and deletions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.inference import BackendHealth
from lai.domain.observability import AlertEvent
from lai.domain.scheduler import SpoolerSnapshot


class ErrorDetail(StrictModel):
    """Machine-readable error detail inside an API error envelope."""

    code: str = Field(min_length=1)
    message: str
    request_id: str | None = None
    details: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(StrictModel):
    """Standard JSON error envelope returned for all handled and fatal failures."""

    error: ErrorDetail


class DeleteResponse(StrictModel):
    """Confirmation envelope returned by DELETE endpoints."""

    id: str = Field(min_length=1)
    deleted: bool


class HealthResponse(StrictModel):
    """Unauthenticated liveness probe payload."""

    status: Literal["ok"] = "ok"
    version: str = "0.1.0"


class GpuStatusDTO(StrictModel):
    """GPU utilization and VRAM snapshot reported by nvidia-smi."""

    gpu_name: str = Field(min_length=1)
    memory_used_mb: int = Field(ge=0)
    memory_total_mb: int = Field(ge=0)
    utilization_percent: int = Field(ge=0, le=100)
    temperature_c: float | None = None


class SystemStatusResponse(StrictModel):
    """Comprehensive runtime health, GPU, guardrail, and spooler telemetry."""

    status: Literal["ok", "degraded"]
    active_model_profile: str = Field(min_length=1)
    gpu: GpuStatusDTO | None = None
    inference_health: BackendHealth
    guardrail_health: BackendHealth | None = None
    spooler: SpoolerSnapshot
    recent_alert_count: int = Field(ge=0)


class AlertsListResponse(StrictModel):
    """Recent operational alerts captured by the AlertManager."""

    items: list[AlertEvent]
    count: int = Field(ge=0)
