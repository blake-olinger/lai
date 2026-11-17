"""Health, runtime status, and operational alert routes."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.common import (
    AlertsListResponse,
    GpuStatusDTO,
    HealthResponse,
    SystemStatusResponse,
)
from lai.domain.auth import ApiScope, AuthPrincipal

router = APIRouter(tags=["system"])

_RECENT_ALERT_COUNT_WINDOW = 200


@router.get("/v1/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Unauthenticated liveness probe."""
    return HealthResponse()


@router.get("/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.CHAT_READ))],
    services: Annotated[AppServices, Depends(get_services)],
) -> SystemStatusResponse:
    """Return GPU telemetry, backend health, spooler snapshot, and alert count."""
    gpu_raw, inference_health = await asyncio.gather(
        asyncio.to_thread(services.gpu_telemetry_fn),
        services.inference_provider.health(),
    )
    guardrail_health = (
        await services.guardrail_health_probe()
        if services.guardrail_health_probe is not None
        else None
    )
    gpu_dto = (
        GpuStatusDTO(
            gpu_name=gpu_raw.gpu_name,
            memory_used_mb=gpu_raw.memory_used_mb,
            memory_total_mb=gpu_raw.memory_total_mb,
            utilization_percent=gpu_raw.utilization_percent,
            temperature_c=gpu_raw.temperature_c,
        )
        if gpu_raw is not None
        else None
    )
    spooler_snapshot = services.spooler.snapshot()
    recent_alerts = services.alert_manager.recent_alerts(limit=_RECENT_ALERT_COUNT_WINDOW)
    backends_ok = inference_health.healthy and (
        guardrail_health is None or guardrail_health.healthy
    )
    return SystemStatusResponse(
        status="ok" if backends_ok else "degraded",
        active_model_profile=services.settings.default_model_profile,
        gpu=gpu_dto,
        inference_health=inference_health,
        guardrail_health=guardrail_health,
        spooler=spooler_snapshot,
        recent_alert_count=len(recent_alerts),
    )


@router.get("/v1/system/alerts", response_model=AlertsListResponse)
async def get_system_alerts(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> AlertsListResponse:
    """Return recent fatal, timeout, crash, storage, and security alerts."""
    alerts = services.alert_manager.recent_alerts(limit=limit)
    return AlertsListResponse(items=alerts, count=len(alerts))
