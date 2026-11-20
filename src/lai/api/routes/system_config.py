"""Admin routes for live alerting, usage analytics, and isolated runtime hot-swapping."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from lai.api._runtime_config import (
    build_alerting_response,
    build_cors_response,
    build_guardrails_response,
    build_limits_response,
    build_oidc_response,
    build_updates_response,
    reload_config_sections,
    update_alerting_config,
    update_analytics_config,
    update_cors_config,
    update_guardrails_config,
    update_limits_config,
    update_oidc_config,
    update_updates_config,
)
from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.system_config import (
    AlertingConfigResponse,
    AnalyticsConfigResponse,
    CorsRuntimeConfigResponse,
    GuardrailsPolicyConfigResponse,
    LimitsRuntimeConfigResponse,
    OidcRuntimeConfigResponse,
    ScopedConfigReloadRequest,
    ScopedConfigReloadResponse,
    TestAlertRequest,
    TestAlertResponse,
    UpdateAlertingConfigRequest,
    UpdateAnalyticsConfigRequest,
    UpdateCorsConfigRequest,
    UpdateGuardrailsPolicyRequest,
    UpdateLimitsConfigRequest,
    UpdateOidcConfigRequest,
    UpdatesRuntimeConfigResponse,
    UpdateUpdatesConfigRequest,
)
from lai.domain.analytics import AnalyticsReport
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.errors import LaiError
from lai.domain.observability import AlertEvent
from lai.observability import AlertManager, deliver_test_alert

router = APIRouter(prefix="/v1/system", tags=["system"])


@router.get("/alerting", response_model=AlertingConfigResponse)
async def get_alerting_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> AlertingConfigResponse:
    """Return active redacted SMTP and webhook alerting configuration."""
    return build_alerting_response(services.settings)


@router.patch("/alerting", response_model=AlertingConfigResponse)
async def patch_alerting_config(
    body: UpdateAlertingConfigRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> AlertingConfigResponse:
    """Hot-swap non-secret alerting settings and persist across restarts."""
    return await update_alerting_config(services, body)


@router.post("/alerting/test", response_model=TestAlertResponse)
async def send_test_alert(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
    body: TestAlertRequest | None = None,
) -> TestAlertResponse:
    """Send a synthetic test alert through active sinks and await delivery status."""
    payload = body or TestAlertRequest()
    event = AlertEvent(
        severity=payload.severity,
        category=payload.category,
        component="system.alerting.test",
        message=payload.message,
        details={"synthetic": "true"},
    )
    if not isinstance(services.alert_manager, AlertManager):
        return TestAlertResponse(
            alert_id=event.id,
            delivered_count=0,
            failed_count=0,
            skipped_count=0,
            results=[],
        )
    results = await deliver_test_alert(
        services.alert_manager,
        event,
        bypass_filters=payload.bypass_filters,
    )
    return TestAlertResponse(
        alert_id=event.id,
        delivered_count=sum(1 for item in results if item.delivered),
        failed_count=sum(1 for item in results if not item.delivered and not item.skipped),
        skipped_count=sum(1 for item in results if item.skipped),
        results=results,
    )


@router.get("/analytics", response_model=AnalyticsReport)
async def get_usage_analytics(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
    window_hours: Annotated[int, Query(ge=1, le=2160)] = 24,
) -> AnalyticsReport:
    """Return endpoint, principal/client-app, token, and hourly rollup usage metrics."""
    if services.analytics_repo is None:
        msg = "Usage analytics repository is not initialized."
        raise LaiError(msg, code="service_unavailable", status_code=503)
    return await services.analytics_repo.get_report(window_hours=window_hours)


@router.get("/analytics/config", response_model=AnalyticsConfigResponse)
async def get_analytics_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> AnalyticsConfigResponse:
    """Return live endpoint usage analytics toggle and retention settings."""
    return AnalyticsConfigResponse(
        enabled=services.settings.analytics.enabled,
        retention_days=services.settings.analytics.retention_days,
    )


@router.patch("/analytics/config", response_model=AnalyticsConfigResponse)
async def patch_analytics_config(
    body: UpdateAnalyticsConfigRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> AnalyticsConfigResponse:
    """Hot-swap endpoint usage analytics enablement and retention days."""
    return await update_analytics_config(services, body)


@router.get("/config/oidc", response_model=OidcRuntimeConfigResponse)
async def get_oidc_runtime_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> OidcRuntimeConfigResponse:
    """Return live OIDC audiences, allowlists, admin emails, scopes, and leeway."""
    return build_oidc_response(services.settings)


@router.patch("/config/oidc", response_model=OidcRuntimeConfigResponse)
async def patch_oidc_runtime_config(
    body: UpdateOidcConfigRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> OidcRuntimeConfigResponse:
    """Hot-swap OIDC authorization policy in place while preserving cached JWKS."""
    return await update_oidc_config(services, body)


@router.get("/config/cors", response_model=CorsRuntimeConfigResponse)
async def get_cors_runtime_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> CorsRuntimeConfigResponse:
    """Return live browser CORS origins and allowed headers."""
    return build_cors_response(services.settings)


@router.patch("/config/cors", response_model=CorsRuntimeConfigResponse)
async def patch_cors_runtime_config(
    body: UpdateCorsConfigRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> CorsRuntimeConfigResponse:
    """Hot-swap browser CORS origins and headers live without restarting."""
    return await update_cors_config(services, body)


@router.get("/config/limits", response_model=LimitsRuntimeConfigResponse)
async def get_limits_runtime_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> LimitsRuntimeConfigResponse:
    """Return live rate limits, principal quotas, queue limits, keepalives, and log level."""
    return build_limits_response(services.settings)


@router.patch("/config/limits", response_model=LimitsRuntimeConfigResponse)
async def patch_limits_runtime_config(
    body: UpdateLimitsConfigRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> LimitsRuntimeConfigResponse:
    """Hot-swap limits, quotas, and log level in place without dropping buckets or queues."""
    return await update_limits_config(services, body)


@router.get("/config/guardrails", response_model=GuardrailsPolicyConfigResponse)
async def get_guardrails_runtime_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> GuardrailsPolicyConfigResponse:
    """Return live pre-flight guardrail enforcement policy."""
    return build_guardrails_response(services.settings)


@router.patch("/config/guardrails", response_model=GuardrailsPolicyConfigResponse)
async def patch_guardrails_runtime_config(
    body: UpdateGuardrailsPolicyRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> GuardrailsPolicyConfigResponse:
    """Hot-swap guardrail enforcement policy without restarting inference subprocesses."""
    return await update_guardrails_config(services, body)


@router.get("/config/updates", response_model=UpdatesRuntimeConfigResponse)
async def get_updates_runtime_config(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> UpdatesRuntimeConfigResponse:
    """Return live self-update scheduling and rollback configuration."""
    return build_updates_response(services.settings)


@router.patch("/config/updates", response_model=UpdatesRuntimeConfigResponse)
async def patch_updates_runtime_config(
    body: UpdateUpdatesConfigRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> UpdatesRuntimeConfigResponse:
    """Hot-swap non-privileged self-update scheduling and rollback toggles."""
    return await update_updates_config(services, body)


@router.post("/config/reload", response_model=ScopedConfigReloadResponse)
async def reload_scoped_config(
    body: ScopedConfigReloadRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ScopedConfigReloadResponse:
    """Reload only the explicitly requested section(s) from YAML and environment variables."""
    return await reload_config_sections(services, body.sections)
