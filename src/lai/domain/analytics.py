"""Framework-free domain models for endpoint usage analytics and retention."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.guardrails import GuardrailDirection, GuardrailVerdict


class AnalyticsConfig(StrictModel):
    """Configuration controlling built-in SQLite endpoint usage analytics."""

    enabled: bool = True
    retention_days: int = Field(default=90, ge=1, le=3650)


class UsageEvent(StrictModel):
    """Single completed HTTP request measurement (privacy-preserving opaque IDs only)."""

    timestamp: datetime = Field(default_factory=utc_now)
    method: str = Field(min_length=1)
    route: str = Field(min_length=1)
    status_code: int = Field(ge=100, le=599)
    duration_ms: float = Field(ge=0.0)
    principal_id: str | None = None
    oauth_client_id: str | None = None


class TokenUsageRecord(StrictModel):
    """Completed inference turn token usage attributed to a principal and optional client app."""

    timestamp: datetime = Field(default_factory=utc_now)
    principal_id: str = Field(min_length=1)
    oauth_client_id: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class EndpointUsageSummary(StrictModel):
    """Aggregated request volume, status class breakdown, and latency per route template."""

    method: str = Field(min_length=1)
    route: str = Field(min_length=1)
    request_count: int = Field(ge=0)
    status_2xx: int = Field(ge=0)
    status_4xx: int = Field(ge=0)
    status_5xx: int = Field(ge=0)
    avg_duration_ms: float = Field(ge=0.0)
    max_duration_ms: float = Field(ge=0.0)


class PrincipalUsageSummary(StrictModel):
    """Aggregated request, error, and token totals per opaque principal and OAuth client app."""

    principal_id: str = Field(min_length=1)
    oauth_client_id: str | None = None
    request_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class HourlyUsageBucket(StrictModel):
    """UTC hourly rollup point for time-series request, error, and token trends."""

    bucket_hour: datetime
    request_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class GuardrailUsageEvent(StrictModel):
    """Single completed guardrail evaluation measurement (no prompt content)."""

    timestamp: datetime = Field(default_factory=utc_now)
    direction: GuardrailDirection = GuardrailDirection.INGRESS
    verdict: GuardrailVerdict
    policy: str | None = None
    fail_closed: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)


class GuardrailDirectionSummary(StrictModel):
    """Aggregated guardrail evaluation volume, block rate, and latency per direction."""

    direction: GuardrailDirection
    evaluations: int = Field(ge=0)
    allowed: int = Field(ge=0)
    blocked: int = Field(ge=0)
    block_rate: float = Field(ge=0.0, le=1.0)
    fail_closed: int = Field(ge=0)
    avg_latency_ms: float = Field(ge=0.0)
    max_latency_ms: float = Field(ge=0.0)


class GuardrailPolicyCount(StrictModel):
    """Aggregated block count per violated safety policy and direction."""

    direction: GuardrailDirection
    policy: str = Field(min_length=1)
    block_count: int = Field(ge=0)


class GuardrailAnalyticsSummary(StrictModel):
    """Windowed guardrail efficacy summary embedded in AnalyticsReport."""

    total_evaluations: int = Field(default=0, ge=0)
    total_allowed: int = Field(default=0, ge=0)
    total_blocked: int = Field(default=0, ge=0)
    block_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    fail_closed_count: int = Field(default=0, ge=0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    max_latency_ms: float = Field(default=0.0, ge=0.0)
    by_direction: list[GuardrailDirectionSummary] = Field(default_factory=list)
    top_policies: list[GuardrailPolicyCount] = Field(default_factory=list)


class AnalyticsReport(StrictModel):
    """Windowed usage analytics report returned by GET /v1/system/analytics."""

    window_hours: int = Field(ge=1)
    generated_at: datetime = Field(default_factory=utc_now)
    total_requests: int = Field(ge=0)
    error_count: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    unique_principals: int = Field(ge=0)
    unique_client_apps: int = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    endpoints: list[EndpointUsageSummary] = Field(default_factory=list)
    principals: list[PrincipalUsageSummary] = Field(default_factory=list)
    hourly: list[HourlyUsageBucket] = Field(default_factory=list)
    guardrails: GuardrailAnalyticsSummary = Field(default_factory=GuardrailAnalyticsSummary)
