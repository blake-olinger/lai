"""Framework-free domain models for endpoint usage analytics and retention."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from lai.domain.base import StrictModel, utc_now


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
