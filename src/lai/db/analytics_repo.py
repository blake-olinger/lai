"""SQLite hourly rollup repository implementing UsageAnalyticsPort."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from lai.db._analytics_queries import (
    query_endpoints,
    query_guardrails,
    query_hourly,
    query_principals,
)
from lai.domain.analytics import (
    AnalyticsReport,
    GuardrailUsageEvent,
    TokenUsageRecord,
    UsageEvent,
)
from lai.domain.base import utc_now
from lai.ports.analytics import UsageAnalyticsPort

if TYPE_CHECKING:
    from lai.db.connection import SqliteDatabase

_STATUS_2XX_MIN = 200
_STATUS_3XX_MIN = 300
_STATUS_4XX_MIN = 400
_STATUS_5XX_MIN = 500
_ERROR_RATE_DIGITS = 4

_UPSERT_ENDPOINT_SQL = """
    INSERT INTO usage_endpoint_hourly (
        bucket_hour, method, route, request_count,
        status_2xx, status_4xx, status_5xx, total_duration_ms, max_duration_ms
    )
    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
    ON CONFLICT(bucket_hour, method, route) DO UPDATE SET
        request_count = usage_endpoint_hourly.request_count + 1,
        status_2xx = usage_endpoint_hourly.status_2xx + excluded.status_2xx,
        status_4xx = usage_endpoint_hourly.status_4xx + excluded.status_4xx,
        status_5xx = usage_endpoint_hourly.status_5xx + excluded.status_5xx,
        total_duration_ms = usage_endpoint_hourly.total_duration_ms + excluded.total_duration_ms,
        max_duration_ms = MAX(usage_endpoint_hourly.max_duration_ms, excluded.max_duration_ms);
"""

_UPSERT_PRINCIPAL_REQ_SQL = """
    INSERT INTO usage_principal_hourly (
        bucket_hour, principal_id, oauth_client_id,
        request_count, error_count, prompt_tokens, completion_tokens, total_tokens
    )
    VALUES (?, ?, ?, 1, ?, 0, 0, 0)
    ON CONFLICT(bucket_hour, principal_id, oauth_client_id) DO UPDATE SET
        request_count = usage_principal_hourly.request_count + 1,
        error_count = usage_principal_hourly.error_count + excluded.error_count;
"""

_UPSERT_PRINCIPAL_TOKENS_SQL = """
    INSERT INTO usage_principal_hourly (
        bucket_hour, principal_id, oauth_client_id,
        request_count, error_count, prompt_tokens, completion_tokens, total_tokens
    )
    VALUES (?, ?, ?, 0, 0, ?, ?, ?)
    ON CONFLICT(bucket_hour, principal_id, oauth_client_id) DO UPDATE SET
        prompt_tokens = usage_principal_hourly.prompt_tokens + excluded.prompt_tokens,
        completion_tokens = usage_principal_hourly.completion_tokens + excluded.completion_tokens,
        total_tokens = usage_principal_hourly.total_tokens + excluded.total_tokens;
"""

_UPSERT_GUARDRAIL_SQL = """
    INSERT INTO usage_guardrail_hourly (
        bucket_hour, direction, policy, verdict,
        evaluation_count, fail_closed_count, total_latency_ms, max_latency_ms
    )
    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
    ON CONFLICT(bucket_hour, direction, policy, verdict) DO UPDATE SET
        evaluation_count = usage_guardrail_hourly.evaluation_count + 1,
        fail_closed_count = usage_guardrail_hourly.fail_closed_count + excluded.fail_closed_count,
        total_latency_ms = usage_guardrail_hourly.total_latency_ms + excluded.total_latency_ms,
        max_latency_ms = MAX(usage_guardrail_hourly.max_latency_ms, excluded.max_latency_ms);
"""


def _to_bucket_hour(ts: datetime) -> str:
    """Truncate a datetime to an ISO-8601 UTC hourly bucket string."""
    utc_dt = ts.astimezone(UTC) if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    floored = utc_dt.replace(minute=0, second=0, microsecond=0)
    return floored.isoformat()


class SqliteUsageAnalyticsRepository(UsageAnalyticsPort):
    """Privacy-preserving UTC hourly rollup store for endpoint, token, and guardrail analytics."""

    def __init__(self, db: SqliteDatabase, *, retention_days: int = 90) -> None:
        self._db = db
        self._retention_days = max(1, retention_days)
        self._last_prune_bucket: str | None = None

    def set_retention_days(self, retention_days: int) -> None:
        """Update the retention window used for automatic hourly bucket pruning."""
        self._retention_days = max(1, retention_days)

    async def _maybe_prune(self, bucket_hour: str, timestamp: datetime) -> None:
        if bucket_hour != self._last_prune_bucket:
            self._last_prune_bucket = bucket_hour
            cutoff = timestamp - timedelta(days=self._retention_days)
            await self.prune_before(cutoff)

    async def record_request(self, event: UsageEvent) -> None:
        """Upsert one completed HTTP request into endpoint and optional principal rollups."""
        bucket_hour = _to_bucket_hour(event.timestamp)
        is_2xx = 1 if _STATUS_2XX_MIN <= event.status_code < _STATUS_3XX_MIN else 0
        is_4xx = 1 if _STATUS_4XX_MIN <= event.status_code < _STATUS_5XX_MIN else 0
        is_5xx = 1 if event.status_code >= _STATUS_5XX_MIN else 0
        is_error = 1 if event.status_code >= _STATUS_4XX_MIN else 0
        duration = max(0.0, float(event.duration_ms))

        async with self._db.write_transaction() as conn:
            await conn.execute(
                _UPSERT_ENDPOINT_SQL,
                (
                    bucket_hour,
                    event.method.upper(),
                    event.route,
                    is_2xx,
                    is_4xx,
                    is_5xx,
                    duration,
                    duration,
                ),
            )
            if event.principal_id:
                await conn.execute(
                    _UPSERT_PRINCIPAL_REQ_SQL,
                    (
                        bucket_hour,
                        event.principal_id,
                        event.oauth_client_id or "",
                        is_error,
                    ),
                )

        await self._maybe_prune(bucket_hour, event.timestamp)

    async def record_tokens(self, record: TokenUsageRecord) -> None:
        """Upsert inference token counts into the UTC hourly principal rollup table."""
        total = record.total_tokens or (record.prompt_tokens + record.completion_tokens)
        if record.prompt_tokens == 0 and record.completion_tokens == 0 and total == 0:
            return
        bucket_hour = _to_bucket_hour(record.timestamp)
        await self._db.execute_write(
            _UPSERT_PRINCIPAL_TOKENS_SQL,
            (
                bucket_hour,
                record.principal_id,
                record.oauth_client_id or "",
                record.prompt_tokens,
                record.completion_tokens,
                total,
            ),
        )

    async def record_guardrail(self, event: GuardrailUsageEvent) -> None:
        """Upsert one guardrail evaluation into the UTC hourly guardrail efficacy rollup."""
        bucket_hour = _to_bucket_hour(event.timestamp)
        latency = max(0.0, float(event.latency_ms))
        await self._db.execute_write(
            _UPSERT_GUARDRAIL_SQL,
            (
                bucket_hour,
                event.direction.value,
                event.policy or "",
                event.verdict.value,
                1 if event.fail_closed else 0,
                latency,
                latency,
            ),
        )
        await self._maybe_prune(bucket_hour, event.timestamp)

    async def prune_before(self, cutoff: datetime) -> int:
        """Delete hourly rollup rows older than cutoff and return total rows removed."""
        cutoff_bucket = _to_bucket_hour(cutoff)
        async with self._db.write_transaction() as conn:
            c1 = await conn.execute(
                "DELETE FROM usage_endpoint_hourly WHERE bucket_hour < ?;",
                (cutoff_bucket,),
            )
            c2 = await conn.execute(
                "DELETE FROM usage_principal_hourly WHERE bucket_hour < ?;",
                (cutoff_bucket,),
            )
            c3 = await conn.execute(
                "DELETE FROM usage_guardrail_hourly WHERE bucket_hour < ?;",
                (cutoff_bucket,),
            )
            return max(0, c1.rowcount) + max(0, c2.rowcount) + max(0, c3.rowcount)

    async def get_report(
        self,
        window_hours: int = 24,
        *,
        now: datetime | None = None,
    ) -> AnalyticsReport:
        """Aggregate endpoints, principals/client apps, hourly trends, and guardrails."""
        resolved_window = max(1, window_hours)
        resolved_now = now or utc_now()
        start_bucket = _to_bucket_hour(resolved_now - timedelta(hours=resolved_window - 1))

        endpoints = await query_endpoints(self._db, start_bucket)
        principals = await query_principals(self._db, start_bucket)
        hourly = await query_hourly(self._db, start_bucket)
        guardrails = await query_guardrails(self._db, start_bucket)

        total_requests = sum(item.request_count for item in endpoints)
        error_count = sum(item.status_4xx + item.status_5xx for item in endpoints)
        error_rate = (
            round(error_count / total_requests, _ERROR_RATE_DIGITS) if total_requests > 0 else 0.0
        )
        unique_principals = len({item.principal_id for item in principals})
        unique_client_apps = len(
            {item.oauth_client_id for item in principals if item.oauth_client_id}
        )

        return AnalyticsReport(
            window_hours=resolved_window,
            generated_at=resolved_now,
            total_requests=total_requests,
            error_count=error_count,
            error_rate=error_rate,
            unique_principals=unique_principals,
            unique_client_apps=unique_client_apps,
            total_prompt_tokens=sum(item.prompt_tokens for item in principals),
            total_completion_tokens=sum(item.completion_tokens for item in principals),
            total_tokens=sum(item.total_tokens for item in principals),
            endpoints=endpoints,
            principals=principals,
            hourly=hourly,
            guardrails=guardrails,
        )
