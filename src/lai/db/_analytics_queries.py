"""Read-aggregation SQL queries for endpoint, principal, hourly, and guardrail analytics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from lai.domain.analytics import (
    EndpointUsageSummary,
    GuardrailAnalyticsSummary,
    GuardrailDirectionSummary,
    GuardrailPolicyCount,
    HourlyUsageBucket,
    PrincipalUsageSummary,
)
from lai.domain.guardrails import GuardrailDirection

if TYPE_CHECKING:
    from lai.db.connection import SqliteDatabase

_RATE_DIGITS = 4
_DURATION_DIGITS = 2


def parse_bucket_hour(raw: str) -> datetime:
    """Parse an ISO-8601 UTC hourly bucket string back into a timezone-aware datetime."""
    parsed = datetime.fromisoformat(raw)
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


async def query_endpoints(
    db: SqliteDatabase,
    start_bucket: str,
) -> list[EndpointUsageSummary]:
    """Return aggregated request volume, status classes, and latency per route template."""
    rows = await db.fetch_all(
        """
        SELECT
            method,
            route,
            SUM(request_count) AS request_count,
            SUM(status_2xx) AS status_2xx,
            SUM(status_4xx) AS status_4xx,
            SUM(status_5xx) AS status_5xx,
            SUM(total_duration_ms) AS total_duration_ms,
            MAX(max_duration_ms) AS max_duration_ms
        FROM usage_endpoint_hourly
        WHERE bucket_hour >= ?
        GROUP BY method, route
        ORDER BY request_count DESC, method ASC, route ASC;
        """,
        (start_bucket,),
    )
    results: list[EndpointUsageSummary] = []
    for row in rows:
        req_count = int(row["request_count"])
        avg_ms = (
            round(float(row["total_duration_ms"]) / req_count, _DURATION_DIGITS)
            if req_count > 0
            else 0.0
        )
        results.append(
            EndpointUsageSummary(
                method=str(row["method"]),
                route=str(row["route"]),
                request_count=req_count,
                status_2xx=int(row["status_2xx"]),
                status_4xx=int(row["status_4xx"]),
                status_5xx=int(row["status_5xx"]),
                avg_duration_ms=avg_ms,
                max_duration_ms=round(float(row["max_duration_ms"]), _DURATION_DIGITS),
            )
        )
    return results


async def query_principals(
    db: SqliteDatabase,
    start_bucket: str,
) -> list[PrincipalUsageSummary]:
    """Return aggregated requests, errors, and tokens per opaque principal and client app."""
    rows = await db.fetch_all(
        """
        SELECT
            principal_id,
            oauth_client_id,
            SUM(request_count) AS request_count,
            SUM(error_count) AS error_count,
            SUM(prompt_tokens) AS prompt_tokens,
            SUM(completion_tokens) AS completion_tokens,
            SUM(total_tokens) AS total_tokens
        FROM usage_principal_hourly
        WHERE bucket_hour >= ?
        GROUP BY principal_id, oauth_client_id
        ORDER BY request_count DESC, total_tokens DESC, principal_id ASC;
        """,
        (start_bucket,),
    )
    return [
        PrincipalUsageSummary(
            principal_id=str(row["principal_id"]),
            oauth_client_id=str(row["oauth_client_id"]) or None,
            request_count=int(row["request_count"]),
            error_count=int(row["error_count"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            total_tokens=int(row["total_tokens"]),
        )
        for row in rows
    ]


async def query_hourly(
    db: SqliteDatabase,
    start_bucket: str,
) -> list[HourlyUsageBucket]:
    """Return merged hourly request, error, and token trend points."""
    ep_rows = await db.fetch_all(
        """
        SELECT
            bucket_hour,
            SUM(request_count) AS request_count,
            SUM(status_4xx + status_5xx) AS error_count
        FROM usage_endpoint_hourly
        WHERE bucket_hour >= ?
        GROUP BY bucket_hour
        ORDER BY bucket_hour ASC;
        """,
        (start_bucket,),
    )
    tok_rows = await db.fetch_all(
        """
        SELECT
            bucket_hour,
            SUM(total_tokens) AS total_tokens
        FROM usage_principal_hourly
        WHERE bucket_hour >= ?
        GROUP BY bucket_hour
        ORDER BY bucket_hour ASC;
        """,
        (start_bucket,),
    )
    merged: dict[str, tuple[int, int, int]] = {}
    for row in ep_rows:
        bucket = str(row["bucket_hour"])
        merged[bucket] = (int(row["request_count"]), int(row["error_count"]), 0)
    for row in tok_rows:
        bucket = str(row["bucket_hour"])
        req_count, err_count, _ = merged.get(bucket, (0, 0, 0))
        merged[bucket] = (req_count, err_count, int(row["total_tokens"]))

    return [
        HourlyUsageBucket(
            bucket_hour=parse_bucket_hour(bucket),
            request_count=counts[0],
            error_count=counts[1],
            total_tokens=counts[2],
        )
        for bucket, counts in sorted(merged.items())
    ]


async def query_guardrails(
    db: SqliteDatabase,
    start_bucket: str,
) -> GuardrailAnalyticsSummary:
    """Return windowed guardrail efficacy metrics across ingress, egress, and tool outputs."""
    dir_rows = await db.fetch_all(
        """
        SELECT
            direction,
            SUM(evaluation_count) AS evaluations,
            SUM(CASE WHEN verdict = 'allow' THEN evaluation_count ELSE 0 END) AS allowed,
            SUM(CASE WHEN verdict = 'block' THEN evaluation_count ELSE 0 END) AS blocked,
            SUM(fail_closed_count) AS fail_closed,
            SUM(total_latency_ms) AS total_latency_ms,
            MAX(max_latency_ms) AS max_latency_ms
        FROM usage_guardrail_hourly
        WHERE bucket_hour >= ?
        GROUP BY direction
        ORDER BY direction ASC;
        """,
        (start_bucket,),
    )
    policy_rows = await db.fetch_all(
        """
        SELECT
            direction,
            policy,
            SUM(evaluation_count) AS block_count
        FROM usage_guardrail_hourly
        WHERE bucket_hour >= ? AND verdict = 'block' AND policy != ''
        GROUP BY direction, policy
        ORDER BY block_count DESC, direction ASC, policy ASC;
        """,
        (start_bucket,),
    )

    by_direction: list[GuardrailDirectionSummary] = []
    total_evals = 0
    total_allowed = 0
    total_blocked = 0
    total_fail_closed = 0
    total_latency_sum = 0.0
    max_latency = 0.0

    for row in dir_rows:
        evals = int(row["evaluations"])
        allowed = int(row["allowed"])
        blocked = int(row["blocked"])
        fail_closed = int(row["fail_closed"])
        latency_sum = float(row["total_latency_ms"])
        row_max_latency = float(row["max_latency_ms"])

        total_evals += evals
        total_allowed += allowed
        total_blocked += blocked
        total_fail_closed += fail_closed
        total_latency_sum += latency_sum
        max_latency = max(max_latency, row_max_latency)

        by_direction.append(
            GuardrailDirectionSummary(
                direction=GuardrailDirection(str(row["direction"])),
                evaluations=evals,
                allowed=allowed,
                blocked=blocked,
                block_rate=round(blocked / evals, _RATE_DIGITS) if evals > 0 else 0.0,
                fail_closed=fail_closed,
                avg_latency_ms=round(latency_sum / evals, _DURATION_DIGITS) if evals > 0 else 0.0,
                max_latency_ms=round(row_max_latency, _DURATION_DIGITS),
            )
        )

    top_policies = [
        GuardrailPolicyCount(
            direction=GuardrailDirection(str(row["direction"])),
            policy=str(row["policy"]),
            block_count=int(row["block_count"]),
        )
        for row in policy_rows
    ]
    return GuardrailAnalyticsSummary(
        total_evaluations=total_evals,
        total_allowed=total_allowed,
        total_blocked=total_blocked,
        block_rate=round(total_blocked / total_evals, _RATE_DIGITS) if total_evals > 0 else 0.0,
        fail_closed_count=total_fail_closed,
        avg_latency_ms=(
            round(total_latency_sum / total_evals, _DURATION_DIGITS) if total_evals > 0 else 0.0
        ),
        max_latency_ms=round(max_latency, _DURATION_DIGITS),
        by_direction=by_direction,
        top_policies=top_policies,
    )
