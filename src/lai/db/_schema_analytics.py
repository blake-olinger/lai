"""SQLite DDL statements for runtime config overrides and hourly usage analytics."""

from __future__ import annotations

ANALYTICS_AND_RUNTIME_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runtime_settings (
        section TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS usage_endpoint_hourly (
        bucket_hour TEXT NOT NULL,
        method TEXT NOT NULL,
        route TEXT NOT NULL,
        request_count INTEGER NOT NULL DEFAULT 0,
        status_2xx INTEGER NOT NULL DEFAULT 0,
        status_4xx INTEGER NOT NULL DEFAULT 0,
        status_5xx INTEGER NOT NULL DEFAULT 0,
        total_duration_ms REAL NOT NULL DEFAULT 0.0,
        max_duration_ms REAL NOT NULL DEFAULT 0.0,
        PRIMARY KEY (bucket_hour, method, route)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_usage_endpoint_bucket
        ON usage_endpoint_hourly(bucket_hour DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS usage_principal_hourly (
        bucket_hour TEXT NOT NULL,
        principal_id TEXT NOT NULL,
        oauth_client_id TEXT NOT NULL DEFAULT '',
        request_count INTEGER NOT NULL DEFAULT 0,
        error_count INTEGER NOT NULL DEFAULT 0,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (bucket_hour, principal_id, oauth_client_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_usage_principal_bucket
        ON usage_principal_hourly(bucket_hour DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS usage_guardrail_hourly (
        bucket_hour TEXT NOT NULL,
        direction TEXT NOT NULL,
        policy TEXT NOT NULL DEFAULT '',
        verdict TEXT NOT NULL,
        evaluation_count INTEGER NOT NULL DEFAULT 0,
        fail_closed_count INTEGER NOT NULL DEFAULT 0,
        total_latency_ms REAL NOT NULL DEFAULT 0.0,
        max_latency_ms REAL NOT NULL DEFAULT 0.0,
        PRIMARY KEY (bucket_hour, direction, policy, verdict)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_usage_guardrail_bucket
        ON usage_guardrail_hourly(bucket_hour DESC);
    """,
)
