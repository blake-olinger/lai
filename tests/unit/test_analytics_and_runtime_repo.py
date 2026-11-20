"""Unit tests for SQLite usage analytics rollups and runtime config persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from lai.db import SqliteDatabase, SqliteRuntimeConfigRepository, SqliteUsageAnalyticsRepository
from lai.db.schema import CURRENT_SCHEMA_VERSION
from lai.domain.analytics import GuardrailUsageEvent, TokenUsageRecord, UsageEvent
from lai.domain.errors import LaiError
from lai.domain.guardrails import GuardrailDirection, GuardrailVerdict

if TYPE_CHECKING:
    from pathlib import Path


async def test_sqlite_usage_analytics_rollups_tokens_and_pruning(tmp_path: Path) -> None:
    """SqliteUsageAnalyticsRepository aggregates endpoints, principals, tokens, and pruning."""
    async with SqliteDatabase(tmp_path / "analytics.db") as db:
        repo = SqliteUsageAnalyticsRepository(db, retention_days=7)
        now = datetime(2026, 11, 19, 15, 30, tzinfo=UTC)
        old = now - timedelta(days=10)
        prev_hour = now - timedelta(hours=1)

        await repo.record_request(
            UsageEvent(
                timestamp=old,
                method="GET",
                route="/v1/projects",
                status_code=200,
                duration_ms=10.0,
                principal_id="usr_old",
            )
        )
        await repo.record_request(
            UsageEvent(
                timestamp=prev_hour,
                method="POST",
                route="/v1/sessions/{session_id}/messages",
                status_code=200,
                duration_ms=100.0,
                principal_id="oidc:u1",
                oauth_client_id="web-client",
            )
        )
        await repo.record_request(
            UsageEvent(
                timestamp=now,
                method="POST",
                route="/v1/sessions/{session_id}/messages",
                status_code=502,
                duration_ms=250.0,
                principal_id="oidc:u1",
                oauth_client_id="web-client",
            )
        )
        await repo.record_request(
            UsageEvent(
                timestamp=now,
                method="GET",
                route="/v1/projects/{project_id}",
                status_code=404,
                duration_ms=5.0,
                principal_id="key:k1",
            )
        )
        await repo.record_tokens(
            TokenUsageRecord(
                timestamp=now,
                principal_id="oidc:u1",
                oauth_client_id="web-client",
                prompt_tokens=120,
                completion_tokens=80,
                total_tokens=200,
            )
        )
        await repo.record_tokens(
            TokenUsageRecord(timestamp=now, principal_id="oidc:u1", total_tokens=0)
        )

        report = await repo.get_report(window_hours=24, now=now)
        assert report.total_requests == 3
        assert report.error_count == 2
        assert report.unique_principals == 2
        assert report.unique_client_apps == 1
        assert report.total_prompt_tokens == 120
        assert report.total_completion_tokens == 80
        assert report.total_tokens == 200
        assert len(report.hourly) == 2

        by_route = {(e.method, e.route): e for e in report.endpoints}
        msg_ep = by_route[("POST", "/v1/sessions/{session_id}/messages")]
        assert (msg_ep.request_count, msg_ep.status_2xx, msg_ep.status_5xx) == (2, 1, 1)
        assert msg_ep.avg_duration_ms == 175.0
        assert msg_ep.max_duration_ms == 250.0

        short = await repo.get_report(window_hours=1, now=now)
        assert short.total_requests == 2

        deleted = await repo.prune_before(now + timedelta(hours=2))
        assert deleted >= 3
        empty = await repo.get_report(window_hours=24, now=now)
        assert empty.total_requests == 0
        assert empty.error_rate == 0.0


async def test_sqlite_runtime_config_repository_crud(tmp_path: Path) -> None:
    """SqliteRuntimeConfigRepository persists, updates, lists, and deletes sections."""
    async with SqliteDatabase(tmp_path / "cfg.db") as db:
        repo = SqliteRuntimeConfigRepository(db)
        assert await repo.get_section("cors") is None
        assert await repo.get_all_sections() == {}

        await repo.upsert_section("cors", {"cors_allowed_origins": ["https://a.test"]})
        await repo.upsert_section("analytics", {"enabled": False, "retention_days": 30})
        assert await repo.get_section("cors") == {"cors_allowed_origins": ["https://a.test"]}
        await repo.upsert_section("cors", {"cors_allowed_origins": ["https://b.test"]})
        assert await repo.get_all_sections() == {
            "analytics": {"enabled": False, "retention_days": 30},
            "cors": {"cors_allowed_origins": ["https://b.test"]},
        }
        assert await repo.delete_section("cors") is True
        assert await repo.delete_section("cors") is False
        assert await repo.get_section("cors") is None


async def test_guardrail_efficacy_analytics_rollups_and_pruning(tmp_path: Path) -> None:
    """SqliteUsageAnalyticsRepository aggregates guardrail efficacy by direction and policy."""
    async with SqliteDatabase(tmp_path / "guardrails_analytics.db") as db:
        repo = SqliteUsageAnalyticsRepository(db, retention_days=7)
        now = datetime(2026, 11, 19, 15, 30, tzinfo=UTC)
        old = now - timedelta(days=10)
        prev_hour = now - timedelta(hours=1)

        # Older than retention: pruned when current-hour event lands.
        await repo.record_guardrail(
            GuardrailUsageEvent(
                timestamp=old,
                direction=GuardrailDirection.INGRESS,
                verdict=GuardrailVerdict.BLOCK,
                policy="Old Policy",
                latency_ms=50.0,
            )
        )
        # Window events across ingress, egress, and tool_output.
        events = (
            GuardrailUsageEvent(
                timestamp=prev_hour,
                direction=GuardrailDirection.INGRESS,
                verdict=GuardrailVerdict.ALLOW,
                latency_ms=20.0,
            ),
            GuardrailUsageEvent(
                timestamp=now,
                direction=GuardrailDirection.INGRESS,
                verdict=GuardrailVerdict.BLOCK,
                policy="No Prompt Injection",
                latency_ms=40.0,
            ),
            GuardrailUsageEvent(
                timestamp=now,
                direction=GuardrailDirection.INGRESS,
                verdict=GuardrailVerdict.BLOCK,
                policy="No Prompt Injection",
                latency_ms=60.0,
            ),
            GuardrailUsageEvent(
                timestamp=now,
                direction=GuardrailDirection.EGRESS,
                verdict=GuardrailVerdict.BLOCK,
                policy="No Dangerous Content",
                fail_closed=True,
                latency_ms=100.0,
            ),
            GuardrailUsageEvent(
                timestamp=now,
                direction=GuardrailDirection.TOOL_OUTPUT,
                verdict=GuardrailVerdict.ALLOW,
                latency_ms=30.0,
            ),
        )
        for ev in events:
            await repo.record_guardrail(ev)

        report = await repo.get_report(window_hours=24, now=now)
        gr = report.guardrails
        assert gr.total_evaluations == 5
        assert gr.total_allowed == 2
        assert gr.total_blocked == 3
        assert gr.block_rate == pytest.approx(0.6)
        assert gr.fail_closed_count == 1
        assert gr.avg_latency_ms == pytest.approx(50.0)
        assert gr.max_latency_ms == pytest.approx(100.0)

        by_dir = {item.direction: item for item in gr.by_direction}
        assert set(by_dir) == {
            GuardrailDirection.INGRESS,
            GuardrailDirection.EGRESS,
            GuardrailDirection.TOOL_OUTPUT,
        }
        ing = by_dir[GuardrailDirection.INGRESS]
        assert (ing.evaluations, ing.allowed, ing.blocked, ing.fail_closed) == (3, 1, 2, 0)
        assert ing.avg_latency_ms == pytest.approx(40.0)
        assert ing.max_latency_ms == pytest.approx(60.0)

        egr = by_dir[GuardrailDirection.EGRESS]
        assert (egr.evaluations, egr.blocked, egr.fail_closed) == (1, 1, 1)

        assert [(p.direction, p.policy, p.block_count) for p in gr.top_policies] == [
            (GuardrailDirection.INGRESS, "No Prompt Injection", 2),
            (GuardrailDirection.EGRESS, "No Dangerous Content", 1),
        ]

        short = await repo.get_report(window_hours=1, now=now)
        assert short.guardrails.total_evaluations == 4

        deleted = await repo.prune_before(now + timedelta(hours=2))
        assert deleted >= 4
        after_prune = await repo.get_report(window_hours=24, now=now)
        assert after_prune.guardrails.total_evaluations == 0
        assert after_prune.guardrails.by_direction == []
        assert after_prune.guardrails.top_policies == []


async def test_sqlite_schema_user_version_stamps_and_rejects_future_version(tmp_path: Path) -> None:
    """apply_schema stamps PRAGMA user_version and rejects unsupported future DB versions."""
    db_path = tmp_path / "versioned.db"
    async with SqliteDatabase(db_path) as db:
        row = await db.fetch_one("PRAGMA user_version;")
        assert row is not None
        assert int(row[0]) == CURRENT_SCHEMA_VERSION
        # Re-running initialize on current schema version is idempotent.
        await db.initialize()
        row_again = await db.fetch_one("PRAGMA user_version;")
        assert row_again is not None
        assert int(row_again[0]) == CURRENT_SCHEMA_VERSION

        # Simulate a database created by a newer release.
        future_version = CURRENT_SCHEMA_VERSION + 1
        await db.execute_write(f"PRAGMA user_version = {future_version};")

    future_db = SqliteDatabase(db_path)
    try:
        with pytest.raises(LaiError) as exc_info:
            await future_db.initialize()
        assert exc_info.value.code == "schema_version_unsupported"
        assert exc_info.value.status_code == 500
    finally:
        await future_db.close()
