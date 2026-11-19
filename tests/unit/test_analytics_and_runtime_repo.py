"""Unit tests for SQLite usage analytics rollups and runtime config persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from lai.db import SqliteDatabase, SqliteRuntimeConfigRepository, SqliteUsageAnalyticsRepository
from lai.domain.analytics import TokenUsageRecord, UsageEvent

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
