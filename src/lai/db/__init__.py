"""SQLite metadata, FTS5, and audit persistence layer."""

from __future__ import annotations

from lai.db.analytics_repo import SqliteUsageAnalyticsRepository
from lai.db.auth_repo import SqliteApiKeyRepository
from lai.db.client_app_repo import SqliteClientAppRepository
from lai.db.connection import SqliteDatabase
from lai.db.file_repo import SqliteFileRepository, sanitize_fts5_query
from lai.db.project_repo import SqliteProjectRepository
from lai.db.runtime_config_repo import SqliteRuntimeConfigRepository
from lai.db.schema import SCHEMA_STATEMENTS, apply_schema
from lai.db.session_repo import SqliteSessionRepository
from lai.db.tool_repo import SqliteToolAuditRepository

__all__ = [
    "SCHEMA_STATEMENTS",
    "SqliteApiKeyRepository",
    "SqliteClientAppRepository",
    "SqliteDatabase",
    "SqliteFileRepository",
    "SqliteProjectRepository",
    "SqliteRuntimeConfigRepository",
    "SqliteSessionRepository",
    "SqliteToolAuditRepository",
    "SqliteUsageAnalyticsRepository",
    "apply_schema",
    "sanitize_fts5_query",
]
