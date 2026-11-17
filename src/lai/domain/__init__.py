"""Pure domain models, prefixed IDs, and error hierarchy."""

from __future__ import annotations

from lai.domain.auth import (
    ApiKeyRecord,
    ApiScope,
    AuthMethod,
    AuthPrincipal,
    CreatedApiKey,
    RequestCredentials,
)
from lai.domain.observability import (
    AlertCategory,
    AlertEvent,
    AlertSeverity,
    LoggingConfig,
)

__all__ = [
    "AlertCategory",
    "AlertEvent",
    "AlertSeverity",
    "ApiKeyRecord",
    "ApiScope",
    "AuthMethod",
    "AuthPrincipal",
    "CreatedApiKey",
    "LoggingConfig",
    "RequestCredentials",
]
