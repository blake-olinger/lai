"""Pure domain models, prefixed IDs, and error hierarchy."""

from __future__ import annotations

from lai.domain.analytics import (
    AnalyticsConfig,
    AnalyticsReport,
    EndpointUsageSummary,
    HourlyUsageBucket,
    PrincipalUsageSummary,
    TokenUsageRecord,
    UsageEvent,
)
from lai.domain.auth import (
    ApiKeyRecord,
    ApiScope,
    AuthMethod,
    AuthPrincipal,
    CreatedApiKey,
    RequestCredentials,
    expand_scopes,
)
from lai.domain.chat import ClientToolCallEvent, InferenceToolCall
from lai.domain.client_apps import ClientAppRecord
from lai.domain.errors import (
    ClientToolDispatchError,
    ConflictError,
    GuardrailBlockedError,
    StorageFailureError,
)
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailVerdict,
    PromptAuditEntry,
    SafetyPrinciple,
)
from lai.domain.observability import (
    AlertCategory,
    AlertEvent,
    AlertSeverity,
    AlertSinkDeliveryResult,
    LoggingConfig,
    SmtpAlertConfig,
    SmtpTlsMode,
)

__all__ = [
    "AlertCategory",
    "AlertEvent",
    "AlertSeverity",
    "AlertSinkDeliveryResult",
    "AnalyticsConfig",
    "AnalyticsReport",
    "ApiKeyRecord",
    "ApiScope",
    "AuthMethod",
    "AuthPrincipal",
    "ClientAppRecord",
    "ClientToolCallEvent",
    "ClientToolDispatchError",
    "ConflictError",
    "CreatedApiKey",
    "EndpointUsageSummary",
    "GuardrailBlockMode",
    "GuardrailBlockedError",
    "GuardrailConfig",
    "GuardrailDecision",
    "GuardrailVerdict",
    "HourlyUsageBucket",
    "InferenceToolCall",
    "LoggingConfig",
    "PrincipalUsageSummary",
    "PromptAuditEntry",
    "RequestCredentials",
    "SafetyPrinciple",
    "SmtpAlertConfig",
    "SmtpTlsMode",
    "StorageFailureError",
    "TokenUsageRecord",
    "UsageEvent",
    "expand_scopes",
]
