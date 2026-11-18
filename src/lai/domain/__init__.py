"""Pure domain models, prefixed IDs, and error hierarchy."""

from __future__ import annotations

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
from lai.domain.client_apps import ClientAppRecord, CreatedClientApp
from lai.domain.errors import ClientToolDispatchError, GuardrailBlockedError
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
    "ClientAppRecord",
    "ClientToolCallEvent",
    "ClientToolDispatchError",
    "CreatedApiKey",
    "CreatedClientApp",
    "GuardrailBlockMode",
    "GuardrailBlockedError",
    "GuardrailConfig",
    "GuardrailDecision",
    "GuardrailVerdict",
    "InferenceToolCall",
    "LoggingConfig",
    "PromptAuditEntry",
    "RequestCredentials",
    "SafetyPrinciple",
    "expand_scopes",
]
