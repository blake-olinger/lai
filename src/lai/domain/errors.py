"""Domain exception hierarchy mapping directly to structured API failures."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from datetime import datetime


class LaiError(Exception):
    """Base domain error with machine-readable code and HTTP status hint."""

    default_code: ClassVar[str] = "internal_error"
    default_status_code: ClassVar[int] = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.status_code = status_code if status_code is not None else self.default_status_code
        self.details: dict[str, str] = dict(details) if details is not None else {}


class AuthenticationError(LaiError):
    """Raised when caller credentials are missing or invalid."""

    default_code: ClassVar[str] = "authentication_failed"
    default_status_code: ClassVar[int] = 401


class AuthorizationError(LaiError):
    """Raised when an authenticated principal lacks required scopes or access."""

    default_code: ClassVar[str] = "authorization_denied"
    default_status_code: ClassVar[int] = 403


class RateLimitExceededError(LaiError):
    """Raised when a caller exceeds the configured request rate limit."""

    default_code: ClassVar[str] = "rate_limit_exceeded"
    default_status_code: ClassVar[int] = 429

    def __init__(
        self,
        message: str = "Rate limit exceeded.",
        *,
        retry_after_seconds: float | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after_seconds = retry_after_seconds


class QueueFullError(LaiError):
    """Raised when the inference spooler queue reaches capacity."""

    default_code: ClassVar[str] = "queue_full"
    default_status_code: ClassVar[int] = 503


class NotFoundError(LaiError):
    """Raised when a requested resource does not exist."""

    default_code: ClassVar[str] = "not_found"
    default_status_code: ClassVar[int] = 404


class ConflictError(LaiError):
    """Raised when a resource state or unique constraint conflicts with the request."""

    default_code: ClassVar[str] = "conflict"
    default_status_code: ClassVar[int] = 409


class ValidationFailedError(LaiError):
    """Raised when domain validation or invariant checks fail."""

    default_code: ClassVar[str] = "validation_failed"
    default_status_code: ClassVar[int] = 422


class PayloadTooLargeError(LaiError):
    """Raised when an uploaded payload exceeds configured size limits."""

    default_code: ClassVar[str] = "payload_too_large"
    default_status_code: ClassVar[int] = 413


class InferenceBackendError(LaiError):
    """Raised when the local inference backend fails or times out."""

    default_code: ClassVar[str] = "inference_backend_error"
    default_status_code: ClassVar[int] = 502


class ToolApprovalRequiredError(LaiError):
    """Raised when tool execution requires explicit human approval."""

    default_code: ClassVar[str] = "tool_approval_required"
    default_status_code: ClassVar[int] = 409

    def __init__(
        self,
        message: str,
        *,
        approval_id: str,
        tool_name: str,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.approval_id = approval_id
        self.tool_name = tool_name


class ClientToolDispatchError(LaiError):
    """Raised when a tool invocation must be dispatched to the client application."""

    default_code: ClassVar[str] = "client_tool_dispatch"
    default_status_code: ClassVar[int] = 409

    def __init__(  # noqa: PLR0913
        self,
        message: str,
        *,
        approval_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_json: str,
        requires_approval: bool,
        expires_at: datetime | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.approval_id = approval_id
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.arguments_json = arguments_json
        self.requires_approval = requires_approval
        self.expires_at = expires_at


class ClientDisconnectedError(LaiError):
    """Raised when the client disconnects before streaming finishes."""

    default_code: ClassVar[str] = "client_disconnected"
    default_status_code: ClassVar[int] = 499


class GuardrailBlockedError(LaiError):
    """Raised when a prompt is rejected by the pre-flight safety guardrail."""

    default_code: ClassVar[str] = "prompt_blocked_by_guardrail"
    default_status_code: ClassVar[int] = 422

    def __init__(
        self,
        message: str = "Prompt blocked by safety guardrail policy.",
        *,
        violated_policy: str | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.violated_policy = violated_policy
