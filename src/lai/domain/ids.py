"""Prefixed identifier generators and validators."""

from __future__ import annotations

import uuid

PROJECT_ID_PREFIX = "prj_"
SESSION_ID_PREFIX = "ses_"
FILE_ID_PREFIX = "fil_"
CHUNK_ID_PREFIX = "chk_"
MESSAGE_ID_PREFIX = "msg_"
API_KEY_ID_PREFIX = "key_"
JOB_ID_PREFIX = "job_"
APPROVAL_ID_PREFIX = "apr_"
AUDIT_ID_PREFIX = "aud_"
ALERT_ID_PREFIX = "alt_"

ALL_ID_PREFIXES: tuple[str, ...] = (
    PROJECT_ID_PREFIX,
    SESSION_ID_PREFIX,
    FILE_ID_PREFIX,
    CHUNK_ID_PREFIX,
    MESSAGE_ID_PREFIX,
    API_KEY_ID_PREFIX,
    JOB_ID_PREFIX,
    APPROVAL_ID_PREFIX,
    AUDIT_ID_PREFIX,
    ALERT_ID_PREFIX,
)


def generate_prefixed_id(prefix: str) -> str:
    """Generate a globally unique identifier carrying the given prefix."""
    if not prefix or not prefix.endswith("_"):
        msg = f"ID prefix must be non-empty and end with '_': {prefix!r}"
        raise ValueError(msg)
    return f"{prefix}{uuid.uuid4().hex}"


def validate_prefixed_id(value: str, prefix: str) -> str:
    """Validate that value begins with prefix and has a non-empty suffix."""
    if not value.startswith(prefix) or len(value) <= len(prefix):
        msg = f"Expected identifier with prefix {prefix!r}, got {value!r}"
        raise ValueError(msg)
    return value


def new_project_id() -> str:
    """Return a new project identifier."""
    return generate_prefixed_id(PROJECT_ID_PREFIX)


def new_session_id() -> str:
    """Return a new chat session identifier."""
    return generate_prefixed_id(SESSION_ID_PREFIX)


def new_file_id() -> str:
    """Return a new uploaded file identifier."""
    return generate_prefixed_id(FILE_ID_PREFIX)


def new_chunk_id() -> str:
    """Return a new file chunk identifier."""
    return generate_prefixed_id(CHUNK_ID_PREFIX)


def new_message_id() -> str:
    """Return a new chat message identifier."""
    return generate_prefixed_id(MESSAGE_ID_PREFIX)


def new_api_key_id() -> str:
    """Return a new API key record identifier."""
    return generate_prefixed_id(API_KEY_ID_PREFIX)


def new_job_id() -> str:
    """Return a new inference spooler job identifier."""
    return generate_prefixed_id(JOB_ID_PREFIX)


def new_approval_id() -> str:
    """Return a new tool approval request identifier."""
    return generate_prefixed_id(APPROVAL_ID_PREFIX)


def new_audit_id() -> str:
    """Return a new tool audit log identifier."""
    return generate_prefixed_id(AUDIT_ID_PREFIX)


def new_alert_id() -> str:
    """Return a new observability alert identifier."""
    return generate_prefixed_id(ALERT_ID_PREFIX)
