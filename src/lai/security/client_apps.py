"""Client application secret generation, hashing, and tool-tier registration validation."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import TYPE_CHECKING

from lai.domain.base import utc_now
from lai.domain.client_apps import ClientAppRecord, CreatedClientApp
from lai.domain.errors import ValidationFailedError
from lai.domain.ids import new_client_app_id
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

CLIENT_APP_SECRET_PREFIX = "lai_app_"  # noqa: S105 - public token format prefix
CLIENT_APP_TOKEN_BYTES = 32
CLIENT_APP_DISPLAY_PREFIX_LEN = 15


def hash_client_app_secret(plaintext_secret: str) -> str:
    """Compute the SHA-256 hex digest of a plaintext client app secret."""
    return hashlib.sha256(plaintext_secret.encode("utf-8")).hexdigest()


def verify_client_app_secret(plaintext_secret: str, expected_hash: str) -> bool:
    """Verify a plaintext client app secret against a stored digest in constant time."""
    if not plaintext_secret or not expected_hash:
        return False
    computed = hash_client_app_secret(plaintext_secret)
    return hmac.compare_digest(computed, expected_hash)


def generate_client_app(  # noqa: PLR0913
    name: str,
    allowed_tiers: Sequence[PermissionTier] = (PermissionTier.READ,),
    registered_tools: Sequence[ToolSpec] = (),
    allowed_project_ids: Sequence[str] = (),
    *,
    expires_at: datetime | None = None,
    app_id: str | None = None,
    created_at: datetime | None = None,
) -> CreatedClientApp:
    """Mint a client app credential and normalize all registered tools to CLIENT target."""
    stripped_name = name.strip()
    if not stripped_name:
        msg = "Client app name must not be empty."
        raise ValidationFailedError(msg, details={"field": "name"})

    dedup_tiers = tuple(dict.fromkeys(allowed_tiers))
    cleaned_projects = tuple(
        dict.fromkeys(pid.strip() for pid in allowed_project_ids if pid.strip())
    )

    normalized_tools: list[ToolSpec] = []
    seen_names: set[str] = set()
    for spec in registered_tools:
        if spec.permission_tier not in dedup_tiers:
            msg = (
                f"Registered tool {spec.name!r} requires tier {spec.permission_tier.value!r}, "
                "which is not in allowed_tiers."
            )
            raise ValidationFailedError(
                msg,
                details={"tool_name": spec.name, "permission_tier": spec.permission_tier.value},
            )
        if spec.name in seen_names:
            msg = f"Duplicate registered tool name {spec.name!r}."
            raise ValidationFailedError(msg, details={"tool_name": spec.name})
        seen_names.add(spec.name)
        normalized_tools.append(
            spec.model_copy(update={"execution_target": ToolExecutionTarget.CLIENT})
        )

    secret_part = secrets.token_urlsafe(CLIENT_APP_TOKEN_BYTES)
    plaintext_secret = f"{CLIENT_APP_SECRET_PREFIX}{secret_part}"
    secret_prefix = plaintext_secret[:CLIENT_APP_DISPLAY_PREFIX_LEN]
    secret_hash = hash_client_app_secret(plaintext_secret)

    record = ClientAppRecord(
        id=app_id or new_client_app_id(),
        name=stripped_name,
        secret_prefix=secret_prefix,
        secret_hash=secret_hash,
        allowed_tiers=dedup_tiers,
        registered_tools=tuple(normalized_tools),
        allowed_project_ids=cleaned_projects,
        created_at=created_at or utc_now(),
        expires_at=expires_at,
    )
    return CreatedClientApp(record=record, plaintext_secret=plaintext_secret)
