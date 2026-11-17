"""API key generation, prefix extraction, and constant-time SHA-256 verification."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import TYPE_CHECKING

from lai.domain.auth import ApiKeyRecord, ApiScope, CreatedApiKey
from lai.domain.base import utc_now
from lai.domain.ids import new_api_key_id

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

API_KEY_PREFIX = "lai_sk_"
API_KEY_TOKEN_BYTES = 32
KEY_DISPLAY_PREFIX_LEN = 14


def hash_api_key(plaintext_key: str) -> str:
    """Compute the SHA-256 hex digest of a plaintext API key."""
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()


def verify_key_hash(plaintext_key: str, expected_hash: str) -> bool:
    """Verify a plaintext API key against a stored hex digest in constant time."""
    if not plaintext_key or not expected_hash:
        return False
    computed = hash_api_key(plaintext_key)
    return hmac.compare_digest(computed, expected_hash)


def generate_api_key(
    name: str,
    scopes: Sequence[ApiScope],
    *,
    expires_at: datetime | None = None,
    key_id: str | None = None,
    created_at: datetime | None = None,
) -> CreatedApiKey:
    """Mint a new API key with `lai_sk_` prefix and return plaintext + hashed record."""
    secret_part = secrets.token_urlsafe(API_KEY_TOKEN_BYTES)
    plaintext_key = f"{API_KEY_PREFIX}{secret_part}"
    key_prefix = plaintext_key[:KEY_DISPLAY_PREFIX_LEN]
    key_hash = hash_api_key(plaintext_key)
    dedup_scopes = tuple(dict.fromkeys(scopes))

    record = ApiKeyRecord(
        id=key_id or new_api_key_id(),
        name=name,
        key_prefix=key_prefix,
        key_hash=key_hash,
        scopes=dedup_scopes,
        created_at=created_at or utc_now(),
        expires_at=expires_at,
    )
    return CreatedApiKey(record=record, plaintext_key=plaintext_key)
