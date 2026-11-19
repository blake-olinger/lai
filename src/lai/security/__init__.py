"""Authentication, API key hashing, scope checks, Cloudflare edge guard, and rate limits."""

from __future__ import annotations

from lai.security.api_key_provider import ApiKeyAuthProvider
from lai.security.chain import CompositeAuthenticator
from lai.security.client_apps import build_client_app
from lai.security.cloudflare import CloudflareEdgeConfig, CloudflareEdgeVerifier
from lai.security.keys import (
    API_KEY_PREFIX,
    generate_api_key,
    hash_api_key,
    verify_key_hash,
)
from lai.security.oidc_provider import OidcJwtAuthProvider, OidcProviderConfig
from lai.security.ratelimit import TokenBucketRateLimiter
from lai.security.scopes import expand_scopes, has_effective_scope, require_scope

__all__ = [
    "API_KEY_PREFIX",
    "ApiKeyAuthProvider",
    "CloudflareEdgeConfig",
    "CloudflareEdgeVerifier",
    "CompositeAuthenticator",
    "OidcJwtAuthProvider",
    "OidcProviderConfig",
    "TokenBucketRateLimiter",
    "build_client_app",
    "expand_scopes",
    "generate_api_key",
    "has_effective_scope",
    "hash_api_key",
    "require_scope",
    "verify_key_hash",
]
