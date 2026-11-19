"""Fail-closed configuration validators for OIDC, Cloudflare Access, and CORS."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from lai.config.settings import LaiSettings

PUBLIC_CONSUMER_EMAIL_DOMAINS: frozenset[str] = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "icloud.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
    }
)


def _has_text(value: str | None) -> bool:
    return bool(value is not None and value.strip())


def _validate_oidc_settings(settings: LaiSettings) -> None:
    """Reject insecure or incomplete OIDC configuration when OIDC auth is enabled."""
    normalized_providers = {provider.strip().lower() for provider in settings.auth_providers}
    if not (settings.oidc_enabled or "oidc" in normalized_providers):
        return

    if not settings.resolved_oidc_audiences():
        msg = (
            "OIDC authentication requires at least one configured audience "
            "(oidc_audiences or oidc_audience)."
        )
        raise ValueError(msg)

    has_allowed_emails = any(email.strip() for email in settings.oidc_allowed_emails)
    cleaned_domains = [
        raw.strip().lower().lstrip("@")
        for raw in settings.oidc_allowed_domains
        if raw.strip().lstrip("@")
    ]
    if not (has_allowed_emails or cleaned_domains):
        msg = "OIDC authentication requires non-empty oidc_allowed_emails or oidc_allowed_domains."
        raise ValueError(msg)

    for domain in cleaned_domains:
        if domain in PUBLIC_CONSUMER_EMAIL_DOMAINS:
            msg = (
                f"Consumer email domain {domain!r} is not permitted in oidc_allowed_domains; "
                "use oidc_allowed_emails for explicit allowlisting."
            )
            raise ValueError(msg)


def _validate_cloudflare_settings(settings: LaiSettings) -> None:
    """Require valid JWT or service-token credentials when Cloudflare Access is enabled."""
    if not settings.cf_access_enabled:
        return
    has_jwt = _has_text(settings.cf_access_team_domain) and _has_text(settings.cf_access_aud)
    has_service = _has_text(settings.cf_access_client_id) and _has_text(
        settings.cf_access_client_secret
    )
    if not (has_jwt or has_service):
        msg = (
            "cf_access_enabled=True requires either (cf_access_team_domain and cf_access_aud) "
            "or (cf_access_client_id and cf_access_client_secret)."
        )
        raise ValueError(msg)


def _validate_single_cors_origin(origin: str) -> None:
    """Validate that a CORS origin is an explicit http:// or https:// scheme://netloc origin."""
    if not origin or origin != origin.strip() or any(ch.isspace() for ch in origin):
        msg = f"Invalid CORS origin {origin!r}: must not be empty or contain whitespace."
        raise ValueError(msg)
    if origin == "*":
        msg = "Wildcard '*' is not allowed in cors_allowed_origins."
        raise ValueError(msg)
    if "?" in origin or "#" in origin:
        msg = f"Invalid CORS origin {origin!r}: query and fragment are not permitted."
        raise ValueError(msg)

    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"}:
        msg = f"Invalid CORS origin {origin!r}: scheme must be 'http' or 'https'."
        raise ValueError(msg)
    if (
        not parsed.netloc
        or not parsed.hostname
        or "@" in parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        msg = f"Invalid CORS origin {origin!r}: must be a valid scheme://netloc origin."
        raise ValueError(msg)
    try:
        _ = parsed.port
    except ValueError as exc:
        msg = f"Invalid CORS origin {origin!r}: invalid port."
        raise ValueError(msg) from exc
    if parsed.path or parsed.query or parsed.fragment:
        msg = (
            f"Invalid CORS origin {origin!r}: must not contain a path, "
            "trailing '/', query, or fragment."
        )
        raise ValueError(msg)


def validate_lai_settings(settings: LaiSettings) -> None:
    """Run all fail-closed security and CORS checks on LaiSettings."""
    _validate_oidc_settings(settings)
    _validate_cloudflare_settings(settings)
    for origin in settings.cors_allowed_origins:
        _validate_single_cors_origin(origin)
