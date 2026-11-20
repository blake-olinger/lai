"""Fail-closed configuration validators for OIDC, proxies, integrity, CORS, and alerting."""

from __future__ import annotations

import ipaddress
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from lai.config.settings import LaiSettings

_CORS_HEADER_RE = re.compile(r"^[A-Za-z0-9-]+$")

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

    if not any(claim.strip() for claim in settings.oidc_identity_claims):
        msg = "OIDC authentication requires at least one non-empty claim in oidc_identity_claims."
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


def _validate_cors_header_name(header: str) -> None:
    """Validate that an extra CORS allow header is an explicit ASCII token without '*'."""
    if not header or header != header.strip() or _CORS_HEADER_RE.fullmatch(header) is None:
        msg = (
            f"Invalid CORS header name {header!r}: "
            "must be a non-empty ASCII header token (letters, digits, '-')."
        )
        raise ValueError(msg)


def _validate_email_address(address: str, field_name: str) -> None:
    """Reject empty, header-injectable, or syntactically invalid email addresses."""
    if "\r" in address or "\n" in address:
        msg = f"Invalid email address in {field_name}: control characters are not permitted."
        raise ValueError(msg)
    cleaned = address.strip()
    local, sep, domain = cleaned.partition("@")
    if (
        not cleaned
        or not sep
        or not local
        or not domain
        or "@" in domain
        or any(ch.isspace() for ch in cleaned)
    ):
        msg = f"Invalid email address {address!r} in {field_name}."
        raise ValueError(msg)


def _validate_webhook_url(url: str) -> None:
    """Require an explicit http:// or https:// URL with hostname for alert webhooks."""
    cleaned = url.strip()
    if not cleaned or cleaned != url or any(ch.isspace() for ch in cleaned):
        msg = f"Invalid alert_webhook_url {url!r}: must not be empty or contain whitespace."
        raise ValueError(msg)
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        msg = f"Invalid alert_webhook_url {url!r}: must be a valid http:// or https:// URL."
        raise ValueError(msg)


def _validate_alerting_settings(settings: LaiSettings) -> None:
    """Enforce fail-closed validation on webhook and outbound SMTP alerting settings."""
    if settings.logging.alert_webhook_url is not None:
        _validate_webhook_url(settings.logging.alert_webhook_url)

    smtp = settings.logging.alert_smtp
    if _has_text(smtp.password) and not _has_text(smtp.username):
        msg = "logging.alert_smtp.password requires logging.alert_smtp.username to be configured."
        raise ValueError(msg)
    if "\r" in smtp.subject_prefix or "\n" in smtp.subject_prefix:
        msg = "logging.alert_smtp.subject_prefix must not contain CR/LF characters."
        raise ValueError(msg)
    if not smtp.enabled:
        return

    if not _has_text(smtp.host) or any(ch.isspace() for ch in (smtp.host or "").strip()):
        msg = "logging.alert_smtp.host must be a non-empty hostname when SMTP alerting is enabled."
        raise ValueError(msg)
    if not _has_text(smtp.from_address):
        msg = "logging.alert_smtp.from_address is required when SMTP alerting is enabled."
        raise ValueError(msg)
    _validate_email_address(smtp.from_address or "", "logging.alert_smtp.from_address")

    explicit = [item.strip() for item in smtp.to_addresses if item.strip()]
    fallback = (
        [item.strip() for item in settings.oidc_admin_emails if item.strip()]
        if smtp.use_admin_emails_fallback
        else []
    )
    recipients = explicit or fallback
    if not recipients:
        msg = (
            "SMTP alerting requires non-empty logging.alert_smtp.to_addresses "
            "or oidc_admin_emails when use_admin_emails_fallback is enabled."
        )
        raise ValueError(msg)
    for recipient in recipients:
        _validate_email_address(recipient, "logging.alert_smtp.to_addresses")


def _validate_trusted_proxy_cidrs(settings: LaiSettings) -> None:
    """Validate trusted_proxy_cidrs entries as strict IPv4/IPv6 networks."""
    for raw in settings.trusted_proxy_cidrs:
        cleaned = raw.strip()
        if not cleaned or cleaned != raw:
            msg = (
                f"Invalid trusted proxy CIDR {raw!r}: "
                "must not be empty or contain surrounding whitespace."
            )
            raise ValueError(msg)
        try:
            ipaddress.ip_network(cleaned, strict=False)
        except ValueError as exc:
            msg = f"Invalid trusted proxy CIDR {raw!r}: {exc}"
            raise ValueError(msg) from exc


def _validate_model_integrity(settings: LaiSettings) -> None:
    """Enforce boot-only model allowlist and SHA-256 pinning for custom profiles."""
    integrity = settings.model_integrity
    if not integrity.strict_allowlist:
        return
    approved_digests = frozenset(integrity.approved_model_sha256s)
    allowed_repos = frozenset(integrity.allowed_hf_repos)
    for name, profile in settings.custom_profiles.items():
        if profile.sha256 is None:
            msg = (
                f"Custom model profile {name!r} must pin sha256 "
                "when model_integrity.strict_allowlist is enabled."
            )
            raise ValueError(msg)
        if profile.sha256 not in approved_digests:
            msg = (
                f"Custom model profile {name!r} sha256 {profile.sha256!r} "
                "is not in model_integrity.approved_model_sha256s."
            )
            raise ValueError(msg)
        if allowed_repos and profile.hf_repo not in allowed_repos:
            msg = (
                f"Custom model profile {name!r} hf_repo {profile.hf_repo!r} "
                "is not in model_integrity.allowed_hf_repos."
            )
            raise ValueError(msg)


def validate_lai_settings(settings: LaiSettings) -> None:
    """Run all fail-closed security, proxy, integrity, CORS, and alerting checks."""
    _validate_oidc_settings(settings)
    _validate_cloudflare_settings(settings)
    _validate_trusted_proxy_cidrs(settings)
    _validate_model_integrity(settings)
    for origin in settings.cors_allowed_origins:
        _validate_single_cors_origin(origin)
    for header in settings.cors_extra_allow_headers:
        _validate_cors_header_name(header)
    _validate_alerting_settings(settings)
