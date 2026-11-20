"""OIDC JWT claim policy enforcement helpers (Entra ID v2.0, Google, and standard OIDC)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from lai.domain.auth import ApiScope
from lai.domain.errors import AuthenticationError, AuthorizationError
from lai.observability.logging import get_component_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lai.security.oidc_provider import OidcProviderConfig


def resolve_identity_claim(config: OidcProviderConfig, claims: Mapping[str, Any]) -> str | None:
    """Return the first non-empty lowercase identity string from configured identity_claims."""
    for claim_name in config.identity_claims:
        raw_val = claims.get(claim_name)
        if isinstance(raw_val, str) and raw_val.strip():
            return raw_val.strip().lower()
    return None


def is_email_verified(claims: Mapping[str, Any]) -> bool:
    """Return True when email ownership is verified via email_verified or Entra xms_edov."""
    if claims.get("email_verified") is True:
        return True
    edov = claims.get("xms_edov")
    if edov is True:
        return True
    if isinstance(edov, int) and not isinstance(edov, bool) and edov == 1:
        return True
    return isinstance(edov, str) and edov.strip().lower() in {"true", "1"}


def validate_guest_policy(
    config: OidcProviderConfig,
    claims: Mapping[str, Any],
    *,
    issuer: str,
) -> None:
    """Reject Entra ID B2B guest tokens unless config.allow_guest_users is True."""
    if config.allow_guest_users:
        return
    acct = claims.get("acct")
    if acct is not None:
        is_member = (isinstance(acct, int) and not isinstance(acct, bool) and acct == 0) or (
            isinstance(acct, str) and acct.strip() == "0"
        )
        if not is_member:
            get_component_logger("security").warning(
                "oidc_auth_denied",
                reason="guest_user_not_allowed",
                acct=str(acct),
            )
            msg = "OIDC B2B guest accounts are not permitted."
            raise AuthorizationError(msg)
        return

    raw_idp = claims.get("idp")
    if isinstance(raw_idp, str) and raw_idp.strip():
        norm_idp = raw_idp.strip().rstrip("/").lower()
        norm_iss = issuer.strip().rstrip("/").lower()
        if norm_idp != norm_iss:
            get_component_logger("security").warning(
                "oidc_auth_denied",
                reason="guest_idp_not_allowed",
                idp=norm_idp,
            )
            msg = "OIDC external identity provider guest accounts are not permitted."
            raise AuthorizationError(msg)


def validate_email_policy(config: OidcProviderConfig, claims: Mapping[str, Any]) -> str | None:
    """Enforce email verification and identity/domain allowlists, returning normalized identity."""
    email = resolve_identity_claim(config, claims)

    if config.require_email_verified and not is_email_verified(claims):
        get_component_logger("security").warning("oidc_auth_failed", reason="email_not_verified")
        msg = "OIDC token email is not verified."
        raise AuthenticationError(msg)

    has_allowlist = bool(config.allowed_emails or config.allowed_domains)
    if not has_allowlist:
        return email

    if email is None:
        get_component_logger("security").warning(
            "oidc_auth_denied", reason="missing_email_for_allowlist"
        )
        msg = "OIDC token missing identity claim required by allowlist."
        raise AuthorizationError(msg)

    allowed_emails = {item.strip().lower() for item in config.allowed_emails if item.strip()}
    allowed_domains = {
        item.strip().lower().lstrip("@") for item in config.allowed_domains if item.strip()
    }
    domain = email.partition("@")[2]
    if email in allowed_emails or (domain and domain in allowed_domains):
        return email

    get_component_logger("security").warning(
        "oidc_auth_denied", reason="email_not_allowed", email=email
    )
    msg = "OIDC caller email is not permitted."
    raise AuthorizationError(msg, details={"email": email})


def resolve_principal_identity(claims: Mapping[str, Any]) -> str:
    """Return Entra tenant-wide oid when present, falling back to sub."""
    raw_oid = claims.get("oid")
    if isinstance(raw_oid, str) and raw_oid.strip():
        return raw_oid.strip()
    return str(claims["sub"]).strip()


def resolve_oauth_client_id(
    claims: Mapping[str, Any],
    *,
    audiences: tuple[str, ...],
    allowed_client_ids: tuple[str, ...] = (),
) -> str:
    """Verify calling OAuth client ID against allowed_client_ids (or audiences fallback)."""
    normalized_allowed = tuple(dict.fromkeys(c.strip() for c in allowed_client_ids if c.strip()))
    client_allowlist = normalized_allowed or audiences

    for claim_key, fail_reason in (("azp", "invalid_azp_claim"), ("appid", "invalid_appid_claim")):
        if claim_key in claims and claims[claim_key] is not None:
            raw_client = claims[claim_key]
            if not isinstance(raw_client, str) or raw_client.strip() not in client_allowlist:
                get_component_logger("security").warning("oidc_auth_failed", reason=fail_reason)
                msg = f"OIDC token {claim_key} claim is not an authorized client."
                raise AuthenticationError(msg)
            return raw_client.strip()

    aud = claims.get("aud")
    aud_candidates: list[str] = []
    if isinstance(aud, str) and aud.strip():
        aud_candidates.append(aud.strip())
    elif isinstance(aud, list):
        aud_candidates.extend(
            item.strip() for item in cast("list[object]", aud) if isinstance(item, str) and item
        )

    for candidate in aud_candidates:
        if candidate in client_allowlist:
            return candidate

    reason = "missing_authorized_client_id" if normalized_allowed else "unmatched_aud_claim"
    get_component_logger("security").warning("oidc_auth_failed", reason=reason)
    msg = "OIDC token does not identify an authorized calling client."
    raise AuthenticationError(msg)


def resolve_scopes(
    config: OidcProviderConfig,
    claims: Mapping[str, Any],
    *,
    email: str | None,
    identity_key: str,
) -> tuple[ApiScope, ...]:
    """Grant ADMIN via admin_subjects, Entra App Roles, or verified admin_emails."""
    admin_subjects = {item.strip() for item in config.admin_subjects if item.strip()}
    if identity_key in admin_subjects:
        return (ApiScope.ADMIN,)

    admin_roles = {item.strip() for item in config.admin_roles if item.strip()}
    raw_roles = claims.get("roles")
    if admin_roles and isinstance(raw_roles, list):
        roles_list = cast("list[object]", raw_roles)
        if any(isinstance(r, str) and r.strip() in admin_roles for r in roles_list):
            return (ApiScope.ADMIN,)

    if email is not None:
        admin_emails = {item.strip().lower() for item in config.admin_emails if item.strip()}
        if email in admin_emails:
            if is_email_verified(claims):
                return (ApiScope.ADMIN,)
            get_component_logger("security").warning(
                "oidc_admin_email_unverified_denied",
                email=email,
                identity_key=identity_key,
            )
    return config.default_scopes
