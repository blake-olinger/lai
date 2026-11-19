# Security Policy

## Supported Versions

Security updates are delivered on the active `main` branch and latest tagged release:

| Version | Supported |
| --- | --- |
| `0.1.x` (`main`) | Yes |
| Pre-`0.1.0` snapshots | No |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities, credential leaks, or
multi-tenant isolation bypasses.

Report vulnerabilities privately through either channel:

- **GitHub Security Advisories** (preferred):
  [https://github.com/blake-olinger/lai/security/advisories/new](https://github.com/blake-olinger/lai/security/advisories/new)
- **Email**: [hello@blakeolinger.dev](mailto:hello@blakeolinger.dev)

Please include:

1. Affected component, endpoint, or configuration surface.
2. Clear reproduction steps or a minimal proof of concept.
3. Expected security invariant vs. observed behavior.
4. Any relevant log excerpts (scrubbed of live secrets or tokens).

We aim to acknowledge reports within 48 hours, validate severity and blast radius promptly, and
coordinate a fix before public disclosure.

## Security Architecture & Boundaries

LAI is built with defense-in-depth across two primary execution planes and strict multi-tenant
isolation:

### Dual-Plane Security Boundaries

1. **Control & API Plane (`lai.api`, `lai.security`, `lai.tools`, `lai.config`)**
   - **Authentication**: Protected endpoints require `Authorization: Bearer <token>` using either
     SHA-256 hashed server-issued API keys (`lai_sk_...`) or RS256/ES256 Google OIDC ID tokens
     verified against TTL-cached JWKS (`exp`, `iss`, `sub`, `aud`, `email_verified`).
   - **Fail-Closed Startup Validation**: Enabling OIDC without explicit `oidc_audiences` (or
     `oidc_audience`) and `oidc_allowed_emails` / corporate `oidc_allowed_domains` fails startup;
     public consumer domains (`gmail.com`, `yahoo.com`, `outlook.com`, etc.) are rejected in
     `oidc_allowed_domains`; wildcard CORS origins (`"*"`) are rejected; enabling
     `cf_access_enabled` without complete team/audience or service-token credentials fails closed.
   - **Header Trust Boundary**: `trust_cloudflare_headers` defaults to `false`. Only enable it
     when network ingress is strictly restricted to a Cloudflare Named Tunnel or trusted proxy so
     `CF-Connecting-IP` and `CF-Access-*` headers cannot be spoofed.
   - **HTTP-Only Ingestion**: File ingestion streams bytes over HTTP only (`/v1/files` and
     `/v1/files/stream`). Server filesystem paths are never accepted as ingestion targets.
   - **Guarded Tool Execution**: Tools are segregated by `PermissionTier` (`READ`, `WRITE`,
     `EXEC`, `NETWORK`), workspace-root confined, and gated behind explicit human approval
     (`PENDING -> APPROVED -> EXECUTED`) with single-use replay protection.

2. **Inference & Guardrail Plane (`lai.inference`, `lai.guardrails`)**
   - **Pre-Flight Safety Gate**: Prompts are evaluated at `t=0ms` by a CPU-only ShieldGemma-2B
     instance (`--ngl 0`, `fail_open: false` by default) before entering the GPU inference spooler.
   - **Audit Scrubbing**: Structured JSONL logs and `prompts_audit.jsonl` scrub bearer tokens,
     secrets, and PII before persistence.

### Multi-Tenant Isolation

- **Principal Ownership & `404` Anti-Enumeration**: Projects, files, sessions, messages, and tool
  approvals carry `owner_principal_id`. Cross-tenant lookups or mutations by non-admin callers
  fail with `404 Not Found` (`not_found`) rather than `403` to prevent resource-ID enumeration.
- **Per-Principal Quotas & Fair Queuing**: Server-enforced limits cap projects
  (`max_projects_per_principal`), cumulative uploaded storage (`max_storage_bytes_per_principal`),
  and queued inference jobs (`max_queued_jobs_per_principal`), paired with two-level
  `principal_id -> session_id` round-robin scheduling in `InferenceSpooler`.

### Public SPA & Desktop OIDC Client Model

Public browser SPAs and desktop clients cannot protect embedded static client secrets. LAI
therefore issues and accepts **zero static client secrets**:

- Client applications are registered via `lai apps create --oauth-client-id <CLIENT_ID>` (or
  `POST /v1/client-apps`), binding the record to an OAuth 2.0 `oauth_client_id`.
- For OIDC callers, `AuthPrincipal.oauth_client_id` is derived cryptographically from the
  verified Google ID token's `azp` claim (when present, required to be in `oidc_audiences`) or
  matched `aud` claim, and enforced against `ClientAppRecord.oauth_client_id` on client-tool turns.

## Automated Security Controls in CI

Every push and pull request runs:

- `uv run python -m ruff check --select S src scripts` (Bandit security lint rules)
- `uv tool run pip-audit -r requirements.txt` (dependency vulnerability audit)
- SHA-pinned GitHub Actions workflows with least-privilege `permissions`
