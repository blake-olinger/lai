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

## Security Model & Boundaries

LAI is designed with defense-in-depth across its API control plane, local inference plane, and
multi-tenant workspace layer.

### Control & API Plane

- **Authentication**: Protected endpoints require `Authorization: Bearer <token>` using either
  SHA-256 hashed API keys (`lai_sk_...`) or RS256/ES256 Google OIDC ID tokens verified against
  TTL-cached JWKS (`exp`, `iss`, `sub`, `aud`, `azp`, and `email_verified`). Only `/v1/health` is
  unauthenticated; interactive OpenAPI docs routes (`/docs`, `/redoc`, `/openapi.json`) are
  disabled on the runtime server.
- **Fail-Closed Configuration Validation**: Startup fails closed if OIDC is enabled without
  explicit audiences and email/domain allowlists, if public consumer email domains (`gmail.com`,
  `yahoo.com`, `outlook.com`, etc.) are placed in `oidc_allowed_domains`, if wildcard `"*"` is used
  in `cors_allowed_origins`, or if Cloudflare Access is enabled without complete credentials.
- **Proxy Header Trust Boundary**: `trust_cloudflare_headers` defaults to `false`. Even when
  enabled, `CF-Connecting-IP` is only honored for connections arriving over loopback (`127.0.0.1` /
  `::1`) so external callers cannot spoof client IPs for rate limiting.
- **HTTP-Only File Ingestion**: Files enter LAI exclusively via HTTP upload (`/v1/files` and
  `/v1/files/stream`). Server filesystem paths are never accepted as ingestion targets.
- **Guarded Tool Execution**: Server-side tools are read-only and strictly confined to uploaded
  attachments within the caller's project and session scope. Client-side tools are classified by
  risk tier (`read`, `write`, `exec`, `network`), bounded by registered app permissions and caller
  scopes, and gated behind explicit human approval with single-use replay protection.

### Inference & Guardrail Plane

- **Pre-Flight Safety Gate**: Prompts are evaluated on CPU by ShieldGemma-2B (`fail_open: false` by
  default) before entering the GPU inference queue. Returned client tool outputs are also scrubbed
  and guardrail-scanned before being fed back to the model.
- **Log & Audit Scrubbing**: Structured JSONL logs and prompt audit logs scrub bearer tokens, API
  keys, and common PII patterns before persistence.

### Multi-Tenant Isolation

- **Ownership Scoping & `404` Anti-Enumeration**: Projects, uploaded files, sessions, messages, and
  tool approvals are bound to their owner. Cross-tenant reads or mutations by non-admin callers
  fail with `404 Not Found` rather than `403` so resource IDs cannot be enumerated.
- **Quotas & Fair Scheduling**: Server-enforced quotas cap per-user projects, cumulative uploaded
  storage, and queued inference jobs, paired with fair round-robin scheduling across users and
  sessions plus pre-auth IP and per-user rate limits.

### Zero-Secret Public SPA & Desktop Client Model

Public browser SPAs and installed desktop applications cannot keep embedded static client secrets
confidential. LAI therefore issues and accepts **zero static client secrets**:

- Client applications are registered against their Google OAuth 2.0 Client ID (`lai apps create` or
  `POST /v1/client-apps`).
- For OIDC callers, client identity is derived cryptographically from the verified Google ID
  token's `azp` claim (when present, required to match configured `oidc_audiences`) or matched
  `aud` claim, and enforced on all client tool dispatches and result submissions.

## Automated Security Controls

Every push and pull request runs automated CI security checks:

- **Secret scanning** across full git history (`gitleaks`)
- **Static security linting** (`ruff` Bandit rules across source and scripts)
- **Dependency vulnerability auditing** (`pip-audit` against the locked dependency graph)
- **Supply-chain hardening** via SHA-pinned GitHub Actions workflows, least-privilege job
  permissions, and automated weekly Dependabot updates
