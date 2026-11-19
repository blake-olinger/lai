# Local AI Platform (LAI)

[![CI](https://github.com/blake-olinger/lai/actions/workflows/ci.yml/badge.svg)](https://github.com/blake-olinger/lai/actions/workflows/ci.yml)
[![Security](https://github.com/blake-olinger/lai/actions/workflows/security.yml/badge.svg)](https://github.com/blake-olinger/lai/actions/workflows/security.yml)
[![Docs](https://github.com/blake-olinger/lai/actions/workflows/docs.yml/badge.svg)](https://blake-olinger.github.io/lai/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Local, multi-tenant, security-hardened AI platform for isolated project workspaces, large log and
source-code root-cause debugging, pre-flight CPU ShieldGemma prompt guardrails, guarded tool
execution, and OpenAI-compatible remote access over Cloudflare Named Tunnels.

## Architecture Overview

LAI enforces a strict 6-layer architecture via `import-linter` (horizontal imports across Layer-3
subsystems are prohibited):

1. **Surface (`lai.cli`, `lai.api`)** — FastAPI HTTP/SSE server (`:8400`) and Typer CLI (`init`,
   `keys`, `apps`, `models`, `serve`, `tunnel`).
2. **Orchestration (`lai.orchestrator`)** — Pre-flight guardrail gate at `t=0ms`, session
   serialization, adaptive context routing (`DIRECT`, `RETRIEVAL`, `MAP_REDUCE`), and bounded
   server/client tool loops.
3. **Subsystems (independent Layer-3 modules)**:
   - `lai.security`: `lai_sk_` SHA-256 API keys, multi-audience RS256/ES256 Google OIDC JWT
     validation (`azp`/`aud` client binding), Cloudflare Access edge checks, scope hierarchy, and
     token-bucket rate limiting.
   - `lai.db`: WAL-mode SQLite (`BEGIN IMMEDIATE` writes, foreign keys, FTS5 porter index, and
     owner-scoped repositories).
   - `lai.ingestion`: HTTP-stream-only `LocalBlobStore`, log/code/prose chunkers, CPU-only
     `FastEmbedCpuAdapter` (`BAAI/bge-small-en-v1.5`), project-isolated `LanceDbVectorStore`, and
     Reciprocal Rank Fusion.
   - `lai.inference`: Supervised CUDA `llama-server` (`:8081`, Gemma 4 26B-A4B MoE with `--jinja`,
     `--flash-attn on`, `-ctk q8_0`, `-ctv q8_0`, `--n-cpu-moe`) or signed Ollama backend, plus
     streaming `<thought>` parsing.
   - `lai.guardrails`: CPU-only `ShieldGemma-2B` supervisor (`:8082`, `--ngl 0`) and
     `CompositeGuardrailPipeline` with scrubbed 24h prompt auditing.
   - `lai.tools`: Policy engine (`READ`, `WRITE`, `EXEC`, `NETWORK`), human approval workflow
     (`PENDING -> APPROVED -> EXECUTED`), read-only workspace tools, and OIDC-bound client tool
     dispatch.
   - `lai.scheduler`: Per-session mutual exclusion (`SessionGate`), two-level fair
     `principal_id -> session_id` `InferenceSpooler` with per-principal queue caps, and Cloudflare
     524 keepalive wrappers.
   - `lai.config`: YAML + `LAI_*` env settings, fail-closed startup validators, and model profiles.
4. **Observability (`lai.observability`)** — `structlog` JSONL 24h daily rotation, PII/secret
   scrubbing, and webhook alerting (`AlertManager`).
5. **Ports (`lai.ports`)** — PEP 544 `Protocol` contracts.
6. **Domain (`lai.domain`)** — Pure immutable Pydantic v2 models, prefixed IDs, and `LaiError`
   hierarchy.

## Prerequisites & Installation

Prerequisites: **Python 3.13** (`>=3.13,<3.14`), [`uv`](https://docs.astral.sh/uv/), and NVIDIA
CUDA drivers for local GPU inference.

```powershell
uv python install 3.13
uv sync --locked
```

Copy the example configuration to customize settings:

```powershell
Copy-Item config/config.example.yaml config/config.yaml
```

## Quickstart (`init`, `keys`, `apps`, `models setup`, `serve`)

Initialize local directories (`data/`, `models/`, `bin/`) and SQLite schema:

```powershell
uv run python -m lai.cli init
```

Mint an admin API key (copy the one-time `lai_sk_...` secret):

```powershell
uv run python -m lai.cli keys create --name "local-admin" --admin
uv run python -m lai.cli keys list
```

Register an OIDC-bound public SPA or desktop client application (no static client secret):

```powershell
uv run python -m lai.cli apps create `
  --name "lai-web-spa" `
  --oauth-client-id "1234567890-webapp.apps.googleusercontent.com" `
  --tier read
uv run python -m lai.cli apps list
```

Download CUDA `llama-server` binaries, Gemma 4 26B-A4B GGUF, and ShieldGemma-2B GGUF:

```powershell
uv run python -m lai.cli models setup
```

Start Gemma 4 (`:8081`), CPU ShieldGemma (`:8082`), and FastAPI (`:8400`):

```powershell
uv run python -m lai.cli serve --config config/config.yaml
```

Or on Windows via the PowerShell launcher:

```powershell
.\scripts\start-lai.ps1 -ConfigPath config\config.yaml
```

## Multi-Tenant Isolation & Per-Principal Quotas

LAI isolates workspaces and compute across authenticated principals:

- **Owner Scoping & `404` Anti-Enumeration**: Projects, uploaded files, chat sessions, messages,
  and tool approvals are bound to `owner_principal_id`. Cross-tenant lookups or mutations by
  non-admin principals fail with `404 Not Found` (`not_found`) so resource IDs cannot be probed.
- **Per-Principal Quotas** (configured in `config/config.yaml`, defaults shown):
  - `max_projects_per_principal: 10`
  - `max_storage_bytes_per_principal: 1073741824` (1 GiB across uploaded files per principal)
  - `max_queued_jobs_per_principal: 4` (enforced by `InferenceSpooler`)
- **Two-Level Fair Scheduling**: `InferenceSpooler` schedules round-robin across `principal_id`
  first and `session_id` lanes second, preventing one caller with many sessions from starving
  other tenants.

## Google OIDC Multi-Audience & Public SPA/Desktop Client Binding

LAI supports multiple Google OAuth 2.0 client IDs (for example, a Web SPA client ID and an
Electron/Desktop native client ID) simultaneously via `oidc_audiences`:

```yaml
auth_providers:
  - "api_key"
  - "oidc"
oidc_enabled: true
oidc_issuer_url: "https://accounts.google.com"
oidc_jwks_url: "https://www.googleapis.com/oauth2/v3/certs"
oidc_audiences:
  - "1234567890-webapp.apps.googleusercontent.com"
  - "1234567890-desktop.apps.googleusercontent.com"
oidc_allowed_domains:
  - "yourcompany.com"
oidc_admin_emails:
  - "owner@yourcompany.com"
```

Fail-closed startup rules:

- Enabling OIDC requires non-empty `oidc_audiences` (or legacy `oidc_audience`) **and** non-empty
  `oidc_allowed_emails` or `oidc_allowed_domains`.
- Public consumer domains (`gmail.com`, `googlemail.com`, `yahoo.com`, `outlook.com`, `hotmail.com`,
  `live.com`, `icloud.com`, `aol.com`, `proton.me`, `protonmail.com`) are rejected in
  `oidc_allowed_domains` — use `oidc_allowed_emails` for individual personal accounts.

### Why Public Clients Use `azp` / `aud` Binding Without Secrets

Public SPAs and desktop applications cannot keep static client secrets confidential inside browser
bundles or user-installed binaries. Instead of issuing or accepting static client secrets, LAI
verifies the Google-signed RS256/ES256 ID token's `azp` claim (when present, required to be in
`oidc_audiences`) or matched `aud` claim and binds `AuthPrincipal.oauth_client_id`
cryptographically to that OAuth client ID. Register each client with
`lai apps create --oauth-client-id <CLIENT_ID>` so client tool execution scopes (`allowed_tiers`,
`allowed_project_ids`) are enforced purely from verified OIDC identity with zero secrets on the
client.

## CORS & Cloudflare Named Tunnel

Expose `127.0.0.1:8400` securely over HTTPS without opening inbound firewall ports:

```powershell
cloudflared tunnel login
cloudflared tunnel create lai-local
cloudflared tunnel route dns lai-local ai.example.com

uv run python -m lai.cli tunnel render-config `
  --tunnel "<TUNNEL_UUID>" `
  --hostname "ai.example.com" `
  --output deploy/cloudflared/config.yml

cloudflared tunnel --config deploy/cloudflared/config.yml run
```

Production reverse-proxy & browser configuration in `config/config.yaml`:

```yaml
cors_allowed_origins:
  - "https://app.example.com"
trust_cloudflare_headers: true
cf_access_enabled: false
```

- **CORS allowlist**: Specify exact `http://` or `https://` origins in `cors_allowed_origins`;
  wildcard `"*"` is rejected at startup.
- **`trust_cloudflare_headers: true`**: Enable **only** when the API server is unreachable outside
  the Cloudflare Named Tunnel (or trusted proxy) so `CF-Connecting-IP` cannot be spoofed.
- **`cf_access_enabled: false` for public SPA/Desktop APIs**: When serving browser SPAs or
  desktop clients authenticated directly via Google OIDC ID tokens (`Authorization: Bearer ...`)
  and CORS, keep `cf_access_enabled: false` so Cloudflare Access interactive redirects do not
  intercept CORS preflight `OPTIONS` or bearer API calls. Enable `cf_access_enabled: true` (with
  `cf_access_team_domain` + `cf_access_aud`, or service token credentials) only for private
  Cloudflare Access-protected deployments.
- **100 MB Tunnel Request Body Limit**: Cloudflare Free and Pro plans cap individual HTTP request
  bodies at **100 MB** (`HTTP 413`). Keep remote tunnel uploads under 100 MB per request (while
  direct LAN access supports up to `max_upload_bytes`).

## API Reference & Examples

- **Interactive Scalar API Reference**:
  [https://blake-olinger.github.io/lai/](https://blake-olinger.github.io/lai/)
- **Checked-in Markdown Reference**: [docs/api-reference.md](docs/api-reference.md)
- **OpenAPI 3.1 JSON Schema**: [openapi.json](openapi.json)

All protected endpoints accept `Authorization: Bearer <google_id_token>` or
`Authorization: Bearer lai_sk_...`.

```powershell
# Unauthenticated liveness
curl.exe http://127.0.0.1:8400/v1/health

# Create a project & stream-upload a log file
curl.exe -X POST http://127.0.0.1:8400/v1/projects `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{"name":"backend-logs","description":"Production crash diagnostics"}'

curl.exe -X POST "http://127.0.0.1:8400/v1/files/stream?project_id=prj_..." `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "X-LAI-Filename: app.log" `
  --data-binary "@app.log"

# OpenAI-compatible chat completions
curl.exe -X POST http://127.0.0.1:8400/v1/chat/completions `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "Content-Type: application/json" `
  -H "X-LAI-Project-Id: prj_..." `
  -H "X-LAI-Context-Mode: auto" `
  -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Explain top errors"}],"stream":false}'
```

When ShieldGemma blocks a prompt in `refusal_message` mode, the response sets
`X-LAI-Guardrail-Blocked: true` and `X-LAI-Guardrail-Policy: <policy>`.

## 24-Hour Rotating Logs & Operational Alerts

LAI rotates structured logs daily at UTC midnight (`data/logs/`):

- `lai.jsonl` — All structured application events (14-day gzip-compressed retention).
- `errors.jsonl` — `ERROR` and `CRITICAL` events with tracebacks (30-day retention).
- `prompts_audit.jsonl` — SHA-256 prompt fingerprints, scrubbed text, and ShieldGemma verdicts
  (30-day retention).
- `alerts.jsonl` — Emitted `AlertEvent` records (`fatal_exception`, `timeout`, `subprocess_crash`,
  `storage_failure`, `security_alert`).
- `llama-server.log` / `shieldgemma-server.log` — Supervised subprocess outputs.

Set `logging.alert_webhook_url` in `config/config.yaml` for real-time webhook notifications (60s
cooldown deduplication), or inspect alerts via `GET /v1/system/alerts` (`admin` scope).

## Verification

Run the full verification suite (lint, 350-line budget, `basedpyright` + `mypy`, `import-linter`
boundaries, `>= 90%` branch coverage, OpenAPI/docs sync, and critical-tier `pytest-gremlins`
mutation testing):

```powershell
uv run python scripts/verify.py all --tier critical --cov-fail-under 90
```

## Security, Contributing & License

- **Security Policy**: [SECURITY.md](SECURITY.md) — private vulnerability disclosure via
  [GitHub Security Advisories](https://github.com/blake-olinger/lai/security/advisories/new) or
  `hello@blakeolinger.dev`.
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow, quality gates,
  and OpenAPI/docs synchronization.
- **License**: [Apache License, Version 2.0](LICENSE) (`Copyright 2026 Blake Olinger`).
