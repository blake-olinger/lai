# Local AI Platform (LAI)

[![CI](https://github.com/blake-olinger/lai/actions/workflows/ci.yml/badge.svg)](https://github.com/blake-olinger/lai/actions/workflows/ci.yml)
[![Security](https://github.com/blake-olinger/lai/actions/workflows/security.yml/badge.svg)](https://github.com/blake-olinger/lai/actions/workflows/security.yml)
[![Docs](https://github.com/blake-olinger/lai/actions/workflows/docs.yml/badge.svg)](https://blake-olinger.github.io/lai/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Local, multi-tenant, security-hardened AI server for isolated project workspaces, large log and
source-code analysis, pre-flight prompt safety guardrails, guarded tool execution, and
OpenAI-compatible remote access over Cloudflare Named Tunnels.

## Key Capabilities

- **Isolated Project Workspaces**: Organize uploaded logs, source files, chat sessions, and search
  indexes per project, with strict per-user ownership isolation and `404` anti-enumeration.
- **Large Log & Code Analysis**: Stream-upload files over HTTP only (no server filesystem access),
  automatically indexed for hybrid full-text and semantic retrieval with adaptive context routing
  (direct, retrieval, or map-reduce for oversized inputs).
- **Pre-Flight Safety Guardrails**: Evaluates prompts on CPU with ShieldGemma-2B before requests
  reach the GPU, with scrubbed prompt auditing and configurable refusal or error responses.
- **OpenAI-Compatible API & Streaming**: Drop-in `/v1/models` and `/v1/chat/completions` endpoints
  plus native project/session routes, SSE streaming, and proxy keepalives to prevent tunnel
  timeouts.
- **Guarded Tool Execution**: Read-only analysis over uploaded project files on the server, plus
  optional client-executed tools gated by permission tiers (`read`, `write`, `exec`, `network`) and
  explicit human approvals.
- **Multi-Tenant Security & Quotas**: Authenticate via API keys or Google OIDC (supporting multiple
  web and desktop clients with zero static client secrets), backed by per-user project, storage,
  and queue quotas, fair scheduling, and rate limiting.
- **Remote Access & Observability**: Expose your local server safely over HTTPS with a Cloudflare
  Named Tunnel, strict CORS allowlisting, 24-hour rotating structured logs, and webhook alerts.

## Prerequisites & Installation

- **Python 3.13** (`>=3.13,<3.14`)
- **[`uv`](https://docs.astral.sh/uv/)** for dependency management
- **NVIDIA GPU with CUDA drivers** (for `llamacpp` local GPU inference) or **Ollama**

Install dependencies:

```powershell
uv python install 3.13
uv sync --locked
```

Copy the example configuration to customize settings:

```powershell
Copy-Item config/config.example.yaml config/config.yaml
```

## Quickstart

1. **Initialize local storage and database** (`data/`, `models/`, `bin/`):

   ```powershell
   uv run python -m lai.cli init
   ```

2. **Create an admin API key** (copy the one-time `lai_sk_...` secret):

   ```powershell
   uv run python -m lai.cli keys create --name "local-admin" --admin
   uv run python -m lai.cli keys list
   ```

3. **Download models and inference runtime**:

   ```powershell
   uv run python -m lai.cli models setup
   ```

   > **Note:** Automated `llama.cpp` binary downloads target Windows CUDA x64. On Linux, either use
   > `inference_backend: "ollama"` (`lai models setup --backend ollama`) or install `llama-server`
   > on your `PATH` and run `lai models setup --skip-binaries`.

4. **Start the server** (inference backend, CPU guardrail, and API server on `127.0.0.1:8400`):

   ```powershell
   uv run python -m lai.cli serve --config config/config.yaml
   ```

   On Windows, you can also use the included launcher script (which optionally supervises
   `cloudflared` alongside LAI):

   ```powershell
   .\scripts\start-lai.ps1 -ConfigPath config\config.yaml
   ```

5. **Verify with a quick API check**:

   ```powershell
   # Unauthenticated liveness probe
   curl.exe http://127.0.0.1:8400/v1/health

   # Create a project and stream-upload a log file
   curl.exe -X POST http://127.0.0.1:8400/v1/projects `
     -H "Authorization: Bearer $env:LAI_API_KEY" `
     -H "Content-Type: application/json" `
     -d '{"name":"backend-logs","description":"Crash diagnostics"}'

   curl.exe -X POST "http://127.0.0.1:8400/v1/files/stream?project_id=prj_..." `
     -H "Authorization: Bearer $env:LAI_API_KEY" `
     -H "X-LAI-Filename: app.log" `
     --data-binary "@app.log"

   # OpenAI-compatible chat completion
   curl.exe -X POST http://127.0.0.1:8400/v1/chat/completions `
     -H "Authorization: Bearer $env:LAI_API_KEY" `
     -H "Content-Type: application/json" `
     -H "X-LAI-Project-Id: prj_..." `
     -H "X-LAI-Context-Mode: auto" `
     -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Summarize the top errors"}],"stream":false}'
   ```

   When the guardrail blocks a prompt in `refusal_message` mode, the response includes
   `X-LAI-Guardrail-Blocked: true` and `X-LAI-Guardrail-Policy: <policy>`.

## Authentication & Multi-Tenancy

All protected endpoints accept `Authorization: Bearer <google_id_token>` or
`Authorization: Bearer lai_sk_...`.

### Google OIDC & Secret-Free Public Clients

LAI supports multiple Google OAuth 2.0 client IDs (such as a browser SPA and a desktop app) at the
same time via `oidc_audiences`:

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

Startup validation fails closed:

- OIDC requires at least one audience in `oidc_audiences` and at least one entry in
  `oidc_allowed_emails` or `oidc_allowed_domains`.
- Public consumer email domains (`gmail.com`, `yahoo.com`, `outlook.com`, `icloud.com`,
  `proton.me`, etc.) are rejected in `oidc_allowed_domains` — list individual accounts in
  `oidc_allowed_emails` instead.

Because public SPAs and desktop binaries cannot safely embed static client secrets, LAI issues and
accepts **zero static client secrets**. Client applications are bound cryptographically to their
verified Google OAuth Client ID and registered with:

```powershell
uv run python -m lai.cli apps create `
  --name "lai-web-spa" `
  --oauth-client-id "1234567890-webapp.apps.googleusercontent.com" `
  --tier read
```

See [Client App & Project Lifecycle](docs/client-apps-and-projects.md) for onboarding, updating,
rotating, and decommissioning client apps and projects.

### Isolation & Quotas

- **Owner Isolation**: Projects, uploaded files, sessions, messages, and tool approvals are scoped
  to their owner. Non-admin access attempts against another user's resources return
  `404 Not Found` to prevent resource ID enumeration.
- **Per-User Quotas** (configurable in `config/config.yaml`, non-admin defaults):
  - `max_projects_per_principal`: `10`
  - `max_storage_bytes_per_principal`: `1073741824` (1 GiB)
  - `max_queued_jobs_per_principal`: `4`
- **Fair Scheduling & Rate Limits**: Inference jobs are scheduled fairly across users and sessions,
  with configurable per-IP (`ip_rate_limit_rpm`) and per-user (`rate_limit_rpm`) limits.

## Remote Access (Cloudflare Named Tunnel & CORS)

Expose your loopback server (`127.0.0.1:8400`) over HTTPS without opening inbound firewall ports:

```powershell
cloudflared tunnel login
cloudflared tunnel create lai-local
cloudflared tunnel route dns lai-local ai.example.com

uv run python -m lai.cli tunnel render-config `
  --tunnel "<TUNNEL_UUID>" `
  --hostname "ai.example.com" `
  --credentials-file "$env:USERPROFILE\.cloudflared\<TUNNEL_UUID>.json" `
  --output deploy/cloudflared/config.yml

cloudflared tunnel --config deploy/cloudflared/config.yml run
```

Configure browser origins and proxy trust in `config/config.yaml`:

```yaml
cors_allowed_origins:
  - "https://app.example.com"
trust_cloudflare_headers: true
cf_access_enabled: false
```

- **Exact CORS origins**: Wildcard `"*"` and trailing slashes are rejected at startup.
- **`trust_cloudflare_headers: true`**: Enable only when traffic reaches LAI exclusively via
  Cloudflare on loopback so client IP headers cannot be spoofed.
- **`cf_access_enabled: false`**: Recommended for browser SPAs and desktop clients authenticating
  with Google OIDC bearer tokens so Cloudflare Access redirects do not block CORS preflights.
- **100 MB Tunnel Upload Limit**: Cloudflare Free and Pro plans cap HTTP request bodies at 100 MB
  (`HTTP 413`).

See the [Deployment & Operations Guide](docs/deployment.md) for the full Google Cloud OIDC and
Cloudflare setup walkthrough.

## Logging & Operational Alerts

Structured JSONL logs rotate daily at UTC midnight under `data/logs/`:

- `lai.jsonl` — Application events (14-day compressed retention by default).
- `errors.jsonl` — Errors and tracebacks (30-day retention).
- `prompts_audit.jsonl` — Scrubbed prompt audit records and guardrail verdicts (30-day retention).
- `alerts.jsonl` — Fatal error, timeout, crash, storage, and security alerts.

Configure `logging.alert_webhook_url` for real-time webhook notifications (with cooldown
deduplication) or inspect recent alerts via `GET /v1/system/alerts` (`admin` scope).

## Documentation

- **Interactive API Reference**: [https://blake-olinger.github.io/lai/](https://blake-olinger.github.io/lai/)
- **Markdown API Reference**: [docs/api-reference.md](docs/api-reference.md)
- **OpenAPI 3.1 Schema**: [openapi.json](openapi.json)
- **Deployment & Operations Guide**: [docs/deployment.md](docs/deployment.md)
- **Client App & Project Lifecycle**: [docs/client-apps-and-projects.md](docs/client-apps-and-projects.md)
- **Security Policy**: [SECURITY.md](SECURITY.md)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Security, Contributing & License

- **Security**: Report vulnerabilities privately via
  [GitHub Security Advisories](https://github.com/blake-olinger/lai/security/advisories/new) or
  `hello@blakeolinger.dev` — see [SECURITY.md](SECURITY.md).
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture rules, verification
  gates, and pull request requirements.
- **License**: [Apache License, Version 2.0](LICENSE) (`Copyright 2026 Blake Olinger`).
