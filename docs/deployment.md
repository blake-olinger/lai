# Deployment & Operations Guide

This guide walks through configuring LAI for remote production access using **Google Cloud OIDC**
and a **Cloudflare Named Tunnel**.

---

## Host & Runtime Prerequisites

- **Python 3.13** (`>=3.13,<3.14`) and **[`uv`](https://docs.astral.sh/uv/)**
- **GPU inference runtime**:
  - **Windows + NVIDIA GPU (`llamacpp`)**: `uv run python -m lai.cli models setup` downloads CUDA
    `llama-server.exe` binaries and GGUF models automatically.
  - **Linux or Ollama (`ollama`)**: Install Ollama and run
    `uv run python -m lai.cli models setup --backend ollama`, or install `llama-server` on `PATH`
    and run `uv run python -m lai.cli models setup --skip-binaries`.
- **Single-process server**: Always run LAI as a **single worker process** (`lai serve` runs a
  single Uvicorn worker). The GPU inference spooler, per-session serialization gate, and rate
  limiters operate in-process.
- **NTP time synchronization**: Ensure the host clock is synchronized via NTP. Google OIDC JWT
  validation enforces `exp` strictly without clock-skew leeway.

---

## Google Cloud OIDC Setup

LAI verifies Google-signed RS256/ES256 ID tokens directly against Google's public JWKS endpoint
(`https://www.googleapis.com/oauth2/v3/certs`). LAI never needs or stores a Google OAuth client
secret.

### 1. Configure the Google Auth Platform / OAuth Consent Screen

1. Open [Google Cloud Console](https://console.cloud.google.com/) and select or create a project.
2. Navigate to **Google Auth Platform** (or **APIs & Services → OAuth consent screen**).
3. Configure **Branding** (app name, support email, authorized domains) and **Audience**:
   - **Internal** (Google Workspace organizations): restricts sign-in to users in your Workspace
     domain automatically.
   - **External** (personal accounts or mixed users): start in **Testing** mode and add explicit
     **Test users** (up to 100), or publish to **In production** once ready. Only basic OIDC scopes
     (`openid`, `email`, `profile`) are needed, which do not require restricted-scope verification.
4. Under **Data Access / Scopes**, ensure `openid`, `email`, and `profile` are enabled (LAI
   requires verified `email` and `email_verified: true` claims).

### 2. Create OAuth 2.0 Client IDs

Create a separate Client ID for each client application type under **Clients → Create client** (or
**APIs & Services → Credentials → Create Credentials → OAuth client ID**):

#### Browser SPA (Web Application)

- **Application type**: **Web application**
- **Authorized JavaScript origins**: Add your exact SPA origin(s), e.g. `https://app.example.com`
  (and `http://localhost:5173` during local development — remove localhost in production).
- **Authorized redirect URIs**: Add your callback URI if using a redirect flow, or leave empty if
  using Google Identity Services (GIS) popup/One Tap ID tokens directly.
- Copy the **Client ID** (`<id>.apps.googleusercontent.com`). Do **not** embed any client secret in
  your SPA bundle; use Google Identity Services (`accounts.google.com/gsi/client`) to obtain a
  signed Google **ID token** (`credential`) and send it to LAI as
  `Authorization: Bearer <id_token>`.

#### Desktop / Native Client (Electron, CLI, Desktop App)

- **Application type**: **Desktop app**
- Use standard loopback OAuth 2.0 Authorization Code + PKCE (`http://127.0.0.1:<port>`) with
  `scope=openid email profile` (plus `access_type=offline` if refreshing tokens) to obtain a Google
  ID token (`id_token`).
- Copy the **Client ID** (`<id>.apps.googleusercontent.com`). *(Note: Google still displays a
  non-confidential client secret for Installed/Desktop apps during Google's `/token` exchange per
  Google's installed-app spec, but LAI never requests, accepts, or stores that secret.)*

### 3. Configure OIDC in `config/config.yaml`

```yaml
auth_providers:
  - "api_key"
  - "oidc"

oidc_enabled: true
oidc_issuer_url: "https://accounts.google.com"
oidc_jwks_url: "https://www.googleapis.com/oauth2/v3/certs"

# List every Google OAuth Client ID permitted to call LAI
oidc_audiences:
  - "1234567890-webapp.apps.googleusercontent.com"
  - "1234567890-desktop.apps.googleusercontent.com"

# Explicit allowlist (at least one of oidc_allowed_emails or oidc_allowed_domains is required)
# Public consumer domains (gmail.com, yahoo.com, outlook.com, etc.) are rejected in oidc_allowed_domains.
oidc_allowed_emails:
  - "alice@gmail.com"
oidc_allowed_domains:
  - "yourcompany.com"

# Emails promoted to full admin scope
oidc_admin_emails:
  - "owner@yourcompany.com"
```

> **Token lifetime:** Google ID tokens expire after 1 hour. Clients should silently refresh ID
> tokens before expiry and retry on `401 Unauthorized`.

If a client application uses **client-side tools**, also register its OAuth Client ID with
`lai apps create` (see [Client App & Project Lifecycle](client-apps-and-projects.md)).

---

## Cloudflare Named Tunnel Setup

A Cloudflare Named Tunnel exposes `http://127.0.0.1:8400` over HTTPS at your custom hostname
without opening any inbound firewall ports on your machine or router.

### 1. Prerequisites in Cloudflare

- A domain managed on Cloudflare DNS (active zone).
- [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
  installed on the same host running LAI (recommended so traffic arrives at LAI over loopback).

### 2. Authenticate, Create the Tunnel, and Bind DNS

```powershell
# Opens a browser window to authorize cloudflared for your domain zone
cloudflared tunnel login

# Create a named tunnel and note the returned <TUNNEL_UUID>
cloudflared tunnel create lai-local

# Create a proxied CNAME record pointing your API hostname to the tunnel
cloudflared tunnel route dns lai-local ai.example.com
```

Verify in the Cloudflare Dashboard under **DNS → Records** that a proxied (`Orange Cloud`) `CNAME`
exists for `ai.example.com` targeting `<TUNNEL_UUID>.cfargotunnel.com`.

### 3. Render `deploy/cloudflared/config.yml`

```powershell
# Windows: pass an explicit absolute --credentials-file path (cloudflared on Windows may not expand ~)
uv run python -m lai.cli tunnel render-config `
  --tunnel "<TUNNEL_UUID>" `
  --hostname "ai.example.com" `
  --credentials-file "$env:USERPROFILE\.cloudflared\<TUNNEL_UUID>.json" `
  --output deploy/cloudflared/config.yml
```

On Linux/macOS, `--credentials-file "$HOME/.cloudflared/<TUNNEL_UUID>.json"` works the same way.
`deploy/cloudflared/config.yml` is gitignored so your tunnel UUID and local paths are never
committed.

### 4. Choose Cloudflare Access Mode (`cf_access_enabled`)

| Deployment Type | Recommended `cf_access_enabled` | Why |
| --- | --- | --- |
| Public Browser SPA / Desktop App using Google OIDC bearer tokens | `false` | Avoids Cloudflare Access login redirects breaking browser CORS preflight (`OPTIONS`) and `Authorization: Bearer` calls. LAI validates Google OIDC directly. |
| Private internal / machine-to-machine deployment behind Cloudflare Zero Trust Access | `true` | Adds an extra Cloudflare Access edge check requiring a valid `Cf-Access-Jwt-Assertion` (`cf_access_team_domain` + `cf_access_aud`) or Service Token (`cf_access_client_id` + `cf_access_client_secret`). |

> If you previously configured a Cloudflare Zero Trust Access Application covering `ai.example.com`
> (or a wildcard `*.example.com`) and are running a public SPA with `cf_access_enabled: false`,
> make sure `ai.example.com` is excluded or has a bypass policy so Access does not intercept CORS
> preflights.

### 5. Configure Tunnel Trust & CORS in `config/config.yaml`

```yaml
host: "127.0.0.1"
port: 8400

# Exact browser origins allowed to call the API (no "*", no trailing "/")
cors_allowed_origins:
  - "https://app.example.com"

# Trust CF-Connecting-IP for per-IP rate limiting (only honored from loopback 127.0.0.1 / ::1)
trust_cloudflare_headers: true

# Keep false for browser SPA / desktop Google OIDC deployments
cf_access_enabled: false
```

> **Loopback requirement:** LAI only honors `CF-Connecting-IP` when `trust_cloudflare_headers: true`
> **and** the TCP connection arrives from `127.0.0.1` or `::1`. Run `cloudflared` on the same host
> as LAI so per-IP rate limiting tracks real client IPs instead of a shared proxy IP.

### 6. Cloudflare Network Limits & Streaming Notes

- **100 MB request body limit:** Cloudflare Free and Pro plans reject individual HTTP requests
  larger than 100 MB (`HTTP 413`). Keep remote file uploads under 100 MB per request (or lower
  `max_upload_bytes` to `104857600` in `config/config.yaml` to match).
- **Origin timeout (`HTTP 524`):** Cloudflare closes idle origin connections after 100 seconds. LAI
  automatically sends SSE keepalive frames (every `keepalive_interval_seconds`, default `10s`) and
  chunked whitespace heartbeats for slow non-streaming JSON turns (after
  `json_keepalive_threshold_seconds`, default `20s`) to keep tunnel connections alive.
- **SSL/TLS mode:** In Cloudflare Dashboard (**SSL/TLS → Overview**), set encryption mode to
  **Full** or **Full (strict)** and enable **Always Use HTTPS** so browser clients never downgrade
  to plaintext HTTP.

### 7. Launch LAI and the Tunnel

Run in two terminals:

```powershell
# Terminal 1: Start LAI (inference backend, CPU guardrail, FastAPI on 127.0.0.1:8400)
uv run python -m lai.cli serve --config config/config.yaml

# Terminal 2: Start Cloudflare Named Tunnel
cloudflared tunnel --config deploy/cloudflared/config.yml run
```

Or on Windows, launch and supervise both together with `scripts/start-lai.ps1`:

```powershell
.\scripts\start-lai.ps1 `
  -ConfigPath config\config.yaml `
  -CloudflaredConfig deploy\cloudflared\config.yml
```

---

## Production Configuration Checklist

Before opening traffic, verify these settings in `config/config.yaml`:

- [ ] `host: "127.0.0.1"` (do not bind `0.0.0.0` when serving through a local Cloudflare Tunnel)
- [ ] `guardrails.enabled: true` and `guardrails.fail_open: false`
- [ ] `oidc_enabled: true`, `auth_providers: ["api_key", "oidc"]`, explicit `oidc_audiences`, and
  explicit `oidc_allowed_emails` / corporate `oidc_allowed_domains`
- [ ] `cors_allowed_origins` lists only production `https://` origins (no `localhost`, no `*`)
- [ ] `trust_cloudflare_headers: true` (with `cloudflared` running on loopback)
- [ ] `cf_access_enabled: false` (for SPA/desktop OIDC) or `true` with team domain + AUD / service
  token credentials (for Zero Trust Access)
- [ ] Quotas & rate limits reviewed (`max_projects_per_principal`,
  `max_storage_bytes_per_principal`, `max_queued_jobs_per_principal`, `rate_limit_rpm`,
  `ip_rate_limit_rpm`)
- [ ] `logging.enable_file_logs: true` and optional `logging.alert_webhook_url` configured

---

## Post-Deployment Verification

```powershell
# 1. Public liveness probe over HTTPS
curl.exe -i https://ai.example.com/v1/health

# 2. Confirm runtime API docs are disabled (expect 404)
curl.exe -i https://ai.example.com/docs

# 3. Verify CORS preflight for your SPA origin (expect 200 + Access-Control-Allow-Origin)
curl.exe -i -X OPTIONS https://ai.example.com/v1/projects `
  -H "Origin: https://app.example.com" `
  -H "Access-Control-Request-Method: POST" `
  -H "Access-Control-Request-Headers: Authorization, Content-Type"

# 4. Verify unauthenticated requests are rejected (expect 401)
curl.exe -i https://ai.example.com/v1/projects

# 5. Verify admin health telemetry (GPU, inference backend, guardrail backend)
curl.exe -i https://ai.example.com/v1/system/status `
  -H "Authorization: Bearer $env:LAI_ADMIN_KEY"
```
