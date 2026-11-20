# Deployment & Operations Guide

LAI targets **Ubuntu 24.04 LTS (x86_64)** with pure `llama.cpp` GPU inference (Gemma 4 on `:8081`),
CPU-only ShieldGemma-2B ingress/egress guardrails (`:8082`), and CPU FastEmbed ONNX retrieval.
This guide covers all three environments:

1. **Dev (WSL2 Ubuntu)** — local development and identical containerized CI verification.
2. **Test (Headless Ubuntu + Cloudflare Named Tunnel)** — systemd-supervised server exposed over HTTPS.
3. **Prod (Azure Ubuntu VM + Nginx / Application Gateway + Microsoft Entra ID)** — FHS enterprise deployment.

---

## Runtime Prerequisites

- **OS & Architecture**: Ubuntu 24.04 LTS x86_64 (or WSL2 Ubuntu for development).
- **NVIDIA GPU & CUDA drivers**: `lai models setup` auto-detects driver CUDA capability via
  `nvidia-smi` and installs SHA-256-verified Linux x64 `llama.cpp` CUDA `13.5` or `12.8` binaries
  into `bin/llama/releases/<tag>` with an atomic `bin/llama/current` symlink.
- **Single-process server**: Always run LAI as a **single Uvicorn worker** (`lai serve`). The GPU
  inference spooler, per-session serialization gate, and rate limiters operate in-process.
- **NTP time synchronization**: Keep the host clock synced via `systemd-timesyncd` or `chrony`
  (LAI enforces `oidc_leeway_seconds`, default `30.0s`, on JWT `exp`/`nbf` claims).
- **Private storage permissions**: `lai init`, SQLite, and the local blob store enforce POSIX
  `0700` on `data/`, `blobs/`, `lancedb/`, and `backups/`, and `0600` on `config.yaml` and `lai.db`.

---

## 1. Dev Environment (WSL2 Ubuntu)

Use [`mise`](https://mise.jdx.dev/) (`mise.toml` is the single source of truth for Python 3.13,
`uv`, and `gitleaks`) together with native Docker inside WSL2 Ubuntu so local checks match CI
byte-for-byte.

```bash
# Bootstrap toolchain, locked Python environment, and container-backed git hooks
mise trust && mise install
mise run setup
mise run hooks   # sets git config core.hooksPath .githooks

# Build the shared Ubuntu 24.04 validation container (cached across runs)
bash scripts/run-in-container.sh --build-only

# Run fast pre-commit checks or the full CI + security + coverage + mutation suite
bash scripts/run-in-container.sh pre-commit
bash scripts/run-in-container.sh pre-push
```

Run LAI locally in WSL2 with GPU passthrough:

```bash
cp config/config.example.yaml config/config.yaml
uv run lai init --config config/config.yaml
uv run lai keys create --config config/config.yaml --name "dev-admin" --admin
uv run lai models setup --config config/config.yaml
uv run lai serve --config config/config.yaml
```

---

## 2. Test Environment (Headless Ubuntu + Cloudflare Named Tunnel)

Provision FHS paths (`/opt/lai`, `/etc/lai`, `/var/lib/lai`, `/var/log/lai`), unprivileged `lai`
service user, and hardened systemd units with the included Ubuntu installer:

```bash
sudo bash deploy/systemd/install-ubuntu.sh --version <RELEASE_VERSION>
sudo cp deploy/cloudflare/config.test.example.yaml /etc/lai/config.yaml
sudo chown root:lai /etc/lai/config.yaml
sudo chmod 0640 /etc/lai/config.yaml
```

### Google Cloud OIDC Setup (Test Host)

LAI verifies Google RS256/ES256 ID tokens directly against Google's JWKS endpoint
(`https://www.googleapis.com/oauth2/v3/certs`) without storing any OAuth client secret:

1. In [Google Cloud Console](https://console.cloud.google.com/), configure **Google Auth Platform**
   with scopes `openid`, `email`, and `profile`.
2. Create an OAuth 2.0 **Web application** Client ID (for browser SPAs) and/or **Desktop app**
   Client ID (for native clients using Authorization Code + PKCE).
3. Populate `oidc_audiences`, `oidc_allowed_emails` or corporate `oidc_allowed_domains`, and
   `oidc_admin_emails` in `/etc/lai/config.yaml` (consumer domains like `gmail.com` are rejected in
   `oidc_allowed_domains`; list explicit addresses in `oidc_allowed_emails`).

### Cloudflare Named Tunnel Setup

**Option A — Token-managed systemd service (recommended for headless hosts):**

```bash
getent group cloudflared >/dev/null || sudo groupadd --system cloudflared
id -u cloudflared >/dev/null 2>&1 || sudo useradd --system --gid cloudflared --no-create-home --shell /usr/sbin/nologin cloudflared
sudo cp deploy/cloudflare/cloudflared.service /etc/systemd/system/cloudflared.service
sudo install -d -m 0750 -o root -g cloudflared /etc/cloudflared
sudo install -m 0600 -o cloudflared -g cloudflared /dev/null /etc/cloudflared/tunnel.env
echo 'TUNNEL_TOKEN=<YOUR_CLOUDFLARE_TUNNEL_TOKEN>' | sudo tee /etc/cloudflared/tunnel.env >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now lai.service cloudflared.service
```

**Option B — Local credentials file via `lai tunnel render-config`:**

```bash
cloudflared tunnel login
cloudflared tunnel create lai-test
cloudflared tunnel route dns lai-test ai-test.example.com
/opt/lai/current/.venv/bin/lai tunnel render-config \
  --config /etc/lai/config.yaml \
  --tunnel "<TUNNEL_UUID>" \
  --hostname "ai-test.example.com" \
  --output /etc/lai/cloudflared.yaml
```

`lai tunnel render-config` writes an absolute credentials path (never literal `~`) and applies
`chmod 0600` to the rendered file.

Key Cloudflare settings in `/etc/lai/config.yaml`:

- `trust_cloudflare_headers: true` and `trusted_proxy_cidrs: ["127.0.0.1/32", "::1/128"]` (only
  when `cloudflared` forwards exclusively over loopback).
- `cf_access_enabled: false` for public SPA/desktop OIDC clients (so Cloudflare Access redirects do
  not intercept CORS preflights), or `true` with `cf_access_team_domain` + `cf_access_aud` for Zero
  Trust Access.
- `max_upload_bytes: 104857600` (matches Cloudflare Free/Pro 100 MB body cap). LAI's SSE keepalives
  (`keepalive_interval_seconds: 10.0`) and JSON heartbeats (`json_keepalive_threshold_seconds: 20.0`)
  prevent Cloudflare `HTTP 524` idle timeouts.

---

## 3. Prod Environment (Azure Ubuntu VM + Nginx / Application Gateway + Entra ID)

Provision an Azure GPU VM (Ubuntu 24.04 LTS with NVIDIA CUDA drivers) using the FHS installer and
Azure production overlay:

```bash
sudo bash deploy/systemd/install-ubuntu.sh --version <RELEASE_VERSION>
sudo cp deploy/azure/config.azure-prod.example.yaml /etc/lai/config.yaml
sudo cp deploy/azure/nginx-lai.conf.example /etc/nginx/sites-available/lai.conf
sudo ln -sf /etc/nginx/sites-available/lai.conf /etc/nginx/sites-enabled/lai.conf
sudo chown root:lai /etc/lai/config.yaml
sudo chmod 0640 /etc/lai/config.yaml
```

Store all secrets exclusively in `/etc/lai/lai.env` (`0640`, `root:lai`):

```bash
sudo install -m 0640 -o root -g lai /dev/null /etc/lai/lai.env
cat <<'EOF' | sudo tee /etc/lai/lai.env >/dev/null
LAI_LOGGING__ALERT_SMTP__PASSWORD=<smtp-password>
LAI_LOGGING__ALERT_WEBHOOK_URL=<optional-webhook-url>
EOF
```

### Microsoft Entra ID (Azure AD) v2.0 OIDC Configuration

1. Register an API application in **Microsoft Entra ID → App registrations** and note the
   **Directory (tenant) ID** and **Application (client) ID**.
2. Expose an API scope for your web/desktop clients, configure v2.0 access tokens
   (`requestedAccessTokenVersion: 2`), and define an App Role (e.g. `LAI.Admin`) for administrators.
3. Configure `/etc/lai/config.yaml`:

```yaml
auth_providers:
  - "api_key"
  - "oidc"

oidc_enabled: true
oidc_issuer_url: "https://login.microsoftonline.com/<TENANT_ID>/v2.0"
oidc_jwks_url: "https://login.microsoftonline.com/<TENANT_ID>/discovery/v2.0/keys"
oidc_audiences:
  - "<ENTRA_API_APP_CLIENT_ID>"
oidc_allowed_client_ids:
  - "<WEB_SPA_CLIENT_ID>"
  - "<DESKTOP_APP_CLIENT_ID>"
oidc_identity_claims:
  - "email"
  - "preferred_username"
  - "upn"
# Entra ID v2 tokens frequently omit `email_verified`; use `oidc_admin_roles` or
# `oidc_admin_subjects` for admin elevation (unverified email never grants ADMIN).
oidc_require_email_verified: false
oidc_allow_guest_users: false
oidc_allowed_domains:
  - "corp.example.com"
oidc_admin_roles:
  - "LAI.Admin"
oidc_admin_subjects: []
oidc_admin_emails: []
```

LAI binds principal identity to `oidc:<normalized_issuer>:<oid_or_sub>` (stable across SPA and
Desktop client apps via Entra `oid`), verifies token `aud` against `oidc_audiences` and OAuth
caller `azp`/`appid` against `oidc_allowed_client_ids`, rejects Entra B2B guests unless
`oidc_allow_guest_users: true`, and requires `oidc_admin_roles`, `oidc_admin_subjects`, or a
verified email (`email_verified=true` / `xms_edov=true`) for `ApiScope.ADMIN`.

### Reverse Proxy Client IP Trust (`trusted_proxy_cidrs` & `proxy_ip_header`)

Configure reverse-proxy trust explicitly so per-IP rate limiting tracks real clients without
allowing header spoofing:

```yaml
trust_cloudflare_headers: false
trusted_proxy_cidrs:
  - "127.0.0.1/32"
  - "::1/128"
  - "10.0.1.0/24" # Optional Azure Application Gateway VNet subnet
proxy_ip_header: "x-forwarded-for" # or "x-azure-clientip" when fronted directly by Azure App Gateway
```

- When `proxy_ip_header: "x-forwarded-for"`, LAI walks the hop chain from **right to left** and
  selects the rightmost IP not inside `trusted_proxy_cidrs` (defeating forged left-side headers and
  failing closed to the socket peer IP on malformed entries).
- Ensure `deploy/azure/nginx-lai.conf.example` keeps `proxy_buffering off;` and `X-Accel-Buffering: no`
  so SSE streams flush immediately.

Initialize storage, download verified models/binaries, and enable services:

```bash
sudo -u lai /opt/lai/current/.venv/bin/lai init --config /etc/lai/config.yaml
sudo -u lai /opt/lai/current/.venv/bin/lai models setup --config /etc/lai/config.yaml
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl enable --now lai.service
```

---

## Model Supply-Chain Integrity & Guardrail Hardening

- **Boot-Only Model Allowlist (`model_integrity`)**:
  - `strict_allowlist: true` (default) keeps built-in pinned Gemma 4 and ShieldGemma-2B profiles
    working out of the box while rejecting any `custom_profiles` entry lacking a pinned `sha256`,
    not listed in `approved_model_sha256s`, or referencing an `hf_repo` outside `allowed_hf_repos`
    (when non-empty).
  - `verify_sha256_on_startup: true` (default) verifies both the primary GGUF and ShieldGemma GGUF
    with streaming 1 MiB constant-time SHA-256 comparison before spawning `llama-server`, logging a
    `SECURITY_ALERT` and refusing to start on any tamper or corruption.
  - `llama.cpp` release tarballs are verified against pinned digests in `src/lai/inference/manifest.py`
    with Tar-Slip and symlink-escape protection.
- **Bidirectional Boot-Locked Guardrails**:
  - Ingress scans `POST /v1/sessions` `system_prompt`, per-turn `system_prompt`, prior turns, and
    user prompts (`direction=ingress`); client tool outputs are scanned as `direction=tool_output`.
  - Egress (`output_mode: "buffered"` default) buffers thinking and content deltas, evaluates them
    with ShieldGemma-2B `n_probs` calibration before streaming or persisting, and emits
    `X-LAI-Guardrail-Direction: egress` on refusal.
  - `allow_runtime_disable: false` and `max_violation_threshold: 0.95` are **boot-only locks**:
    neither `PATCH /v1/system/config/guardrails`, scoped reloads, nor tampered SQLite rows can
    disable guardrails, enable `fail_open`, set `output_mode: "off"`, or raise `violation_threshold`
    above `max_violation_threshold` (attempts return `422` and emit `AlertCategory.SECURITY_ALERT`).

---

## Atomic Self-Upgrades & Rollback (`lai upgrade`)

LAI supports verified, zero-guesswork wheel slot upgrades under `/opt/lai/releases/v<version>` with
an atomic `/opt/lai/current` symlink swap, WAL-safe online SQLite backup (`PRAGMA integrity_check`
+ POSIX `uid`/`gid` preservation), systemd restart, health verification, and automatic rollback.

```bash
# Check GitHub Releases for a newer verified wheel (reads SHA256SUMS.txt)
sudo /opt/lai/current/.venv/bin/lai upgrade check --config /etc/lai/config.yaml

# Dry-run or apply latest verified release (optionally refreshing pinned llama.cpp binaries)
sudo /opt/lai/current/.venv/bin/lai upgrade apply --config /etc/lai/config.yaml --dry-run
sudo /opt/lai/current/.venv/bin/lai upgrade apply --config /etc/lai/config.yaml --with-binaries

# CI/CD air-gapped push deploy from a staged local wheel + locked hashed requirements
sudo /opt/lai/current/.venv/bin/lai upgrade apply \
  --config /etc/lai/config.yaml \
  --wheel /tmp/lai-0.2.0-py3-none-any.whl \
  --sha256 "<64-hex-wheel-sha256>" \
  --requirements /tmp/requirements-lock.txt \
  --requirements-sha256 "<64-hex-requirements-sha256>"

# Roll back immediately to the previous slot (and optionally restore a pre-upgrade DB backup)
sudo /opt/lai/current/.venv/bin/lai upgrade rollback \
  --config /etc/lai/config.yaml \
  --backup /var/lib/lai/backups/lai-pre-v0.2.0-<timestamp>.db
```

### Opt-In Scheduled Upgrades (`lai-upgrade.timer`)

`deploy/systemd/lai-upgrade.service` runs `lai upgrade apply --config /etc/lai/config.yaml --if-enabled`,
which merges any persisted runtime overrides from SQLite before evaluating `updates.enabled` and
routes upgrade/checksum/health failures through `AlertManager` (SMTP/webhook/file alerts):

```bash
# Enable weekly scheduled checks (remains a safe no-op while updates.enabled is false)
sudo systemctl enable --now lai-upgrade.timer
```

### Dependency Automation (Renovate)

[`renovate.json`](../renovate.json) keeps Python dependencies, `.github/tools/pip-audit`,
`mise.toml`, `docker/ci.Dockerfile`, GitHub Actions, Scalar CDN, `ggml-org/llama.cpp`, and `mise`
fresh with a mandatory **7-day release cooldown** (`minimumReleaseAge: "7 days"`). Checksum-coupled
custom pins disable automerge so operators review SHA-256 digests explicitly.

---

## Operational Alerting, Analytics & Live Configuration Hot-Swapping

### Outbound SMTP & Webhook Alerts

Configure `logging.alert_smtp` in `/etc/lai/config.yaml` and supply `LAI_LOGGING__ALERT_SMTP__PASSWORD`
via `/etc/lai/lai.env`. Verify end-to-end delivery with:

```bash
curl -fsS -X POST https://ai.corp.example.com/v1/system/alerting/test \
  -H "Authorization: Bearer ${LAI_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"message":"Pre-flight SMTP verification"}'
```

### Usage & Guardrail Efficacy Analytics

`GET /v1/system/analytics?window_hours=24` returns privacy-preserving hourly rollups for endpoints,
opaque principals/client apps, tokens, and **guardrail efficacy** (`guardrails.total_evaluations`,
`total_blocked`, `block_rate`, `fail_closed_count`, `by_direction` across `ingress`/`egress`/`tool_output`,
and `top_policies`).

### Zero-Restart Isolated Configuration Hot-Swapping

| Subsystem | Admin Endpoints | Live Editable Settings |
| --- | --- | --- |
| **Alerting** | `GET` / `PATCH /v1/system/alerting`, `POST /v1/system/alerting/test` | SMTP host/port/TLS/recipients/filters, webhook toggle/filters, cooldown, `alert_on_errors`, `alert_on_timeouts` |
| **Analytics** | `GET /v1/system/analytics`, `GET` / `PATCH /v1/system/analytics/config` | `enabled`, `retention_days` (prunes expired buckets immediately) |
| **OIDC / Auth** | `GET` / `PATCH /v1/system/config/oidc` | `oidc_audiences`, `oidc_allowed_client_ids`, `oidc_allowed_emails`, `oidc_allowed_domains`, `oidc_admin_emails`, `oidc_admin_roles`, `oidc_admin_subjects`, `oidc_allow_guest_users`, `oidc_default_scopes`, `oidc_require_email_verified`, `oidc_identity_claims`, `oidc_leeway_seconds` |
| **CORS** | `GET` / `PATCH /v1/system/config/cors` | `cors_allowed_origins`, `cors_extra_allow_headers` |
| **Limits & Proxy** | `GET` / `PATCH /v1/system/config/limits` | `rate_limit_rpm`, `ip_rate_limit_rpm`, quotas, queue bounds, `max_upload_bytes`, keepalives, `trust_cloudflare_headers`, `trusted_proxy_cidrs`, `proxy_ip_header`, `log_level` |
| **Guardrails Policy** | `GET` / `PATCH /v1/system/config/guardrails` | `enabled`, `block_mode`, `output_mode`, `refusal_text`, `violation_threshold`, `fail_open`, `timeout_seconds` (enforcing boot locks!) |
| **Updates Policy** | `GET` / `PATCH /v1/system/config/updates` | `enabled`, `keep_releases`, `auto_rollback`, `allow_prerelease`, `health_timeout_seconds`, `health_poll_interval_seconds` |

Reload specific sections from YAML/env (clearing only their SQLite overrides):

```bash
curl -fsS -X POST https://ai.corp.example.com/v1/system/config/reload \
  -H "Authorization: Bearer ${LAI_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"sections":["alerting","cors","updates"]}'
```

---

## Production Security Checklist & Post-Deployment Verification

- [ ] `host: "127.0.0.1"` (never bind `0.0.0.0` directly behind Nginx or Cloudflare Tunnel)
- [ ] `model_integrity.strict_allowlist: true` and `model_integrity.verify_sha256_on_startup: true`
- [ ] `guardrails.enabled: true`, `guardrails.output_mode: "buffered"`, `guardrails.fail_open: false`,
  `guardrails.allow_runtime_disable: false`
- [ ] `oidc_enabled: true` with explicit `oidc_audiences` and corporate `oidc_allowed_domains` / `oidc_allowed_emails`
- [ ] `cors_allowed_origins` lists only exact production `https://` origins
- [ ] `trusted_proxy_cidrs` and `proxy_ip_header` (or `trust_cloudflare_headers`) match your exact proxy hops
- [ ] Secrets stored strictly in `/etc/lai/lai.env` (`chmod 0640`, `root:lai`), never committed to YAML
- [ ] Outbound SMTP/webhook alerts verified via `POST /v1/system/alerting/test`

```bash
# 1. Public liveness probe & runtime docs disabled (expect 200, then 404)
curl -i https://ai.corp.example.com/v1/health
curl -i https://ai.corp.example.com/docs

# 2. Verify CORS preflight and unauthenticated rejection (expect 200, then 401)
curl -i -X OPTIONS https://ai.corp.example.com/v1/projects \
  -H "Origin: https://app.corp.example.com" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Authorization, Content-Type"
curl -i https://ai.corp.example.com/v1/projects

# 3. Verify GPU, llama-server, ShieldGemma, alerting, and guardrail analytics
curl -fsS https://ai.corp.example.com/v1/system/status \
  -H "Authorization: Bearer ${LAI_ADMIN_KEY}"
curl -fsS "https://ai.corp.example.com/v1/system/analytics?window_hours=1" \
  -H "Authorization: Bearer ${LAI_ADMIN_KEY}"
```
