# Local AI Platform (LAI)

[![CI](https://github.com/blake-olinger/lai/actions/workflows/ci.yml/badge.svg)](https://github.com/blake-olinger/lai/actions/workflows/ci.yml)
[![Security](https://github.com/blake-olinger/lai/actions/workflows/security.yml/badge.svg)](https://github.com/blake-olinger/lai/actions/workflows/security.yml)
[![Docs](https://github.com/blake-olinger/lai/actions/workflows/docs.yml/badge.svg)](https://blake-olinger.github.io/lai/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Linux-first, multi-tenant, security-hardened local AI server for isolated project workspaces, large
log and source-code analysis, bidirectional ingress/egress safety guardrails, guarded tool
execution, verified atomic self-upgrades, and OpenAI-compatible remote access behind Cloudflare
Named Tunnels or Nginx / Azure Application Gateway.

## Key Capabilities

- **Pure Linux `llama.cpp` Runtime & Model Integrity**: Supervises CUDA-accelerated `llama-server`
  (Gemma 4 on `:8081`) and CPU-only ShieldGemma-2B (`:8082`) with Linux `PR_SET_PDEATHSIG` orphan
  protection, pinned GitHub/HuggingFace revisions, constant-time SHA-256 startup verification, and
  boot-only `model_integrity.strict_allowlist` enforcement.
- **Bidirectional Safety Guardrails (Ingress, Tool Output & Buffered Egress)**: Evaluates session
  `system_prompt`, user messages, client tool outputs (`direction=tool_output`), and model
  thinking/content (`output_mode=buffered` or `audit_only`) on CPU with calibrated `n_probs`
  scoring, boot-locked `allow_runtime_disable: false` protection, scrubbed JSONL auditing, and
  `X-LAI-Guardrail-Blocked` / `X-LAI-Guardrail-Policy` / `X-LAI-Guardrail-Direction` headers.
- **Isolated Project Workspaces & Large Log/Code Analysis**: HTTP-upload-only file ingestion
  (no server filesystem access), CPU FastEmbed + LanceDB + SQLite FTS5 hybrid RRF retrieval, and
  adaptive context routing (`direct`, `retrieval`, or `map_reduce`) with strict per-user ownership
  isolation (`404` anti-enumeration) and POSIX `0700`/`0600` storage permissions.
- **OpenAI-Compatible API & Streaming**: Drop-in `/v1/models` and `/v1/chat/completions` endpoints
  plus native project/session routes, SSE streaming, and proxy keepalives to prevent edge timeouts.
- **Enterprise Multi-Tenant Auth (API Keys, Google OIDC & Microsoft Entra ID)**: Supports multiple
  web and desktop OAuth client IDs with zero static client secrets, configurable identity claim
  precedence (`email`, `preferred_username`, `upn`), `azp`/`appid`/`aud` binding, and per-user
  project/storage/queue quotas.
- **Observability, Guardrail Analytics, Live Hot-Swap & Atomic Upgrades**: 24-hour rotating JSONL
  logs, outbound SMTP email & webhook alerts, hourly endpoint/token/guardrail efficacy analytics
  (`GET /v1/system/analytics`), zero-restart per-section config hot-swapping (`/v1/system/config/*`),
  and SHA-256-verified atomic release slot upgrades with SQLite WAL backup and automatic rollback
  (`lai upgrade`).

## Prerequisites & Installation (Linux / Ubuntu 24.04 or WSL2 Ubuntu)

- **Ubuntu 24.04 LTS** (bare-metal, Azure GPU VM, or WSL2 Ubuntu for local development)
- **[`mise`](https://mise.jdx.dev/)** (single source of truth in [`mise.toml`](mise.toml) for Python
  `3.13`, `uv`, and `gitleaks`)
- **Docker** (used for identical local and CI verification via `scripts/run-in-container.sh`)
- **NVIDIA GPU + CUDA drivers** (for local `llama.cpp` GPU inference; `lai models setup`
  auto-selects CUDA `13.5` or `12.8`)

Install toolchain and locked dependencies:

```bash
mise trust && mise install
mise run setup
mise run hooks
```

Copy the example configuration to customize settings:

```bash
cp config/config.example.yaml config/config.yaml
```

## Quickstart

1. **Initialize local storage, permissions (`0700`/`0600`), and SQLite database**:

   ```bash
   uv run lai init --config config/config.yaml
   ```

2. **Create an admin API key** (copy the one-time `lai_sk_...` secret):

   ```bash
   uv run lai keys create --config config/config.yaml --name "local-admin" --admin
   uv run lai keys list --config config/config.yaml
   ```

3. **Download verified Linux `llama.cpp` CUDA binaries, GGUF models, and CPU FastEmbed ONNX cache**:

   ```bash
   uv run lai models setup --config config/config.yaml
   ```

   `lai models setup` verifies every `llama.cpp` release tarball and HuggingFace GGUF against pinned
   SHA-256 digests, atomically activates `bin/llama/current/llama-server`, and pre-warms FastEmbed
   into `models/fastembed` (pass `--skip-embeddings` to opt out).

4. **Start the server** (Gemma 4 `:8081`, CPU ShieldGemma-2B `:8082`, FastAPI `:8400`):

   ```bash
   uv run lai serve --config config/config.yaml
   ```

5. **Verify with a quick API check**:

   ```bash
   # Unauthenticated liveness probe
   curl -fsS http://127.0.0.1:8400/v1/health

   # Create a project and stream-upload a log file
   curl -fsS -X POST http://127.0.0.1:8400/v1/projects \
     -H "Authorization: Bearer ${LAI_API_KEY}" \
     -H "Content-Type: application/json" \
     -d '{"name":"backend-logs","description":"Crash diagnostics"}'

   curl -fsS -X POST "http://127.0.0.1:8400/v1/files/stream?project_id=prj_..." \
     -H "Authorization: Bearer ${LAI_API_KEY}" \
     -H "X-LAI-Filename: app.log" \
     --data-binary "@app.log"

   # OpenAI-compatible chat completion
   curl -fsS -X POST http://127.0.0.1:8400/v1/chat/completions \
     -H "Authorization: Bearer ${LAI_API_KEY}" \
     -H "Content-Type: application/json" \
     -H "X-LAI-Project-Id: prj_..." \
     -H "X-LAI-Context-Mode: auto" \
     -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Summarize the top errors"}],"stream":false}'
   ```

   When guardrails block an ingress prompt or buffered model response, the response includes
   `X-LAI-Guardrail-Blocked: true`, `X-LAI-Guardrail-Policy: <policy>`, and
   `X-LAI-Guardrail-Direction: ingress|egress`.

## Authentication & Multi-Tenancy

All protected endpoints accept `Authorization: Bearer <oidc_jwt>` (Google OIDC or Microsoft Entra ID
v2.0) or `Authorization: Bearer lai_sk_...`.

```yaml
auth_providers:
  - "api_key"
  - "oidc"
oidc_enabled: true
# Example: Microsoft Entra ID v2.0 (see docs/deployment.md for Google OIDC too)
oidc_issuer_url: "https://login.microsoftonline.com/<TENANT_ID>/v2.0"
oidc_jwks_url: "https://login.microsoftonline.com/<TENANT_ID>/discovery/v2.0/keys"
oidc_audiences:
  - "<ENTRA_API_APP_CLIENT_ID>"
oidc_identity_claims: ["email", "preferred_username", "upn"]
oidc_require_email_verified: false
oidc_allowed_domains:
  - "corp.example.com"
oidc_admin_emails:
  - "ai-admin@corp.example.com"
```

Startup validation fails closed:

- OIDC requires at least one audience in `oidc_audiences`, non-empty `oidc_identity_claims`, and at
  least one entry in `oidc_allowed_emails` or `oidc_allowed_domains`.
- Public consumer email domains (`gmail.com`, `yahoo.com`, `outlook.com`, `icloud.com`,
  `proton.me`, etc.) are rejected in `oidc_allowed_domains` — list individual accounts in
  `oidc_allowed_emails` instead.
- Register secret-free OAuth client apps for client-executed tools via `lai apps create` — see
  [Client App & Project Lifecycle](docs/client-apps-and-projects.md).

## Deployment Topologies & Self-Upgrades

LAI ships production-hardened Linux systemd units (`deploy/systemd/`), Cloudflare Named Tunnel
overlays (`deploy/cloudflare/`), and Azure Ubuntu VM + Nginx + Entra ID overlays (`deploy/azure/`).

- **Dev (WSL2 Ubuntu)**: Local development and identical containerized CI verification via
  `bash scripts/run-in-container.sh pre-commit` and `bash scripts/run-in-container.sh pre-push`.
- **Test (Headless Ubuntu + Cloudflare Named Tunnel)**: Systemd-managed `lai.service` +
  `cloudflared.service` with `trust_cloudflare_headers: true` and `trusted_proxy_cidrs`.
- **Prod (Azure Ubuntu VM + Nginx / Application Gateway + Entra ID)**: FHS layout under `/opt/lai`,
  `/etc/lai`, `/var/lib/lai`, `/var/log/lai` provisioned by `deploy/systemd/install-ubuntu.sh`,
  boot-locked guardrails, `model_integrity.strict_allowlist: true`, and `lai upgrade` atomic slot
  deployments with WAL-safe SQLite backup and automatic health-checked rollback.

See the [Deployment & Operations Guide](docs/deployment.md) for step-by-step instructions across all
three environments.

## Documentation

- **Interactive API Reference**: [https://blake-olinger.github.io/lai/](https://blake-olinger.github.io/lai/)
- **Markdown API Reference**: [docs/api-reference.md](docs/api-reference.md)
- **OpenAPI 3.1 Schema**: [openapi.json](openapi.json)
- **Deployment & Operations Guide**: [docs/deployment.md](docs/deployment.md)
- **Client App & Project Lifecycle**: [docs/client-apps-and-projects.md](docs/client-apps-and-projects.md)
- **Security Policy**: [SECURITY.md](SECURITY.md)
- **Releases & Release Notes**: [GitHub Releases](https://github.com/blake-olinger/lai/releases)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Security, Contributing & License

- **Security**: Report vulnerabilities privately via
  [GitHub Security Advisories](https://github.com/blake-olinger/lai/security/advisories/new) or
  `hello@blakeolinger.dev` — see [SECURITY.md](SECURITY.md).
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture rules, container-backed
  git hooks, verification gates, and pull request requirements.
- **License**: [Apache License, Version 2.0](LICENSE) (`Copyright 2026 Blake Olinger`).
