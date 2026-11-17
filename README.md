# Local AI Platform (LAI)

Local, security-hardened AI platform for multi-project chat, large log and source-code root-cause debugging, pre-flight ShieldGemma prompt guardrails, guarded tool execution, and OpenAI-compatible remote access via Cloudflare Named Tunnel.

## Architecture Overview

LAI follows a strict 6-layer architecture enforced by `import-linter`:

1. **Surface (`lai.api`, `lai.cli`)** — FastAPI HTTP/SSE server (`:8400`) and Typer CLI (`init`, `keys`, `models`, `serve`, `tunnel`).
2. **Orchestration (`lai.orchestrator`)** — Pre-flight guardrail gate at `t=0ms`, session serialization, adaptive context routing (`DIRECT`, `RETRIEVAL`, `MAP_REDUCE`), and bounded tool-call loops.
3. **Subsystems (independent Layer-3 modules)**:
   - `lai.security`: `lai_sk_` SHA-256 API keys, RS256/ES256 OIDC JWT validation, Cloudflare Access edge checks, scope hierarchy, token-bucket rate limiter.
   - `lai.db`: WAL-mode SQLite (`BEGIN IMMEDIATE` writes, foreign keys, FTS5 porter index).
   - `lai.ingestion`: HTTP-stream-only `LocalBlobStore`, log/code/prose chunkers, CPU-only `FastEmbedCpuAdapter` (`BAAI/bge-small-en-v1.5`), project-isolated `LanceDbVectorStore`, and Reciprocal Rank Fusion.
   - `lai.inference`: Supervised CUDA `llama-server` (`:8081`, Gemma 4 26B-A4B MoE with `--jinja`, `--flash-attn on`, `-ctk q8_0`, `-ctv q8_0`, `--n-cpu-moe`) and streaming `<thought>` parser.
   - `lai.guardrails`: CPU-only `ShieldGemma-2B` supervisor (`:8082`, `--ngl 0`) and `CompositeGuardrailPipeline` with scrubbed 24h prompt auditing.
   - `lai.tools`: Policy engine (`READ`, `WRITE`, `EXEC`, `NETWORK`), approval workflow (`PENDING -> APPROVED -> EXECUTED`), and Phase 1 read-only tools (`search_files`, `read_file_lines`, `get_file_overview`).
   - `lai.scheduler`: Per-session mutual exclusion (`SessionGate`), single-slot priority `InferenceSpooler`, and Cloudflare 524 keepalive wrappers.
   - `lai.config`: YAML + `LAI_*` env settings and built-in model profiles.
4. **Observability (`lai.observability`)** — `structlog` JSONL 24h daily rotation, PII/secret scrubbing, and webhook alerting (`AlertManager`).
5. **Ports (`lai.ports`)** — PEP 544 `Protocol` contracts.
6. **Domain (`lai.domain`)** — Pure immutable Pydantic v2 models, prefixed IDs, and `LaiError` hierarchy.

## Installation

Prerequisites: Python **3.13**, [`uv`](https://docs.astral.sh/uv/), and NVIDIA CUDA drivers for local GPU inference.

```powershell
uv python install 3.13
uv sync --locked
```

Copy the example configuration if you want custom settings:

```powershell
Copy-Item config/config.example.yaml config/config.yaml
```

## Quickstart (`init`, `keys`, `models setup`, `serve`)

Initialize local directories (`data/`, `models/`, `bin/`) and SQLite database:

```powershell
uv run python -m lai.cli init
```

Mint an admin API key (copy the one-time `lai_sk_...` secret):

```powershell
uv run python -m lai.cli keys create --name "local-admin" --admin
uv run python -m lai.cli keys list
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

## API Examples

All protected endpoints accept `Authorization: Bearer lai_sk_...` (or an OIDC JWT when configured).

### Health & System Status

```powershell
# Unauthenticated liveness
curl.exe http://127.0.0.1:8400/v1/health

# GPU VRAM, primary Gemma 4, ShieldGemma, spooler & alerts summary
curl.exe -H "Authorization: Bearer $env:LAI_API_KEY" http://127.0.0.1:8400/v1/system/status
```

### Create Project & Upload Files (HTTP-Only Ingestion)

```powershell
# 1. Create a project
curl.exe -X POST http://127.0.0.1:8400/v1/projects `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{"name":"backend-logs","description":"Production crash diagnostics"}'

# 2. Multipart upload (synchronous; pass ?wait=false for HTTP 202 background CPU indexing)
curl.exe -X POST "http://127.0.0.1:8400/v1/files?wait=true" `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -F "project_id=prj_..." `
  -F "file=@app.log"

# 3. Raw streaming upload (for large files without multipart framing overhead)
curl.exe -X POST "http://127.0.0.1:8400/v1/files/stream?project_id=prj_..." `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "X-LAI-Filename: huge.log" `
  --data-binary "@huge.log"
```

### Chat Sessions & SSE Streaming

```powershell
# Create session
curl.exe -X POST http://127.0.0.1:8400/v1/sessions `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{"project_id":"prj_...","title":"Root Cause Investigation"}'

# Send SSE streaming turn (emits keepalives every 10s to prevent proxy 524 timeouts)
curl.exe -N -X POST http://127.0.0.1:8400/v1/sessions/ses_.../messages `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{"content":"Summarize all errors in the uploaded log","stream":true}'
```

### OpenAI-Compatible Endpoint

Point any OpenAI SDK or IDE plugin at `http://127.0.0.1:8400/v1`:

```powershell
curl.exe -X POST http://127.0.0.1:8400/v1/chat/completions `
  -H "Authorization: Bearer $env:LAI_API_KEY" `
  -H "Content-Type: application/json" `
  -H "X-LAI-Project-Id: prj_..." `
  -H "X-LAI-Context-Mode: auto" `
  -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Explain the top exception"}],"stream":false}'
```

When ShieldGemma blocks a prompt in `refusal_message` mode, the response carries `X-LAI-Guardrail-Blocked: true` and `X-LAI-Guardrail-Policy: <policy>`.

## Google OIDC JWT Authentication

To accept Google OIDC ID tokens alongside `lai_sk_` API keys:

1. Create an OAuth 2.0 Client ID in Google Cloud Console.
2. Update `config/config.yaml`:

```yaml
auth_providers:
  - "api_key"
  - "oidc"
oidc_enabled: true
oidc_issuer_url: "https://accounts.google.com"
oidc_jwks_url: "https://www.googleapis.com/oauth2/v3/certs"
oidc_audience: "your-client-id.apps.googleusercontent.com"
oidc_allowed_domains:
  - "example.com"
oidc_admin_emails:
  - "owner@example.com"
```

Tokens are validated via RS256/ES256 against TTL-cached JWKS (`exp`, `iss`, `aud`, `email_verified`) with automatic `kid` rotation refresh.

## Cloudflare Named Tunnel Setup

Expose `127.0.0.1:8400` securely without opening inbound router ports:

1. Authenticate and create a Named Tunnel:

```powershell
cloudflared tunnel login
cloudflared tunnel create lai-local
cloudflared tunnel route dns lai-local ai.example.com
```

2. Render `cloudflared` configuration (or customize `deploy/cloudflared/config.template.yml`):

```powershell
uv run python -m lai.cli tunnel render-config `
  --tunnel "<TUNNEL_UUID>" `
  --hostname "ai.example.com" `
  --output deploy/cloudflared/config.yml
```

3. Enable Cloudflare Access edge verification in `config/config.yaml` (recommended):

```yaml
cf_access_enabled: true
cf_access_team_domain: "your-team.cloudflareaccess.com"
cf_access_aud: "<ACCESS_APPLICATION_AUD_TAG>"
```

Service tokens (`CF-Access-Client-Id` / `CF-Access-Client-Secret`) are also supported for automated clients.

4. Run the tunnel:

```powershell
cloudflared tunnel --config deploy/cloudflared/config.yml run
```

## 24-Hour Rotating Logs & Operational Alerts

LAI rotates structured logs daily at UTC midnight (`data/logs/`):

- `lai.jsonl` — All structured application events (14-day gzip-compressed retention).
- `errors.jsonl` — `ERROR` and `CRITICAL` events with tracebacks (30-day retention).
- `prompts_audit.jsonl` — SHA-256 prompt fingerprints, scrubbed text, and ShieldGemma verdicts (30-day retention).
- `alerts.jsonl` — Emitted `AlertEvent` records (`fatal_exception`, `timeout`, `subprocess_crash`, `storage_failure`, `security_alert`).
- `llama-server.log` / `shieldgemma-server.log` — Supervised subprocess outputs.

Set `logging.alert_webhook_url` in `config/config.yaml` to receive real-time Slack/Discord/Teams webhook alerts (with 60s cooldown deduplication), or inspect recent alerts via `GET /v1/system/alerts` (requires `admin` scope).

## Phase 2 PowerShell Tool Plugin Guide

`lai.tools` isolates every tool behind `ToolPluginPort`, `DefaultToolPolicy`, and `GuardedToolExecutor`. Phase 1 registers read-only workspace tools (`PermissionTier.READ`). To add a Phase 2 guarded PowerShell execution plugin:

1. Implement `ToolPluginPort` with `PermissionTier.EXEC` and `requires_approval_by_default=True`:

```python
from typing import Any
from lai.domain.tools import PermissionTier, ToolExecutionContext, ToolResult, ToolSpec
from lai.ports.tools import ToolPluginPort

class RunPowerShellPlugin(ToolPluginPort):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="run_powershell",
            description="Execute a read-only diagnostic PowerShell command.",
            permission_tier=PermissionTier.EXEC,
            requires_approval_by_default=True,
            timeout_seconds=15.0,
            parameters_schema={
                "type": "object",
                "properties": {"script": {"type": "string"}},
                "required": ["script"],
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        # Enforce workspace_root confinement, allowlists, and non-interactive -NoProfile execution.
        ...
```

2. Register the plugin in `ToolRegistryPort` (requiring callers to hold `tools:exec` or `admin`).
3. Because `requires_approval_by_default=True`, invoking the tool persists a `PENDING` `ToolApprovalRecord` and raises `ToolApprovalRequiredError` (`HTTP 409`) or emits `ApprovalRequiredEvent` during streaming.
4. An operator holding `tools:approve` resolves the request via `POST /v1/tool-approvals/{approval_id}/resolve` (`{"approved": true, "note": "reviewed"}`), transitioning `PENDING -> APPROVED -> EXECUTED` with full audit logging and replay protection.

## Verification

```powershell
uv run python scripts/verify.py all --skip-mutate --cov-fail-under 85
```
