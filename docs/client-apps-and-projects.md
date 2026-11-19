# Client App & Project Lifecycle Management

This guide covers day-to-day administration of **client applications** (OIDC-bound web SPAs and
desktop apps), **project workspaces**, **API keys**, and **user access** in LAI — from initial
onboarding through updates, credential rotation, and decommissioning.

---

## Understanding the Two-Layer Client Access Model

When external apps connect to LAI using Google OIDC, access is controlled at **two separate
layers**:

1. **Authentication Layer (`config/config.yaml` or live Admin API)**
   - `oidc_audiences` defines which Google OAuth 2.0 Client IDs are accepted for login.
   - `oidc_allowed_emails` / `oidc_allowed_domains` define which verified users may authenticate.
   - `cors_allowed_origins` defines which browser origins may call LAI directly from a SPA.
   - These settings can be hot-swapped live without restarting LAI via
     `PATCH /v1/system/config/oidc` and `PATCH /v1/system/config/cors` (persisted across restarts),
     or reloaded cleanly from `config/config.yaml` via `POST /v1/system/config/reload` with
     `{"sections": ["oidc", "cors"]}`.

2. **Client App Registration Layer (`lai apps` CLI / `/v1/client-apps` Admin API)**
   - Registers a Google OAuth 2.0 Client ID (`oauth_client_id`) as a named application with
     explicit **client-side tool capabilities**:
     - `allowed_tiers`: permitted tool risk tiers (`read`, `write`, `exec`, `network`)
     - `registered_tools`: optional static client tool definitions stored on the server
     - `allowed_project_ids`: optional allowlist of project IDs (empty means any project the
       authenticated user can access)
     - `expires_at`: optional expiration timestamp
   - Takes effect immediately in the database (no server restart required).
   - Uses **zero static client secrets** — LAI binds client identity cryptographically from the
     verified Google ID token's `azp` / `aud` claim.

> **Important distinction:** Revoking a client app registration disables **client-side tool
> execution and tool result submission** immediately, but it does **not** prevent an allowlisted
> user from calling standard chat, project, or file endpoints with tokens from that OAuth Client ID
> until that Client ID is also removed from `oidc_audiences` (and `cors_allowed_origins`, if a SPA).
> See [Decommissioning a Client App](#decommissioning-a-client-app) for full offboarding.

---

## Client Application Lifecycle

### 1. Adding a New Client App

#### Step 1: Create the OAuth 2.0 Client ID in Google Cloud

Provision a dedicated OAuth 2.0 Client ID in Google Cloud Console for the application (**Web
application** for a browser SPA, or **Desktop app** for a native/Electron client). See the
[Deployment Guide](deployment.md#google-cloud-oidc-setup) for detailed Google Cloud instructions.

#### Step 2: Allow the Client ID in `config/config.yaml`

Add the new Client ID to `oidc_audiences`. If the app is a browser SPA, also add its exact origin
(no trailing `/`, no wildcard `*`) to `cors_allowed_origins`:

```yaml
auth_providers:
  - "api_key"
  - "oidc"
oidc_enabled: true
oidc_audiences:
  - "1234567890-existing.apps.googleusercontent.com"
  - "1234567890-newapp.apps.googleusercontent.com"
cors_allowed_origins:
  - "https://newapp.example.com"
```

Apply the change live without restarting via `PATCH /v1/system/config/oidc` and
`PATCH /v1/system/config/cors` (or edit `config/config.yaml` and run
`POST /v1/system/config/reload` with `{"sections": ["oidc", "cors"]}`).

#### Step 3: Register the Client App in LAI

If the app only calls standard project/file/chat endpoints without client-side tools, Step 2 is
sufficient. If the app registers or invokes **client-executed tools**, register it in LAI via
either the CLI or the Admin REST API.

**Option A — CLI (`lai apps create`):**

```powershell
uv run python -m lai.cli apps create `
  --config config/config.yaml `
  --name "my-desktop-app" `
  --oauth-client-id "1234567890-newapp.apps.googleusercontent.com" `
  --tier read `
  --tier write `
  --project-id "prj_0123456789abcdef" `
  --registered-tools tools.json `
  --expires-days 180
```

- `--tier` is repeatable (`read`, `write`, `exec`, `network`); defaults to `read`.
- `--project-id` is optional and repeatable; omit it to allow client tools across any project the
  caller owns (or all projects for admins).
- `--registered-tools` is optional; pass a path to a JSON array of `ToolSpec` definitions to
  pre-register static client tools (apps can also send dynamic `client_tools` per turn within
  their `allowed_tiers`).
- `--expires-days` is optional; omit for no automatic expiry.

**Option B — Admin REST API (`POST /v1/client-apps`):**

```powershell
curl.exe -X POST http://127.0.0.1:8400/v1/client-apps `
  -H "Authorization: Bearer $env:LAI_ADMIN_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "name": "my-desktop-app",
    "oauth_client_id": "1234567890-newapp.apps.googleusercontent.com",
    "allowed_tiers": ["read", "exec"],
    "allowed_project_ids": ["prj_0123456789abcdef"],
    "registered_tools": [
      {
        "name": "read_local_config",
        "description": "Read a local workspace config file",
        "permission_tier": "read",
        "parameters_schema": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"],
          "additionalProperties": false
        }
      }
    ]
  }'
```

**Uniqueness rule:** Only **one active registration** may exist per `oauth_client_id`. Registering
a duplicate while an active, non-expired record exists returns `409 Conflict`. If a previous
registration for the same `oauth_client_id` has already expired, LAI automatically revokes the
expired record before creating the replacement.

**Caller scope requirement for privileged client tools:** Registering an app with `write`, `exec`,
or `network` tiers is necessary but not sufficient — the authenticated caller must also hold
`tools:exec` (or `admin`) scope, and an approver holding `tools:approve` (or `admin`) must approve
each privileged invocation before the client receives an execution lease.

### 2. Inspecting & Auditing Client Apps

List all registered apps (active, expired, and revoked) via CLI or Admin API:

```powershell
uv run python -m lai.cli apps list --config config/config.yaml
```

```powershell
curl.exe http://127.0.0.1:8400/v1/client-apps `
  -H "Authorization: Bearer $env:LAI_ADMIN_KEY"
```

Each record reports:

- `id` (`app_...`) and `name`
- `oauth_client_id`
- `active` status (`active`, `expired`, or `revoked` in CLI output)
- `allowed_tiers`, `allowed_project_ids` (`*` in CLI when unrestricted), and `registered_tools`
- `created_at`, `last_used_at`, `expires_at`, and `revoked_at`

### 3. Updating, Re-scoping, or Renewing a Client App

Active (non-revoked) client app registrations can be updated **in place** without disrupting
in-flight client tool approvals bound to the `app_...` ID. The bound `oauth_client_id` stays
immutable (use [Rotating an OAuth Client ID](#4-rotating-an-oauth-client-id-zero-downtime) to
change it), and revoked apps cannot be updated.

**Via CLI (`lai apps update`):**

```powershell
# Update allowed tiers, project allowlist, or extend expiry in place
uv run python -m lai.cli apps update app_<ID> `
  --config config/config.yaml `
  --tier read `
  --tier write `
  --project-id "prj_old" `
  --project-id "prj_new" `
  --expires-days 180

# Clear project allowlist (allow all accessible projects) or remove expiration
uv run python -m lai.cli apps update app_<ID> --clear-projects --clear-expiry --config config/config.yaml

# Replace or clear static registered_tools
uv run python -m lai.cli apps update app_<ID> --registered-tools tools.json --config config/config.yaml
uv run python -m lai.cli apps update app_<ID> --clear-tools --config config/config.yaml
```

**Via Admin REST API (`PATCH /v1/client-apps/{app_id}`):**

```powershell
curl.exe -X PATCH http://127.0.0.1:8400/v1/client-apps/app_<ID> `
  -H "Authorization: Bearer $env:LAI_ADMIN_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "allowed_project_ids": ["prj_old", "prj_new"],
    "expires_at": null
  }'
```

- Only fields included in the JSON payload are modified; sending `"expires_at": null` clears the
  expiration limit, whereas omitting `expires_at` leaves it unchanged.
- If you reduce `allowed_tiers`, any retained `registered_tools` must still fit within the new
  tiers (otherwise the update fails with `422`).
- If an app has already expired and you prefer a fresh registration ID, calling
  `lai apps create` / `POST /v1/client-apps` with the same `oauth_client_id` automatically revokes
  the expired row and preserves it in audit history.

### 4. Rotating an OAuth Client ID (Zero Downtime)

When migrating an app to a new Google OAuth Client ID without downtime:

1. Create the new OAuth 2.0 Client ID in Google Cloud Console.
2. Add the new Client ID **alongside** the old Client ID in `oidc_audiences` via
   `PATCH /v1/system/config/oidc` (or edit `config/config.yaml` and run
   `POST /v1/system/config/reload` with `{"sections": ["oidc"]}`).
3. Register the new Client ID in LAI (`lai apps create` or `POST /v1/client-apps`) with matching
   tiers, project scope, and tools.
4. Deploy the updated Client ID to your SPA or desktop app.
5. Watch `last_used_at` in `GET /v1/client-apps` until all clients have migrated off the old ID.
6. Decommission the old Client ID using the steps below.

### 5. Decommissioning a Client App

To completely retire a client application so it can neither execute client tools nor authenticate:

1. **Revoke the client app registration immediately** (blocks client tools and `/result`
   submissions with no restart required):
   ```powershell
   uv run python -m lai.cli apps revoke app_<ID> --config config/config.yaml
   ```
   *(Or `DELETE /v1/client-apps/app_<ID>`.)*
2. **Remove the OAuth Client ID from `oidc_audiences`** via `PATCH /v1/system/config/oidc` (or
   edit `config/config.yaml` and reload `["oidc"]`) so tokens minted for that client can no longer
   log in to LAI at all.
3. **Remove the app's origin from `cors_allowed_origins`** via `PATCH /v1/system/config/cors` (for
   browser SPAs), or edit `config/config.yaml` and reload `["cors"]`.
4. **Verify live configuration** with `GET /v1/system/config/oidc` and `GET /v1/system/config/cors`
   (no server restart required).
5. **Delete or disable the OAuth 2.0 Client ID in Google Cloud Console** so no new ID tokens can be
   issued.
6. **Optional cleanup:** Revoke any service/automation API keys created for the app
   (`lai keys revoke key_...`) and delete any dedicated projects that are no longer needed.

---

## Client Tool Execution Flow (Runtime Summary)

When a registered client app sends a turn with client tools:

1. **Tool resolution:** LAI merges any static `registered_tools` on the app record with per-turn
   `client_tools` in the request, rejecting unknown/inactive apps (`401`/`403`), out-of-scope
   projects (`403`), tiers exceeding `allowed_tiers` (`403`), or names shadowing built-in server
   tools (`422`).
2. **Dispatch & approval:**
   - `read` tools (without `requires_approval_by_default`): LAI issues a single-use 5-minute
     execution lease (`requires_approval: false`) and returns `pending_client_tool_call` (or a
     `client_tool_call` SSE event).
   - `write`, `exec`, `network` tools (or tools with `requires_approval_by_default: true`): LAI
     pauses in `pending` state (`requires_approval: true`). An approver with `tools:approve` (or
     `admin`) resolves it via `POST /v1/tool-approvals/{approval_id}/resolve`, which transitions it
     to `approved` with a 5-minute lease (or `denied`).
3. **Result submission & resume:** The client executes the tool locally and posts the output to
   `POST /v1/tool-approvals/{approval_id}/result` using the same authenticated user and bound OAuth
   Client ID. LAI validates ownership, scrubs secrets/PII, scans the output with ShieldGemma,
   atomically consumes the single-use approval, and resumes the model turn.

---

## Project Workspace Lifecycle

### 1. Creating a Project

Create a project via `POST /v1/projects` (requires `projects:write` or `admin`):

```powershell
curl.exe -X POST http://127.0.0.1:8400/v1/projects `
  -H "Authorization: Bearer $env:LAI_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{
    "name": "payments-incident-2026-11",
    "description": "Checkout timeout investigation"
  }'
```

- The project is owned by the calling user (`owner_principal_id` is assigned automatically).
- Non-admin callers are capped at `max_projects_per_principal` (default `10`; returns `403` with
  code `quota_exceeded` when reached).
- All project files must be uploaded over HTTP — LAI never reads files from the server filesystem.
- If a registered client app uses a restricted `allowed_project_ids` allowlist and needs client-tool
  access to the new project, update the app in place via `lai apps update` or
  `PATCH /v1/client-apps/{app_id}` to include the new `prj_...` ID.

### 2. Uploading Files & Running Sessions

- **Upload files:**
  - Multipart form: `POST /v1/files` (`file`, `project_id`, optional `wait=true|false`)
  - Raw streaming: `POST /v1/files/stream?project_id=prj_...` with header `X-LAI-Filename: app.log`
  - For large files, pass `wait=false` to receive `202 Accepted` immediately, then poll
    `GET /v1/files/{file_id}` until `status` is `"ready"`.
  - Respects both `max_upload_bytes` (default 250 MiB, or 100 MB through Cloudflare Free/Pro
    Tunnel) and `max_storage_bytes_per_principal` (default 1 GiB for non-admins).
- **Create and query sessions:**
  - `POST /v1/sessions` with `{"project_id": "prj_..."}`, then `POST /v1/sessions/{session_id}/messages`
  - Or call `POST /v1/chat/completions` with `project_id` (or `X-LAI-Project-Id`) and optional
    `session_id` / `file_ids`.
- **Inspect workspace contents:**
  - `GET /v1/projects` (caller's projects, or all projects for admins)
  - `GET /v1/files?project_id=prj_...` and `GET /v1/files/{file_id}/chunks`
  - `GET /v1/sessions?project_id=prj_...` and `GET /v1/sessions/{session_id}`

### 3. Cleaning Up Files & Sessions

- **Delete an individual file:** `DELETE /v1/files/{file_id}` removes the stored blob on disk,
  indexed chunks, full-text search rows, and vector embeddings, immediately reclaiming storage
  quota.
- **Delete a chat session:** `DELETE /v1/sessions/{session_id}` deletes the session and its message
  history.

### 4. Decommissioning a Project

Delete a project via `DELETE /v1/projects/{project_id}` (permitted for the project owner or an
admin):

```powershell
curl.exe -X DELETE http://127.0.0.1:8400/v1/projects/prj_0123456789abcdef `
  -H "Authorization: Bearer $env:LAI_TOKEN"
```

Deleting a project performs a full cascade cleanup:

- Deletes all uploaded file blobs on disk
- Deletes all vector embeddings for the project's files
- Deletes all SQLite rows for the project, its files, chunks, full-text index entries, chat
  sessions, messages, tool approvals, and tool audit records
- Immediately reclaims the caller's project-count and storage-byte quotas

**Post-deletion notes:**

- If any registered client app explicitly referenced the deleted `prj_...` ID in
  `allowed_project_ids`, remove it via `lai apps update` or `PATCH /v1/client-apps/{app_id}`.
- Historical scrubbed entries in rotating log files (`data/logs/lai.jsonl`,
  `data/logs/prompts_audit.jsonl`) age out automatically according to `logging.retention_days` and
  `logging.error_retention_days`.

---

## API Keys & User Offboarding

### API Key Lifecycle

```powershell
# Mint a scoped user key (default scopes: projects, files, chat, tools:safe)
uv run python -m lai.cli keys create --config config/config.yaml --name "ci-runner" --expires-days 90

# Mint an admin key
uv run python -m lai.cli keys create --config config/config.yaml --name "ops-admin" --admin

# Audit keys (secrets are never shown again)
uv run python -m lai.cli keys list --config config/config.yaml

# Revoke a key (takes effect immediately, no restart required)
uv run python -m lai.cli keys revoke key_0123456789abcdef --config config/config.yaml
```

To rotate an API key with zero downtime: mint a new key, update the consumer, verify traffic, then
revoke the old key.

### Offboarding an OIDC User

1. Remove the user's email from `oidc_allowed_emails` and `oidc_admin_emails` live via
   `PATCH /v1/system/config/oidc` (or update `config/config.yaml` and call
   `POST /v1/system/config/reload` with `{"sections": ["oidc"]}` — or suspend/delete the account in
   Google Workspace if using `oidc_allowed_domains`).
2. Access is revoked immediately on the next request without restarting LAI.
   *(Note: if relying solely on Google Workspace suspension without updating the allowlist,
   already-issued Google ID tokens remain cryptographically valid until their 1-hour `exp` claim
   expires.)*
3. As an admin, audit and delete or transfer any projects owned by the departed user via
   `GET /v1/projects` and `DELETE /v1/projects/{project_id}`.
