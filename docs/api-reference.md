# LAI Local AI Server API Reference

- **API Version**: `0.1.0`
- **OpenAPI Specification**: `3.1.0`
- **Interactive Docs**: [GitHub Pages Scalar Reference](https://blake-olinger.github.io/lai/)

> Auto-generated from the FastAPI OpenAPI 3.1 schema via `scripts/export_openapi.py`. Do not edit manually.

## Authentication

All protected endpoints require a bearer token in the `Authorization` header:

- **Google OIDC ID token**: `Authorization: Bearer <google_id_token>` (RS256/ES256 verified against
  configured JWKS, enforcing `exp`, `iss`, `sub`, `aud`, and verified email/domain rules).
- **LAI API key**: `Authorization: Bearer lai_sk_...` (SHA-256 hashed keys via `lai keys create`).
- **Liveness probe**: `GET /v1/health` is unauthenticated.
- **Optional Cloudflare Access edge headers**: When `cf_access_enabled: true`, requests must also
  carry a valid `Cf-Access-Jwt-Assertion` or `CF-Access-Client-Id` / `CF-Access-Client-Secret`.

Authorization is scope-gated (`projects:read`, `projects:write`, `files:read`, `files:write`,
`chat:read`, `chat:write`, `tools:safe`, `tools:exec`, `tools:approve`, `admin`).

## Public SPA & Desktop OIDC Client Binding

Browser SPAs and desktop clients cannot store static client secrets securely. Instead of issuing
or accepting static client secrets, LAI binds registered client apps to Google OIDC client IDs:

1. Register the client ID via `POST /v1/client-apps` or `lai apps create --oauth-client-id <ID>`.
2. On OIDC requests, LAI extracts `azp` (when present, required to match `oidc_audiences`) or the
   matched `aud` claim as the caller's verified `oauth_client_id`.
3. Client-executed tool dispatch verifies that the caller's verified `oauth_client_id` matches an
   active registered client app with zero client secrets transmitted or stored.

## Multi-Tenant Isolation & Per-Principal Quotas

- **Owner scoping & `404` anti-enumeration**: Projects, files, sessions, messages, and approvals are
  scoped to the principal that created them. Non-admin lookups or mutations targeting another
  principal's resources fail with `404 Not Found` (`not_found`) so resource IDs cannot be probed.
- **Per-principal quotas**: Fail-closed limits cap active projects (`max_projects_per_principal`),
  cumulative file bytes (`max_storage_bytes_per_principal`), and queued jobs
  (`max_queued_jobs_per_principal`).
- **Fair scheduling**: Queued inference jobs are scheduled round-robin across users first and
  sessions second so one tenant cannot starve others.

## Standard Headers & Error Envelope

| Header | Direction | Purpose |
| --- | --- | --- |
| `Authorization` | Request | `Bearer <google_id_token>` or `Bearer lai_sk_...` |
| `X-Request-Id` | Request / Response | Correlation ID echoed or minted as `req_<hex>` |
| `X-LAI-Filename` | Request | Original filename for raw streaming `/v1/files/stream` uploads |
| `X-LAI-Project-Id` | Request | Project scope for `/v1/files/stream` or `/v1/chat/completions` |
| `X-LAI-Session-Id` | Request | Existing or explicit session ID for `/v1/chat/completions` |
| `X-LAI-File-Ids` | Request | Comma-separated file IDs for `/v1/chat/completions` |
| `X-LAI-Context-Mode` | Request | `auto`, `direct`, `map_reduce`, or `retrieval` |
| `X-LAI-Guardrail-Blocked` | Response | `true` when ShieldGemma blocked the turn |
| `X-LAI-Guardrail-Policy` | Response | Policy label reported by ShieldGemma on blocks |
| `Retry-After` | Response | Backoff seconds returned with `429` or `503` responses |

All handled and unhandled failures return the `ErrorResponse` JSON envelope:

```json
{"error": {"code": "not_found", "message": "Resource not found.",
           "request_id": "req_0123456789abcdef", "details": {}}}
```

## SSE Event Framing

Streaming responses set `Content-Type: text/event-stream`, `Cache-Control: no-cache`, and
`X-Accel-Buffering: no`, emitting heartbeats every `keepalive_interval_seconds` to prevent
Cloudflare 524 origin timeouts:

- **`POST /v1/sessions/{session_id}/messages` (`stream: true`)**: Named SSE frames
  (`event: <event_type>\ndata: <json>\n\n`) for `keepalive`, `progress`, `thinking_delta`,
  `content_delta`, `approval_required`, `client_tool_call`, `turn_completed`, and `error`.
- **`POST /v1/chat/completions` (`stream: true`)**: OpenAI-compatible
  `data: <OpenAIChatCompletionChunk>\n\n` frames plus `: keepalive\n\n` comments, including
  `lai_client_tool_call` on client tool pauses, terminating with `data: [DONE]\n\n`.
- **Non-streaming (`stream: false`)**: Turns exceeding `json_keepalive_threshold_seconds` switch
  to chunked whitespace heartbeats before emitting the JSON body.

## Endpoints

### Tag: `client-apps`

### `GET /v1/client-apps`

**List Client Apps**

List all registered client applications and their bound OAuth client IDs.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ClientAppListResponse`](#schema-clientapplistresponse) |

### `POST /v1/client-apps`

**Create Client App**

Register a new client application bound to an OAuth client ID.

**Request Body** (required)

- `application/json`: [`CreateClientAppRequest`](#schema-createclientapprequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `allowed_project_ids` | array[string] | no | default=`[]` |
| `allowed_tiers` | array[[`PermissionTier`](#schema-permissiontier)] | no | default=`['read']` |
| `expires_at` | string (date-time) \| null | no | - |
| `name` | string | yes | minLength=1; maxLength=120 |
| `oauth_client_id` | string | yes | minLength=1; maxLength=256 |
| `registered_tools` | array[[`ToolSpec`](#schema-toolspec)] | no | default=`[]` |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `201` | Successful Response | `application/json`: [`ClientAppResponse`](#schema-clientappresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `DELETE /v1/client-apps/{app_id}`

**Revoke Client App**

Revoke an existing client application by ID.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `app_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`DeleteResponse`](#schema-deleteresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `PATCH /v1/client-apps/{app_id}`

**Patch Client App**

Update mutable settings on an active (non-revoked) client application.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `app_id` | path | yes | string | - |

**Request Body** (required)

- `application/json`: [`UpdateClientAppRequest`](#schema-updateclientapprequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `allowed_project_ids` | array[string] \| null | no | - |
| `allowed_tiers` | array[[`PermissionTier`](#schema-permissiontier)] \| null | no | - |
| `expires_at` | string (date-time) \| null | no | - |
| `name` | string \| null | no | - |
| `registered_tools` | array[[`ToolSpec`](#schema-toolspec)] \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ClientAppResponse`](#schema-clientappresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |


### Tag: `files`

### `GET /v1/files`

**List Files**

List uploaded files scoped to an accessible project.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `project_id` | query | yes | string | minLength=1 |
| `limit` | query | no | integer | default=`200`; minimum=1; maximum=500 |
| `offset` | query | no | integer | default=`0`; minimum=0 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`FileListResponse`](#schema-filelistresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/files`

**Upload File Multipart**

Upload a project file via multipart/form-data and ingest synchronously or asynchronously.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `wait` | query | no | boolean | default=`True` |

**Request Body** (required)

- `multipart/form-data`: [`Body_upload_file_multipart_v1_files_post`](#schema-body_upload_file_multipart_v1_files_post)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `file` | string | yes | - |
| `project_id` | string | yes | minLength=1 |
| `wait` | boolean \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `201` | Successful Response | `application/json`: [`FileRecord`](#schema-filerecord) |
| `202` | Accepted for asynchronous background indexing when wait=false. | `application/json`: [`FileRecord`](#schema-filerecord) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/files/stream`

**Upload File Stream**

Stream a raw HTTP request body directly into project blob storage and ingestion.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `project_id` | query | no | string \| null | - |
| `wait` | query | no | boolean | default=`True` |
| `X-LAI-Filename` | header | yes | string | minLength=1 |
| `X-LAI-Project-Id` | header | no | string \| null | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `201` | Successful Response | `application/json`: [`FileRecord`](#schema-filerecord) |
| `202` | Accepted for asynchronous background indexing when wait=false. | `application/json`: [`FileRecord`](#schema-filerecord) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `DELETE /v1/files/{file_id}`

**Delete File**

Delete a file's blob, SQLite metadata/chunks/FTS5 rows, and LanceDB vectors.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `file_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`DeleteResponse`](#schema-deleteresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/files/{file_id}`

**Get File**

Fetch metadata and deterministic summaries for an accessible uploaded file.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `file_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`FileRecord`](#schema-filerecord) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/files/{file_id}/chunks`

**List File Chunks**

Return ordered indexed chunks for file_id, optionally filtered by line range.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `file_id` | path | yes | string | - |
| `start_line` | query | no | integer \| null | - |
| `end_line` | query | no | integer \| null | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`FileChunkListResponse`](#schema-filechunklistresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |


### Tag: `openai`

### `POST /v1/chat/completions`

**Create Openai Chat Completion**

OpenAI-compatible chat completions supporting project/file context and keepalives.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `X-LAI-Project-Id` | header | no | string \| null | - |
| `X-LAI-File-Ids` | header | no | string \| null | - |
| `X-LAI-Session-Id` | header | no | string \| null | - |
| `X-LAI-Context-Mode` | header | no | string \| null | - |

**Request Body** (required)

- `application/json`: [`OpenAIChatCompletionRequest`](#schema-openaichatcompletionrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `client_tools` | array[[`ToolSpec`](#schema-toolspec)] | no | - |
| `context_mode` | [`ContextMode`](#schema-contextmode) \| null | no | - |
| `enable_thinking` | boolean | no | default=`True` |
| `enable_tools` | boolean | no | default=`True` |
| `file_ids` | array[string] \| null | no | - |
| `max_completion_tokens` | integer \| null | no | - |
| `max_tokens` | integer \| null | no | - |
| `messages` | array[[`OpenAIChatMessageInput`](#schema-openaichatmessageinput)] | yes | minItems=1 |
| `model` | string \| null | no | - |
| `project_id` | string \| null | no | - |
| `session_id` | string \| null | no | - |
| `stream` | boolean | no | default=`False` |
| `temperature` | number \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`OpenAIChatCompletionResponse`](#schema-openaichatcompletionresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/models`

**List Openai Models**

List configured model profiles in OpenAI /v1/models format.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`OpenAIModelListResponse`](#schema-openaimodellistresponse) |


### Tag: `projects`

### `GET /v1/projects`

**List Projects**

List workspace projects scoped to the caller (or all projects for admins).

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `limit` | query | no | integer | default=`100`; minimum=1; maximum=500 |
| `offset` | query | no | integer | default=`0`; minimum=0 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ProjectListResponse`](#schema-projectlistresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/projects`

**Create Project**

Create a new isolated project workspace.

**Request Body** (required)

- `application/json`: [`ProjectCreateRequest`](#schema-projectcreaterequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `default_model_profile` | string | no | default=`'gemma4-26b-a4b'` |
| `description` | string | no | default=`''` |
| `name` | string | yes | minLength=1; maxLength=120 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `201` | Successful Response | `application/json`: [`Project`](#schema-project) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `DELETE /v1/projects/{project_id}`

**Delete Project**

Delete a project along with its stored file blobs, vectors, and metadata.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `project_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`DeleteResponse`](#schema-deleteresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/projects/{project_id}`

**Get Project**

Fetch a project by ID.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `project_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`Project`](#schema-project) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |


### Tag: `sessions`

### `GET /v1/sessions`

**List Sessions**

List chat sessions belonging to an accessible project.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `project_id` | query | yes | string | minLength=1 |
| `limit` | query | no | integer | default=`100`; minimum=1; maximum=500 |
| `offset` | query | no | integer | default=`0`; minimum=0 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`SessionListResponse`](#schema-sessionlistresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/sessions`

**Create Session**

Create a new chat session scoped to an accessible project.

**Request Body** (required)

- `application/json`: [`SessionCreateRequest`](#schema-sessioncreaterequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `model_profile` | string \| null | no | - |
| `project_id` | string | yes | minLength=1 |
| `system_prompt` | string \| null | no | - |
| `title` | string | no | default=`'New Session'`; minLength=1; maxLength=200 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `201` | Successful Response | `application/json`: [`Session`](#schema-session) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `DELETE /v1/sessions/{session_id}`

**Delete Session**

Delete a chat session and its messages after verifying ownership.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`DeleteResponse`](#schema-deleteresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/sessions/{session_id}`

**Get Session**

Fetch session metadata along with its chronological message history.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`SessionDetailResponse`](#schema-sessiondetailresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/sessions/{session_id}/messages`

**Post Session Message**

Run a user turn in session_id as JSON (with keepalive) or SSE stream.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string | - |

**Request Body** (required)

- `application/json`: [`SessionMessageRequest`](#schema-sessionmessagerequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `attached_file_ids` | array[string] | no | default=`[]` |
| `client_tools` | array[[`ToolSpec`](#schema-toolspec)] | no | default=`[]` |
| `content` | string | yes | minLength=1 |
| `context_mode` | [`ContextMode`](#schema-contextmode) | no | default=`'auto'` |
| `enable_thinking` | boolean | no | default=`True` |
| `enable_tools` | boolean | no | default=`True` |
| `max_tokens` | integer \| null | no | - |
| `stream` | boolean | no | default=`False` |
| `temperature` | number \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`TurnOutput`](#schema-turnoutput) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |


### Tag: `system`

### `GET /v1/health`

**Get Health**

Unauthenticated liveness probe.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`HealthResponse`](#schema-healthresponse) |

### `GET /v1/system/alerting`

**Get Alerting Config**

Return active redacted SMTP and webhook alerting configuration.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`AlertingConfigResponse`](#schema-alertingconfigresponse) |

### `PATCH /v1/system/alerting`

**Patch Alerting Config**

Hot-swap non-secret alerting settings and persist across restarts.

**Request Body** (required)

- `application/json`: [`UpdateAlertingConfigRequest`](#schema-updatealertingconfigrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `alert_cooldown_seconds` | number \| null | no | - |
| `alert_on_errors` | boolean \| null | no | - |
| `alert_on_timeouts` | boolean \| null | no | - |
| `alert_webhook_categories` | array[[`AlertCategory`](#schema-alertcategory)] \| null | no | - |
| `alert_webhook_enabled` | boolean \| null | no | - |
| `alert_webhook_min_severity` | [`AlertSeverity`](#schema-alertseverity) \| null | no | - |
| `smtp_categories` | array[[`AlertCategory`](#schema-alertcategory)] \| null | no | - |
| `smtp_enabled` | boolean \| null | no | - |
| `smtp_from_address` | string \| null | no | - |
| `smtp_host` | string \| null | no | - |
| `smtp_min_severity` | [`AlertSeverity`](#schema-alertseverity) \| null | no | - |
| `smtp_port` | integer \| null | no | - |
| `smtp_subject_prefix` | string \| null | no | - |
| `smtp_timeout_seconds` | number \| null | no | - |
| `smtp_tls_mode` | [`SmtpTlsMode`](#schema-smtptlsmode) \| null | no | - |
| `smtp_to_addresses` | array[string] \| null | no | - |
| `smtp_use_admin_emails_fallback` | boolean \| null | no | - |
| `smtp_username` | string \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`AlertingConfigResponse`](#schema-alertingconfigresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/system/alerting/test`

**Send Test Alert**

Send a synthetic test alert through active sinks and await delivery status.

**Request Body** (optional)

- `application/json`: [`TestAlertRequest`](#schema-testalertrequest) \| null

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`TestAlertResponse`](#schema-testalertresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/alerts`

**Get System Alerts**

Return recent fatal, timeout, crash, storage, and security alerts.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `limit` | query | no | integer | default=`50`; minimum=1; maximum=200 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`AlertsListResponse`](#schema-alertslistresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/analytics`

**Get Usage Analytics**

Return endpoint, principal/client-app, token, and hourly rollup usage metrics.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `window_hours` | query | no | integer | default=`24`; minimum=1; maximum=2160 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`AnalyticsReport`](#schema-analyticsreport) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/analytics/config`

**Get Analytics Config**

Return live endpoint usage analytics toggle and retention settings.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`AnalyticsConfigResponse`](#schema-analyticsconfigresponse) |

### `PATCH /v1/system/analytics/config`

**Patch Analytics Config**

Hot-swap endpoint usage analytics enablement and retention days.

**Request Body** (required)

- `application/json`: [`UpdateAnalyticsConfigRequest`](#schema-updateanalyticsconfigrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `enabled` | boolean \| null | no | - |
| `retention_days` | integer \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`AnalyticsConfigResponse`](#schema-analyticsconfigresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/config/cors`

**Get Cors Runtime Config**

Return live browser CORS origins and allowed headers.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`CorsRuntimeConfigResponse`](#schema-corsruntimeconfigresponse) |

### `PATCH /v1/system/config/cors`

**Patch Cors Runtime Config**

Hot-swap browser CORS origins and headers live without restarting.

**Request Body** (required)

- `application/json`: [`UpdateCorsConfigRequest`](#schema-updatecorsconfigrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `cors_allowed_origins` | array[string] \| null | no | - |
| `cors_extra_allow_headers` | array[string] \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`CorsRuntimeConfigResponse`](#schema-corsruntimeconfigresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/config/guardrails`

**Get Guardrails Runtime Config**

Return live pre-flight guardrail enforcement policy.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`GuardrailsPolicyConfigResponse`](#schema-guardrailspolicyconfigresponse) |

### `PATCH /v1/system/config/guardrails`

**Patch Guardrails Runtime Config**

Hot-swap guardrail enforcement policy without restarting inference subprocesses.

**Request Body** (required)

- `application/json`: [`UpdateGuardrailsPolicyRequest`](#schema-updateguardrailspolicyrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `block_mode` | [`GuardrailBlockMode`](#schema-guardrailblockmode) \| null | no | - |
| `enabled` | boolean \| null | no | - |
| `fail_open` | boolean \| null | no | - |
| `refusal_text` | string \| null | no | - |
| `timeout_seconds` | number \| null | no | - |
| `violation_threshold` | number \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`GuardrailsPolicyConfigResponse`](#schema-guardrailspolicyconfigresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/config/limits`

**Get Limits Runtime Config**

Return live rate limits, principal quotas, queue limits, keepalives, and log level.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`LimitsRuntimeConfigResponse`](#schema-limitsruntimeconfigresponse) |

### `PATCH /v1/system/config/limits`

**Patch Limits Runtime Config**

Hot-swap limits, quotas, and log level in place without dropping buckets or queues.

**Request Body** (required)

- `application/json`: [`UpdateLimitsConfigRequest`](#schema-updatelimitsconfigrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `ip_rate_limit_rpm` | integer \| null | no | - |
| `json_keepalive_threshold_seconds` | number \| null | no | - |
| `keepalive_interval_seconds` | number \| null | no | - |
| `log_level` | string \| null | no | - |
| `max_projects_per_principal` | integer \| null | no | - |
| `max_queued_jobs` | integer \| null | no | - |
| `max_queued_jobs_per_principal` | integer \| null | no | - |
| `max_storage_bytes_per_principal` | integer \| null | no | - |
| `max_upload_bytes` | integer \| null | no | - |
| `rate_limit_rpm` | integer \| null | no | - |
| `trust_cloudflare_headers` | boolean \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`LimitsRuntimeConfigResponse`](#schema-limitsruntimeconfigresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/config/oidc`

**Get Oidc Runtime Config**

Return live OIDC audiences, allowlists, admin emails, scopes, and leeway.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`OidcRuntimeConfigResponse`](#schema-oidcruntimeconfigresponse) |

### `PATCH /v1/system/config/oidc`

**Patch Oidc Runtime Config**

Hot-swap OIDC authorization policy in place while preserving cached JWKS.

**Request Body** (required)

- `application/json`: [`UpdateOidcConfigRequest`](#schema-updateoidcconfigrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `oidc_admin_emails` | array[string] \| null | no | - |
| `oidc_allowed_domains` | array[string] \| null | no | - |
| `oidc_allowed_emails` | array[string] \| null | no | - |
| `oidc_audiences` | array[string] \| null | no | - |
| `oidc_default_scopes` | array[string] \| null | no | - |
| `oidc_leeway_seconds` | number \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`OidcRuntimeConfigResponse`](#schema-oidcruntimeconfigresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/system/config/reload`

**Reload Scoped Config**

Reload only the explicitly requested section(s) from YAML and environment variables.

**Request Body** (required)

- `application/json`: [`ScopedConfigReloadRequest`](#schema-scopedconfigreloadrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `sections` | array[string] | yes | minItems=1 |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ScopedConfigReloadResponse`](#schema-scopedconfigreloadresponse) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/system/status`

**Get System Status**

Return GPU telemetry, backend health, spooler snapshot, and alert count.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`SystemStatusResponse`](#schema-systemstatusresponse) |


### Tag: `tools`

### `GET /v1/tool-approvals/{approval_id}`

**Get Tool Approval**

Fetch status and details for an accessible tool approval request.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `approval_id` | path | yes | string | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ToolApprovalRecord`](#schema-toolapprovalrecord) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/tool-approvals/{approval_id}/resolve`

**Resolve Tool Approval**

Approve or deny a pending tool execution request (requires TOOLS_APPROVE).

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `approval_id` | path | yes | string | - |

**Request Body** (required)

- `application/json`: [`ResolveApprovalRequest`](#schema-resolveapprovalrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `approved` | boolean | yes | - |
| `note` | string \| null | no | - |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ToolApprovalRecord`](#schema-toolapprovalrecord) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `POST /v1/tool-approvals/{approval_id}/result`

**Submit Tool Result**

Submit a client-executed tool result and resume the paused session turn.

**Parameters**

| Name | In | Required | Type | Details |
| --- | --- | --- | --- | --- |
| `approval_id` | path | yes | string | - |

**Request Body** (required)

- `application/json`: [`SubmitToolResultRequest`](#schema-submittoolresultrequest)

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `duration_ms` | number | no | default=`0.0`; minimum=0.0 |
| `error` | string \| null | no | - |
| `output` | string | yes | - |
| `stream` | boolean | no | default=`False` |
| `success` | boolean | no | default=`True` |

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`TurnOutput`](#schema-turnoutput) |
| `422` | Validation Error | `application/json`: [`HTTPValidationError`](#schema-httpvalidationerror) |

### `GET /v1/tools`

**List Tools**

List registered tool plugins and their permission tiers.

**Responses**

| Status | Description | Schema |
| --- | --- | --- |
| `200` | Successful Response | `application/json`: [`ToolListResponse`](#schema-toollistresponse) |


## Component Schemas

<a id="schema-alertcategory"></a>
### `AlertCategory`

Failure classification attached to automated alert events.

**Enum values**: `timeout`, `fatal_exception`, `subprocess_crash`, `storage_failure`, `security_alert`, `application_error`

<a id="schema-alertevent"></a>
### `AlertEvent`

Structured alert emitted on fatal errors, crashes, or timeouts.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `category` | [`AlertCategory`](#schema-alertcategory) | yes | - |
| `component` | string | yes | minLength=1 |
| `created_at` | string (date-time) | no | - |
| `details` | object[str, string] | no | - |
| `exception_type` | string \| null | no | - |
| `id` | string | no | - |
| `job_id` | string \| null | no | - |
| `message` | string | yes | - |
| `project_id` | string \| null | no | - |
| `request_id` | string \| null | no | - |
| `session_id` | string \| null | no | - |
| `severity` | [`AlertSeverity`](#schema-alertseverity) | yes | - |
| `traceback_excerpt` | string \| null | no | - |

<a id="schema-alertseverity"></a>
### `AlertSeverity`

Severity level for automated operational alerts.

**Enum values**: `error`, `critical`

<a id="schema-alertsinkdeliveryresult"></a>
### `AlertSinkDeliveryResult`

Per-sink delivery outcome returned when sending synthetic test alerts.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `delivered` | boolean | yes | - |
| `error` | string \| null | no | - |
| `sink_name` | string | yes | minLength=1 |
| `skipped` | boolean | no | default=`False` |

<a id="schema-alertingconfigresponse"></a>
### `AlertingConfigResponse`

Redacted live operational alerting configuration (never returns secrets).

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `alert_cooldown_seconds` | number | yes | minimum=0.0 |
| `alert_on_errors` | boolean | yes | - |
| `alert_on_timeouts` | boolean | yes | - |
| `alert_webhook_categories` | array[[`AlertCategory`](#schema-alertcategory)] | no | - |
| `alert_webhook_configured` | boolean | yes | - |
| `alert_webhook_enabled` | boolean | yes | - |
| `alert_webhook_min_severity` | [`AlertSeverity`](#schema-alertseverity) | yes | - |
| `alert_webhook_target` | string \| null | no | - |
| `effective_smtp_recipients` | array[string] | no | - |
| `smtp_categories` | array[[`AlertCategory`](#schema-alertcategory)] | no | - |
| `smtp_enabled` | boolean | yes | - |
| `smtp_from_address` | string \| null | no | - |
| `smtp_host` | string \| null | no | - |
| `smtp_min_severity` | [`AlertSeverity`](#schema-alertseverity) | yes | - |
| `smtp_password_configured` | boolean | yes | - |
| `smtp_port` | integer | yes | minimum=1.0; maximum=65535.0 |
| `smtp_subject_prefix` | string | yes | - |
| `smtp_timeout_seconds` | number | yes | maximum=60.0; exclusiveMinimum=0.0 |
| `smtp_tls_mode` | [`SmtpTlsMode`](#schema-smtptlsmode) | yes | - |
| `smtp_to_addresses` | array[string] | no | - |
| `smtp_use_admin_emails_fallback` | boolean | yes | - |
| `smtp_username` | string \| null | no | - |

<a id="schema-alertslistresponse"></a>
### `AlertsListResponse`

Recent operational alerts captured by the AlertManager.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `items` | array[[`AlertEvent`](#schema-alertevent)] | yes | - |

<a id="schema-analyticsconfigresponse"></a>
### `AnalyticsConfigResponse`

Live configuration for built-in endpoint usage analytics.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `enabled` | boolean | yes | - |
| `retention_days` | integer | yes | minimum=1.0; maximum=3650.0 |

<a id="schema-analyticsreport"></a>
### `AnalyticsReport`

Windowed usage analytics report returned by GET /v1/system/analytics.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `endpoints` | array[[`EndpointUsageSummary`](#schema-endpointusagesummary)] | no | - |
| `error_count` | integer | yes | minimum=0.0 |
| `error_rate` | number | yes | minimum=0.0; maximum=1.0 |
| `generated_at` | string (date-time) | no | - |
| `hourly` | array[[`HourlyUsageBucket`](#schema-hourlyusagebucket)] | no | - |
| `principals` | array[[`PrincipalUsageSummary`](#schema-principalusagesummary)] | no | - |
| `total_completion_tokens` | integer | yes | minimum=0.0 |
| `total_prompt_tokens` | integer | yes | minimum=0.0 |
| `total_requests` | integer | yes | minimum=0.0 |
| `total_tokens` | integer | yes | minimum=0.0 |
| `unique_client_apps` | integer | yes | minimum=0.0 |
| `unique_principals` | integer | yes | minimum=0.0 |
| `window_hours` | integer | yes | minimum=1.0 |

<a id="schema-apiscope"></a>
### `ApiScope`

Fine-grained permission scopes granted to API keys and principals.

**Enum values**: `projects:read`, `projects:write`, `files:read`, `files:write`, `chat:read`, `chat:write`, `tools:safe`, `tools:exec`, `tools:approve`, `admin`

<a id="schema-approvalstatus"></a>
### `ApprovalStatus`

Lifecycle status of an explicit human tool approval request.

**Enum values**: `pending`, `approved`, `denied`, `expired`, `executed`

<a id="schema-backendhealth"></a>
### `BackendHealth`

Health snapshot reported by the supervised llama-server process.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `error` | string \| null | no | - |
| `healthy` | boolean | yes | - |
| `latency_ms` | number \| null | no | - |
| `model_loaded` | string \| null | no | - |
| `slots_idle` | integer | no | default=`0`; minimum=0.0 |
| `slots_total` | integer | no | default=`0`; minimum=0.0 |
| `status` | string | yes | minLength=1 |

<a id="schema-body_upload_file_multipart_v1_files_post"></a>
### `Body_upload_file_multipart_v1_files_post`

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `file` | string | yes | - |
| `project_id` | string | yes | minLength=1 |
| `wait` | boolean \| null | no | - |

<a id="schema-citation"></a>
### `Citation`

Line-accurate provenance pointer into a project file.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `chunk_id` | string \| null | no | - |
| `end_line` | integer \| null | no | - |
| `excerpt` | string \| null | no | - |
| `file_id` | string | yes | minLength=1 |
| `filename` | string | yes | minLength=1 |
| `start_line` | integer \| null | no | - |

<a id="schema-clientapplistresponse"></a>
### `ClientAppListResponse`

List of registered client application metadata records.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `items` | array[[`ClientAppResponse`](#schema-clientappresponse)] | yes | - |

<a id="schema-clientappresponse"></a>
### `ClientAppResponse`

Public metadata for a registered OIDC-bound client application.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `active` | boolean | yes | - |
| `allowed_project_ids` | array[string] | yes | - |
| `allowed_tiers` | array[[`PermissionTier`](#schema-permissiontier)] | yes | - |
| `created_at` | string (date-time) | yes | - |
| `expires_at` | string (date-time) \| null | no | - |
| `id` | string | yes | minLength=1 |
| `last_used_at` | string (date-time) \| null | no | - |
| `name` | string | yes | minLength=1; maxLength=120 |
| `oauth_client_id` | string | yes | minLength=1; maxLength=256 |
| `registered_tools` | array[[`ToolSpec`](#schema-toolspec)] | yes | - |
| `revoked_at` | string (date-time) \| null | no | - |

<a id="schema-clienttoolcallevent"></a>
### `ClientToolCallEvent`

Event emitted when a client-executed tool invocation pauses for caller execution.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `approval_id` | string | yes | minLength=1 |
| `arguments_json` | string | yes | - |
| `event_type` | string | no | const=`'client_tool_call'`; default=`'client_tool_call'` |
| `expires_at` | string (date-time) \| null | no | - |
| `requires_approval` | boolean | yes | - |
| `tool_call_id` | string | yes | minLength=1 |
| `tool_name` | string | yes | minLength=1 |

<a id="schema-codefilesummary"></a>
### `CodeFileSummary`

Deterministic structural summary extracted from a source code file.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `imports` | array[string] | no | default=`[]` |
| `language` | string | yes | minLength=1 |
| `symbols` | array[string] | no | default=`[]` |
| `total_lines` | integer | yes | minimum=0.0 |

<a id="schema-contextmode"></a>
### `ContextMode`

Strategy used by the orchestrator to feed attached files into context.

**Enum values**: `auto`, `direct`, `map_reduce`, `retrieval`

<a id="schema-corsruntimeconfigresponse"></a>
### `CorsRuntimeConfigResponse`

Live browser CORS origin and allowed-header policy.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `cors_allowed_origins` | array[string] | no | - |
| `cors_extra_allow_headers` | array[string] | no | - |
| `effective_allow_headers` | array[string] | no | - |

<a id="schema-createclientapprequest"></a>
### `CreateClientAppRequest`

Payload for registering a new OIDC-bound client application.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `allowed_project_ids` | array[string] | no | default=`[]` |
| `allowed_tiers` | array[[`PermissionTier`](#schema-permissiontier)] | no | default=`['read']` |
| `expires_at` | string (date-time) \| null | no | - |
| `name` | string | yes | minLength=1; maxLength=120 |
| `oauth_client_id` | string | yes | minLength=1; maxLength=256 |
| `registered_tools` | array[[`ToolSpec`](#schema-toolspec)] | no | default=`[]` |

<a id="schema-deleteresponse"></a>
### `DeleteResponse`

Confirmation envelope returned by DELETE endpoints.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `deleted` | boolean | yes | - |
| `id` | string | yes | minLength=1 |

<a id="schema-endpointusagesummary"></a>
### `EndpointUsageSummary`

Aggregated request volume, status class breakdown, and latency per route template.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `avg_duration_ms` | number | yes | minimum=0.0 |
| `max_duration_ms` | number | yes | minimum=0.0 |
| `method` | string | yes | minLength=1 |
| `request_count` | integer | yes | minimum=0.0 |
| `route` | string | yes | minLength=1 |
| `status_2xx` | integer | yes | minimum=0.0 |
| `status_4xx` | integer | yes | minimum=0.0 |
| `status_5xx` | integer | yes | minimum=0.0 |

<a id="schema-exceptionfingerprint"></a>
### `ExceptionFingerprint`

Aggregated signature for repeated exceptions or error patterns in a log.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=1.0 |
| `exception_type` | string | yes | minLength=1 |
| `fingerprint` | string | yes | minLength=1 |
| `first_line` | integer | yes | minimum=1.0 |
| `last_line` | integer | yes | minimum=1.0 |
| `message_sample` | string | yes | - |

<a id="schema-filechunk"></a>
### `FileChunk`

Indexed line-ranged chunk extracted from a project file.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `chunk_index` | integer | yes | minimum=0.0 |
| `end_line` | integer | yes | minimum=1.0 |
| `file_id` | string | yes | minLength=1 |
| `id` | string | no | - |
| `metadata` | object[str, string] | no | - |
| `project_id` | string | yes | minLength=1 |
| `severity` | [`LogSeverity`](#schema-logseverity) \| null | no | - |
| `start_line` | integer | yes | minimum=1.0 |
| `symbol_name` | string \| null | no | - |
| `text` | string | yes | - |
| `token_count` | integer | yes | minimum=0.0 |

<a id="schema-filechunklistresponse"></a>
### `FileChunkListResponse`

Ordered list of indexed chunks for a specific project file.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `file_id` | string | yes | minLength=1 |
| `items` | array[[`FileChunk`](#schema-filechunk)] | yes | - |

<a id="schema-filekind"></a>
### `FileKind`

Logical classification of an ingested project file.

**Enum values**: `log`, `code`, `text`, `markdown`, `json`, `other`

<a id="schema-filelistresponse"></a>
### `FileListResponse`

List of uploaded files within a project.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `items` | array[[`FileRecord`](#schema-filerecord)] | yes | - |

<a id="schema-filerecord"></a>
### `FileRecord`

Persisted metadata for an uploaded file within a project.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `blob_path` | string | yes | minLength=1 |
| `chunk_count` | integer | no | default=`0`; minimum=0.0 |
| `code_summary` | [`CodeFileSummary`](#schema-codefilesummary) \| null | no | - |
| `content_type` | string | no | default=`'application/octet-stream'` |
| `created_at` | string (date-time) | no | - |
| `error_message` | string \| null | no | - |
| `filename` | string | yes | minLength=1 |
| `id` | string | no | - |
| `kind` | [`FileKind`](#schema-filekind) | yes | - |
| `log_summary` | [`LogFileSummary`](#schema-logfilesummary) \| null | no | - |
| `project_id` | string | yes | minLength=1 |
| `sha256` | string | yes | minLength=64; maxLength=64 |
| `size_bytes` | integer | yes | minimum=0.0 |
| `status` | [`FileStatus`](#schema-filestatus) | no | default=`'pending'` |
| `token_count` | integer | no | default=`0`; minimum=0.0 |
| `updated_at` | string (date-time) | no | - |

<a id="schema-filestatus"></a>
### `FileStatus`

Ingestion lifecycle status for an uploaded file.

**Enum values**: `pending`, `processing`, `ready`, `failed`

<a id="schema-gpustatusdto"></a>
### `GpuStatusDTO`

GPU utilization and VRAM snapshot reported by nvidia-smi.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `gpu_name` | string | yes | minLength=1 |
| `memory_total_mb` | integer | yes | minimum=0.0 |
| `memory_used_mb` | integer | yes | minimum=0.0 |
| `temperature_c` | number \| null | no | - |
| `utilization_percent` | integer | yes | minimum=0.0; maximum=100.0 |

<a id="schema-guardrailblockmode"></a>
### `GuardrailBlockMode`

Response mode when a prompt is blocked by the guardrail policy.

**Enum values**: `refusal_message`, `http_error`

<a id="schema-guardrailspolicyconfigresponse"></a>
### `GuardrailsPolicyConfigResponse`

Live pre-flight guardrail enforcement policy.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `backend` | string | yes | enum: `llamacpp`, `ollama` |
| `block_mode` | [`GuardrailBlockMode`](#schema-guardrailblockmode) | yes | - |
| `enabled` | boolean | yes | - |
| `fail_open` | boolean | yes | - |
| `refusal_text` | string | yes | - |
| `timeout_seconds` | number | yes | exclusiveMinimum=0.0 |
| `violation_threshold` | number | yes | exclusiveMinimum=0.0 |

<a id="schema-httpvalidationerror"></a>
### `HTTPValidationError`

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `detail` | array[[`ValidationError`](#schema-validationerror)] | no | - |

<a id="schema-healthresponse"></a>
### `HealthResponse`

Unauthenticated liveness probe payload.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `status` | string | no | const=`'ok'`; default=`'ok'` |
| `version` | string | no | default=`'0.1.0'` |

<a id="schema-hourlyusagebucket"></a>
### `HourlyUsageBucket`

UTC hourly rollup point for time-series request, error, and token trends.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `bucket_hour` | string (date-time) | yes | - |
| `error_count` | integer | yes | minimum=0.0 |
| `request_count` | integer | yes | minimum=0.0 |
| `total_tokens` | integer | yes | minimum=0.0 |

<a id="schema-inferencetoolcall"></a>
### `InferenceToolCall`

Tool invocation emitted by the inference model.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `arguments_json` | string | yes | - |
| `id` | string | yes | minLength=1 |
| `name` | string | yes | minLength=1 |

<a id="schema-jobpriority"></a>
### `JobPriority`

Priority order for inference spooler scheduling (lower runs first).

**Enum values**: `10`, `20`, `30`

<a id="schema-limitsruntimeconfigresponse"></a>
### `LimitsRuntimeConfigResponse`

Live rate limits, per-principal quotas, upload limits, keepalive, and log level.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `ip_rate_limit_rpm` | integer | yes | minimum=1.0 |
| `json_keepalive_threshold_seconds` | number | yes | exclusiveMinimum=0.0 |
| `keepalive_interval_seconds` | number | yes | exclusiveMinimum=0.0 |
| `log_level` | string | yes | - |
| `max_projects_per_principal` | integer | yes | minimum=1.0 |
| `max_queued_jobs` | integer | yes | minimum=1.0 |
| `max_queued_jobs_per_principal` | integer | yes | minimum=1.0 |
| `max_storage_bytes_per_principal` | integer | yes | minimum=1.0 |
| `max_upload_bytes` | integer | yes | minimum=1.0 |
| `rate_limit_rpm` | integer | yes | minimum=1.0 |
| `trust_cloudflare_headers` | boolean | yes | - |

<a id="schema-logfilesummary"></a>
### `LogFileSummary`

Deterministic structural summary extracted from a log file.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `error_count` | integer | no | default=`0`; minimum=0.0 |
| `first_timestamp` | string \| null | no | - |
| `last_timestamp` | string \| null | no | - |
| `severity_counts` | object[str, integer] | no | - |
| `top_exceptions` | array[[`ExceptionFingerprint`](#schema-exceptionfingerprint)] | no | default=`[]` |
| `total_lines` | integer | yes | minimum=0.0 |
| `warning_count` | integer | no | default=`0`; minimum=0.0 |

<a id="schema-logseverity"></a>
### `LogSeverity`

Normalized log entry severity levels.

**Enum values**: `trace`, `debug`, `info`, `warning`, `error`, `critical`

<a id="schema-message"></a>
### `Message`

Persisted message within a chat session.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `citations` | array[[`Citation`](#schema-citation)] | no | default=`[]` |
| `content` | string | yes | - |
| `created_at` | string (date-time) | no | - |
| `id` | string | no | - |
| `project_id` | string | yes | minLength=1 |
| `role` | [`MessageRole`](#schema-messagerole) | yes | - |
| `session_id` | string | yes | minLength=1 |
| `thinking` | string \| null | no | - |
| `tool_call_id` | string \| null | no | - |
| `tool_calls` | array[[`InferenceToolCall`](#schema-inferencetoolcall)] | no | default=`[]` |
| `tool_name` | string \| null | no | - |
| `usage` | [`TokenUsage`](#schema-tokenusage) \| null | no | - |

<a id="schema-messagerole"></a>
### `MessageRole`

Conversation message role.

**Enum values**: `system`, `user`, `assistant`, `tool`

<a id="schema-oidcruntimeconfigresponse"></a>
### `OidcRuntimeConfigResponse`

Live OIDC authorization policy settings.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `enabled` | boolean | yes | - |
| `issuer_url` | string | yes | - |
| `jwks_url` | string | yes | - |
| `oidc_admin_emails` | array[string] | no | - |
| `oidc_allowed_domains` | array[string] | no | - |
| `oidc_allowed_emails` | array[string] | no | - |
| `oidc_audiences` | array[string] | no | - |
| `oidc_default_scopes` | array[string] | no | - |
| `oidc_leeway_seconds` | number | yes | minimum=0.0; maximum=300.0 |

<a id="schema-openaichatchoice"></a>
### `OpenAIChatChoice`

Single completion choice entry.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `finish_reason` | string \| null | no | default=`'stop'` |
| `index` | integer | no | default=`0` |
| `message` | [`OpenAIChatChoiceMessage`](#schema-openaichatchoicemessage) | yes | - |

<a id="schema-openaichatchoicemessage"></a>
### `OpenAIChatChoiceMessage`

Assistant message object returned in a non-streaming completion choice.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `content` | string | yes | - |
| `reasoning_content` | string \| null | no | - |
| `role` | string | no | const=`'assistant'`; default=`'assistant'` |
| `tool_calls` | array[[`OpenAIToolCall`](#schema-openaitoolcall)] \| null | no | - |

<a id="schema-openaichatcompletionrequest"></a>
### `OpenAIChatCompletionRequest`

Drop-in OpenAI /v1/chat/completions payload with optional LAI routing fields.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `client_tools` | array[[`ToolSpec`](#schema-toolspec)] | no | - |
| `context_mode` | [`ContextMode`](#schema-contextmode) \| null | no | - |
| `enable_thinking` | boolean | no | default=`True` |
| `enable_tools` | boolean | no | default=`True` |
| `file_ids` | array[string] \| null | no | - |
| `max_completion_tokens` | integer \| null | no | - |
| `max_tokens` | integer \| null | no | - |
| `messages` | array[[`OpenAIChatMessageInput`](#schema-openaichatmessageinput)] | yes | minItems=1 |
| `model` | string \| null | no | - |
| `project_id` | string \| null | no | - |
| `session_id` | string \| null | no | - |
| `stream` | boolean | no | default=`False` |
| `temperature` | number \| null | no | - |

<a id="schema-openaichatcompletionresponse"></a>
### `OpenAIChatCompletionResponse`

Non-streaming OpenAI chat completion response envelope.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `choices` | array[[`OpenAIChatChoice`](#schema-openaichatchoice)] | yes | - |
| `created` | integer | yes | - |
| `id` | string | yes | minLength=1 |
| `lai_client_tool_call` | [`ClientToolCallEvent`](#schema-clienttoolcallevent) \| null | no | - |
| `model` | string | yes | minLength=1 |
| `object` | string | no | const=`'chat.completion'`; default=`'chat.completion'` |
| `usage` | [`OpenAIUsage`](#schema-openaiusage) | yes | - |

<a id="schema-openaichatmessageinput"></a>
### `OpenAIChatMessageInput`

Incoming OpenAI message supporting text strings or multimodal text parts.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `content` | string \| array[object] \| null | no | - |
| `name` | string \| null | no | - |
| `role` | string | yes | minLength=1 |

<a id="schema-openaimodelinfo"></a>
### `OpenAIModelInfo`

Single model descriptor returned by GET /v1/models.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `created` | integer | no | default=`0` |
| `id` | string | yes | minLength=1 |
| `object` | string | no | const=`'model'`; default=`'model'` |
| `owned_by` | string | no | default=`'local'` |

<a id="schema-openaimodellistresponse"></a>
### `OpenAIModelListResponse`

OpenAI-compatible model listing envelope.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `data` | array[[`OpenAIModelInfo`](#schema-openaimodelinfo)] | yes | - |
| `object` | string | no | const=`'list'`; default=`'list'` |

<a id="schema-openaitoolcall"></a>
### `OpenAIToolCall`

Non-streaming OpenAI assistant tool_calls item.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `function` | [`OpenAIToolFunction`](#schema-openaitoolfunction) | yes | - |
| `id` | string | yes | minLength=1 |
| `type` | string | no | const=`'function'`; default=`'function'` |

<a id="schema-openaitoolfunction"></a>
### `OpenAIToolFunction`

Function name and JSON-encoded arguments inside an OpenAI tool call.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `arguments` | string | yes | - |
| `name` | string | yes | minLength=1 |

<a id="schema-openaiusage"></a>
### `OpenAIUsage`

Token usage breakdown compatible with OpenAI responses.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `completion_tokens` | integer | no | default=`0`; minimum=0.0 |
| `prompt_tokens` | integer | no | default=`0`; minimum=0.0 |
| `total_tokens` | integer | no | default=`0`; minimum=0.0 |

<a id="schema-permissiontier"></a>
### `PermissionTier`

Risk classification governing default tool execution policy.

**Enum values**: `read`, `write`, `exec`, `network`

<a id="schema-principalusagesummary"></a>
### `PrincipalUsageSummary`

Aggregated request, error, and token totals per opaque principal and OAuth client app.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `completion_tokens` | integer | yes | minimum=0.0 |
| `error_count` | integer | yes | minimum=0.0 |
| `oauth_client_id` | string \| null | no | - |
| `principal_id` | string | yes | minLength=1 |
| `prompt_tokens` | integer | yes | minimum=0.0 |
| `request_count` | integer | yes | minimum=0.0 |
| `total_tokens` | integer | yes | minimum=0.0 |

<a id="schema-project"></a>
### `Project`

Isolated workspace project containing files, sessions, and indexes.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `created_at` | string (date-time) | no | - |
| `default_model_profile` | string | no | default=`'gemma4-26b-a4b'` |
| `description` | string | no | default=`''` |
| `id` | string | no | - |
| `name` | string | yes | minLength=1; maxLength=120 |
| `owner_principal_id` | string | no | default=`'system'`; minLength=1 |
| `updated_at` | string (date-time) | no | - |

<a id="schema-projectcreaterequest"></a>
### `ProjectCreateRequest`

Payload for creating a new isolated workspace project.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `default_model_profile` | string | no | default=`'gemma4-26b-a4b'` |
| `description` | string | no | default=`''` |
| `name` | string | yes | minLength=1; maxLength=120 |

<a id="schema-projectlistresponse"></a>
### `ProjectListResponse`

Paginated list of workspace projects.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `items` | array[[`Project`](#schema-project)] | yes | - |

<a id="schema-resolveapprovalrequest"></a>
### `ResolveApprovalRequest`

Payload for approving or denying a pending high-risk tool invocation.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `approved` | boolean | yes | - |
| `note` | string \| null | no | - |

<a id="schema-scopedconfigreloadrequest"></a>
### `ScopedConfigReloadRequest`

Explicit list of configuration sections to reload from YAML and environment variables.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `sections` | array[string] | yes | minItems=1 |

<a id="schema-scopedconfigreloadresponse"></a>
### `ScopedConfigReloadResponse`

Summary of sections reloaded from YAML/env versus left untouched.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `config_path` | string \| null | no | - |
| `reloaded_sections` | array[string] | no | - |
| `unchanged_sections` | array[string] | no | - |

<a id="schema-session"></a>
### `Session`

Conversation session scoped to a single project.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `created_at` | string (date-time) | no | - |
| `id` | string | no | - |
| `model_profile` | string | no | default=`'gemma4-26b-a4b'` |
| `project_id` | string | yes | minLength=1 |
| `system_prompt` | string \| null | no | - |
| `title` | string | no | default=`'New Session'`; minLength=1; maxLength=200 |
| `updated_at` | string (date-time) | no | - |

<a id="schema-sessioncreaterequest"></a>
### `SessionCreateRequest`

Payload for creating a new chat session within a project.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `model_profile` | string \| null | no | - |
| `project_id` | string | yes | minLength=1 |
| `system_prompt` | string \| null | no | - |
| `title` | string | no | default=`'New Session'`; minLength=1; maxLength=200 |

<a id="schema-sessiondetailresponse"></a>
### `SessionDetailResponse`

Session metadata paired with chronological message history.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `messages` | array[[`Message`](#schema-message)] | yes | - |
| `session` | [`Session`](#schema-session) | yes | - |

<a id="schema-sessionlistresponse"></a>
### `SessionListResponse`

Paginated list of chat sessions for a project.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `items` | array[[`Session`](#schema-session)] | yes | - |

<a id="schema-sessionmessagerequest"></a>
### `SessionMessageRequest`

User turn payload posted to a chat session.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `attached_file_ids` | array[string] | no | default=`[]` |
| `client_tools` | array[[`ToolSpec`](#schema-toolspec)] | no | default=`[]` |
| `content` | string | yes | minLength=1 |
| `context_mode` | [`ContextMode`](#schema-contextmode) | no | default=`'auto'` |
| `enable_thinking` | boolean | no | default=`True` |
| `enable_tools` | boolean | no | default=`True` |
| `max_tokens` | integer \| null | no | - |
| `stream` | boolean | no | default=`False` |
| `temperature` | number \| null | no | - |

<a id="schema-smtptlsmode"></a>
### `SmtpTlsMode`

Transport security mode used when connecting to an outbound SMTP server.

**Enum values**: `starttls`, `ssl`, `none`

<a id="schema-spoolersnapshot"></a>
### `SpoolerSnapshot`

Point-in-time status of the single-slot inference spooler.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `active_job_id` | string \| null | no | - |
| `active_priority` | [`JobPriority`](#schema-jobpriority) \| null | no | - |
| `active_session_id` | string \| null | no | - |
| `avg_wait_ms` | number | no | default=`0.0`; minimum=0.0 |
| `max_queued_jobs` | integer | no | default=`64`; minimum=1.0 |
| `queued_by_priority` | object[str, integer] | no | - |
| `queued_count` | integer | no | default=`0`; minimum=0.0 |
| `total_completed` | integer | no | default=`0`; minimum=0.0 |
| `total_evicted` | integer | no | default=`0`; minimum=0.0 |
| `total_rejected` | integer | no | default=`0`; minimum=0.0 |

<a id="schema-submittoolresultrequest"></a>
### `SubmitToolResultRequest`

Payload posted by a client application to return client-executed tool output.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `duration_ms` | number | no | default=`0.0`; minimum=0.0 |
| `error` | string \| null | no | - |
| `output` | string | yes | - |
| `stream` | boolean | no | default=`False` |
| `success` | boolean | no | default=`True` |

<a id="schema-systemstatusresponse"></a>
### `SystemStatusResponse`

Comprehensive runtime health, GPU, guardrail, and spooler telemetry.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `active_model_profile` | string | yes | minLength=1 |
| `gpu` | [`GpuStatusDTO`](#schema-gpustatusdto) \| null | no | - |
| `guardrail_health` | [`BackendHealth`](#schema-backendhealth) \| null | no | - |
| `inference_health` | [`BackendHealth`](#schema-backendhealth) | yes | - |
| `recent_alert_count` | integer | yes | minimum=0.0 |
| `spooler` | [`SpoolerSnapshot`](#schema-spoolersnapshot) | yes | - |
| `status` | string | yes | enum: `ok`, `degraded` |

<a id="schema-testalertrequest"></a>
### `TestAlertRequest`

Payload for dispatching a synthetic test alert through active alert sinks.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `bypass_filters` | boolean | no | default=`True` |
| `category` | [`AlertCategory`](#schema-alertcategory) | no | default=`'application_error'` |
| `message` | string | no | default=`'Synthetic pre-flight test alert from LAI.'`; minLength=1; maxLength=500 |
| `severity` | [`AlertSeverity`](#schema-alertseverity) | no | default=`'critical'` |

<a id="schema-testalertresponse"></a>
### `TestAlertResponse`

Per-sink delivery results from POST /v1/system/alerting/test.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `alert_id` | string | yes | minLength=1 |
| `delivered_count` | integer | yes | minimum=0.0 |
| `failed_count` | integer | yes | minimum=0.0 |
| `results` | array[[`AlertSinkDeliveryResult`](#schema-alertsinkdeliveryresult)] | no | - |
| `skipped_count` | integer | yes | minimum=0.0 |

<a id="schema-tokenusage"></a>
### `TokenUsage`

Token accounting reported by an inference turn.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `completion_tokens` | integer | no | default=`0`; minimum=0.0 |
| `prompt_tokens` | integer | no | default=`0`; minimum=0.0 |
| `total_tokens` | integer | no | default=`0`; minimum=0.0 |

<a id="schema-toolapprovalrecord"></a>
### `ToolApprovalRecord`

Persisted approval gate record for a high-risk tool call.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `allowed_file_ids` | array[string] | no | default=`[]` |
| `arguments_json` | string | yes | - |
| `client_app_id` | string \| null | no | - |
| `created_at` | string (date-time) | no | - |
| `decided_at` | string (date-time) \| null | no | - |
| `decided_by` | string \| null | no | - |
| `decision_note` | string \| null | no | - |
| `execution_result` | [`ToolResult`](#schema-toolresult) \| null | no | - |
| `execution_target` | [`ToolExecutionTarget`](#schema-toolexecutiontarget) | no | default=`'server_attachment'` |
| `expires_at` | string (date-time) \| null | no | - |
| `id` | string | no | - |
| `permission_tier` | [`PermissionTier`](#schema-permissiontier) | yes | - |
| `principal_scopes` | array[[`ApiScope`](#schema-apiscope)] | no | default=`[]` |
| `project_id` | string | yes | minLength=1 |
| `reason` | string | yes | - |
| `requested_by` | string | yes | minLength=1 |
| `session_id` | string | yes | minLength=1 |
| `status` | [`ApprovalStatus`](#schema-approvalstatus) | no | default=`'pending'` |
| `tool_call_id` | string \| null | no | - |
| `tool_name` | string | yes | minLength=1 |

<a id="schema-toolexecutiontarget"></a>
### `ToolExecutionTarget`

Execution plane responsible for running a tool plugin.

**Enum values**: `server_attachment`, `client`

<a id="schema-toollistresponse"></a>
### `ToolListResponse`

Catalog of registered tool specifications.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `count` | integer | yes | minimum=0.0 |
| `items` | array[[`ToolSpec`](#schema-toolspec)] | yes | - |

<a id="schema-toolresult"></a>
### `ToolResult`

Normalized output returned by a tool execution.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `duration_ms` | number | no | default=`0.0`; minimum=0.0 |
| `error` | string \| null | no | - |
| `output` | string | yes | - |
| `success` | boolean | yes | - |
| `tool_name` | string | yes | minLength=1 |
| `truncated` | boolean | no | default=`False` |

<a id="schema-toolspec"></a>
### `ToolSpec`

Declarative specification for a registered tool plugin.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `description` | string | yes | - |
| `execution_target` | [`ToolExecutionTarget`](#schema-toolexecutiontarget) | no | default=`'server_attachment'` |
| `name` | string | yes | minLength=1 |
| `parameters_schema` | object | no | - |
| `permission_tier` | [`PermissionTier`](#schema-permissiontier) | yes | - |
| `requires_approval_by_default` | boolean | no | default=`False` |
| `timeout_seconds` | number | no | default=`30.0`; exclusiveMinimum=0.0 |

<a id="schema-turnoutput"></a>
### `TurnOutput`

Final result of a completed orchestrator turn.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `citations` | array[[`Citation`](#schema-citation)] | no | default=`[]` |
| `context_mode_used` | [`ContextMode`](#schema-contextmode) | yes | - |
| `guardrail_blocked` | boolean | no | default=`False` |
| `guardrail_policy` | string \| null | no | - |
| `message` | [`Message`](#schema-message) | yes | - |
| `pending_approval_id` | string \| null | no | - |
| `pending_client_tool_call` | [`ClientToolCallEvent`](#schema-clienttoolcallevent) \| null | no | - |
| `session_id` | string | yes | minLength=1 |
| `usage` | [`TokenUsage`](#schema-tokenusage) | yes | - |

<a id="schema-updatealertingconfigrequest"></a>
### `UpdateAlertingConfigRequest`

Partial update for live non-secret alerting settings (secrets stay YAML/env-only).

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `alert_cooldown_seconds` | number \| null | no | - |
| `alert_on_errors` | boolean \| null | no | - |
| `alert_on_timeouts` | boolean \| null | no | - |
| `alert_webhook_categories` | array[[`AlertCategory`](#schema-alertcategory)] \| null | no | - |
| `alert_webhook_enabled` | boolean \| null | no | - |
| `alert_webhook_min_severity` | [`AlertSeverity`](#schema-alertseverity) \| null | no | - |
| `smtp_categories` | array[[`AlertCategory`](#schema-alertcategory)] \| null | no | - |
| `smtp_enabled` | boolean \| null | no | - |
| `smtp_from_address` | string \| null | no | - |
| `smtp_host` | string \| null | no | - |
| `smtp_min_severity` | [`AlertSeverity`](#schema-alertseverity) \| null | no | - |
| `smtp_port` | integer \| null | no | - |
| `smtp_subject_prefix` | string \| null | no | - |
| `smtp_timeout_seconds` | number \| null | no | - |
| `smtp_tls_mode` | [`SmtpTlsMode`](#schema-smtptlsmode) \| null | no | - |
| `smtp_to_addresses` | array[string] \| null | no | - |
| `smtp_use_admin_emails_fallback` | boolean \| null | no | - |
| `smtp_username` | string \| null | no | - |

<a id="schema-updateanalyticsconfigrequest"></a>
### `UpdateAnalyticsConfigRequest`

Partial live update for endpoint usage analytics collection and retention.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `enabled` | boolean \| null | no | - |
| `retention_days` | integer \| null | no | - |

<a id="schema-updateclientapprequest"></a>
### `UpdateClientAppRequest`

Partial update payload for an active OIDC-bound client application.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `allowed_project_ids` | array[string] \| null | no | - |
| `allowed_tiers` | array[[`PermissionTier`](#schema-permissiontier)] \| null | no | - |
| `expires_at` | string (date-time) \| null | no | - |
| `name` | string \| null | no | - |
| `registered_tools` | array[[`ToolSpec`](#schema-toolspec)] \| null | no | - |

<a id="schema-updatecorsconfigrequest"></a>
### `UpdateCorsConfigRequest`

Partial live update for CORS allowed origins and extra request headers.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `cors_allowed_origins` | array[string] \| null | no | - |
| `cors_extra_allow_headers` | array[string] \| null | no | - |

<a id="schema-updateguardrailspolicyrequest"></a>
### `UpdateGuardrailsPolicyRequest`

Partial live update for guardrail enforcement policy without restarting subprocesses.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `block_mode` | [`GuardrailBlockMode`](#schema-guardrailblockmode) \| null | no | - |
| `enabled` | boolean \| null | no | - |
| `fail_open` | boolean \| null | no | - |
| `refusal_text` | string \| null | no | - |
| `timeout_seconds` | number \| null | no | - |
| `violation_threshold` | number \| null | no | - |

<a id="schema-updatelimitsconfigrequest"></a>
### `UpdateLimitsConfigRequest`

Partial live update for rate limits, quotas, queue bounds, keepalive, and log level.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `ip_rate_limit_rpm` | integer \| null | no | - |
| `json_keepalive_threshold_seconds` | number \| null | no | - |
| `keepalive_interval_seconds` | number \| null | no | - |
| `log_level` | string \| null | no | - |
| `max_projects_per_principal` | integer \| null | no | - |
| `max_queued_jobs` | integer \| null | no | - |
| `max_queued_jobs_per_principal` | integer \| null | no | - |
| `max_storage_bytes_per_principal` | integer \| null | no | - |
| `max_upload_bytes` | integer \| null | no | - |
| `rate_limit_rpm` | integer \| null | no | - |
| `trust_cloudflare_headers` | boolean \| null | no | - |

<a id="schema-updateoidcconfigrequest"></a>
### `UpdateOidcConfigRequest`

Partial live update for OIDC audiences, allowlists, scopes, and clock leeway.

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `oidc_admin_emails` | array[string] \| null | no | - |
| `oidc_allowed_domains` | array[string] \| null | no | - |
| `oidc_allowed_emails` | array[string] \| null | no | - |
| `oidc_audiences` | array[string] \| null | no | - |
| `oidc_default_scopes` | array[string] \| null | no | - |
| `oidc_leeway_seconds` | number \| null | no | - |

<a id="schema-validationerror"></a>
### `ValidationError`

| Field | Type | Required | Details |
| --- | --- | --- | --- |
| `ctx` | object | no | - |
| `input` | any | no | - |
| `loc` | array[string \| integer] | yes | - |
| `msg` | string | yes | - |
| `type` | string | yes | - |
