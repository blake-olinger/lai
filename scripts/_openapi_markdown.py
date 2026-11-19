"""Deterministic Markdown API reference renderer from an OpenAPI 3.1 schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "options", "head"})
_MAX_ENUM_PREVIEW = 8

_OVERVIEW_SECTION = """## Authentication

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
   matched `aud` claim and attaches it to `AuthPrincipal.oauth_client_id`.
3. Client-executed tool dispatch verifies that the caller's verified `oauth_client_id` matches the
   active `ClientAppRecord.oauth_client_id` with zero client secrets transmitted or stored.

## Multi-Tenant Isolation & Per-Principal Quotas

- **Owner scoping & `404` anti-enumeration**: Projects, files, sessions, messages, and approvals are
  owned by `owner_principal_id`. Non-admin lookups or mutations targeting another principal's
  resources fail with `404 Not Found` (`not_found`) so resource IDs cannot be probed.
- **Per-principal quotas**: Fail-closed limits cap active projects (`max_projects_per_principal`),
  cumulative file bytes (`max_storage_bytes_per_principal`), and queued jobs
  (`max_queued_jobs_per_principal`).
- **Two-level fair scheduling**: The spooler schedules round-robin across `principal_id` first and
  `session_id` lanes second so one tenant cannot starve others.

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
  (`event: <event_type>\\ndata: <json>\\n\\n`) for `keepalive`, `progress`, `thinking_delta`,
  `content_delta`, `approval_required`, `client_tool_call`, `turn_completed`, and `error`.
- **`POST /v1/chat/completions` (`stream: true`)**: OpenAI-compatible
  `data: <OpenAIChatCompletionChunk>\\n\\n` frames plus `: keepalive\\n\\n` comments, including
  `lai_client_tool_call` on client tool pauses, terminating with `data: [DONE]\\n\\n`.
- **Non-streaming (`stream: false`)**: Turns exceeding `json_keepalive_threshold_seconds` switch
  to chunked whitespace heartbeats before emitting the JSON body."""


def _as_mapping(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return cast("list[object]", value) if isinstance(value, list) else []


def _escape_cell(value: object) -> str:
    return str(value).replace("\r\n", " ").replace("\n", " ").strip().replace("|", "\\|")


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", maxsplit=1)[-1]


def _format_ref_link(ref: str) -> str:
    name = _ref_name(ref)
    return f"[`{name}`](#schema-{name.lower()})"


def _format_union_type(variants: list[object]) -> str:
    parts: list[str] = []
    has_null = False
    for item in variants:
        variant = _as_mapping(item)
        if variant.get("type") == "null":
            has_null = True
        else:
            parts.append(_format_type(variant))
    if has_null:
        parts.append("null")
    return " \\| ".join(parts) if parts else "any"


def _format_scalar_or_container_type(prop: Mapping[str, object]) -> str:
    raw_type = str(prop.get("type", "any"))
    result = raw_type
    if raw_type == "array":
        item_schema = _as_mapping(prop.get("items"))
        inner = _format_type(item_schema) if item_schema else "any"
        result = f"array[{inner}]"
    elif raw_type == "object":
        addl = _as_mapping(prop.get("additionalProperties"))
        if addl:
            result = f"object[str, {_format_type(addl)}]"
    else:
        fmt = prop.get("format")
        if isinstance(fmt, str) and fmt:
            result = f"{raw_type} ({fmt})"
    return result


def _format_type(prop: Mapping[str, object]) -> str:
    ref = prop.get("$ref")
    if isinstance(ref, str):
        return _format_ref_link(ref)
    if "anyOf" in prop:
        return _format_union_type(_as_list(prop["anyOf"]))
    if "oneOf" in prop:
        return _format_union_type(_as_list(prop["oneOf"]))
    if "allOf" in prop:
        items = [_format_type(_as_mapping(v)) for v in _as_list(prop["allOf"])]
        return " & ".join(items) if items else "object"
    return _format_scalar_or_container_type(prop)


def _format_constraints(prop: Mapping[str, object]) -> str:
    parts: list[str] = []
    if "const" in prop:
        parts.append(f"const=`{prop['const']!r}`")
    if "enum" in prop:
        values = _as_list(prop["enum"])
        preview = ", ".join(f"`{v}`" for v in values[:_MAX_ENUM_PREVIEW])
        suffix = ", ..." if len(values) > _MAX_ENUM_PREVIEW else ""
        parts.append(f"enum: {preview}{suffix}")
    if "default" in prop:
        parts.append(f"default=`{prop['default']!r}`")
    bound_keys = ("minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "minItems")
    parts.extend(f"{key}={prop[key]}" for key in bound_keys if key in prop)
    desc = prop.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(_escape_cell(desc))
    return "; ".join(parts) if parts else "-"


def _render_properties_table(schema: Mapping[str, object]) -> list[str]:
    props = _as_mapping(schema.get("properties"))
    if not props:
        return []
    required = {str(item) for item in _as_list(schema.get("required"))}
    lines = ["| Field | Type | Required | Details |", "| --- | --- | --- | --- |"]
    for name in sorted(props):
        prop_map = _as_mapping(props[name])
        req_label = "yes" if name in required else "no"
        lines.append(
            f"| `{_escape_cell(name)}` | {_format_type(prop_map)} "
            f"| {req_label} | {_format_constraints(prop_map)} |"
        )
    return lines


def _render_parameters(params: list[object]) -> list[str]:
    if not params:
        return []
    lines = [
        "",
        "**Parameters**",
        "",
        "| Name | In | Required | Type | Details |",
        "| --- | --- | --- | --- | --- |",
    ]
    for raw_param in params:
        param = _as_mapping(raw_param)
        name = _escape_cell(param.get("name", ""))
        location = _escape_cell(param.get("in", ""))
        req_label = "yes" if param.get("required") else "no"
        param_schema = _as_mapping(param.get("schema"))
        type_str = _format_type(param_schema) if param_schema else "any"
        lines.append(
            f"| `{name}` | {location} | {req_label} | {type_str} "
            f"| {_format_constraints(param_schema)} |"
        )
    return lines


def _render_request_body(body_obj: object, schemas: Mapping[str, object]) -> list[str]:
    body = _as_mapping(body_obj)
    content = _as_mapping(body.get("content"))
    if not content:
        return []
    req_text = "required" if body.get("required") else "optional"
    lines = ["", f"**Request Body** ({req_text})", ""]
    for media_type in sorted(content):
        media_schema = _as_mapping(_as_mapping(content[media_type]).get("schema"))
        ref = media_schema.get("$ref")
        type_label = _format_type(media_schema) if media_schema else "any"
        lines.append(f"- `{_escape_cell(media_type)}`: {type_label}")
        target_schema = (
            _as_mapping(schemas.get(_ref_name(ref))) if isinstance(ref, str) else media_schema
        )
        table = _render_properties_table(target_schema)
        if table:
            lines.extend(["", *table])
    return lines


def _render_responses(responses_obj: object) -> list[str]:
    responses = _as_mapping(responses_obj)
    if not responses:
        return []
    lines = ["", "**Responses**", "", "| Status | Description | Schema |", "| --- | --- | --- |"]
    for status_code in sorted(responses):
        resp = _as_mapping(responses[status_code])
        desc = _escape_cell(resp.get("description", "-")) or "-"
        content = _as_mapping(resp.get("content"))
        schema_parts = [
            f"`{mt}`: {_format_type(_as_mapping(_as_mapping(content[mt]).get('schema')))}"
            for mt in sorted(content)
        ]
        schema_cell = "<br>".join(schema_parts) if schema_parts else "-"
        lines.append(f"| `{_escape_cell(status_code)}` | {desc} | {schema_cell} |")
    return lines


def _render_endpoint(
    path: str,
    method: str,
    op: Mapping[str, object],
    schemas: Mapping[str, object],
) -> list[str]:
    lines = [f"### `{method.upper()} {path}`"]
    summary = op.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.extend(["", f"**{_escape_cell(summary)}**"])
    description = op.get("description")
    if isinstance(description, str) and description.strip():
        lines.extend(["", description.strip()])
    lines.extend(_render_parameters(_as_list(op.get("parameters"))))
    lines.extend(_render_request_body(op.get("requestBody"), schemas))
    lines.extend(_render_responses(op.get("responses")))
    lines.append("")
    return lines


def _collect_operations_by_tag(
    paths: Mapping[str, object],
) -> dict[str, list[tuple[str, str, Mapping[str, object]]]]:
    grouped: dict[str, list[tuple[str, str, Mapping[str, object]]]] = {}
    for path in sorted(paths):
        path_item = _as_mapping(paths[path])
        for method in sorted(path_item):
            if method.lower() not in _HTTP_METHODS:
                continue
            op = _as_mapping(path_item[method])
            raw_tags = _as_list(op.get("tags"))
            tags = [str(tag) for tag in raw_tags if str(tag).strip()] or ["default"]
            for tag in tags:
                grouped.setdefault(tag, []).append((path, method.lower(), op))
    for entries in grouped.values():
        entries.sort(key=lambda item: (item[0], item[1]))
    return grouped


def _render_schema_entry(name: str, schema: Mapping[str, object]) -> list[str]:
    lines = [f'<a id="schema-{name.lower()}"></a>', f"### `{name}`"]
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        lines.extend(["", description.strip()])
    if "enum" in schema:
        values = ", ".join(f"`{item}`" for item in _as_list(schema["enum"]))
        lines.extend(["", f"**Enum values**: {values}"])
    table = _render_properties_table(schema)
    if table:
        lines.extend(["", *table])
    elif "enum" not in schema:
        lines.extend(["", f"**Type**: {_format_type(schema)}"])
    lines.append("")
    return lines


def render_api_reference_markdown(schema: Mapping[str, Any]) -> str:
    """Render deterministic Markdown API documentation from an OpenAPI 3.1 schema dict."""
    root = _as_mapping(schema)
    info = _as_mapping(root.get("info"))
    title = str(info.get("title", "LAI Local AI Server"))
    version = str(info.get("version", "0.1.0"))
    openapi_version = str(root.get("openapi", "3.1.0"))
    info_desc = info.get("description")

    lines: list[str] = [
        f"# {title} API Reference",
        "",
        f"- **API Version**: `{version}`",
        f"- **OpenAPI Specification**: `{openapi_version}`",
        "- **Interactive Docs**: [GitHub Pages Scalar Reference]"
        "(https://blake-olinger.github.io/lai/)",
        "",
        "> Auto-generated from the FastAPI OpenAPI 3.1 schema via "
        "`scripts/export_openapi.py`. Do not edit manually.",
    ]
    if isinstance(info_desc, str) and info_desc.strip():
        lines.extend(["", info_desc.strip()])
    lines.extend(["", _OVERVIEW_SECTION])

    schemas = _as_mapping(_as_mapping(root.get("components")).get("schemas"))
    grouped_ops = _collect_operations_by_tag(_as_mapping(root.get("paths")))

    lines.extend(["", "## Endpoints"])
    for tag in sorted(grouped_ops):
        lines.extend(["", f"### Tag: `{tag}`", ""])
        for path, method, op in grouped_ops[tag]:
            lines.extend(_render_endpoint(path, method, op, schemas))

    lines.extend(["", "## Component Schemas", ""])
    for name in sorted(schemas):
        lines.extend(_render_schema_entry(name, _as_mapping(schemas[name])))

    return "\n".join(lines).rstrip() + "\n"
