"""Unit tests for domain models, IDs, errors, settings, profiles, and logging."""

from __future__ import annotations

import secrets
from datetime import UTC
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from lai import domain as dom
from lai.config.logging import bind_request_id, configure_logging, get_logger, get_request_id
from lai.config.profiles import (
    DEFAULT_MODEL_PROFILE_NAME,
    GEMMA4_26B_A4B_PROFILE,
    get_model_profile,
)
from lai.config.settings import LaiSettings, load_settings
from lai.domain import ApiScope, AuthMethod, AuthPrincipal, ClientAppRecord, RequestCredentials
from lai.domain import errors as derr
from lai.domain import ids as did
from lai.domain.base import MutableStateModel, StrictModel, utc_now
from lai.domain.inference import ModelProfile
from lai.domain.projects import CreateProjectInput, Project
from lai.domain.scheduler import JobPriority, LeaseRequest, SpoolerSnapshot
from lai.domain.tools import PermissionTier, ToolExecutionContext

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_GEMMA4_FIELDS: dict[str, object] = {
    "name": "gemma4-26b-a4b",
    "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
    "fallback_hf_repo": "bartowski/google_gemma-4-26B-A4B-it-GGUF",
    "hf_filename": "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    "ctx_size": 24576,
    "flash_attn": True,
    "kv_cache_type_k": "q8_0",
    "kv_cache_type_v": "q8_0",
    "parallel_slots": 1,
    "n_gpu_layers": 999,
    "cpu_moe_layers": 8,
    "batch_size": 2048,
    "ubatch_size": 512,
    "jinja": True,
}

EXPECTED_SETTINGS_DEFAULTS: dict[str, object] = {
    "host": "127.0.0.1",
    "port": 8400,
    "llama_host": "127.0.0.1",
    "llama_port": 8081,
    "max_upload_bytes": 262_144_000,
    "direct_file_budget_tokens": 12000,
    "map_window_tokens": 9000,
    "map_window_overlap_tokens": 300,
    "max_queued_jobs": 64,
    "max_projects_per_principal": 10,
    "max_storage_bytes_per_principal": 1_073_741_824,
    "max_queued_jobs_per_principal": 4,
    "keepalive_interval_seconds": 10.0,
    "json_keepalive_threshold_seconds": 20.0,
    "rate_limit_rpm": 120,
    "cf_access_enabled": False,
    "trust_cloudflare_headers": False,
    "cors_allowed_origins": (),
    "auth_providers": ("api_key",),
    "oidc_enabled": False,
    "oidc_issuer_url": "https://accounts.google.com",
    "oidc_jwks_url": "https://www.googleapis.com/oauth2/v3/certs",
    "oidc_audience": None,
    "oidc_audiences": (),
    "oidc_allowed_emails": (),
    "oidc_allowed_domains": (),
    "oidc_default_scopes": (
        "projects:read",
        "projects:write",
        "files:read",
        "files:write",
        "chat:read",
        "chat:write",
        "tools:safe",
    ),
    "oidc_admin_emails": (),
}

UPDATED_COUNT, OVERRIDDEN_PORT_ENV, OVERRIDDEN_PORT_ARG, YAML_RATE_LIMIT = 2, 9123, 9222, 240


class _MutableSample(MutableStateModel):
    count: int = 0
    label: str = "init"


def test_strict_and_mutable_models_and_utc_now() -> None:
    """StrictModel forbids extras/mutation; MutableStateModel allows updates; utc_now is UTC."""
    project = Project(name="alpha")
    assert isinstance(project, StrictModel)
    assert project.id.startswith(did.PROJECT_ID_PREFIX)
    assert project.owner_principal_id == "system"
    assert Project(name="owned", owner_principal_id="oidc:u1").owner_principal_id == "oidc:u1"
    with pytest.raises(ValidationError):
        CreateProjectInput.model_validate({"name": "beta", "unknown_field": 123})
    with pytest.raises(ValidationError):
        project.name = "mutated"  # type: ignore[misc]

    sample = _MutableSample(count=1)
    sample.count = UPDATED_COUNT
    assert sample.count == UPDATED_COUNT
    with pytest.raises(ValidationError):
        _MutableSample.model_validate({"count": 1, "extra_key": True})
    assert utc_now().tzinfo is UTC


def test_prefixed_id_generators_and_validation() -> None:
    """All ID helpers produce expected prefixes and validate non-empty suffixes."""
    id_fns = (
        (did.new_project_id, did.PROJECT_ID_PREFIX),
        (did.new_session_id, did.SESSION_ID_PREFIX),
        (did.new_file_id, did.FILE_ID_PREFIX),
        (did.new_chunk_id, did.CHUNK_ID_PREFIX),
        (did.new_message_id, did.MESSAGE_ID_PREFIX),
        (did.new_api_key_id, did.API_KEY_ID_PREFIX),
        (did.new_job_id, did.JOB_ID_PREFIX),
        (did.new_approval_id, did.APPROVAL_ID_PREFIX),
        (did.new_audit_id, did.AUDIT_ID_PREFIX),
        (did.new_alert_id, did.ALERT_ID_PREFIX),
        (did.new_client_app_id, did.CLIENT_APP_ID_PREFIX),
    )
    assert did.CLIENT_APP_ID_PREFIX in did.ALL_ID_PREFIXES
    assert len(id_fns) == len(did.ALL_ID_PREFIXES)
    pairs = [(fn(), pfx) for fn, pfx in id_fns]
    assert len({did.validate_prefixed_id(g, p) for g, p in pairs}) == len(pairs)
    for g, p in pairs:
        assert (g.startswith(p), len(g) > len(p)) == (True, True)
    for bad_prefix in ("bad", ""):
        with pytest.raises(ValueError, match="prefix"):
            did.generate_prefixed_id(bad_prefix)
    for bad_id in ("prj_", "ses_abc"):
        with pytest.raises(ValueError, match="Expected identifier"):
            did.validate_prefixed_id(bad_id, "prj_")


def test_error_hierarchy_codes_and_statuses() -> None:
    """Domain error classes carry distinct codes, HTTP statuses, and metadata."""
    apr = derr.ToolApprovalRequiredError("needs", approval_id="apr_1", tool_name="exec")
    disp = derr.ClientToolDispatchError(
        "d",
        approval_id="apr_2",
        tool_call_id="tc_1",
        tool_name="run_local",
        arguments_json="{}",
        requires_approval=True,
    )
    cases: list[tuple[derr.LaiError, int, str]] = [
        (derr.AuthenticationError("a"), 401, "authentication_failed"),
        (derr.AuthorizationError("b"), 403, "authorization_denied"),
        (derr.RateLimitExceededError("c", retry_after_seconds=5.0), 429, "rate_limit_exceeded"),
        (derr.QueueFullError("d"), 503, "queue_full"),
        (derr.NotFoundError("e"), 404, "not_found"),
        (derr.ValidationFailedError("f"), 422, "validation_failed"),
        (derr.PayloadTooLargeError("g"), 413, "payload_too_large"),
        (derr.InferenceBackendError("h"), 502, "inference_backend_error"),
        (apr, 409, "tool_approval_required"),
        (disp, 409, "client_tool_dispatch"),
        (derr.ClientDisconnectedError("i"), 499, "client_disconnected"),
    ]
    for err, status_code, code in cases:
        assert isinstance(err, derr.LaiError)
        assert (err.status_code, err.code) == (status_code, code)
    assert (apr.approval_id, apr.tool_name) == ("apr_1", "exec")
    assert (disp.approval_id, disp.tool_call_id) == ("apr_2", "tc_1")
    assert (disp.tool_name, disp.arguments_json) == ("run_local", "{}")
    assert (disp.requires_approval, disp.expires_at) == (True, None)


def test_gemma4_profile_defaults_and_resolution() -> None:
    """Default Gemma 4 26B A4B profile matches hardware-tuned llama.cpp flags."""
    profile = get_model_profile(DEFAULT_MODEL_PROFILE_NAME)
    assert profile == GEMMA4_26B_A4B_PROFILE
    dumped = profile.model_dump()
    for key, expected_value in EXPECTED_GEMMA4_FIELDS.items():
        assert dumped[key] == expected_value
    custom = ModelProfile(name="custom", hf_repo="example/model", hf_filename="model.gguf")
    assert get_model_profile("custom", custom_profiles={"custom": custom}) == custom
    with pytest.raises(derr.NotFoundError):
        get_model_profile("nonexistent-profile")


def test_settings_defaults_yaml_and_env_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LaiSettings defaults match spec and overrides > env > YAML precedence holds."""
    dumped = LaiSettings().model_dump()
    for key, expected_value in EXPECTED_SETTINGS_DEFAULTS.items():
        assert dumped[key] == expected_value

    empty_cfg = tmp_path / "empty.yaml"
    empty_cfg.write_text("", encoding="utf-8")
    assert load_settings(empty_cfg).port == EXPECTED_SETTINGS_DEFAULTS["port"]

    cfg_file = tmp_path / "lai.yaml"
    cfg_file.write_text(f"port: 9001\nrate_limit_rpm: {YAML_RATE_LIMIT}\n", encoding="utf-8")
    monkeypatch.setenv("LAI_CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("LAI_PORT", str(OVERRIDDEN_PORT_ENV))

    loaded = load_settings()
    assert (loaded.port, loaded.rate_limit_rpm) == (OVERRIDDEN_PORT_ENV, YAML_RATE_LIMIT)
    assert load_settings(cfg_file, port=OVERRIDDEN_PORT_ARG).port == OVERRIDDEN_PORT_ARG

    with pytest.raises(derr.ValidationFailedError):
        load_settings(tmp_path / "missing.yaml")
    bad_root = tmp_path / "bad.yaml"
    bad_root.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(derr.ValidationFailedError):
        load_settings(bad_root)


def test_settings_fail_closed_security_and_cors_validation(tmp_path: Path) -> None:
    """Invalid OIDC, Cloudflare Access, and CORS configs fail closed; valid ones pass."""
    cfg_file = tmp_path / "invalid.yaml"
    cfg_file.write_text("", encoding="utf-8")
    bad_origins = (
        "*",
        " * ",
        "ftp://a.co",
        "https://a.co/",
        "https://a.co/api",
        "https://a.co?q=1",
        "https://a.co#f",
        "https://u@a.co",
        "https://a.co:99999",
    )
    cors_bads: list[dict[str, object]] = [
        {"cors_allowed_origins": (origin,)} for origin in bad_origins
    ]
    bad_configs: tuple[dict[str, object], ...] = (
        {"oidc_enabled": True, "oidc_allowed_domains": ("corp.example",)},
        {"auth_providers": ("oidc",), "oidc_allowed_domains": ("corp.example",)},
        {"oidc_enabled": True, "oidc_audiences": ("aud-1",)},
        {"oidc_enabled": True, "oidc_audiences": ("aud-1",), "oidc_allowed_domains": (" ",)},
        {"oidc_enabled": True, "oidc_audience": "aud-1", "oidc_allowed_domains": ("@Gmail.com ",)},
        {"oidc_enabled": True, "oidc_audiences": ("aud-1",), "oidc_allowed_domains": ("aol.com",)},
        {"cf_access_enabled": True},
        {"cf_access_enabled": True, "cf_access_team_domain": "t.cf.com", "cf_access_aud": "   "},
        {"cf_access_enabled": True, "cf_access_client_id": "id", "cf_access_client_secret": ""},
        *cors_bads,
    )
    for bad_kwargs in bad_configs:
        with pytest.raises(ValidationError):
            LaiSettings.model_validate(bad_kwargs)
        with pytest.raises(derr.ValidationFailedError):
            load_settings(cfg_file, **bad_kwargs)

    s1 = LaiSettings(
        oidc_enabled=True,
        oidc_audience="legacy-aud",
        oidc_audiences=("  web-aud  ", "", "legacy-aud", "desktop-aud"),
        oidc_allowed_emails=("alice@gmail.com",),
        cf_access_enabled=True,
        cf_access_team_domain="team.cloudflareaccess.com",
        cf_access_aud="cf-aud",
        cors_allowed_origins=("https://app.example.com", "http://localhost:5173"),
    )
    assert s1.resolved_oidc_audiences() == ("web-aud", "legacy-aud", "desktop-aud")
    s2 = LaiSettings(
        auth_providers=("oidc",),
        oidc_audiences=("aud-1",),
        oidc_allowed_domains=("acme.test",),
        cf_access_enabled=True,
        cf_access_client_id="svc-id",
        cf_access_client_secret=secrets.token_hex(8),
    )
    assert s2.resolved_oidc_audiences() == ("aud-1",)


def test_logging_and_request_id_context() -> None:
    """Logging configuration and request_id context manager propagate and reset cleanly."""
    configure_logging(level="DEBUG", json_logs=True)
    logger = get_logger("lai.test")
    assert get_request_id() is None
    with bind_request_id("req_abc123"):
        assert get_request_id() == "req_abc123"
        logger.info("test_event", ok=True)
    assert get_request_id() is None


def test_pluggable_auth_scopes_credentials_and_context_fields() -> None:
    """AuthPrincipal, ClientAppRecord, ToolExecutionContext, and LeaseRequest fields validate."""
    assert (ApiScope.CHAT_READ.value, ApiScope.TOOLS_SAFE.value) == ("chat:read", "tools:safe")
    assert ApiScope.TOOLS_EXEC.value == "tools:exec"
    assert set(AuthMethod) == {
        AuthMethod.API_KEY,
        AuthMethod.OIDC,
        AuthMethod.CLOUDFLARE_ACCESS,
        AuthMethod.LOCAL_CLI,
    }
    assert not hasattr(dom, "CreatedClientApp")

    sample_bearer = did.new_job_id()
    creds = RequestCredentials(
        authorization_scheme="Bearer",
        bearer_token=sample_bearer,
        headers={"cf-access-jwt-assertion": "edge_jwt"},
        client_host="127.0.0.1",
    )
    assert creds.bearer_token == sample_bearer

    principal = AuthPrincipal(
        principal_id="oidc:sub_1",
        auth_method=AuthMethod.OIDC,
        scopes=(ApiScope.CHAT_READ, ApiScope.TOOLS_SAFE),
        email="user@example.com",
        issuer="https://accounts.google.com",
        subject="sub_1",
        oauth_client_id="client_123",
    )
    assert (principal.auth_method, principal.oauth_client_id) == (AuthMethod.OIDC, "client_123")
    p_empty = AuthPrincipal(principal_id="u", auth_method=AuthMethod.API_KEY, scopes=())
    assert p_empty.oauth_client_id is None
    app_rec = ClientAppRecord(name="web-spa", oauth_client_id="client_123")
    assert app_rec.oauth_client_id == "client_123"
    assert (hasattr(app_rec, "secret_prefix"), hasattr(app_rec, "secret_hash")) == (False, False)
    ctx = ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id=principal.principal_id,
        principal_scopes=principal.scopes,
        allowed_file_ids=("fil_1", "fil_2"),
        client_app_id="app_1",
        client_allowed_tiers=(PermissionTier.READ,),
        tool_call_id="tc_1",
    )
    assert ctx.allowed_file_ids == ("fil_1", "fil_2")
    assert (ctx.client_app_id, ctx.tool_call_id) == ("app_1", "tc_1")
    assert ctx.client_allowed_tiers == (PermissionTier.READ,)
    prios = {JobPriority.INTERACTIVE, JobPriority.APPROVAL_RESUME, JobPriority.MAP_REDUCE}
    assert set(JobPriority) == prios
    lease = LeaseRequest(session_id="ses_1", project_id="prj_1")
    assert (lease.principal_id, lease.priority) == ("anonymous", JobPriority.INTERACTIVE)
    assert lease.bypass_principal_quota is False
    lease_bp = LeaseRequest(session_id="s", project_id="p", bypass_principal_quota=True)
    assert lease_bp.bypass_principal_quota is True
    snap = SpoolerSnapshot()
    assert (snap.total_evicted, snap.avg_wait_ms) == (0, 0.0)
