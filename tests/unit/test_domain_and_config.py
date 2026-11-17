"""Unit tests for domain models, IDs, errors, settings, profiles, and logging."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from lai.config.logging import (
    bind_request_id,
    configure_logging,
    get_logger,
    get_request_id,
)
from lai.config.profiles import (
    DEFAULT_MODEL_PROFILE_NAME,
    GEMMA4_26B_A4B_PROFILE,
    get_model_profile,
)
from lai.config.settings import LaiSettings, load_settings
from lai.domain import (
    ApiScope,
    AuthMethod,
    AuthPrincipal,
    RequestCredentials,
)
from lai.domain.base import MutableStateModel, StrictModel, utc_now
from lai.domain.errors import (
    AuthenticationError,
    AuthorizationError,
    ClientDisconnectedError,
    InferenceBackendError,
    LaiError,
    NotFoundError,
    PayloadTooLargeError,
    QueueFullError,
    RateLimitExceededError,
    ToolApprovalRequiredError,
    ValidationFailedError,
)
from lai.domain.ids import (
    API_KEY_ID_PREFIX,
    APPROVAL_ID_PREFIX,
    AUDIT_ID_PREFIX,
    CHUNK_ID_PREFIX,
    FILE_ID_PREFIX,
    JOB_ID_PREFIX,
    MESSAGE_ID_PREFIX,
    PROJECT_ID_PREFIX,
    SESSION_ID_PREFIX,
    generate_prefixed_id,
    new_api_key_id,
    new_approval_id,
    new_audit_id,
    new_chunk_id,
    new_file_id,
    new_job_id,
    new_message_id,
    new_project_id,
    new_session_id,
    validate_prefixed_id,
)
from lai.domain.inference import ModelProfile
from lai.domain.projects import CreateProjectInput, Project
from lai.domain.scheduler import JobPriority, LeaseRequest, SpoolerSnapshot
from lai.domain.tools import ToolExecutionContext

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
    "keepalive_interval_seconds": 10.0,
    "json_keepalive_threshold_seconds": 20.0,
    "rate_limit_rpm": 120,
    "cf_access_enabled": False,
    "auth_providers": ("api_key",),
    "oidc_enabled": False,
    "oidc_issuer_url": "https://accounts.google.com",
    "oidc_jwks_url": "https://www.googleapis.com/oauth2/v3/certs",
    "oidc_audience": None,
    "oidc_allowed_emails": (),
    "oidc_allowed_domains": (),
    "oidc_default_scopes": (
        "projects:read",
        "files:read",
        "files:write",
        "chat:read",
        "chat:write",
        "tools:safe",
    ),
    "oidc_admin_emails": (),
}

UPDATED_COUNT = 2
OVERRIDDEN_PORT_ENV = 9123
OVERRIDDEN_PORT_ARG = 9222
YAML_RATE_LIMIT = 240


class _MutableSample(MutableStateModel):
    count: int = 0
    label: str = "init"


def test_strict_model_forbids_extra_and_is_immutable() -> None:
    """StrictModel subclasses reject extra keys and forbid post-init mutation."""
    project = Project(name="alpha")
    assert isinstance(project, StrictModel)
    assert project.id.startswith(PROJECT_ID_PREFIX)

    with pytest.raises(ValidationError):
        CreateProjectInput.model_validate({"name": "beta", "unknown_field": 123})

    with pytest.raises(ValidationError):
        project.name = "mutated"  # type: ignore[misc]


def test_mutable_state_model_allows_mutation_but_forbids_extra() -> None:
    """MutableStateModel allows field updates while still forbidding extra fields."""
    sample = _MutableSample(count=1)
    sample.count = UPDATED_COUNT
    assert sample.count == UPDATED_COUNT

    with pytest.raises(ValidationError):
        _MutableSample.model_validate({"count": 1, "extra_key": True})


def test_utc_now_is_timezone_aware_utc() -> None:
    """utc_now returns timezone-aware UTC datetimes."""
    assert utc_now().tzinfo is UTC


def test_prefixed_id_generators_and_validation() -> None:
    """All ID helpers produce expected prefixes and validate non-empty suffixes."""
    pairs = [
        (new_project_id(), PROJECT_ID_PREFIX),
        (new_session_id(), SESSION_ID_PREFIX),
        (new_file_id(), FILE_ID_PREFIX),
        (new_chunk_id(), CHUNK_ID_PREFIX),
        (new_message_id(), MESSAGE_ID_PREFIX),
        (new_api_key_id(), API_KEY_ID_PREFIX),
        (new_job_id(), JOB_ID_PREFIX),
        (new_approval_id(), APPROVAL_ID_PREFIX),
        (new_audit_id(), AUDIT_ID_PREFIX),
    ]
    seen: set[str] = set()
    for generated, prefix in pairs:
        assert generated.startswith(prefix)
        assert len(generated) > len(prefix)
        assert validate_prefixed_id(generated, prefix) == generated
        seen.add(generated)
    assert len(seen) == len(pairs)

    for bad_prefix in ("bad", ""):
        with pytest.raises(ValueError, match="prefix"):
            generate_prefixed_id(bad_prefix)

    for bad_id in ("prj_", "ses_abc"):
        with pytest.raises(ValueError, match="Expected identifier"):
            validate_prefixed_id(bad_id, "prj_")


def test_error_hierarchy_codes_and_statuses() -> None:
    """Domain error classes carry distinct codes, HTTP statuses, and metadata."""
    approval_err = ToolApprovalRequiredError(
        "needs approval", approval_id="apr_1", tool_name="exec"
    )
    expected: list[tuple[LaiError, int, str]] = [
        (AuthenticationError("bad token"), 401, "authentication_failed"),
        (AuthorizationError("forbidden"), 403, "authorization_denied"),
        (RateLimitExceededError("slow down", retry_after_seconds=5.0), 429, "rate_limit_exceeded"),
        (QueueFullError("busy"), 503, "queue_full"),
        (NotFoundError("missing"), 404, "not_found"),
        (ValidationFailedError("invalid"), 422, "validation_failed"),
        (PayloadTooLargeError("too big"), 413, "payload_too_large"),
        (InferenceBackendError("llama down"), 502, "inference_backend_error"),
        (approval_err, 409, "tool_approval_required"),
        (ClientDisconnectedError("gone"), 499, "client_disconnected"),
    ]
    for err, status_code, code in expected:
        assert isinstance(err, LaiError)
        assert err.status_code == status_code
        assert err.code == code

    assert approval_err.approval_id == "apr_1"
    assert approval_err.tool_name == "exec"


def test_gemma4_profile_defaults_and_resolution() -> None:
    """Default Gemma 4 26B A4B profile matches hardware-tuned llama.cpp flags."""
    profile = get_model_profile(DEFAULT_MODEL_PROFILE_NAME)
    assert profile == GEMMA4_26B_A4B_PROFILE
    dumped = profile.model_dump()
    for key, expected_value in EXPECTED_GEMMA4_FIELDS.items():
        assert dumped[key] == expected_value

    custom = ModelProfile(name="custom", hf_repo="example/model", hf_filename="model.gguf")
    assert get_model_profile("custom", custom_profiles={"custom": custom}) == custom

    with pytest.raises(NotFoundError):
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
    cfg_file.write_text(
        f"port: 9001\nrate_limit_rpm: {YAML_RATE_LIMIT}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LAI_CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("LAI_PORT", str(OVERRIDDEN_PORT_ENV))

    loaded = load_settings()
    assert loaded.port == OVERRIDDEN_PORT_ENV
    assert loaded.rate_limit_rpm == YAML_RATE_LIMIT
    assert load_settings(cfg_file, port=OVERRIDDEN_PORT_ARG).port == OVERRIDDEN_PORT_ARG

    with pytest.raises(ValidationFailedError):
        load_settings(tmp_path / "missing.yaml")

    bad_root = tmp_path / "bad.yaml"
    bad_root.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValidationFailedError):
        load_settings(bad_root)


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
    """New AuthMethod, ApiScope, RequestCredentials, tool context, and scheduler fields work."""
    assert ApiScope.CHAT_READ.value == "chat:read"
    assert ApiScope.TOOLS_SAFE.value == "tools:safe"
    assert ApiScope.TOOLS_EXEC.value == "tools:exec"
    assert set(AuthMethod) == {
        AuthMethod.API_KEY,
        AuthMethod.OIDC,
        AuthMethod.CLOUDFLARE_ACCESS,
        AuthMethod.LOCAL_CLI,
    }

    sample_bearer = new_job_id()
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
    )
    assert principal.auth_method is AuthMethod.OIDC
    assert principal.issuer == "https://accounts.google.com"
    assert principal.subject == "sub_1"

    ctx = ToolExecutionContext(
        project_id="prj_1",
        session_id="ses_1",
        principal_id=principal.principal_id,
        principal_scopes=principal.scopes,
        allowed_file_ids=("fil_1", "fil_2"),
    )
    assert ctx.principal_scopes == (ApiScope.CHAT_READ, ApiScope.TOOLS_SAFE)
    assert ctx.allowed_file_ids == ("fil_1", "fil_2")

    lease = LeaseRequest(session_id="ses_1", project_id="prj_1")
    assert lease.principal_id == "anonymous"
    assert lease.priority == JobPriority.INTERACTIVE
    snapshot = SpoolerSnapshot()
    assert snapshot.total_evicted == 0
    assert snapshot.avg_wait_ms == 0.0
