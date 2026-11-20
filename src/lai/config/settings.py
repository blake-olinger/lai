"""Typed application configuration with YAML and environment variable loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self, cast

import yaml
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from lai.config._validators import validate_lai_settings
from lai.config.profiles import DEFAULT_MODEL_PROFILE_NAME
from lai.domain.analytics import AnalyticsConfig
from lai.domain.errors import ValidationFailedError
from lai.domain.guardrails import GuardrailConfig
from lai.domain.inference import ModelIntegrityConfig, ModelProfile
from lai.domain.observability import LoggingConfig
from lai.domain.updates import UpdateConfig

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo


class _MappingSettingsSource(PydanticBaseSettingsSource):
    """Settings source backed by an in-memory key/value mapping."""

    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, object]) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[object, str, bool]:
        """Return raw field value from the backing mapping."""
        del field
        value = self._data.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, object]:
        """Return a shallow copy of the backing settings dictionary."""
        return dict(self._data)


class LaiSettings(BaseSettings):
    """Runtime settings for the local AI server, storage, and inference backend."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="LAI_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8400, ge=1, le=65535)
    llama_host: str = "127.0.0.1"
    llama_port: int = Field(default=8081, ge=1, le=65535)

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    bin_dir: Path = Path("bin")

    max_upload_bytes: int = Field(default=262_144_000, ge=1)
    max_json_body_bytes: int = Field(default=2_097_152, ge=1024)
    direct_file_budget_tokens: int = Field(default=12000, ge=256)
    map_window_tokens: int = Field(default=9000, ge=256)
    map_window_overlap_tokens: int = Field(default=300, ge=0)
    map_max_windows: int = Field(default=32, ge=1, le=256)
    max_queued_jobs: int = Field(default=64, ge=1)
    max_projects_per_principal: int = Field(default=10, ge=1)
    max_storage_bytes_per_principal: int = Field(default=1_073_741_824, ge=1)
    max_queued_jobs_per_principal: int = Field(default=4, ge=1)

    keepalive_interval_seconds: float = Field(default=10.0, gt=0.0)
    json_keepalive_threshold_seconds: float = Field(default=20.0, gt=0.0)
    rate_limit_rpm: int = Field(default=120, ge=1)
    ip_rate_limit_rpm: int = Field(default=240, ge=1)

    cf_access_enabled: bool = False
    trust_cloudflare_headers: bool = False
    trusted_proxy_cidrs: tuple[str, ...] = ()
    proxy_ip_header: Literal["cf-connecting-ip", "x-forwarded-for", "x-azure-clientip"] = (
        "x-forwarded-for"
    )
    cors_allowed_origins: tuple[str, ...] = ()
    cors_extra_allow_headers: tuple[str, ...] = ()
    cf_access_client_id: str | None = None
    cf_access_client_secret: str | None = None
    cf_access_team_domain: str | None = None
    cf_access_aud: str | None = None

    auth_providers: tuple[str, ...] = ("api_key",)
    oidc_enabled: bool = False
    oidc_issuer_url: str = "https://accounts.google.com"
    oidc_jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    oidc_audience: str | None = None
    oidc_audiences: tuple[str, ...] = ()
    oidc_allowed_client_ids: tuple[str, ...] = ()
    oidc_allowed_emails: tuple[str, ...] = ()
    oidc_allowed_domains: tuple[str, ...] = ()
    oidc_require_email_verified: bool = True
    oidc_identity_claims: tuple[str, ...] = ("email", "preferred_username", "upn")
    oidc_leeway_seconds: float = Field(default=30.0, ge=0.0, le=300.0)
    oidc_default_scopes: tuple[str, ...] = (
        "projects:read",
        "projects:write",
        "files:read",
        "files:write",
        "chat:read",
        "chat:write",
        "tools:safe",
    )
    oidc_admin_emails: tuple[str, ...] = ()
    oidc_admin_roles: tuple[str, ...] = ()
    oidc_admin_subjects: tuple[str, ...] = ()
    oidc_allow_guest_users: bool = False

    default_model_profile: str = DEFAULT_MODEL_PROFILE_NAME
    custom_profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    model_integrity: ModelIntegrityConfig = Field(default_factory=ModelIntegrityConfig)
    updates: UpdateConfig = Field(default_factory=UpdateConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    log_level: str = "INFO"
    log_json: bool = False

    def resolved_oidc_audiences(self) -> tuple[str, ...]:
        """Return deduplicated, non-empty OIDC audiences from oidc_audiences and oidc_audience."""
        items = [item.strip() for item in self.oidc_audiences if item.strip()]
        if self.oidc_audience is not None:
            legacy = self.oidc_audience.strip()
            if legacy:
                items.append(legacy)
        return tuple(dict.fromkeys(items))

    @model_validator(mode="after")
    def _validate_security_constraints(self) -> Self:
        """Enforce fail-closed OIDC, Cloudflare Access, and CORS origin settings."""
        validate_lai_settings(self)
        return self


def load_yaml_mapping(config_path: Path) -> dict[str, object]:
    """Load and validate a YAML configuration file as a dictionary."""
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise ValidationFailedError(msg, details={"path": str(config_path)})
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        loaded: object = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML syntax in configuration file {config_path}: {exc}"
        raise ValidationFailedError(msg, details={"path": str(config_path)}) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"Configuration file root must be a YAML mapping: {config_path}"
        raise ValidationFailedError(msg, details={"path": str(config_path)})
    mapping = cast("dict[object, object]", loaded)
    return {str(key): value for key, value in mapping.items()}


def _wrap_settings_validation_error(
    exc: ValidationError,
    resolved_path: Path | None,
) -> ValidationFailedError:
    """Convert raw Pydantic ValidationError into structured ValidationFailedError."""
    errors = exc.errors()
    summary = "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'config'}: {err['msg']}" for err in errors
    )
    details: dict[str, str] = {"error_count": str(len(errors))}
    if resolved_path is not None:
        details["path"] = str(resolved_path)
    msg = f"Invalid configuration settings: {summary}"
    return ValidationFailedError(msg, details=details)


def load_settings(
    config_path: Path | None = None,
    **overrides: object,
) -> LaiSettings:
    """Load settings with precedence: explicit overrides > env vars > YAML > defaults."""
    resolved_path = config_path
    if resolved_path is None:
        env_path = os.environ.get("LAI_CONFIG_FILE")
        if env_path:
            resolved_path = Path(env_path)

    yaml_data = load_yaml_mapping(resolved_path) if resolved_path is not None else {}
    override_data = dict(overrides)

    class _ConfiguredLaiSettings(LaiSettings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                _MappingSettingsSource(settings_cls, override_data),
                env_settings,
                dotenv_settings,
                _MappingSettingsSource(settings_cls, yaml_data),
                file_secret_settings,
            )

    try:
        loaded = _ConfiguredLaiSettings()
        validated = LaiSettings.model_validate(loaded.model_dump())
    except ValidationError as exc:
        raise _wrap_settings_validation_error(exc, resolved_path) from exc
    return validated
