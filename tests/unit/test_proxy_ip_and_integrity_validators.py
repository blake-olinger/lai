"""Unit tests for trusted proxy IP resolution, model integrity allowlists, and POSIX modes."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lai.api.deps import resolve_client_ip
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings, load_settings
from lai.domain.auth import RequestCredentials
from lai.domain.errors import ValidationFailedError
from lai.domain.inference import ModelIntegrityConfig, ModelProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
_GOOD_SHA = "a" * 64
_OTHER_SHA = "b" * 64


@dataclass(frozen=True)
class _ProxyCase:
    client_host: str | None
    headers: dict[str, str]
    trust_cf: bool
    cidrs: tuple[str, ...]
    header_name: str
    expected: str


_PROXY_CASES: tuple[_ProxyCase, ...] = (
    _ProxyCase(
        "127.0.0.1",
        {"CF-Connecting-IP": "198.51.100.10"},
        True,
        (),
        "x-forwarded-for",
        "198.51.100.10",
    ),
    _ProxyCase(
        "::1", {"cf-connecting-ip": "2001:db8::5"}, True, (), "x-forwarded-for", "2001:db8::5"
    ),
    _ProxyCase(
        "198.51.100.7",
        {"X-Forwarded-For": "203.0.113.1"},
        False,
        ("10.0.1.0/24",),
        "x-forwarded-for",
        "198.51.100.7",
    ),
    _ProxyCase(
        "198.51.100.7",
        {"X-Azure-ClientIP": "203.0.113.1"},
        False,
        ("10.0.1.0/24",),
        "x-azure-clientip",
        "198.51.100.7",
    ),
    _ProxyCase(
        "127.0.0.1",
        {"X-Forwarded-For": "1.2.3.4, 203.0.113.42, 10.0.1.5"},
        False,
        ("127.0.0.1/32", "10.0.1.0/24"),
        "x-forwarded-for",
        "203.0.113.42",
    ),
    _ProxyCase(
        "127.0.0.1",
        {"X-Forwarded-For": "10.0.1.9, 10.0.1.5"},
        False,
        ("127.0.0.1/32", "10.0.1.0/24"),
        "x-forwarded-for",
        "10.0.1.9",
    ),
    _ProxyCase(
        "127.0.0.1",
        {"X-Forwarded-For": "not-an-ip, 203.0.113.42"},
        False,
        ("127.0.0.1/32",),
        "x-forwarded-for",
        "127.0.0.1",
    ),
    _ProxyCase(
        "127.0.0.1",
        {"X-Forwarded-For": " , "},
        False,
        ("127.0.0.1/32",),
        "x-forwarded-for",
        "127.0.0.1",
    ),
    _ProxyCase(
        "10.0.1.8",
        {"X-Azure-ClientIP": "203.0.113.88"},
        False,
        ("10.0.1.0/24",),
        "x-azure-clientip",
        "203.0.113.88",
    ),
    _ProxyCase(
        "10.0.1.8",
        {"X-Azure-ClientIP": "bad-ip"},
        False,
        ("10.0.1.0/24",),
        "x-azure-clientip",
        "10.0.1.8",
    ),
    _ProxyCase(
        "10.0.1.8",
        {"CF-Connecting-IP": "198.51.100.44"},
        False,
        ("10.0.1.0/24",),
        "cf-connecting-ip",
        "198.51.100.44",
    ),
    _ProxyCase(
        "testclient",
        {"X-Forwarded-For": "203.0.113.1"},
        False,
        ("127.0.0.1/32",),
        "x-forwarded-for",
        "testclient",
    ),
    _ProxyCase(None, {}, False, ("127.0.0.1/32",), "x-forwarded-for", "unknown"),
)


@pytest.mark.parametrize("case", _PROXY_CASES)
def test_resolve_client_ip_proxy_headers_and_spoofing_resistance(case: _ProxyCase) -> None:
    """Verify _resolve_client_ip honors trusted CIDRs and resists header spoofing."""
    creds = RequestCredentials(client_host=case.client_host, headers=case.headers)
    actual = resolve_client_ip(
        creds,
        trust_cf_header=case.trust_cf,
        trusted_proxy_cidrs=case.cidrs,
        proxy_ip_header=case.header_name,
    )
    assert actual == case.expected


def test_model_integrity_strict_allowlist_proxy_cidrs_and_deploy_overlays(tmp_path: Path) -> None:
    """Verify strict_allowlist, trusted_proxy_cidrs, oidc_identity_claims, and deploy YAMLs."""
    defaults = LaiSettings()
    assert defaults.model_integrity.strict_allowlist is True

    unpinned = ModelProfile(name="u", hf_repo="org/repo", hf_filename="m.gguf")
    with pytest.raises(ValueError, match="must pin sha256"):
        LaiSettings(custom_profiles={"u": unpinned})

    pinned = ModelProfile(name="p", hf_repo="org/repo", hf_filename="m.gguf", sha256=_GOOD_SHA)
    with pytest.raises(ValueError, match="approved_model_sha256s"):
        LaiSettings(
            custom_profiles={"p": pinned},
            model_integrity=ModelIntegrityConfig(approved_model_sha256s=(_OTHER_SHA,)),
        )

    with pytest.raises(ValueError, match="allowed_hf_repos"):
        LaiSettings(
            custom_profiles={"p": pinned},
            model_integrity=ModelIntegrityConfig(
                approved_model_sha256s=(_GOOD_SHA,),
                allowed_hf_repos=("trusted/repo",),
            ),
        )

    ok_strict = LaiSettings(
        custom_profiles={"p": pinned},
        model_integrity=ModelIntegrityConfig(
            approved_model_sha256s=(_GOOD_SHA,),
            allowed_hf_repos=("org/repo",),
        ),
    )
    assert "p" in ok_strict.custom_profiles
    ok_relaxed = LaiSettings(
        custom_profiles={"u": unpinned},
        model_integrity=ModelIntegrityConfig(strict_allowlist=False),
    )
    assert "u" in ok_relaxed.custom_profiles

    for bad_cidr in ("", " 127.0.0.1/32", "999.0.0.1/24", "not-cidr"):
        with pytest.raises(ValueError, match="Invalid trusted proxy CIDR"):
            LaiSettings(trusted_proxy_cidrs=(bad_cidr,))

    with pytest.raises(ValueError, match="oidc_identity_claims"):
        LaiSettings(
            oidc_enabled=True,
            oidc_audiences=("aud",),
            oidc_allowed_emails=("u@example.test",),
            oidc_identity_claims=(),
        )

    azure = load_settings(REPO_ROOT / "deploy" / "azure" / "config.azure-prod.example.yaml")
    cf = load_settings(REPO_ROOT / "deploy" / "cloudflare" / "config.test.example.yaml")
    assert azure.oidc_require_email_verified is False
    assert cf.trust_cloudflare_headers is True

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("trusted_proxy_cidrs:\n  - invalid\n", encoding="utf-8")
    with pytest.raises(ValidationFailedError):
        load_settings(bad_yaml)

    cfg_out = tmp_path / "init_cfg.yaml"
    data_out = tmp_path / "init_data"
    res = CliRunner().invoke(
        cli_app,
        [
            "init",
            "--config",
            str(cfg_out),
            "--data-dir",
            str(data_out),
            "--models-dir",
            str(tmp_path / "m"),
            "--bin-dir",
            str(tmp_path / "b"),
        ],
    )
    assert res.exit_code == 0, res.output
    if os.name == "posix":
        assert stat.S_IMODE(data_out.stat().st_mode) == 0o700
        assert stat.S_IMODE((data_out / "backups").stat().st_mode) == 0o700
        assert stat.S_IMODE((data_out / "lai.db").stat().st_mode) == 0o600
        assert stat.S_IMODE(cfg_out.stat().st_mode) == 0o600
