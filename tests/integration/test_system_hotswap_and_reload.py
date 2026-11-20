"""Integration tests for isolated per-subsystem hot-swapping, boot locks, and scoped reload."""

from __future__ import annotations

from http import HTTPStatus as S
from typing import TYPE_CHECKING

import yaml

from lai.api.runtime import build_app_services, close_app_services
from lai.config.settings import LaiSettings
from lai.db import SqliteDatabase, SqliteRuntimeConfigRepository
from lai.domain.auth import ApiScope
from lai.domain.guardrails import GuardrailConfig, GuardrailOutputMode
from lai.domain.observability import AlertCategory, LoggingConfig
from lai.domain.updates import UpdateConfig
from tests.integration._helpers import (
    IntegrationHarness,
    configure_test_oidc,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from pathlib import Path


async def _assert_guardrails_and_updates_hotswap(
    h: IntegrationHarness,
    admin_hdrs: dict[str, str],
) -> None:
    """Verify guardrail boot-lock rejections + SECURITY_ALERT, safe mutations, and updates API."""
    for forbidden in (
        {"enabled": False},
        {"fail_open": True},
        {"output_mode": "off"},
        {"violation_threshold": 0.95},
    ):
        r_lock = await h.client.patch(
            "/v1/system/config/guardrails",
            headers=admin_hdrs,
            json=forbidden,
        )
        assert r_lock.status_code == S.UNPROCESSABLE_ENTITY, forbidden
    await h.svc.alert_manager.flush()
    sec_alerts = [
        e for e in h.svc.alert_manager.recent_alerts() if e.category is AlertCategory.SECURITY_ALERT
    ]
    assert len(sec_alerts) >= 1

    r_gr = await h.client.patch(
        "/v1/system/config/guardrails",
        headers=admin_hdrs,
        json={
            "block_mode": "http_error",
            "output_mode": "audit_only",
            "violation_threshold": 0.75,
            "refusal_text": "Blocked live",
        },
    )
    assert r_gr.status_code == S.OK
    gr_body = r_gr.json()
    assert gr_body["block_mode"] == "http_error"
    assert gr_body["output_mode"] == GuardrailOutputMode.AUDIT_ONLY.value
    assert gr_body["allow_runtime_disable"] is False
    assert gr_body["max_violation_threshold"] == 0.90

    r_upd_get = await h.client.get("/v1/system/config/updates", headers=admin_hdrs)
    assert r_upd_get.status_code == S.OK
    assert r_upd_get.json()["enabled"] is False

    r_upd_patch = await h.client.patch(
        "/v1/system/config/updates",
        headers=admin_hdrs,
        json={"enabled": True, "keep_releases": 5, "allow_prerelease": True},
    )
    assert r_upd_patch.status_code == S.OK
    assert r_upd_patch.json()["enabled"] is True
    assert r_upd_patch.json()["keep_releases"] == 5
    assert r_upd_patch.json()["allow_prerelease"] is True


async def _assert_scoped_reloads(
    h: IntegrationHarness,
    admin_hdrs: dict[str, str],
    live_origin: str,
) -> None:
    """Verify scoped YAML reloads reset only requested sections and preserve untouched ones."""
    await h.client.patch(
        "/v1/system/alerting",
        headers=admin_hdrs,
        json={"alert_cooldown_seconds": 120.0},
    )
    r_reload = await h.client.post(
        "/v1/system/config/reload",
        headers=admin_hdrs,
        json={"sections": ["alerting"]},
    )
    assert r_reload.status_code == S.OK
    assert r_reload.json()["reloaded_sections"] == ["alerting"]
    assert {"cors", "updates"}.issubset(r_reload.json()["unchanged_sections"])
    assert (await h.client.get("/v1/system/alerting", headers=admin_hdrs)).json()[
        "alert_cooldown_seconds"
    ] == 45.0
    assert (await h.client.get("/v1/system/config/cors", headers=admin_hdrs)).json()[
        "cors_allowed_origins"
    ] == [live_origin]
    assert (await h.client.get("/v1/system/config/updates", headers=admin_hdrs)).json()[
        "enabled"
    ] is True

    r_reload_rest = await h.client.post(
        "/v1/system/config/reload",
        headers=admin_hdrs,
        json={"sections": ["cors", "limits", "guardrails", "analytics", "updates"]},
    )
    assert r_reload_rest.status_code == S.OK
    assert (await h.client.get("/v1/system/config/cors", headers=admin_hdrs)).json()[
        "cors_allowed_origins"
    ] == ["https://yaml-origin.test"]
    assert (await h.client.get("/v1/system/config/updates", headers=admin_hdrs)).json()[
        "enabled"
    ] is False
    assert (await h.client.get("/v1/system/config/oidc", headers=admin_hdrs)).json()[
        "oidc_audiences"
    ] == ["aud-1", "aud-2"]


async def test_isolated_hotswap_sections_and_scoped_yaml_reload(tmp_path: Path) -> None:
    """Verify CORS, OIDC, limits, guardrails locks, updates hot-swaps, and scoped YAML reload."""
    cfg_path = tmp_path / "config.yaml"
    initial_yaml = {
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "bin_dir": str(tmp_path / "bin"),
        "cors_allowed_origins": ["https://yaml-origin.test"],
        "logging": {
            "log_dir": str(tmp_path / "data" / "logs"),
            "enable_file_logs": False,
            "alert_cooldown_seconds": 45.0,
        },
        "guardrails": {"enabled": True, "allow_runtime_disable": False},
        "updates": {"enabled": False, "keep_releases": 3},
    }
    cfg_path.write_text(yaml.safe_dump(initial_yaml), encoding="utf-8")
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        cors_allowed_origins=(),
        logging=LoggingConfig(
            log_dir=tmp_path / "data" / "logs",
            enable_file_logs=False,
            alert_cooldown_seconds=0.0,
        ),
        guardrails=GuardrailConfig(
            enabled=True, allow_runtime_disable=False, max_violation_threshold=0.90
        ),
        updates=UpdateConfig(enabled=False, keep_releases=3),
    )

    async with open_services_harness(settings) as h:
        h.svc.config_path = cfg_path
        admin_hdrs = await mint_auth_headers(h.svc, "cfg-adm", (ApiScope.ADMIN,))
        oidc = configure_test_oidc(h.svc, audiences=("aud-1",), allowed_domains=("acme.test",))
        h.svc.settings = h.svc.settings.model_copy(
            update={
                "oidc_enabled": True,
                "oidc_audiences": ("aud-1",),
                "oidc_allowed_domains": ("acme.test",),
            }
        )

        origin = "https://live-spa.example.test"
        r_pre_before = await h.client.options(
            "/v1/projects",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in r_pre_before.headers

        r_cors = await h.client.patch(
            "/v1/system/config/cors",
            headers=admin_hdrs,
            json={"cors_allowed_origins": [origin], "cors_extra_allow_headers": ["X-Trace-Id"]},
        )
        assert r_cors.status_code == S.OK
        r_pre_after = await h.client.options(
            "/v1/projects",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization, X-Trace-Id",
            },
        )
        assert r_pre_after.status_code == S.OK
        assert r_pre_after.headers["access-control-allow-origin"] == origin

        assert (
            await h.client.patch(
                "/v1/system/config/oidc",
                headers=admin_hdrs,
                json={"oidc_allowed_domains": ["gmail.com"]},
            )
        ).status_code == S.UNPROCESSABLE_ENTITY

        r_oidc_ok = await h.client.patch(
            "/v1/system/config/oidc",
            headers=admin_hdrs,
            json={
                "oidc_audiences": ["aud-1", "aud-2"],
                "oidc_leeway_seconds": 15.0,
                "oidc_require_email_verified": False,
                "oidc_identity_claims": ["preferred_username", "upn", "email"],
            },
        )
        assert r_oidc_ok.status_code == S.OK
        assert r_oidc_ok.json()["oidc_require_email_verified"] is False
        assert r_oidc_ok.json()["oidc_identity_claims"] == ["preferred_username", "upn", "email"]
        tok_aud2 = oidc.mint_headers(sub="u-aud2", email="dev@acme.test", aud="aud-2")
        assert (await h.client.get("/v1/projects", headers=tok_aud2)).status_code == S.OK

        for bad_lim in ({"log_level": "NOT_A_LEVEL"}, {"trusted_proxy_cidrs": ["not-a-cidr"]}):
            assert (
                await h.client.patch("/v1/system/config/limits", headers=admin_hdrs, json=bad_lim)
            ).status_code == S.UNPROCESSABLE_ENTITY

        r_lim = await h.client.patch(
            "/v1/system/config/limits",
            headers=admin_hdrs,
            json={
                "rate_limit_rpm": 90,
                "max_upload_bytes": 52_428_800,
                "trusted_proxy_cidrs": ["127.0.0.1/32", "10.0.1.0/24"],
                "proxy_ip_header": "x-azure-clientip",
                "log_level": "WARNING",
            },
        )
        assert r_lim.status_code == S.OK
        assert r_lim.json()["trusted_proxy_cidrs"] == ["127.0.0.1/32", "10.0.1.0/24"]
        assert r_lim.json()["proxy_ip_header"] == "x-azure-clientip"

        await _assert_guardrails_and_updates_hotswap(h, admin_hdrs)
        await _assert_scoped_reloads(h, admin_hdrs, origin)


async def test_tampered_sqlite_guardrail_override_rejected_at_boot(tmp_path: Path) -> None:
    """Direct SQLite tampering cannot weaken boot-locked guardrails while valid sections load."""
    data_dir = tmp_path / "data"
    db = SqliteDatabase(data_dir / "lai.db")
    await db.initialize()
    try:
        repo = SqliteRuntimeConfigRepository(db)
        await repo.upsert_section(
            "guardrails",
            {"enabled": False, "fail_open": True, "output_mode": "off"},
        )
        await repo.upsert_section("updates", {"enabled": True, "keep_releases": 4})
    finally:
        await db.close()

    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(
            log_dir=data_dir / "logs", enable_file_logs=False, alert_cooldown_seconds=0.0
        ),
        guardrails=GuardrailConfig(enabled=True, allow_runtime_disable=False),
        updates=UpdateConfig(enabled=False, keep_releases=3),
    )
    svc = await build_app_services(settings)
    try:
        assert svc.settings.guardrails.enabled is True
        assert svc.settings.guardrails.fail_open is False
        assert svc.settings.guardrails.output_mode is GuardrailOutputMode.BUFFERED
        assert svc.settings.updates.enabled is True
        assert svc.settings.updates.keep_releases == 4
        await svc.alert_manager.flush()
        sec_alerts = [
            e
            for e in svc.alert_manager.recent_alerts()
            if e.category is AlertCategory.SECURITY_ALERT
        ]
        assert len(sec_alerts) >= 1
    finally:
        await close_app_services(svc)
