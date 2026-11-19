"""Integration tests for isolated per-subsystem hot-swapping and scoped YAML reload."""

from __future__ import annotations

from http import HTTPStatus as S
from typing import TYPE_CHECKING

import yaml

from lai.config.settings import LaiSettings
from lai.domain.auth import ApiScope
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import LoggingConfig
from tests.integration._helpers import (
    configure_test_oidc,
    mint_auth_headers,
    open_services_harness,
)

if TYPE_CHECKING:
    from pathlib import Path


async def test_isolated_hotswap_sections_and_scoped_yaml_reload(tmp_path: Path) -> None:
    """Verify live CORS, OIDC, limits, guardrails hot-swaps and isolated scoped YAML reload."""
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
        "guardrails": {"enabled": False},
    }
    cfg_path.write_text(yaml.safe_dump(initial_yaml), encoding="utf-8")
    settings = LaiSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        cors_allowed_origins=(),
        logging=LoggingConfig(log_dir=tmp_path / "data" / "logs", enable_file_logs=False),
        guardrails=GuardrailConfig(enabled=False),
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

        # 1. Live CORS hot-swap activates preflight immediately without restart
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

        # 2. Live OIDC hot-swap adds aud-2 live and rejects consumer domain fail-closed
        r_oidc_bad = await h.client.patch(
            "/v1/system/config/oidc",
            headers=admin_hdrs,
            json={"oidc_allowed_domains": ["gmail.com"]},
        )
        assert r_oidc_bad.status_code == S.UNPROCESSABLE_ENTITY

        r_oidc_ok = await h.client.patch(
            "/v1/system/config/oidc",
            headers=admin_hdrs,
            json={"oidc_audiences": ["aud-1", "aud-2"], "oidc_leeway_seconds": 15.0},
        )
        assert r_oidc_ok.status_code == S.OK
        tok_aud2 = oidc.mint_headers(sub="u-aud2", email="dev@acme.test", aud="aud-2")
        assert (await h.client.get("/v1/projects", headers=tok_aud2)).status_code == S.OK

        # 3. Live limits & guardrails hot-swaps
        assert (
            await h.client.patch(
                "/v1/system/config/limits", headers=admin_hdrs, json={"log_level": "NOT_A_LEVEL"}
            )
        ).status_code == S.UNPROCESSABLE_ENTITY
        r_lim = await h.client.patch(
            "/v1/system/config/limits",
            headers=admin_hdrs,
            json={"rate_limit_rpm": 90, "max_upload_bytes": 52_428_800, "log_level": "WARNING"},
        )
        assert r_lim.status_code == S.OK
        assert (r_lim.json()["rate_limit_rpm"], r_lim.json()["log_level"]) == (90, "WARNING")

        r_gr = await h.client.patch(
            "/v1/system/config/guardrails",
            headers=admin_hdrs,
            json={"block_mode": "http_error", "fail_open": True, "refusal_text": "Blocked live"},
        )
        assert r_gr.status_code == S.OK
        assert r_gr.json()["block_mode"] == "http_error"

        # 4. Scoped reload of ONLY ["alerting"] pulls YAML cooldown=45 while keeping live CORS!
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
        assert "cors" in r_reload.json()["unchanged_sections"]

        assert (await h.client.get("/v1/system/alerting", headers=admin_hdrs)).json()[
            "alert_cooldown_seconds"
        ] == 45.0
        assert (await h.client.get("/v1/system/config/cors", headers=admin_hdrs)).json()[
            "cors_allowed_origins"
        ] == [origin]

        # 5. Reloading ["cors", "limits", "guardrails", "analytics"] resets only those sections
        r_reload_rest = await h.client.post(
            "/v1/system/config/reload",
            headers=admin_hdrs,
            json={"sections": ["cors", "limits", "guardrails", "analytics"]},
        )
        assert r_reload_rest.status_code == S.OK
        assert (await h.client.get("/v1/system/config/cors", headers=admin_hdrs)).json()[
            "cors_allowed_origins"
        ] == ["https://yaml-origin.test"]
        # Live OIDC audiences remained untouched
        assert (await h.client.get("/v1/system/config/oidc", headers=admin_hdrs)).json()[
            "oidc_audiences"
        ] == ["aud-1", "aud-2"]
