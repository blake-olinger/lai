"""Unit tests for client app in-place updates, expired auto-revoke, and CLI JSON tools."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus as S
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings
from lai.db import SqliteClientAppRepository, SqliteDatabase
from lai.domain.auth import ApiScope
from lai.domain.errors import ConflictError, NotFoundError, ValidationFailedError
from lai.domain.observability import LoggingConfig
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec
from lai.security import build_client_app, generate_api_key, update_client_app

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


def test_update_client_app_validation_and_tier_enforcement() -> None:
    """update_client_app validates name, tiers, tools, projects, expiry, and rejects revoked."""
    now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
    read_tool = ToolSpec(
        name="read_cfg",
        description="Read config",
        permission_tier=PermissionTier.READ,
        execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
    )
    exec_tool = ToolSpec(
        name="run_cmd",
        description="Run command",
        permission_tier=PermissionTier.EXEC,
    )
    base = build_client_app(
        "desktop",
        oauth_client_id="cid-1",
        allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC),
        registered_tools=(read_tool, exec_tool),
        allowed_project_ids=("prj_1",),
        expires_at=now + timedelta(days=30),
        created_at=now,
    )

    updated = update_client_app(
        base,
        name="  desktop-v2  ",
        allowed_tiers=(PermissionTier.READ,),
        registered_tools=(read_tool,),
        allowed_project_ids=(" prj_2 ", "", "prj_2"),
        expires_at=None,
        update_expires_at=True,
    )
    assert updated.id == base.id
    assert updated.oauth_client_id == "cid-1"
    assert updated.name == "desktop-v2"
    assert updated.allowed_tiers == (PermissionTier.READ,)
    assert updated.allowed_project_ids == ("prj_2",)
    assert updated.expires_at is None
    assert updated.registered_tools[0].execution_target is ToolExecutionTarget.CLIENT

    unchanged_exp = update_client_app(base, name="renamed")
    assert unchanged_exp.expires_at == base.expires_at

    with pytest.raises(ValidationFailedError, match="At least one allowed permission tier"):
        build_client_app("empty-tiers", "cid-empty", allowed_tiers=())
    with pytest.raises(ValidationFailedError, match="At least one allowed permission tier"):
        update_client_app(base, allowed_tiers=())
    with pytest.raises(ValidationFailedError, match="must not be empty"):
        update_client_app(base, name="   ")
    with pytest.raises(ValidationFailedError, match="allowed_tiers"):
        update_client_app(base, allowed_tiers=(PermissionTier.READ,))
    with pytest.raises(ValidationFailedError, match="revoked"):
        update_client_app(base.model_copy(update={"revoked_at": now}), name="nope")


async def test_repo_auto_revokes_expired_duplicate_and_updates_in_place(tmp_path: Path) -> None:
    """Creating a duplicate oauth_client_id auto-revokes expired rows but conflicts on active."""
    async with SqliteDatabase(tmp_path / "apps.db") as db:
        repo = SqliteClientAppRepository(db)
        expired = build_client_app(
            "expired-app",
            oauth_client_id="cid-shared",
            expires_at=datetime.now(UTC) - timedelta(seconds=5),
        )
        await repo.create(expired)

        replacement = build_client_app(
            "fresh-app",
            oauth_client_id="cid-shared",
            allowed_tiers=(PermissionTier.READ, PermissionTier.WRITE),
        )
        saved = await repo.create(replacement)
        assert saved.id == replacement.id

        old_row = await repo.get_by_id(expired.id)
        assert old_row is not None
        assert old_row.revoked_at is not None

        with pytest.raises(ConflictError):
            await repo.create(build_client_app("dup-active", oauth_client_id="cid-shared"))

        mutated = update_client_app(
            saved,
            name="fresh-renamed",
            allowed_project_ids=("prj_a",),
        )
        persisted = await repo.update(mutated)
        assert persisted.name == "fresh-renamed"
        fetched = await repo.get_by_id(saved.id)
        assert fetched is not None
        assert fetched.allowed_project_ids == ("prj_a",)

        await repo.revoke(saved.id, datetime.now(UTC))
        with pytest.raises(NotFoundError):
            await repo.update(mutated)


async def test_patch_client_app_route_updates_and_rejects_revoked(tmp_path: Path) -> None:
    """PATCH /v1/client-apps/{app_id} updates fields in place, clears expires_at, rejects bad."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
    )
    svc = await build_app_services(settings)
    app = create_app(settings, services=svc)
    try:
        admin_key = generate_api_key("admin", (ApiScope.ADMIN,))
        user_key = generate_api_key("user", (ApiScope.CHAT_WRITE,))
        await svc.api_key_repo.create(admin_key.record)
        await svc.api_key_repo.create(user_key.record)
        adm = {"Authorization": f"Bearer {admin_key.plaintext_key}"}
        usr = {"Authorization": f"Bearer {user_key.plaintext_key}"}

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            r_cr = await client.post(
                "/v1/client-apps",
                headers=adm,
                json={
                    "name": "spa",
                    "oauth_client_id": "cid-spa",
                    "allowed_tiers": ["read"],
                    "expires_at": (datetime.now(UTC) + timedelta(days=10)).isoformat(),
                },
            )
            assert r_cr.status_code == S.CREATED
            app_id = r_cr.json()["id"]

            r_forb = await client.patch(
                f"/v1/client-apps/{app_id}",
                headers=usr,
                json={"name": "nope"},
            )
            assert r_forb.status_code == S.FORBIDDEN

            r_missing = await client.patch(
                "/v1/client-apps/app_missing",
                headers=adm,
                json={"name": "nope"},
            )
            assert r_missing.status_code == S.NOT_FOUND

            r_patch = await client.patch(
                f"/v1/client-apps/{app_id}",
                headers=adm,
                json={
                    "name": "spa-updated",
                    "allowed_tiers": ["read", "write"],
                    "allowed_project_ids": ["prj_9"],
                    "expires_at": None,
                },
            )
            assert r_patch.status_code == S.OK
            body = r_patch.json()
            assert body["name"] == "spa-updated"
            assert body["allowed_tiers"] == ["read", "write"]
            assert body["allowed_project_ids"] == ["prj_9"]
            assert body["expires_at"] is None

            await client.delete(f"/v1/client-apps/{app_id}", headers=adm)
            r_rev = await client.patch(
                f"/v1/client-apps/{app_id}",
                headers=adm,
                json={"name": "cannot-resurrect"},
            )
            assert r_rev.status_code == S.UNPROCESSABLE_CONTENT
    finally:
        await close_app_services(svc)


def test_cli_apps_registered_tools_and_update_command(tmp_path: Path) -> None:
    """Verify `lai apps create --registered-tools` and `lai apps update` flags & errors."""
    runner = CliRunner()
    db_path = tmp_path / "cli_apps.db"
    tools_json = tmp_path / "tools.json"
    tools_json.write_text(
        json.dumps(
            [
                {
                    "name": "read_local",
                    "description": "Read local file",
                    "permission_tier": "read",
                }
            ]
        ),
        encoding="utf-8",
    )

    res_create = runner.invoke(
        cli_app,
        [
            "apps",
            "create",
            "--db-path",
            str(db_path),
            "--name",
            "desktop",
            "--oauth-client-id",
            "cid-desk",
            "--tier",
            "read",
            "--project-id",
            "prj_1",
            "--registered-tools",
            str(tools_json),
            "--expires-days",
            "30",
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    assert "Tools: 1" in res_create.output

    list_line = runner.invoke(cli_app, ["apps", "list", "--db-path", str(db_path)]).output.strip()
    app_id = list_line.split("\t")[0]

    res_upd = runner.invoke(
        cli_app,
        [
            "apps",
            "update",
            app_id,
            "--db-path",
            str(db_path),
            "--name",
            "desktop-v2",
            "--tier",
            "read",
            "--tier",
            "write",
            "--clear-projects",
            "--clear-tools",
            "--clear-expiry",
        ],
    )
    assert res_upd.exit_code == 0, res_upd.output
    assert "Projects: * | Tools: 0" in res_upd.output

    for bad_flags in (
        ["--project-id", "prj_1", "--clear-projects"],
        ["--registered-tools", str(tools_json), "--clear-tools"],
        ["--expires-days", "5", "--clear-expiry"],
    ):
        r_bad = runner.invoke(
            cli_app,
            ["apps", "update", app_id, "--db-path", str(db_path), *bad_flags],
        )
        assert r_bad.exit_code == 1

    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"not": "a list"}', encoding="utf-8")
    r_bad_file = runner.invoke(
        cli_app,
        [
            "apps",
            "update",
            app_id,
            "--db-path",
            str(db_path),
            "--registered-tools",
            str(bad_json),
        ],
    )
    assert r_bad_file.exit_code == 1
    assert "must be an array" in r_bad_file.output

    missing_tools = tmp_path / "missing.json"
    r_missing_file = runner.invoke(
        cli_app,
        [
            "apps",
            "update",
            app_id,
            "--db-path",
            str(db_path),
            "--registered-tools",
            str(missing_tools),
        ],
    )
    assert r_missing_file.exit_code == 1

    async def _verify_record() -> None:
        async with SqliteDatabase(db_path) as db:
            rec = await SqliteClientAppRepository(db).get_by_id(app_id)
            assert rec is not None
            assert (rec.name, rec.allowed_project_ids, rec.registered_tools, rec.expires_at) == (
                "desktop-v2",
                (),
                (),
                None,
            )

    asyncio.run(_verify_record())
