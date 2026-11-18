"""Unit tests for client app security, CLI `lai apps`, and Admin `/v1/client-apps` router."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from http import HTTPStatus as S
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

from lai.api.deps import AppServices, get_services
from lai.api.errors import register_exception_handlers
from lai.api.routes._openai_helpers import StreamMeta, format_openai_sse
from lai.api.routes.client_apps import router as client_apps_router
from lai.api.runtime import build_app_services, close_app_services
from lai.cli.app import app as cli_app
from lai.config.settings import LaiSettings
from lai.db import SqliteClientAppRepository, SqliteDatabase
from lai.domain.auth import ApiScope
from lai.domain.chat import (
    ClientToolCallEvent,
    ContextMode,
    Message,
    MessageRole,
    OrchestratorEvent,
    TokenUsage,
    TurnCompletedEvent,
    TurnOutput,
)
from lai.domain.errors import ValidationFailedError
from lai.domain.ids import CLIENT_APP_ID_PREFIX
from lai.domain.observability import LoggingConfig
from lai.domain.tools import PermissionTier, ToolExecutionTarget, ToolSpec
from lai.security import (
    CLIENT_APP_SECRET_PREFIX,
    generate_api_key,
    generate_client_app,
    hash_client_app_secret,
    verify_client_app_secret,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator
    from pathlib import Path

    from lai.ports.repositories import ClientAppRepositoryPort

_EXPECTED_HASH_LEN = 64
_EXPECTED_PREFIX_LEN = 15


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


class _ServicesWithApps(AppServices):
    def __init__(self, base: AppServices, repo: ClientAppRepositoryPort) -> None:
        for f_info in fields(AppServices):
            setattr(self, f_info.name, getattr(base, f_info.name))
        self.client_app_repo = repo


class _ServicesWithoutApps:
    def __init__(self, base: AppServices) -> None:
        self._base = base

    def __getattr__(self, name: str) -> object:
        if name == "client_app_repo":
            return None
        return getattr(self._base, name)


def test_client_app_security_generation_hashing_and_is_active() -> None:
    """generate_client_app validates tiers, coerces CLIENT target, and verifies secrets."""
    assert CLIENT_APP_SECRET_PREFIX == "lai_app_"  # noqa: S105
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    tool_in = ToolSpec(
        name="run_local",
        description="Run local command",
        permission_tier=PermissionTier.EXEC,
        execution_target=ToolExecutionTarget.SERVER_ATTACHMENT,
    )
    created = generate_client_app(
        "  Desktop Helper  ",
        allowed_tiers=(PermissionTier.READ, PermissionTier.EXEC, PermissionTier.READ),
        registered_tools=(tool_in,),
        allowed_project_ids=(" prj_1 ", "", "prj_1", "prj_2"),
        expires_at=now + timedelta(days=7),
        created_at=now,
    )
    rec = created.record
    assert rec.id.startswith(CLIENT_APP_ID_PREFIX)
    assert rec.name == "Desktop Helper"
    assert rec.secret_prefix.startswith(CLIENT_APP_SECRET_PREFIX)
    assert len(rec.secret_prefix) == _EXPECTED_PREFIX_LEN
    assert len(rec.secret_hash) == _EXPECTED_HASH_LEN
    assert rec.allowed_tiers == (PermissionTier.READ, PermissionTier.EXEC)
    assert rec.allowed_project_ids == ("prj_1", "prj_2")
    assert rec.registered_tools[0].execution_target is ToolExecutionTarget.CLIENT
    assert created.plaintext_secret.startswith(CLIENT_APP_SECRET_PREFIX)
    assert hash_client_app_secret(created.plaintext_secret) == rec.secret_hash
    assert verify_client_app_secret(created.plaintext_secret, rec.secret_hash) is True
    assert verify_client_app_secret("wrong_secret", rec.secret_hash) is False
    assert verify_client_app_secret(created.plaintext_secret, "0" * 64) is False
    assert verify_client_app_secret("", rec.secret_hash) is False
    assert verify_client_app_secret(created.plaintext_secret, "") is False
    assert verify_client_app_secret("", hash_client_app_secret("")) is False

    assert rec.is_active(now=now) is True
    assert rec.is_active(now=now + timedelta(days=7)) is False
    assert rec.model_copy(update={"revoked_at": now}).is_active(now=now) is False

    with pytest.raises(ValidationFailedError, match="must not be empty"):
        generate_client_app("   ")

    with pytest.raises(ValidationFailedError, match="allowed_tiers"):
        generate_client_app("bad-tier", (PermissionTier.READ,), (tool_in,))

    dup_a = ToolSpec(name="dup", description="a", permission_tier=PermissionTier.READ)
    dup_b = ToolSpec(name="dup", description="b", permission_tier=PermissionTier.READ)
    with pytest.raises(ValidationFailedError, match="Duplicate"):
        generate_client_app("dup-tools", (PermissionTier.READ,), (dup_a, dup_b))


def test_cli_apps_create_list_revoke_and_errors(tmp_path: Path) -> None:
    """Verify `lai apps create/list/revoke` lifecycle and error handling."""
    runner = CliRunner()
    db_path = tmp_path / "apps.db"

    res_empty = runner.invoke(cli_app, ["apps", "list", "--db-path", str(db_path)])
    assert res_empty.exit_code == 0
    assert "No client apps found." in res_empty.output

    res_bad = runner.invoke(cli_app, ["apps", "create", "--db-path", str(db_path), "--name", "   "])
    assert res_bad.exit_code == 1
    assert "validation_failed" in res_bad.output

    res_create = runner.invoke(
        cli_app,
        [
            "apps",
            "create",
            "--db-path",
            str(db_path),
            "--name",
            "vscode-ext",
            "--tier",
            "read",
            "--tier",
            "exec",
            "--project-id",
            "prj_1",
            "--project-id",
            "prj_2",
            "--expires-days",
            "14",
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    assert "Allowed tiers: read,exec | Projects: prj_1,prj_2" in res_create.output
    assert "lai_app_" in res_create.output

    res_list = runner.invoke(cli_app, ["apps", "list", "--db-path", str(db_path)])
    assert res_list.exit_code == 0
    line = res_list.output.strip().splitlines()[0]
    app_id = line.split("\t")[0]
    assert "\tactive\t[read,exec]\t[prj_1,prj_2]\ttools=0" in line

    expired_app = generate_client_app(
        "old-app",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    ).record

    async def _seed_expired() -> None:
        async with SqliteDatabase(db_path) as db:
            await SqliteClientAppRepository(db).create(expired_app)

    asyncio.run(_seed_expired())

    res_revoke = runner.invoke(cli_app, ["apps", "revoke", app_id, "--db-path", str(db_path)])
    assert res_revoke.exit_code == 0

    res_list_after = runner.invoke(cli_app, ["apps", "list", "--db-path", str(db_path)])
    assert "\trevoked\t" in res_list_after.output
    assert "\texpired\t" in res_list_after.output

    res_missing = runner.invoke(
        cli_app, ["apps", "revoke", "app_missing", "--db-path", str(db_path)]
    )
    assert res_missing.exit_code == 1


async def test_admin_client_apps_router_and_schemas(tmp_path: Path) -> None:
    """Admin `/v1/client-apps` router enforces ADMIN scope, hides secret_hash, and revokes."""
    data_dir = tmp_path / "data"
    settings = LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=False),
    )
    base_svc = await build_app_services(settings)
    db = SqliteDatabase(data_dir / "lai.db")
    await db.initialize()
    try:
        wrapped = _ServicesWithApps(base_svc, SqliteClientAppRepository(db))
        admin_key = generate_api_key("admin", (ApiScope.ADMIN,))
        user_key = generate_api_key("user", (ApiScope.CHAT_WRITE, ApiScope.TOOLS_EXEC))
        await wrapped.api_key_repo.create(admin_key.record)
        await wrapped.api_key_repo.create(user_key.record)

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(client_apps_router)
        app.dependency_overrides[get_services] = lambda: _ServicesWithoutApps(base_svc)

        admin_hdrs = {"Authorization": f"Bearer {admin_key.plaintext_key}"}
        user_hdrs = {"Authorization": f"Bearer {user_key.plaintext_key}"}
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r_503 = await client.get("/v1/client-apps", headers=admin_hdrs)
            assert (r_503.status_code, r_503.json()["error"]["code"]) == (
                S.SERVICE_UNAVAILABLE,
                "service_unavailable",
            )

            app.dependency_overrides[get_services] = lambda: wrapped
            payload = {
                "name": "agent-cli",
                "allowed_tiers": ["read", "exec"],
                "registered_tools": [
                    {
                        "name": "run_ps",
                        "description": "Run local script",
                        "permission_tier": "exec",
                        "execution_target": "server_attachment",
                    }
                ],
                "allowed_project_ids": ["prj_1"],
            }
            assert (
                await client.post("/v1/client-apps", headers=user_hdrs, json=payload)
            ).status_code == S.FORBIDDEN

            r_bad = await client.post(
                "/v1/client-apps",
                headers=admin_hdrs,
                json={
                    "name": "bad-tier",
                    "allowed_tiers": ["read"],
                    "registered_tools": payload["registered_tools"],
                },
            )
            assert r_bad.status_code == S.UNPROCESSABLE_CONTENT

            r_create = await client.post("/v1/client-apps", headers=admin_hdrs, json=payload)
            assert r_create.status_code == S.CREATED
            created_json = r_create.json()
            assert "secret_hash" not in r_create.text
            assert created_json["client_secret"].startswith(CLIENT_APP_SECRET_PREFIX)
            app_obj = created_json["app"]
            assert (app_obj["active"], app_obj["registered_tools"][0]["execution_target"]) == (
                True,
                "client",
            )

            r_list = await client.get("/v1/client-apps", headers=admin_hdrs)
            assert r_list.status_code == S.OK
            assert "secret_hash" not in r_list.text
            assert r_list.json()["count"] == 1

            r_del = await client.delete(f"/v1/client-apps/{app_obj['id']}", headers=admin_hdrs)
            assert r_del.status_code == S.OK
            assert r_del.json() == {"id": app_obj["id"], "deleted": True}

            r_list_after = await client.get("/v1/client-apps", headers=admin_hdrs)
            assert r_list_after.json()["items"][0]["active"] is False

            r_missing = await client.delete("/v1/client-apps/app_nope", headers=admin_hdrs)
            assert r_missing.status_code == S.NOT_FOUND
    finally:
        await db.close()
        await close_app_services(base_svc)


async def test_openai_sse_handles_client_tool_call_event() -> None:
    """format_openai_sse translates ClientToolCallEvent cleanly until runtime tool chunks."""

    async def _events() -> AsyncIterator[OrchestratorEvent]:
        yield ClientToolCallEvent(
            approval_id="apr_1",
            tool_call_id="tc_1",
            tool_name="run_ps",
            arguments_json="{}",
            requires_approval=True,
        )
        yield TurnCompletedEvent(
            output=TurnOutput(
                session_id="ses_1",
                message=Message(
                    session_id="ses_1",
                    project_id="prj_1",
                    role=MessageRole.ASSISTANT,
                    content="",
                ),
                context_mode_used=ContextMode.RETRIEVAL,
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )

    meta = StreamMeta(completion_id="chatcmpl_1", created=1, model_name="gemma4-26b-a4b")
    frames = [frame async for frame in format_openai_sse(_events(), meta)]
    joined = "".join(frames)
    assert "Client tool call: tc_1 (run_ps)" in joined
    assert "data: [DONE]\n\n" in joined
