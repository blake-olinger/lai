"""Unit tests for tool approval routes, error envelopes, 500 alerts, 429 limits, and CLI."""

from __future__ import annotations

import json
from http import HTTPStatus as S
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from lai.cli.app import app as cli_app
from lai.domain import errors as errs
from lai.domain import tools as dt
from lai.domain.auth import ApiScope
from lai.domain.observability import AlertCategory
from tests._helpers import bearer_headers
from tests.unit.test_api_routes import FakeOrch, open_routes_env

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests._helpers import FailingEventStream

HTTP_CLIENT_DISCONNECTED = 499
EXPECTED_MIN_TOOLS = 3
_CHAT_URL = "/v1/chat/completions"
_CHAT_JSON: dict[str, object] = {
    "model": "gemma4-26b-a4b",
    "messages": [{"role": "user", "content": "hi"}],
}


async def test_tools_list_and_approval_resolve_requires_admin(tmp_path: Path) -> None:
    """Verify /v1/tools listing and admin-gated approval resolution."""
    async with open_routes_env(tmp_path) as env:
        hdrs = bearer_headers(env.toks["writer"])
        assert (await env.client.get("/v1/tools", headers=hdrs)).json()["count"] >= (
            EXPECTED_MIN_TOOLS
        )
        pid = (await env.client.post("/v1/projects", headers=hdrs, json={"name": "tp"})).json()[
            "id"
        ]
        sid = (
            await env.client.post("/v1/sessions", headers=hdrs, json={"project_id": pid})
        ).json()["id"]
        ctx = dt.ToolExecutionContext(
            project_id=pid,
            session_id=sid,
            principal_id="usr_t",
            principal_scopes=(ApiScope.TOOLS_SAFE,),
        )
        with pytest.raises(errs.ToolApprovalRequiredError) as exc_info:
            await env.svc.tool_executor.invoke("read_note", {"note": "hi"}, ctx)
        apr_url = f"/v1/tool-approvals/{exc_info.value.approval_id}/resolve"
        assert (
            await env.client.post(apr_url, headers=hdrs, json={"approved": True})
        ).status_code == S.FORBIDDEN
        r_a = await env.client.post(
            apr_url, headers=bearer_headers(env.toks["admin"]), json={"approved": True}
        )
        assert r_a.json()["status"] == dt.ApprovalStatus.EXECUTED.value


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (errs.ToolApprovalRequiredError("ok", approval_id="a", tool_name="s"), S.CONFLICT),
        (errs.PayloadTooLargeError("big"), S.REQUEST_ENTITY_TOO_LARGE),
        (errs.GuardrailBlockedError("no", violated_policy="p"), S.UNPROCESSABLE_CONTENT),
        (errs.ClientDisconnectedError("bye"), HTTP_CLIENT_DISCONNECTED),
        (errs.InferenceBackendError("bad"), S.BAD_GATEWAY),
        (errs.QueueFullError("full"), S.SERVICE_UNAVAILABLE),
    ],
)
async def test_lai_error_mapping_to_http_envelopes(
    tmp_path: Path,
    err: errs.LaiError,
    expected: int,
) -> None:
    """Verify domain LaiError subclasses map to expected HTTP status codes and request_id."""
    async with open_routes_env(tmp_path, orch=FakeOrch(next_error=err)) as env:
        res = await env.client.post(
            _CHAT_URL, headers=bearer_headers(env.toks["writer"]), json=_CHAT_JSON
        )
        assert res.status_code == expected
        assert res.json()["error"]["request_id"]


async def test_unhandled_runtime_error_returns_500_and_fatal_alert(tmp_path: Path) -> None:
    """Verify unhandled RuntimeError returns sanitized 500 and emits FATAL_EXCEPTION alert."""
    orch = FakeOrch(next_error=RuntimeError("secret crash"))
    async with open_routes_env(tmp_path, orch=orch, raise_app_exceptions=False) as env:
        hdrs = bearer_headers(env.toks["writer"], **{"X-Request-Id": "r5"})
        r5 = await env.client.post(_CHAT_URL, headers=hdrs, json=_CHAT_JSON)
        assert r5.status_code == S.INTERNAL_SERVER_ERROR
        assert "secret crash" not in json.dumps(r5.json())
        items = (
            await env.client.get("/v1/system/alerts", headers=bearer_headers(env.toks["admin"]))
        ).json()["items"]
        cat = AlertCategory.FATAL_EXCEPTION.value
        assert any(a["category"] == cat and a["request_id"] == "r5" for a in items)


async def test_rate_limiter_enforces_429_and_retry_after(
    tmp_path: Path,
    auth_headers_factory: Callable[..., dict[str, str]],
    failing_stream_cls: type[FailingEventStream],
) -> None:
    """Verify rate_limit_rpm=1 enforced upfront returns 429 with Retry-After on second request."""
    with pytest.raises(NotImplementedError):
        await anext(failing_stream_cls(NotImplementedError()))
    async with open_routes_env(tmp_path, rate_limit_rpm=1) as env:
        hdrs = auth_headers_factory(env.toks["writer"])
        assert (await env.client.get("/v1/projects", headers=hdrs)).status_code == S.OK
        r_429 = await env.client.get("/v1/projects", headers=hdrs)
        assert r_429.status_code == S.TOO_MANY_REQUESTS
        assert int(r_429.headers["Retry-After"]) >= 1


def test_cli_runner_smoke(tmp_path: Path) -> None:
    """Smoke-test Typer CLI init, keys create/list/revoke, and tunnel render-config."""
    runner, cfg, out = CliRunner(), tmp_path / "cli" / "config.yaml", tmp_path / "cf.yml"
    init_cmd = ["init", "-c", str(cfg), "--data-dir", str(tmp_path / "d")]
    assert runner.invoke(cli_app, init_cmd).exit_code == 0
    key_res = runner.invoke(cli_app, ["keys", "create", "-c", str(cfg), "-n", "k", "--admin"])
    assert "lai_sk_" in key_res.output
    kids = runner.invoke(cli_app, ["keys", "list", "-c", str(cfg)]).output.splitlines()
    rev_res = runner.invoke(cli_app, ["keys", "revoke", kids[0].split("\t")[0], "-c", str(cfg)])
    assert rev_res.exit_code == 0
    tun_cmd = ["tunnel", "render-config", "-c", str(cfg), "-H", "ai.t", "-o", str(out)]
    assert runner.invoke(cli_app, tun_cmd).exit_code == 0
    assert "ai.t" in out.read_text(encoding="utf-8")
