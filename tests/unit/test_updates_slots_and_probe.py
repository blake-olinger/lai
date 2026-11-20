"""Unit tests for ReleaseSlotManager, SystemdServiceController, and HttpHealthProbe."""

from __future__ import annotations

import hashlib
import sys
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from lai.domain.errors import LaiError
from lai.updates import HttpHealthProbe, ReleaseSlotManager, SystemdServiceController

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence
    from pathlib import Path

_WHL = "lai-0.2.0-py3-none-any.whl"
_WHL_BYTES = b"PK\x03\x04fake-wheel-payload-0.2.0"
_REQ = "requirements-lock.txt"
_REQ_BYTES = (
    b"anyio==4.16.0 \\\n"
    b"    --hash=sha256:59bb03251907983d1603d27c3087ddb1e93299ea2163adcdba37899a4fb12ffb\n"
)
_WHL_SHA = hashlib.sha256(_WHL_BYTES).hexdigest()


async def _expect(code: str, coro: Awaitable[object]) -> None:
    with pytest.raises(LaiError) as exc_info:
        await coro
    assert exc_info.value.code == code


async def _os_err_runner(args: Sequence[str]) -> tuple[int, str, str]:
    del args
    msg = "missing"
    raise OSError(msg)


async def _rc_err_runner(args: Sequence[str]) -> tuple[int, str, str]:
    del args
    return 1, "", "boom"


async def test_slot_manager_lifecycle_pruning_and_errors(tmp_path: Path) -> None:
    """ReleaseSlotManager creates slots with --python 3.13, --require-hashes, and --no-deps."""
    rel, cur = tmp_path / "releases", tmp_path / "current"
    wheel, req = tmp_path / _WHL, tmp_path / _REQ
    wheel.write_bytes(_WHL_BYTES)
    req.write_bytes(_REQ_BYTES)
    cmds: list[list[str]] = []

    async def ok_runner(args: Sequence[str]) -> tuple[int, str, str]:
        cmds.append(list(args))
        return 0, "ok", ""

    mgr = ReleaseSlotManager(rel, cur, command_runner=ok_runner)
    assert (mgr.current_slot(), mgr.previous_slot()) == (None, None)
    await _expect("upgrade_requirements_missing", mgr.install_slot("0.1.0", wheel))
    v1 = await mgr.install_slot("0.1.0", wheel, requirements_path=req)
    assert cmds[0][:4] == ["uv", "venv", "--python", "3.13"]
    assert "--require-hashes" in cmds[1]
    assert ["-r", str(req)] == cmds[1][-2:]
    assert "--no-deps" in cmds[2]
    assert cmds[2][-1] == str(wheel)
    assert mgr.activate_slot(v1) is None
    await mgr.install_slot("0.2.0", wheel, requirements_path=req)
    v2 = await mgr.install_slot("0.2.0", wheel, requirements_path=req)
    assert mgr.activate_slot(v2) == v1

    fresh = ReleaseSlotManager(rel, cur, command_runner=ok_runner)
    assert (fresh.current_slot(), fresh.previous_slot()) == (v2, v1)
    (rel / ".previous").unlink()
    assert fresh.previous_slot() == v1

    (rel / "v-custom").mkdir()
    await mgr.install_slot("0.3.0", wheel, requirements_path=req)
    v4 = await mgr.install_slot("0.4.0", wheel, requirements_path=req)
    mgr.activate_slot(v4)
    pruned = {p.name for p in mgr.prune_old_slots(keep=1, active=v4)}
    assert {"v0.1.0", "v0.3.0", "v-custom"}.issubset(pruned)
    assert (v4.is_dir(), v2.is_dir()) == (True, True)

    with pytest.raises(ValueError, match="keep"):
        mgr.prune_old_slots(keep=0)
    for code, coro in (
        ("upgrade_slot_active", mgr.install_slot("0.4.0", wheel, requirements_path=req)),
        ("upgrade_invalid_version", mgr.install_slot("../evil", wheel, requirements_path=req)),
        (
            "upgrade_invalid_wheel",
            mgr.install_slot("0.5.0", tmp_path / "missing.whl", requirements_path=req),
        ),
        (
            "upgrade_requirements_missing",
            mgr.install_slot("0.5.0", wheel, requirements_path=tmp_path / "no-req.txt"),
        ),
    ):
        await _expect(code, coro)
    with pytest.raises(LaiError) as err_miss:
        mgr.activate_slot(tmp_path / "missing-slot")
    assert err_miss.value.code == "upgrade_slot_missing"

    for runner in (_os_err_runner, _rc_err_runner):
        fail_mgr = ReleaseSlotManager(rel, cur, command_runner=runner)
        await _expect(
            "upgrade_slot_install_failed",
            fail_mgr.install_slot("0.5.0", wheel, requirements_path=req),
        )
        assert not (rel / "v0.5.0").exists()


@respx.mock
async def test_systemd_controller_and_http_health_probe() -> None:
    """SystemdServiceController and HttpHealthProbe handle stop/restart and HTTP polling."""
    cmds: list[list[str]] = []

    async def ok_runner(args: Sequence[str]) -> tuple[int, str, str]:
        cmds.append(list(args))
        return 0, "", ""

    ctrl_ok = SystemdServiceController(command_runner=ok_runner)
    await ctrl_ok.restart("lai.service")
    await ctrl_ok.stop("lai.service")
    assert cmds == [["systemctl", "restart", "lai.service"], ["systemctl", "stop", "lai.service"]]
    for ctrl, svc in (
        (SystemdServiceController(systemctl_executable=sys.executable), "missing.service"),
        (SystemdServiceController(command_runner=ok_runner), "   "),
        (SystemdServiceController(command_runner=_os_err_runner), "lai.service"),
    ):
        await _expect("upgrade_service_restart_failed", ctrl.restart(svc))
        await _expect("upgrade_service_stop_failed", ctrl.stop(svc))

    now, url = [0.0], "http://127.0.0.1:8400/v1/health"

    async def sleeper(dt: float) -> None:
        now[0] += dt

    resps: list[httpx.Response | Exception] = [
        httpx.ConnectError("boot"),
        httpx.Response(503, text="starting"),
        httpx.Response(200, text="bad-json"),
        httpx.Response(200, json=["list"]),
        httpx.Response(200, json={"status": "degraded", "version": "0.2.0"}),
        httpx.Response(200, json={"status": "ok", "version": "0.1.0"}),
        httpx.Response(200, json={"status": "ok", "version": "0.2.0"}),
    ]
    route = respx.get(url).mock(side_effect=resps)
    async with httpx.AsyncClient() as shared:
        p1 = HttpHealthProbe(http_client=shared, clock=lambda: now[0], sleeper=sleeper)
        assert await p1.wait_healthy(
            url, expected_version="0.2.0", timeout_seconds=10.0, poll_interval_seconds=1.0
        )
        assert route.call_count == len(resps)
    now[0] = 0.0
    respx.get(url).mock(return_value=httpx.Response(200, json={"status": "ok", "version": "0.1.0"}))
    p2 = HttpHealthProbe(clock=lambda: now[0], sleeper=sleeper)
    assert not await p2.wait_healthy(
        url, expected_version="0.2.0", timeout_seconds=2.0, poll_interval_seconds=1.0
    )
