"""Unit tests for update domain models, release sources, slot manager, service, and health probe."""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import TYPE_CHECKING

import httpx
import pytest
import respx
from pydantic import ValidationError

from lai.domain.errors import LaiError
from lai.domain.observability import AlertCategory, AlertEvent, LoggingConfig
from lai.domain.updates import ReleaseArtifact, UpdateConfig
from lai.observability import AlertManager, configure_observability
from lai.ports.observability import AlertSinkPort
from lai.updates import (
    GitHubReleaseSource,
    HttpHealthProbe,
    LocalWheelSource,
    ReleaseSlotManager,
    SystemdServiceController,
    compute_sha256,
    parse_semver_tuple,
    parse_sha256sums,
    verify_wheel_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator, Sequence
    from pathlib import Path

_WHL = "lai-0.2.0-py3-none-any.whl"
_WHL_BYTES = b"PK\x03\x04fake-wheel-payload-0.2.0"
_WHL_SHA = hashlib.sha256(_WHL_BYTES).hexdigest()
_BAD_SHA = "0" * 64
_DL = "https://dl.example.test"
_GH = "https://api.github.com/repos/org/lai"


class _MemorySink(AlertSinkPort):
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send_alert(self, event: AlertEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        h.flush()
        h.close()
    root.handlers.clear()


def _install_alerts() -> tuple[AlertManager, _MemorySink]:
    sink = _MemorySink()
    mgr = AlertManager(sinks=(sink,), cooldown_seconds=0.0)
    configure_observability(
        LoggingConfig(enable_file_logs=False, alert_cooldown_seconds=0.0, alert_on_errors=True),
        alert_manager=mgr,
    )
    return mgr, sink


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


def _rel(
    tag: str = "v0.2.0", *, draft: bool = False, wheel: str | None = _WHL, sums: bool = True
) -> dict[str, object]:
    assets: list[object] = ["ignored", {"name": 123, "browser_download_url": None}]
    if wheel is not None:
        assets.append({"name": wheel, "browser_download_url": f"{_DL}/{wheel}"})
    if sums:
        assets.append({"name": "SHA256SUMS.txt", "browser_download_url": f"{_DL}/SHA256SUMS"})
    return {"tag_name": tag, "draft": draft, "assets": assets}


def test_domain_models_and_sha256_semver_helpers(tmp_path: Path) -> None:
    """Domain models, SemVer parser, SHA256SUMS parser, and verifier validate strictly."""
    cfg = UpdateConfig()
    assert (cfg.enabled, cfg.auto_rollback, cfg.keep_releases) == (False, True, 3)
    for bad in ({"github_repo": "bad repo"}, {"keep_releases": 0}):
        with pytest.raises(ValidationError):
            UpdateConfig.model_validate(bad)
    art = ReleaseArtifact(
        version="0.2.0", tag="v0.2.0", wheel_filename=_WHL, sha256=_WHL_SHA.upper()
    )
    assert art.sha256 == _WHL_SHA
    with pytest.raises(ValidationError):
        ReleaseArtifact(version="0.2.0", tag="v0.2.0", wheel_filename=_WHL, sha256="short")
    assert parse_semver_tuple("0.2.0") > parse_semver_tuple("0.1.0")
    assert parse_semver_tuple("v1.10.3-rc1") == (1, 10, 3)
    with pytest.raises(ValueError, match="Invalid SemVer"):
        parse_semver_tuple("bad-ver")
    assert parse_sha256sums(f"# c\n\n{_WHL_SHA.upper()}  *dist/{_WHL}\n") == {_WHL: _WHL_SHA}
    for bad_sums in ("no_whitespace_here", f"{'z' * 64}  {_WHL}"):
        with pytest.raises(LaiError) as err:
            parse_sha256sums(bad_sums)
        assert err.value.code == "upgrade_invalid_checksums"
    wheel = tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    assert compute_sha256(wheel) == _WHL_SHA
    verify_wheel_sha256(wheel, _WHL_SHA.upper())
    mgr, _ = _install_alerts()
    with pytest.raises(LaiError) as mismatch:
        verify_wheel_sha256(wheel, _BAD_SHA)
    assert (mismatch.value.code, mismatch.value.status_code) == ("upgrade_checksum_mismatch", 500)
    assert any(a.category is AlertCategory.SECURITY_ALERT for a in mgr.recent_alerts())


@respx.mock
async def test_github_source_happy_paths_and_tamper_unlink(tmp_path: Path) -> None:
    """GitHubReleaseSource supports latest, prerelease, tag, injected client, and tamper abort."""
    respx.get(f"{_GH}/releases/latest").mock(return_value=httpx.Response(200, json=_rel()))
    rels = [
        _rel("v0.3.0-draft", draft=True),
        _rel("v0.3.0-rc1", wheel="lai-0.3.0rc1-py3-none-any.whl"),
    ]
    respx.get(f"{_GH}/releases").mock(return_value=httpx.Response(200, json=rels))
    respx.get(f"{_GH}/releases/tags/v0.2.0").mock(return_value=httpx.Response(200, json=_rel()))
    sums_txt = f"{_WHL_SHA}  {_WHL}\n{_WHL_SHA}  lai-0.3.0rc1-py3-none-any.whl\n"
    respx.get(f"{_DL}/SHA256SUMS").mock(return_value=httpx.Response(200, text=sums_txt))
    dl_seq: list[httpx.Response | Exception] = [
        httpx.Response(200, content=_WHL_BYTES),
        httpx.Response(200, content=b"tampered-bytes"),
        httpx.ConnectError("stream reset"),
    ]
    respx.get(f"{_DL}/{_WHL}").mock(side_effect=dl_seq)
    async with httpx.AsyncClient() as client:
        auth_val = "test-bearer-value"
        src = GitHubReleaseSource("org/lai", token=auth_val, http_client=client)
        latest = await src.get_latest(allow_prerelease=False)
        assert (latest.version, latest.sha256) == ("0.2.0", _WHL_SHA)
        assert (await src.get_latest(allow_prerelease=True)).tag == "v0.3.0-rc1"
        tagged = await src.get_version("v0.2.0")
        assert (await src.download_wheel(tagged, tmp_path / "ok")).read_bytes() == _WHL_BYTES
        mgr, sink = _install_alerts()
        bad_stage = tmp_path / "bad"
        await _expect("upgrade_checksum_mismatch", src.download_wheel(tagged, bad_stage))
        assert not (bad_stage / _WHL).exists()
        await mgr.flush()
        assert any(e.category is AlertCategory.SECURITY_ALERT for e in sink.events)
        await _expect("upgrade_download_failed", src.download_wheel(tagged, tmp_path / "net"))


@respx.mock
async def test_github_source_error_branches(tmp_path: Path) -> None:
    """GitHubReleaseSource fails closed on API errors, bad payloads, missing assets, traversal."""
    src = GitHubReleaseSource("org/lai")
    respx.get(f"{_GH}/releases/tags/v9.9.9").mock(return_value=httpx.Response(404))
    await _expect("upgrade_release_not_found", src.get_version("9.9.9"))

    r200 = httpx.Response
    cases: list[tuple[httpx.Response | Exception, str]] = [
        (httpx.ConnectError("offline"), "upgrade_source_unreachable"),
        (r200(500), "upgrade_source_error"),
        (r200(200, json=["not-a-dict"]), "upgrade_invalid_release"),
        (r200(200, json={"assets": []}), "upgrade_invalid_release"),
        (r200(200, json={"tag_name": "v0.2.0", "assets": "bad"}), "upgrade_asset_missing"),
        (r200(200, json=_rel(wheel=None)), "upgrade_asset_missing"),
        (r200(200, json=_rel(sums=False)), "upgrade_checksums_missing"),
        (r200(200, json=_rel()), "upgrade_download_failed"),
        (r200(200, json=_rel()), "upgrade_download_failed"),
        (r200(200, json=_rel()), "upgrade_checksum_missing"),
    ]
    respx.get(f"{_GH}/releases/latest").mock(side_effect=[c[0] for c in cases])
    sums_fx: list[httpx.Response | Exception] = [
        httpx.ConnectError("err"),
        r200(503),
        r200(200, text=f"{_WHL_SHA}  other.whl\n"),
    ]
    respx.get(f"{_DL}/SHA256SUMS").mock(side_effect=sums_fx)
    for _, code in cases:
        await _expect(code, src.get_latest())
    respx.get(f"{_GH}/releases").mock(side_effect=[r200(200, json={"bad": 1}), r200(200, json=[])])
    await _expect("upgrade_invalid_release", src.get_latest(allow_prerelease=True))
    await _expect("upgrade_release_not_found", src.get_latest(allow_prerelease=True))
    no_url = ReleaseArtifact(version="0.2.0", tag="v0.2.0", wheel_filename=_WHL, sha256=_WHL_SHA)
    trav = ReleaseArtifact(
        version="0.2.0", tag="v0.2.0", wheel_filename="../x.whl", sha256=_WHL_SHA, download_url=_DL
    )
    await _expect("upgrade_download_failed", src.download_wheel(no_url, tmp_path))
    await _expect("upgrade_invalid_wheel", src.download_wheel(trav, tmp_path))


async def test_local_wheel_source_variants_and_errors(tmp_path: Path) -> None:
    """LocalWheelSource handles sidecar, SHA256SUMS, implicit hash, and tamper aborts."""
    wheel = tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    implicit = await LocalWheelSource(wheel).get_latest()
    assert (implicit.version, implicit.sha256) == ("0.2.0", _WHL_SHA)
    assert await LocalWheelSource(wheel).download_wheel(implicit, tmp_path) == wheel

    (tmp_path / "SHA256SUMS.txt").write_text(f"{_WHL_SHA}  {_WHL}\n", encoding="utf-8")
    assert (await LocalWheelSource(wheel).get_latest()).sha256 == _WHL_SHA

    other_sha = tmp_path / "c.sha"
    other_sha.write_text(f"{_WHL_SHA}  other.whl\n", encoding="utf-8")
    for code, s_path in (
        ("upgrade_checksum_missing", other_sha),
        ("upgrade_checksums_missing", tmp_path / "no.sha"),
    ):
        await _expect(code, LocalWheelSource(wheel, sha256_path=s_path).get_latest())

    (tmp_path / f"{_WHL}.sha256").write_text(f"{_WHL_SHA}\n", encoding="utf-8")
    src = LocalWheelSource(wheel)
    art = await src.get_version("v0.2.0")
    assert (await src.download_wheel(art, tmp_path / "staged")).read_bytes() == _WHL_BYTES
    await _expect("upgrade_release_not_found", src.get_version("0.9.9"))

    mgr, sink = _install_alerts()
    bad = LocalWheelSource(wheel, expected_sha256=_BAD_SHA)
    await _expect(
        "upgrade_checksum_mismatch", bad.download_wheel(await bad.get_latest(), tmp_path / "b")
    )
    await mgr.flush()
    assert any(e.category is AlertCategory.SECURITY_ALERT for e in sink.events)

    bad_name = tmp_path / "bad.whl"
    bad_name.write_bytes(b"x")
    missing = LocalWheelSource(tmp_path / "missing.whl")
    for coro in (
        LocalWheelSource(bad_name).get_latest(),
        missing.get_latest(),
        missing.download_wheel(art, tmp_path),
    ):
        await _expect("upgrade_invalid_wheel", coro)


async def test_slot_manager_lifecycle_pruning_and_errors(tmp_path: Path) -> None:
    """ReleaseSlotManager creates slots, flips symlinks, prunes oldest, and rejects errors."""
    rel, cur, wheel = tmp_path / "releases", tmp_path / "current", tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    cmds: list[list[str]] = []

    async def ok_runner(args: Sequence[str]) -> tuple[int, str, str]:
        cmds.append(list(args))
        return 0, "ok", ""

    mgr = ReleaseSlotManager(rel, cur, command_runner=ok_runner)
    assert (mgr.current_slot(), mgr.previous_slot()) == (None, None)
    v1 = await mgr.install_slot("0.1.0", wheel)
    assert (cmds[0][:2], cmds[1][:3]) == (["uv", "venv"], ["uv", "pip", "install"])
    assert mgr.activate_slot(v1) is None
    await mgr.install_slot("0.2.0", wheel)
    v2 = await mgr.install_slot("0.2.0", wheel)
    assert mgr.activate_slot(v2) == v1

    fresh = ReleaseSlotManager(rel, cur, command_runner=ok_runner)
    assert (fresh.current_slot(), fresh.previous_slot()) == (v2, v1)
    (rel / ".previous").unlink()
    assert fresh.previous_slot() == v1

    (rel / "v-custom").mkdir()
    await mgr.install_slot("0.3.0", wheel)
    v4 = await mgr.install_slot("0.4.0", wheel)
    mgr.activate_slot(v4)
    pruned = {p.name for p in mgr.prune_old_slots(keep=1, active=v4)}
    assert {"v0.1.0", "v0.3.0", "v-custom"}.issubset(pruned)
    assert (v4.is_dir(), v2.is_dir()) == (True, True)

    with pytest.raises(ValueError, match="keep"):
        mgr.prune_old_slots(keep=0)
    for code, coro in (
        ("upgrade_slot_active", mgr.install_slot("0.4.0", wheel)),
        ("upgrade_invalid_version", mgr.install_slot("../evil", wheel)),
        ("upgrade_invalid_wheel", mgr.install_slot("0.5.0", tmp_path / "missing.whl")),
    ):
        await _expect(code, coro)
    with pytest.raises(LaiError) as err_miss:
        mgr.activate_slot(tmp_path / "missing-slot")
    assert err_miss.value.code == "upgrade_slot_missing"

    for runner in (_os_err_runner, _rc_err_runner):
        fail_mgr = ReleaseSlotManager(rel, cur, command_runner=runner)
        await _expect("upgrade_slot_install_failed", fail_mgr.install_slot("0.5.0", wheel))
        assert not (rel / "v0.5.0").exists()


@respx.mock
async def test_systemd_controller_and_http_health_probe() -> None:
    """SystemdServiceController and HttpHealthProbe handle subprocess execution and polling."""
    cmds: list[list[str]] = []

    async def ok_runner(args: Sequence[str]) -> tuple[int, str, str]:
        cmds.append(list(args))
        return 0, "", ""

    await SystemdServiceController(command_runner=ok_runner).restart("lai.service")
    assert cmds == [["systemctl", "restart", "lai.service"]]
    for ctrl, svc in (
        (SystemdServiceController(systemctl_executable=sys.executable), "missing.service"),
        (SystemdServiceController(command_runner=ok_runner), "   "),
        (SystemdServiceController(command_runner=_os_err_runner), "lai.service"),
    ):
        await _expect("upgrade_service_restart_failed", ctrl.restart(svc))

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
