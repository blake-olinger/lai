"""Unit tests for update domain models, GitHubReleaseSource, and LocalWheelSource."""

from __future__ import annotations

import hashlib
import logging
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
    LocalWheelSource,
    compute_sha256,
    parse_semver_tuple,
    parse_sha256sums,
    verify_wheel_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator
    from pathlib import Path

_WHL = "lai-0.2.0-py3-none-any.whl"
_WHL_BYTES = b"PK\x03\x04fake-wheel-payload-0.2.0"
_WHL_SHA = hashlib.sha256(_WHL_BYTES).hexdigest()
_REQ = "requirements-lock.txt"
_REQ_BYTES = (
    b"anyio==4.16.0 \\\n"
    b"    --hash=sha256:59bb03251907983d1603d27c3087ddb1e93299ea2163adcdba37899a4fb12ffb\n"
)
_REQ_SHA = hashlib.sha256(_REQ_BYTES).hexdigest()
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


def _rel(
    tag: str = "v0.2.0",
    *,
    draft: bool = False,
    wheel: str | None = _WHL,
    sums: bool = True,
    reqs: bool = True,
) -> dict[str, object]:
    assets: list[object] = ["ignored", {"name": 123, "browser_download_url": None}]
    if wheel is not None:
        assets.append({"name": wheel, "browser_download_url": f"{_DL}/{wheel}"})
    if sums:
        assets.append({"name": "SHA256SUMS.txt", "browser_download_url": f"{_DL}/SHA256SUMS"})
    if reqs:
        assets.append({"name": _REQ, "browser_download_url": f"{_DL}/{_REQ}"})
    return {"tag_name": tag, "draft": draft, "assets": assets}


def test_domain_models_and_sha256_semver_helpers(tmp_path: Path) -> None:
    """Domain models, SemVer parser, SHA256SUMS parser, and verifier validate strictly."""
    cfg = UpdateConfig()
    assert (cfg.enabled, cfg.auto_rollback, cfg.keep_releases, cfg.health_timeout_seconds) == (
        False,
        True,
        3,
        300.0,
    )
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
    """GitHubReleaseSource supports latest, prerelease, tag, requirements lock, and tamper abort."""
    respx.get(f"{_GH}/releases/latest").mock(return_value=httpx.Response(200, json=_rel()))
    rels = [
        _rel("v0.3.0-draft", draft=True),
        _rel("v0.3.0-rc1", wheel="lai-0.3.0rc1-py3-none-any.whl"),
    ]
    respx.get(f"{_GH}/releases").mock(return_value=httpx.Response(200, json=rels))
    respx.get(f"{_GH}/releases/tags/v0.2.0").mock(return_value=httpx.Response(200, json=_rel()))
    sums_txt = (
        f"{_WHL_SHA}  {_WHL}\n{_WHL_SHA}  lai-0.3.0rc1-py3-none-any.whl\n{_REQ_SHA}  {_REQ}\n"
    )
    respx.get(f"{_DL}/SHA256SUMS").mock(return_value=httpx.Response(200, text=sums_txt))
    respx.get(f"{_DL}/{_WHL}").mock(
        side_effect=[
            httpx.Response(200, content=_WHL_BYTES),
            httpx.Response(200, content=b"tampered-bytes"),
            httpx.ConnectError("stream reset"),
        ]
    )
    respx.get(f"{_DL}/{_REQ}").mock(
        side_effect=[
            httpx.Response(200, content=_REQ_BYTES),
            httpx.Response(200, content=b"tampered-req"),
        ]
    )
    async with httpx.AsyncClient() as client:
        auth_val = "test-bearer-value"
        src = GitHubReleaseSource("org/lai", token=auth_val, http_client=client)
        latest = await src.get_latest(allow_prerelease=False)
        assert (latest.version, latest.sha256, latest.requirements_sha256) == (
            "0.2.0",
            _WHL_SHA,
            _REQ_SHA,
        )
        assert (await src.get_latest(allow_prerelease=True)).tag == "v0.3.0-rc1"
        tagged = await src.get_version("v0.2.0")
        assert (await src.download_wheel(tagged, tmp_path / "ok")).read_bytes() == _WHL_BYTES
        dl_req = await src.download_requirements(tagged, tmp_path / "ok")
        assert dl_req is not None
        assert dl_req.read_bytes() == _REQ_BYTES
        mgr, sink = _install_alerts()
        bad_stage = tmp_path / "bad"
        await _expect("upgrade_checksum_mismatch", src.download_wheel(tagged, bad_stage))
        await _expect("upgrade_checksum_mismatch", src.download_requirements(tagged, bad_stage))
        assert not (bad_stage / _WHL).exists()
        assert not (bad_stage / _REQ).exists()
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
        (r200(200, json=_rel(reqs=False)), "upgrade_requirements_missing"),
        (r200(200, json=_rel()), "upgrade_checksum_missing"),
    ]
    respx.get(f"{_GH}/releases/latest").mock(side_effect=[c[0] for c in cases])
    sums_fx: list[httpx.Response | Exception] = [
        httpx.ConnectError("err"),
        r200(503),
        r200(200, text=f"{_WHL_SHA}  other.whl\n"),
        r200(200, text=f"{_WHL_SHA}  {_WHL}\n"),
        r200(200, text=f"{_WHL_SHA}  {_WHL}\n"),
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
    await _expect("upgrade_requirements_missing", src.download_requirements(no_url, tmp_path))


async def test_local_wheel_source_variants_and_errors(tmp_path: Path) -> None:
    """LocalWheelSource fails closed without checksum/lockfile and verifies staged files."""
    wheel = tmp_path / _WHL
    wheel.write_bytes(_WHL_BYTES)
    mgr, sink = _install_alerts()
    await _expect("upgrade_checksum_missing", LocalWheelSource(wheel).get_latest())
    await mgr.flush()
    assert any(e.category is AlertCategory.SECURITY_ALERT for e in sink.events)

    req = tmp_path / _REQ
    req.write_bytes(_REQ_BYTES)
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(f"{_WHL_SHA}  {_WHL}\n{_REQ_SHA}  {_REQ}\n", encoding="utf-8")
    art_sums = await LocalWheelSource(wheel).get_latest()
    assert (art_sums.sha256, art_sums.requirements_sha256) == (_WHL_SHA, _REQ_SHA)
    assert await LocalWheelSource(wheel).download_wheel(art_sums, tmp_path) == wheel
    assert await LocalWheelSource(wheel).download_requirements(art_sums, tmp_path) == req

    other_sha = tmp_path / "c.sha"
    other_sha.write_text(f"{_WHL_SHA}  other.whl\n", encoding="utf-8")
    for code, s_path in (
        ("upgrade_checksum_missing", other_sha),
        ("upgrade_checksums_missing", tmp_path / "no.sha"),
    ):
        await _expect(code, LocalWheelSource(wheel, sha256_path=s_path).get_latest())

    (tmp_path / f"{_WHL}.sha256").write_text(f"{_WHL_SHA}\n", encoding="utf-8")
    (tmp_path / f"{_REQ}.sha256").write_text(f"{_REQ_SHA}\n", encoding="utf-8")
    src = LocalWheelSource(wheel)
    art = await src.get_version("v0.2.0")
    assert (await src.download_wheel(art, tmp_path / "staged")).read_bytes() == _WHL_BYTES
    staged_req = await src.download_requirements(art, tmp_path / "staged")
    assert staged_req is not None
    assert staged_req.read_bytes() == _REQ_BYTES
    await _expect("upgrade_release_not_found", src.get_version("0.9.9"))

    bad = LocalWheelSource(wheel, expected_sha256=_BAD_SHA)
    await _expect(
        "upgrade_checksum_mismatch", bad.download_wheel(await bad.get_latest(), tmp_path / "b")
    )

    bad_name = tmp_path / "bad.whl"
    bad_name.write_bytes(b"x")
    missing = LocalWheelSource(tmp_path / "missing.whl")
    for coro in (
        LocalWheelSource(bad_name).get_latest(),
        missing.get_latest(),
        missing.download_wheel(art, tmp_path),
    ):
        await _expect("upgrade_invalid_wheel", coro)
