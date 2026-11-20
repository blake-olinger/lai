"""Local wheel release source with fail-closed SHA-256 and locked requirements verification."""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING

from lai.domain.observability import AlertCategory
from lai.domain.updates import ReleaseArtifact
from lai.ports.updates import ReleaseSourcePort
from lai.updates._checksums import (
    REQUIREMENTS_LOCK_FILENAME,
    SHA256_RE,
    WHEEL_RE,
    make_upgrade_error,
    parse_sha256sums,
    prepare_dest_path,
    updates_logger,
    verify_file_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["LocalWheelSource"]


def _parse_sidecar_sha256(sidecar: Path, filename: str) -> str:
    text = sidecar.read_text(encoding="utf-8").strip()
    if SHA256_RE.fullmatch(text.lower()):
        return text.lower()
    digest = parse_sha256sums(text).get(filename)
    if digest is None:
        raise make_upgrade_error(
            "upgrade_checksum_missing", f"Sidecar {sidecar} omits {filename!r}."
        )
    return digest


def _lookup_in_manifest(manifest: Path, filename: str) -> str | None:
    if not manifest.is_file():
        return None
    text = manifest.read_text(encoding="utf-8").strip()
    if SHA256_RE.fullmatch(text.lower()):
        return None
    return parse_sha256sums(text).get(filename)


def _stage_local_file(src: Path, dest_dir: Path, filename: str, expected_sha256: str) -> Path:
    if not src.is_file():
        raise make_upgrade_error("upgrade_invalid_wheel", f"Local artifact missing: {src}", 404)
    verify_file_sha256(src, expected_sha256)
    dest = prepare_dest_path(dest_dir, filename)
    if dest.resolve() != src.resolve():
        shutil.copyfile(src, dest)
    return dest


class LocalWheelSource(ReleaseSourcePort):
    """ReleaseSourcePort reading a staged local wheel file for CI/CD push deployments."""

    def __init__(  # noqa: PLR0913
        self,
        wheel_path: Path,
        *,
        expected_sha256: str | None = None,
        sha256_path: Path | None = None,
        requirements_path: Path | None = None,
        requirements_sha256: str | None = None,
        require_requirements_lock: bool = True,
    ) -> None:
        self._wheel_path = wheel_path
        self._expected_sha256 = expected_sha256
        self._sha256_path = sha256_path
        self._requirements_path = requirements_path
        self._requirements_sha256 = requirements_sha256
        self._require_requirements_lock = require_requirements_lock

    def _resolve_wheel_sha256(self) -> str:
        if self._expected_sha256 is not None:
            return self._expected_sha256.strip().lower()
        if self._sha256_path is not None:
            if not self._sha256_path.is_file():
                raise make_upgrade_error(
                    "upgrade_checksums_missing", f"Missing: {self._sha256_path}", 404
                )
            return _parse_sidecar_sha256(self._sha256_path, self._wheel_path.name)
        sidecar = self._wheel_path.with_name(f"{self._wheel_path.name}.sha256")
        if sidecar.is_file():
            return _parse_sidecar_sha256(sidecar, self._wheel_path.name)
        found = _lookup_in_manifest(
            self._wheel_path.parent / "SHA256SUMS.txt", self._wheel_path.name
        )
        if found is not None:
            return found
        updates_logger().error(
            "upgrade_local_wheel_checksum_missing",
            wheel_path=str(self._wheel_path),
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        msg = (
            f"Refusing unverified local wheel {self._wheel_path.name}: provide --sha256, "
            f"{self._wheel_path.name}.sha256, or SHA256SUMS.txt."
        )
        raise make_upgrade_error("upgrade_checksum_missing", msg)

    def _resolve_requirements_path(self) -> Path | None:
        if self._requirements_path is not None:
            if not self._requirements_path.is_file():
                msg = f"Requirements lockfile missing: {self._requirements_path}"
                raise make_upgrade_error("upgrade_requirements_missing", msg, 404)
            return self._requirements_path
        sibling = self._wheel_path.parent / REQUIREMENTS_LOCK_FILENAME
        if sibling.is_file():
            return sibling
        if self._require_requirements_lock:
            msg = (
                f"Missing locked requirements file ({REQUIREMENTS_LOCK_FILENAME}) beside "
                f"{self._wheel_path.name}; supply --requirements or stage "
                f"{REQUIREMENTS_LOCK_FILENAME}."
            )
            raise make_upgrade_error("upgrade_requirements_missing", msg, 404)
        return None

    def _resolve_requirements_sha256(self, req_path: Path) -> str:
        if self._requirements_sha256 is not None:
            return self._requirements_sha256.strip().lower()
        sidecar = req_path.with_name(f"{req_path.name}.sha256")
        if sidecar.is_file():
            return _parse_sidecar_sha256(sidecar, req_path.name)
        for manifest in (
            self._sha256_path,
            req_path.parent / "SHA256SUMS.txt",
            self._wheel_path.parent / "SHA256SUMS.txt",
        ):
            if manifest is not None:
                found = _lookup_in_manifest(manifest, req_path.name)
                if found is not None:
                    return found
        updates_logger().error(
            "upgrade_local_requirements_checksum_missing",
            requirements_path=str(req_path),
            alert_category=AlertCategory.SECURITY_ALERT.value,
        )
        msg = (
            f"Refusing unverified requirements lockfile {req_path.name}: provide "
            f"{req_path.name}.sha256, SHA256SUMS.txt, or --requirements-sha256."
        )
        raise make_upgrade_error("upgrade_checksum_missing", msg)

    def _build_local_artifact(self) -> ReleaseArtifact:
        if not self._wheel_path.is_file():
            raise make_upgrade_error(
                "upgrade_invalid_wheel", f"Local wheel missing: {self._wheel_path}", 404
            )
        match = WHEEL_RE.fullmatch(self._wheel_path.name)
        if match is None:
            raise make_upgrade_error(
                "upgrade_invalid_wheel", f"Invalid wheel name: {self._wheel_path.name}", 400
            )
        ver = match.group(1)
        wheel_sha256 = self._resolve_wheel_sha256()
        req_path = self._resolve_requirements_path()
        req_sha256 = self._resolve_requirements_sha256(req_path) if req_path is not None else None
        return ReleaseArtifact(
            version=ver,
            tag=f"v{ver}",
            wheel_filename=self._wheel_path.name,
            sha256=wheel_sha256,
            local_path=self._wheel_path,
            requirements_filename=req_path.name if req_path is not None else None,
            requirements_sha256=req_sha256,
            requirements_local_path=req_path,
        )

    async def get_latest(self, *, allow_prerelease: bool = False) -> ReleaseArtifact:
        """Return artifact metadata for the configured local wheel."""
        del allow_prerelease
        return await asyncio.to_thread(self._build_local_artifact)

    async def get_version(self, version: str) -> ReleaseArtifact:
        """Return artifact metadata if the local wheel matches the requested version."""
        artifact = await asyncio.to_thread(self._build_local_artifact)
        if artifact.version != version.strip().lstrip("vV"):
            raise make_upgrade_error(
                "upgrade_release_not_found", f"Wheel {artifact.version} != {version}", 404
            )
        return artifact

    async def download_wheel(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path:
        """Verify local wheel SHA-256 strictly and stage a verified copy into dest_dir."""
        return await asyncio.to_thread(
            _stage_local_file,
            self._wheel_path,
            dest_dir,
            artifact.wheel_filename,
            artifact.sha256,
        )

    async def download_requirements(self, artifact: ReleaseArtifact, dest_dir: Path) -> Path | None:
        """Verify local requirements-lock.txt SHA-256 and stage a verified copy into dest_dir."""
        if artifact.requirements_local_path is None:
            if self._require_requirements_lock:
                msg = "Release artifact lacks a locked requirements file."
                raise make_upgrade_error("upgrade_requirements_missing", msg, 400)
            return None
        if artifact.requirements_sha256 is None:
            msg = f"Missing SHA-256 checksum for {artifact.requirements_local_path.name}."
            raise make_upgrade_error("upgrade_checksum_missing", msg)
        filename = artifact.requirements_filename or artifact.requirements_local_path.name
        return await asyncio.to_thread(
            _stage_local_file,
            artifact.requirements_local_path,
            dest_dir,
            filename,
            artifact.requirements_sha256,
        )
