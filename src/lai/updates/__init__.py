"""Layer-3 self-update subsystem: release sources, slot management, and rollback coordination."""

from __future__ import annotations

from lai.updates.coordinator import UpgradeCoordinator
from lai.updates.slot_manager import (
    CommandRunner,
    HttpHealthProbe,
    ReleaseSlotManager,
    SystemdServiceController,
)
from lai.updates.sources import (
    GitHubReleaseSource,
    LocalWheelSource,
    compute_sha256,
    parse_semver_tuple,
    parse_sha256sums,
    verify_wheel_sha256,
)

__all__ = [
    "CommandRunner",
    "GitHubReleaseSource",
    "HttpHealthProbe",
    "LocalWheelSource",
    "ReleaseSlotManager",
    "SystemdServiceController",
    "UpgradeCoordinator",
    "compute_sha256",
    "parse_semver_tuple",
    "parse_sha256sums",
    "verify_wheel_sha256",
]
