"""Mutation target definitions and tier resolution for scripts/check_mutation.py."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

MutationTier = Literal["critical", "full"]

CRITICAL_MUTATION_TARGETS: tuple[str, ...] = (
    # 12 core algorithmic, security, scheduling, and guardrail modules
    "src/lai/observability/alerts.py",
    "src/lai/observability/scrubber.py",
    "src/lai/security/keys.py",
    "src/lai/security/scopes.py",
    "src/lai/security/ratelimit.py",
    "src/lai/guardrails/prompts.py",
    "src/lai/tools/policy.py",
    "src/lai/scheduler/fair_queue.py",
    "src/lai/scheduler/session_gate.py",
    "src/lai/domain/rrf.py",
    "src/lai/orchestrator/classifier.py",
    "src/lai/orchestrator/windower.py",
    # High-risk domain, security, guardrail, tool dispatch, and API modules
    "src/lai/domain/auth.py",
    "src/lai/security/cloudflare.py",
    "src/lai/security/oidc_provider.py",
    "src/lai/security/api_key_provider.py",
    "src/lai/security/client_apps.py",
    "src/lai/guardrails/pipeline.py",
    "src/lai/guardrails/shieldgemma.py",
    "src/lai/tools/executor.py",
    "src/lai/tools/_client_dispatch.py",
    "src/lai/tools/plugins/_scoping.py",
    "src/lai/api/errors.py",
    "src/lai/api/middleware.py",
    "src/lai/api/deps.py",
    "src/lai/api/_tenancy.py",
    "src/lai/api/_client_auth.py",
    "src/lai/config/_validators.py",
)

FULL_MUTATION_TARGETS: tuple[str, ...] = (
    *CRITICAL_MUTATION_TARGETS,
    "src/lai/api/routes/projects.py",
    "src/lai/api/routes/sessions.py",
    "src/lai/api/routes/files.py",
    "src/lai/api/routes/openai.py",
    "src/lai/api/routes/tools.py",
    "src/lai/api/routes/client_apps.py",
    "src/lai/tools/plugins/search_files.py",
    "src/lai/tools/plugins/read_lines.py",
    "src/lai/tools/plugins/file_overview.py",
    "src/lai/scheduler/spooler.py",
    "src/lai/scheduler/keepalive.py",
    "src/lai/orchestrator/service.py",
    "src/lai/orchestrator/_helpers.py",
    "src/lai/orchestrator/_resume.py",
    "src/lai/orchestrator/aggregator.py",
    "src/lai/db/project_repo.py",
    "src/lai/db/tool_repo.py",
    "src/lai/db/client_app_repo.py",
    "src/lai/db/session_repo.py",
    "src/lai/db/file_repo.py",
)

MUTATION_TIERS: dict[MutationTier, tuple[str, ...]] = {
    "critical": CRITICAL_MUTATION_TARGETS,
    "full": FULL_MUTATION_TARGETS,
}


def resolve_mutation_targets(tier: MutationTier, targets: str | None = None) -> str:
    """Resolve the comma-separated gremlins target string for a tier or explicit override."""
    if targets is not None:
        cleaned = targets.strip()
        if not cleaned:
            msg = "Mutation targets override must not be empty."
            raise ValueError(msg)
        return cleaned
    if tier not in MUTATION_TIERS:
        msg = f"Unknown mutation tier: {tier!r}"
        raise ValueError(msg)
    return ",".join(MUTATION_TIERS[tier])


_TEMP_ENV_KEYS: tuple[str, ...] = ("TEMP", "TMP", "TMPDIR")


@contextmanager
def isolated_temp_env() -> Generator[Path]:
    """Run a block with TEMP/TMP/TMPDIR pointed at an isolated temporary directory."""
    previous = {key: os.environ.get(key) for key in _TEMP_ENV_KEYS}
    with tempfile.TemporaryDirectory(prefix="lai-mutate-", ignore_cleanup_errors=True) as raw_dir:
        tmp_path = Path(raw_dir)
        tmp_str = str(tmp_path)
        for key in _TEMP_ENV_KEYS:
            os.environ[key] = tmp_str
        try:
            yield tmp_path
        finally:
            for key, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
