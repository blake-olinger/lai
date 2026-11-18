"""Mutation target definitions and tier resolution for scripts/verify.py."""

from __future__ import annotations

from typing import Literal

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
    "src/lai/ingestion/rrf.py",
    "src/lai/orchestrator/classifier.py",
    "src/lai/orchestrator/windower.py",
    # High-risk security, guardrail, tool dispatch, and API modules
    "src/lai/security/cloudflare.py",
    "src/lai/security/oidc_provider.py",
    "src/lai/security/api_key_provider.py",
    "src/lai/security/client_apps.py",
    "src/lai/guardrails/pipeline.py",
    "src/lai/guardrails/shieldgemma.py",
    "src/lai/guardrails/ollama_provider.py",
    "src/lai/tools/executor.py",
    "src/lai/tools/_client_dispatch.py",
    "src/lai/tools/plugins/_scoping.py",
    "src/lai/api/errors.py",
    "src/lai/api/middleware.py",
    "src/lai/api/deps.py",
)

FULL_MUTATION_TARGETS: tuple[str, ...] = (
    *CRITICAL_MUTATION_TARGETS,
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
