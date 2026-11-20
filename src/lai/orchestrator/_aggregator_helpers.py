"""Pure budgeting, batching, and clamping helpers for MapReduceAggregator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.auth import ApiScope
from lai.domain.chat import Citation
from lai.domain.errors import ValidationFailedError
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.ingestion.detector import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.inference import InferenceMessage, ModelProfile
    from lai.orchestrator.windower import TextWindow

DEFAULT_MAX_WINDOWS = 32
MAX_COMPRESSION_PASSES = 4
_MIN_REDUCE_FINDINGS_BUDGET = 1024
_REDUCE_PROMPT_OVERHEAD_TOKENS = 2048
_CLAMP_SUFFIX = "\n...[truncated to fit context budget]"


@dataclass(slots=True)
class ReduceParams:
    """Bundled inputs for the final Map-Reduce synthesis turn."""

    turn_input: TurnInput
    principal: AuthPrincipal
    model_profile: ModelProfile
    files: Sequence[FileRecord]
    file_summaries: Sequence[str]
    findings: Sequence[str]
    fallback_citations: Sequence[Citation]
    session_system_prompt: str | None
    history_messages: Sequence[InferenceMessage]


def build_map_reduce_lease(turn_input: TurnInput, principal: AuthPrincipal) -> LeaseRequest:
    """Build a MAP_REDUCE LeaseRequest with admin quota bypass when authorized."""
    return LeaseRequest(
        session_id=turn_input.session_id,
        project_id=turn_input.project_id,
        principal_id=principal.principal_id,
        priority=JobPriority.MAP_REDUCE,
        bypass_principal_quota=principal.has_scope(ApiScope.ADMIN),
    )


def pack_finding_batches(findings: Sequence[str], *, batch_limit_tokens: int) -> list[list[str]]:
    """Group findings into token-bounded batches for hierarchical compression."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for item in findings:
        item_tokens = max(1, estimate_tokens(item) + 4)
        if current and current_tokens + item_tokens > batch_limit_tokens:
            batches.append(current)
            current = [item]
            current_tokens = item_tokens
        else:
            current.append(item)
            current_tokens += item_tokens
    if current:
        batches.append(current)
    return batches


def compute_reduce_findings_budget(
    profile: ModelProfile,
    *,
    max_output_tokens: int,
    compression_threshold_tokens: int,
) -> int:
    """Return the maximum findings token budget that fits safely in the Reduce prompt."""
    ctx_budget = max(
        _MIN_REDUCE_FINDINGS_BUDGET,
        profile.ctx_size - max_output_tokens - _REDUCE_PROMPT_OVERHEAD_TOKENS,
    )
    return min(compression_threshold_tokens, ctx_budget)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Deterministically clip a single string to fit within max_tokens."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    suffix_tokens = estimate_tokens(_CLAMP_SUFFIX)
    target = max(1, max_tokens - suffix_tokens)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= target:
            lo = mid
        else:
            hi = mid - 1
    return f"{text[:lo]}{_CLAMP_SUFFIX}" if lo > 0 else text[: max(1, target)]


def clamp_findings_to_budget(findings: Sequence[str], *, max_tokens: int) -> list[str]:
    """Guarantee joined findings stay within max_tokens after bounded compression passes."""
    if not findings or estimate_tokens("\n\n".join(findings)) <= max_tokens:
        return list(findings)

    clamped: list[str] = []
    used = 0
    for item in findings:
        item_tokens = estimate_tokens(item)
        sep_tokens = 2 if clamped else 0
        remaining = max_tokens - used - sep_tokens
        if remaining <= 0:
            break
        if item_tokens <= remaining:
            clamped.append(item)
            used += item_tokens + sep_tokens
        else:
            clipped = _truncate_text_to_tokens(item, remaining)
            if clipped:
                clamped.append(clipped)
            break
    return clamped


def raise_max_windows_exceeded(*, estimated_count: int, max_windows: int) -> None:
    """Raise ValidationFailedError when a Map-Reduce request exceeds max_windows."""
    msg = (
        f"Map-Reduce request requires at least {estimated_count} windows, exceeding "
        f"configured limit of {max_windows}; narrow attached_file_ids or use "
        "context_mode='retrieval'."
    )
    raise ValidationFailedError(
        msg,
        details={
            "context_mode": "map_reduce",
            "max_windows": str(max_windows),
            "window_count": str(estimated_count),
        },
    )


def build_window_fallback_citations(
    window_items: Sequence[tuple[TextWindow, str]],
    files: Sequence[FileRecord],
) -> list[Citation]:
    """Build fallback Citation spans from mapped windows or attached file records."""
    if window_items:
        return [
            Citation(
                file_id=w.file_id,
                filename=w.filename,
                start_line=w.start_line,
                end_line=w.end_line,
            )
            for w, _ in window_items
        ]
    return [Citation(file_id=f.id, filename=f.filename) for f in files]
