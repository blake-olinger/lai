"""Direct full-context execution path for small/non-file turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.auth import ApiScope
from lai.domain.chat import MessageRole, OrchestratorEvent
from lai.domain.errors import ValidationFailedError
from lai.domain.inference import InferenceMessage, InferenceRequest
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.ingestion.detector import estimate_tokens
from lai.orchestrator._helpers import (
    build_trimmed_history,
    extract_citations,
    read_file_text,
)
from lai.orchestrator._stream import (
    StreamedInferenceAccumulator,
    TurnOutcome,
    stream_leased_inference,
)
from lai.orchestrator.prompts import build_system_prompt, format_direct_files_context

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import Message, TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.inference import ModelProfile
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.repositories import FileRepositoryPort
    from lai.ports.scheduler import InferenceSpoolerPort
    from lai.ports.storage import BlobStorePort


@dataclass(slots=True)
class DirectDeps:
    """Injected ports needed by Direct context execution."""

    file_repo: FileRepositoryPort
    blob_store: BlobStorePort
    inference_provider: InferenceProviderPort
    spooler: InferenceSpoolerPort


_MIN_DIRECT_TOKENS = 256
_MIN_DIRECT_BYTES = 262_144
_DIRECT_PROMPT_RESERVE_TOKENS = 1024
_DIRECT_BYTES_PER_TOKEN_CEILING = 8


def _enforce_direct_file_limits(
    target_files: Sequence[FileRecord],
    *,
    profile: ModelProfile,
    max_tokens: int,
) -> tuple[int, int]:
    """Verify attached files fit within Direct mode token and byte ceilings before reading."""
    max_direct_tokens = max(
        _MIN_DIRECT_TOKENS,
        profile.ctx_size - max_tokens - _DIRECT_PROMPT_RESERVE_TOKENS,
    )
    max_direct_bytes = max(
        max_direct_tokens * _DIRECT_BYTES_PER_TOKEN_CEILING,
        _MIN_DIRECT_BYTES,
    )
    total_tokens = sum(max(0, record.token_count) for record in target_files)
    total_bytes = sum(max(0, record.size_bytes) for record in target_files)
    if total_tokens > max_direct_tokens or total_bytes > max_direct_bytes:
        msg = (
            "Attached files exceed the Direct context ceiling "
            f"(tokens={total_tokens}/{max_direct_tokens}, bytes={total_bytes}/{max_direct_bytes}); "
            "use context_mode='retrieval' or 'map_reduce'."
        )
        raise ValidationFailedError(
            msg,
            details={
                "context_mode": "direct",
                "max_direct_tokens": str(max_direct_tokens),
                "max_direct_bytes": str(max_direct_bytes),
            },
        )
    return max_direct_tokens, max_direct_bytes


async def run_direct_stream(  # noqa: PLR0913
    deps: DirectDeps,
    *,
    turn_input: TurnInput,
    principal: AuthPrincipal,
    profile: ModelProfile,
    target_files: Sequence[FileRecord],
    session_system_prompt: str | None,
    existing_messages: Sequence[Message],
) -> AsyncGenerator[OrchestratorEvent | TurnOutcome]:
    """Inject full line-numbered file text, acquire INTERACTIVE lease, and stream."""
    max_tokens = turn_input.max_tokens if turn_input.max_tokens is not None else profile.max_tokens
    max_direct_tokens = _MIN_DIRECT_TOKENS
    if target_files:
        max_direct_tokens, _ = _enforce_direct_file_limits(
            target_files,
            profile=profile,
            max_tokens=max_tokens,
        )

    file_entries: list[tuple[FileRecord, str]] = []
    loaded_tokens = 0
    for record in target_files:
        content = await read_file_text(deps.blob_store, record, file_repo=deps.file_repo)
        loaded_tokens += estimate_tokens(content)
        if loaded_tokens > max_direct_tokens:
            msg = (
                f"Loaded file content ({loaded_tokens} tokens) exceeds the Direct context "
                f"ceiling ({max_direct_tokens} tokens); use 'retrieval' or 'map_reduce'."
            )
            raise ValidationFailedError(
                msg,
                details={
                    "context_mode": "direct",
                    "max_direct_tokens": str(max_direct_tokens),
                },
            )
        file_entries.append((record, content))

    system_prompt = build_system_prompt(
        session_system_prompt,
        has_files=bool(file_entries),
        has_tools=False,
    )
    user_content = (
        f"{format_direct_files_context(file_entries)}\n\nUser request:\n{turn_input.user_message}"
        if file_entries
        else turn_input.user_message
    )
    temperature = (
        turn_input.temperature if turn_input.temperature is not None else profile.temperature
    )
    reserved = estimate_tokens(system_prompt) + estimate_tokens(user_content) + 32
    history = build_trimmed_history(
        existing_messages,
        profile=profile,
        reserved_prompt_tokens=reserved,
        max_output_tokens=max_tokens,
    )
    request = InferenceRequest(
        model=profile.name,
        messages=(
            InferenceMessage(role=MessageRole.SYSTEM, content=system_prompt),
            *history,
            InferenceMessage(role=MessageRole.USER, content=user_content),
        ),
        tools=(),
        temperature=temperature,
        top_p=profile.top_p,
        max_tokens=max_tokens,
        stream=True,
        enable_thinking=turn_input.enable_thinking,
    )
    lease_req = LeaseRequest(
        session_id=turn_input.session_id,
        project_id=turn_input.project_id,
        principal_id=principal.principal_id,
        priority=JobPriority.INTERACTIVE,
        bypass_principal_quota=principal.has_scope(ApiScope.ADMIN),
    )
    acc = StreamedInferenceAccumulator()
    async for event in stream_leased_inference(
        spooler=deps.spooler,
        inference_provider=deps.inference_provider,
        lease_request=lease_req,
        inference_request=request,
        acc=acc,
    ):
        yield event

    final_content = acc.content
    yield TurnOutcome(
        content=final_content,
        thinking=acc.thinking,
        citations=extract_citations(final_content, target_files),
        usage=acc.usage,
    )
