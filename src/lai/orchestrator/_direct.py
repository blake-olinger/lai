"""Direct full-context execution path for small/non-file turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.chat import MessageRole, OrchestratorEvent
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
    file_entries: list[tuple[FileRecord, str]] = []
    for record in target_files:
        content = await read_file_text(deps.blob_store, record, file_repo=deps.file_repo)
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
    max_tokens = turn_input.max_tokens if turn_input.max_tokens is not None else profile.max_tokens
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
