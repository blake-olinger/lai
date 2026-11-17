"""Shared orchestrator helpers for files, history budgeting, tools, and citations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lai.config.profiles import DEFAULT_MODEL_PROFILE_NAME, get_model_profile
from lai.domain.chat import Citation, Message, MessageRole, TokenUsage
from lai.domain.errors import AuthorizationError, NotFoundError
from lai.domain.files import FileStatus
from lai.domain.inference import InferenceMessage, InferenceToolDef, ModelProfile
from lai.ingestion.detector import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.config.settings import LaiSettings
    from lai.domain.chat import Session, TurnInput
    from lai.domain.files import FileRecord, ScoredChunk
    from lai.ports.repositories import FileRepositoryPort, SessionRepositoryPort
    from lai.ports.storage import BlobStorePort
    from lai.ports.tools import ToolRegistryPort

_CITATION_RE = re.compile(r"\[([^\]\r\n:]+):L(\d+)(?:-L?(\d+))?\]")
_DEFAULT_HISTORY_SHARE_NUMERATOR = 1
_DEFAULT_HISTORY_SHARE_DENOMINATOR = 2
_MAX_FALLBACK_CITATIONS = 8


def resolve_model_profile(
    settings: LaiSettings,
    session: Session | None,
    explicit_profile: ModelProfile | None,
) -> ModelProfile:
    """Select the effective ModelProfile from session override, explicit default, or settings."""
    if session is not None:
        name = session.model_profile
        if explicit_profile is not None and name in {
            explicit_profile.name,
            DEFAULT_MODEL_PROFILE_NAME,
        }:
            return explicit_profile
        try:
            return get_model_profile(name, settings.custom_profiles)
        except NotFoundError:
            if explicit_profile is not None:
                return explicit_profile
            return ModelProfile(
                name=name,
                hf_repo="local/custom",
                hf_filename=f"{name}.gguf",
            )

    if explicit_profile is not None:
        return explicit_profile
    return get_model_profile(settings.default_model_profile, settings.custom_profiles)


async def resolve_target_files(
    file_repo: FileRepositoryPort,
    project_id: str,
    attached_file_ids: Sequence[str],
) -> list[FileRecord]:
    """Load attached files (strictly project-scoped) or all ready project files when empty."""
    if attached_file_ids:
        resolved: list[FileRecord] = []
        for file_id in dict.fromkeys(attached_file_ids):
            record = await file_repo.get_file(file_id)
            if record is None:
                msg = f"Attached file {file_id!r} was not found."
                raise NotFoundError(msg, details={"file_id": file_id})
            if record.project_id != project_id:
                msg = f"File {file_id!r} does not belong to project {project_id!r}."
                raise AuthorizationError(
                    msg,
                    details={"file_id": file_id, "project_id": project_id},
                )
            resolved.append(record)
        return resolved

    project_files = await file_repo.list_files(project_id, limit=200)
    return [
        record
        for record in project_files
        if record.project_id == project_id and record.status is not FileStatus.FAILED
    ]


async def read_file_text(
    blob_store: BlobStorePort,
    record: FileRecord,
    *,
    file_repo: FileRepositoryPort | None = None,
) -> str:
    """Read full UTF-8 file text from blob storage, falling back to indexed chunks if needed."""
    try:
        return await blob_store.read_text_range(Path(record.blob_path), start_line=1, end_line=None)
    except (NotFoundError, FileNotFoundError):
        if file_repo is None:
            return ""
        chunks = await file_repo.list_chunks(record.id)
        return "\n".join(chunk.text for chunk in chunks)


def trim_history_to_budget(
    messages: Sequence[Message],
    *,
    max_history_tokens: int,
) -> tuple[InferenceMessage, ...]:
    """Retain the most recent chronological USER/ASSISTANT messages within token budget."""
    if max_history_tokens <= 0 or not messages:
        return ()

    eligible = [msg for msg in messages if msg.role in {MessageRole.USER, MessageRole.ASSISTANT}]
    selected: list[InferenceMessage] = []
    used_tokens = 0

    for msg in reversed(eligible):
        msg_tokens = max(1, estimate_tokens(msg.content) + 4)
        if used_tokens + msg_tokens > max_history_tokens:
            break
        used_tokens += msg_tokens
        selected.append(InferenceMessage(role=msg.role, content=msg.content))

    selected.reverse()
    return tuple(selected)


def compute_history_budget(
    profile: ModelProfile,
    *,
    reserved_prompt_tokens: int,
    max_output_tokens: int,
) -> int:
    """Compute token budget available for prior conversation turns."""
    remaining = profile.ctx_size - reserved_prompt_tokens - max_output_tokens
    if remaining <= 0:
        return 0
    capped = (
        profile.ctx_size * _DEFAULT_HISTORY_SHARE_NUMERATOR
    ) // _DEFAULT_HISTORY_SHARE_DENOMINATOR
    return min(remaining, capped)


def build_trimmed_history(
    messages: Sequence[Message],
    *,
    profile: ModelProfile,
    reserved_prompt_tokens: int,
    max_output_tokens: int,
) -> tuple[InferenceMessage, ...]:
    """Compute history budget from profile and trim prior turns to fit."""
    budget = compute_history_budget(
        profile,
        reserved_prompt_tokens=reserved_prompt_tokens,
        max_output_tokens=max_output_tokens,
    )
    return trim_history_to_budget(messages, max_history_tokens=budget)


async def persist_turn_if_session(
    session_repo: SessionRepositoryPort,
    session: Session | None,
    turn_input: TurnInput,
    user_content: str,
    assistant_msg: Message,
) -> None:
    """Persist user and assistant Messages chronologically when a session exists."""
    if session is None:
        return
    user_msg = Message(
        session_id=turn_input.session_id,
        project_id=turn_input.project_id,
        role=MessageRole.USER,
        content=user_content,
        created_at=assistant_msg.created_at,
    )
    await session_repo.add_message(user_msg)
    await session_repo.add_message(assistant_msg)


def add_token_usage(base: TokenUsage, delta: TokenUsage) -> TokenUsage:
    """Sum two TokenUsage counters into a new TokenUsage instance."""
    prompt = base.prompt_tokens + delta.prompt_tokens
    completion = base.completion_tokens + delta.completion_tokens
    total = base.total_tokens + delta.total_tokens
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=max(total, prompt + completion),
    )


def build_inference_tool_defs(
    registry: ToolRegistryPort | None,
    *,
    enable_tools: bool,
) -> tuple[InferenceToolDef, ...]:
    """Convert registered ToolSpecs to InferenceToolDefs when tools are enabled."""
    if not enable_tools or registry is None:
        return ()
    return tuple(
        InferenceToolDef(
            name=spec.name,
            description=spec.description,
            parameters_schema=spec.parameters_schema,
        )
        for spec in registry.list_specs()
    )


def parse_tool_arguments(arguments_json: str) -> dict[str, Any]:
    """Parse tool-call JSON arguments safely into a string-keyed dictionary."""
    stripped = arguments_json.strip()
    if not stripped:
        return {}
    parsed: object = json.loads(stripped)
    if not isinstance(parsed, dict):
        msg = "Tool arguments JSON must decode to an object."
        raise TypeError(msg)
    return cast("dict[str, Any]", parsed)


def _match_file_by_label(label: str, files: Sequence[FileRecord]) -> FileRecord | None:
    """Match a citation filename label against target project FileRecords."""
    clean = label.strip()
    for record in files:
        if record.filename == clean:
            return record
    basename_matches = [record for record in files if Path(record.filename).name == clean]
    if len(basename_matches) == 1:
        return basename_matches[0]
    lowered = clean.lower()
    ci_matches = [
        record
        for record in files
        if record.filename.lower() == lowered or Path(record.filename).name.lower() == lowered
    ]
    return ci_matches[0] if len(ci_matches) == 1 else None


def _find_matching_chunk_id(
    file_id: str,
    start_line: int,
    end_line: int,
    chunks: Sequence[ScoredChunk],
) -> str | None:
    """Return the first retrieved chunk_id overlapping [start_line, end_line] for file_id."""
    for hit in chunks:
        chunk = hit.chunk
        if (
            chunk.file_id == file_id
            and chunk.start_line <= end_line
            and chunk.end_line >= start_line
        ):
            return chunk.id
    return None


def extract_citations(
    text: str,
    files: Sequence[FileRecord],
    *,
    retrieved_chunks: Sequence[ScoredChunk] = (),
    fallback_citations: Sequence[Citation] = (),
) -> tuple[Citation, ...]:
    """Extract deduplicated line-accurate Citations from model output or provenance fallbacks."""
    if not files:
        return ()

    citations: list[Citation] = []
    seen: set[tuple[str, int | None, int | None]] = set()

    for match in _CITATION_RE.finditer(text):
        label, raw_start, raw_end = match.group(1), match.group(2), match.group(3)
        record = _match_file_by_label(label, files)
        if record is None:
            continue
        start_line = max(1, int(raw_start))
        end_line = max(start_line, int(raw_end)) if raw_end is not None else start_line
        key = (record.id, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        chunk_id = _find_matching_chunk_id(record.id, start_line, end_line, retrieved_chunks)
        citations.append(
            Citation(
                file_id=record.id,
                filename=record.filename,
                chunk_id=chunk_id,
                start_line=start_line,
                end_line=end_line,
            )
        )

    if citations:
        return tuple(citations)

    deduped_fallback: list[Citation] = []
    for item in fallback_citations:
        fb_key = (item.file_id, item.start_line, item.end_line)
        if fb_key in seen:
            continue
        seen.add(fb_key)
        deduped_fallback.append(item)
        if len(deduped_fallback) >= _MAX_FALLBACK_CITATIONS:
            break
    return tuple(deduped_fallback)
