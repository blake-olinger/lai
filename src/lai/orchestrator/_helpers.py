"""Shared orchestrator helpers for files, history budgeting, tools, and citations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lai.config.profiles import get_model_profile
from lai.domain.chat import Message, MessageRole, TokenUsage
from lai.domain.errors import AuthorizationError, NotFoundError
from lai.domain.files import FileStatus
from lai.domain.inference import InferenceMessage, InferenceToolDef, ModelProfile
from lai.domain.tools import ToolExecutionTarget
from lai.ingestion.detector import estimate_tokens
from lai.orchestrator._citations import extract_citations

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lai.config.settings import LaiSettings
    from lai.domain.chat import Session, TurnInput
    from lai.domain.files import FileRecord
    from lai.domain.tools import ToolSpec
    from lai.ports.repositories import FileRepositoryPort, SessionRepositoryPort
    from lai.ports.storage import BlobStorePort
    from lai.ports.tools import ToolRegistryPort

_DEFAULT_HISTORY_SHARE_NUMERATOR = 1
_DEFAULT_HISTORY_SHARE_DENOMINATOR = 2

__all__ = [
    "add_token_usage",
    "build_inference_tool_defs",
    "build_trimmed_history",
    "compute_history_budget",
    "extract_citations",
    "parse_tool_arguments",
    "persist_turn_if_session",
    "read_file_text",
    "resolve_model_profile",
    "resolve_target_files",
    "trim_history_to_budget",
]


def resolve_model_profile(
    settings: LaiSettings,
    session: Session | None,
    explicit_profile: ModelProfile | None,
    requested_model: str | None = None,
) -> ModelProfile:
    """Select the effective ModelProfile from turn override, session, explicit, or settings."""
    if requested_model is not None:
        return get_model_profile(requested_model, settings.custom_profiles)
    if session is None:
        return explicit_profile or get_model_profile(
            settings.default_model_profile,
            settings.custom_profiles,
        )

    name = session.model_profile
    if explicit_profile is not None and name == explicit_profile.name:
        return explicit_profile
    try:
        return get_model_profile(name, settings.custom_profiles)
    except NotFoundError:
        return explicit_profile or ModelProfile(
            name=name,
            hf_repo="local/custom",
            hf_filename=f"{name}.gguf",
        )


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
    """Retain recent USER/ASSISTANT/TOOL messages within budget, stripping orphaned TOOLs."""
    if max_history_tokens <= 0 or not messages:
        return ()

    eligible_roles = {MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL}
    eligible = [msg for msg in messages if msg.role in eligible_roles]
    selected: list[InferenceMessage] = []
    used_tokens = 0

    for msg in reversed(eligible):
        tool_text = "".join(f"{tc.name}{tc.arguments_json}" for tc in msg.tool_calls)
        msg_tokens = max(1, estimate_tokens(msg.content) + estimate_tokens(tool_text) + 4)
        if used_tokens + msg_tokens > max_history_tokens:
            break
        used_tokens += msg_tokens
        selected.append(
            InferenceMessage(
                role=msg.role,
                content=msg.content,
                name=msg.tool_name,
                tool_call_id=msg.tool_call_id,
                tool_calls=msg.tool_calls,
            )
        )

    selected.reverse()
    cleaned: list[InferenceMessage] = []
    seen_call_ids: set[str] = set()
    for item in selected:
        if item.role is MessageRole.TOOL:
            if not item.tool_call_id or item.tool_call_id not in seen_call_ids:
                continue
        elif item.role is MessageRole.ASSISTANT and item.tool_calls:
            seen_call_ids.update(tc.id for tc in item.tool_calls)
        cleaned.append(item)
    return tuple(cleaned)


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


async def persist_turn_if_session(  # noqa: PLR0913
    session_repo: SessionRepositoryPort,
    session: Session | None,
    turn_input: TurnInput,
    user_content: str,
    assistant_msg: Message,
    *,
    intermediate_messages: Sequence[InferenceMessage] = (),
) -> None:
    """Persist user, any intermediate tool-loop messages, and assistant Message chronologically."""
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
    for item in intermediate_messages:
        await session_repo.add_message(
            Message(
                session_id=turn_input.session_id,
                project_id=turn_input.project_id,
                role=item.role,
                content=item.content,
                tool_calls=item.tool_calls,
                tool_call_id=item.tool_call_id,
                tool_name=item.name,
                created_at=assistant_msg.created_at,
            )
        )
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
    client_tools: Sequence[ToolSpec] = (),
    client_app_id: str | None = None,
) -> tuple[InferenceToolDef, ...]:
    """Convert SERVER_ATTACHMENT specs and authenticated client_tools to InferenceToolDefs."""
    if not enable_tools:
        return ()
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    if registry is not None:
        for spec in registry.list_specs():
            if (
                spec.execution_target is ToolExecutionTarget.SERVER_ATTACHMENT
                and spec.name not in seen
            ):
                specs.append(spec)
                seen.add(spec.name)
    if client_app_id and client_app_id.strip():
        for spec in client_tools:
            if spec.name not in seen:
                specs.append(spec)
                seen.add(spec.name)
    return tuple(
        InferenceToolDef(
            name=spec.name,
            description=spec.description,
            parameters_schema=spec.parameters_schema,
        )
        for spec in specs
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
