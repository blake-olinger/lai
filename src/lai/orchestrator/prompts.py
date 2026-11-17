"""System prompts and structured context formatters for turn orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.chat import MessageRole
from lai.domain.inference import InferenceMessage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lai.domain.files import CodeFileSummary, FileRecord, LogFileSummary, ScoredChunk
    from lai.orchestrator.windower import TextWindow

BASE_SYSTEM_PROMPT = (
    "You are an accurate local engineering and general-purpose AI assistant.\n"
    "When analyzing source code, logs, or documents:\n"
    "- Base every factual claim strictly on the provided files, tool outputs, and conversation.\n"
    "- Cite exact evidence with bracketed line references formatted as [filename:Lstart-Lend] "
    "or [filename:Lline].\n"
    "- Identify root causes, failure chains, timestamps, and concrete fixes clearly.\n"
    "- Never fabricate file contents, stack frames, symbol names, or line numbers."
)

_MAP_SYSTEM_PROMPT = (
    "You are extracting structured findings from a single file window in a Map-Reduce pass.\n"
    "Rules:\n"
    "- Extract only facts, errors, stack traces, timelines, symbols, or counts visible in "
    "this window.\n"
    "- Preserve exact line citations formatted as [filename:Lstart-Lend].\n"
    "- Be concise and factual. If this window contains nothing relevant to the query, reply "
    "briefly with 'No relevant entries in this window.'"
)

_COMPRESSION_SYSTEM_PROMPT = (
    "You are condensing intermediate Map-Reduce findings into a tighter summary.\n"
    "Rules:\n"
    "- Merge duplicates and aggregate counts accurately.\n"
    "- Preserve all critical [filename:Lstart-Lend] line citations, timestamps, and error types.\n"
    "- Do not drop distinct failure modes or root-cause clues."
)

_REDUCE_GUIDANCE = (
    "Synthesize the deterministic file structural summaries and extracted window findings below "
    "into ONE unified, coherent final answer addressing the user's request.\n"
    "Cite supporting evidence using [filename:Lstart-Lend] references."
)


def build_system_prompt(
    custom_prompt: str | None = None,
    *,
    has_files: bool = False,
    has_tools: bool = False,
) -> str:
    """Compose the session system prompt with capability guidance."""
    sections = [BASE_SYSTEM_PROMPT]
    if has_files:
        sections.append(
            "Project files are attached below. Always include [filename:Lstart-Lend] citations "
            "for code or log evidence."
        )
    if has_tools:
        sections.append(
            "Read-only file inspection tools are available (search_files, read_file_lines, "
            "get_file_overview). Use file_id values from the file catalog when calling tools."
        )
    if custom_prompt and custom_prompt.strip():
        sections.append(f"Session instructions:\n{custom_prompt.strip()}")
    return "\n\n".join(sections)


def format_line_numbered_block(filename: str, text: str, *, start_line: int = 1) -> str:
    """Format text with a [filename:Lstart-Lend] header and explicit per-line numbers."""
    lines = text.splitlines()
    if not lines:
        return f"[{filename}:L{start_line}-L{start_line}]\n(empty file)"
    end_line = start_line + len(lines) - 1
    numbered = "\n".join(f"{start_line + offset}: {line}" for offset, line in enumerate(lines))
    return f"[{filename}:L{start_line}-L{end_line}]\n{numbered}"


def format_direct_files_context(file_entries: Sequence[tuple[FileRecord, str]]) -> str:
    """Render full line-numbered file payloads for Direct context injection."""
    blocks = [
        format_line_numbered_block(record.filename, content, start_line=1)
        for record, content in file_entries
    ]
    return "Attached project files:\n\n" + "\n\n".join(blocks)


def _format_log_summary(summary: LogFileSummary) -> list[str]:
    """Render deterministic LogFileSummary details into readable lines."""
    lines = [
        f"  log_lines={summary.total_lines} errors={summary.error_count} "
        f"warnings={summary.warning_count}",
    ]
    if summary.first_timestamp or summary.last_timestamp:
        lines.append(
            f"  time_range={summary.first_timestamp or '?'} .. {summary.last_timestamp or '?'}"
        )
    lines.extend(
        f"  exception {exc.exception_type} count={exc.count} "
        f"lines=L{exc.first_line}-L{exc.last_line}: {exc.message_sample}"
        for exc in summary.top_exceptions[:10]
    )
    return lines


def _format_code_summary(summary: CodeFileSummary) -> list[str]:
    """Render deterministic CodeFileSummary details into readable lines."""
    lines = [f"  language={summary.language} code_lines={summary.total_lines}"]
    if summary.symbols:
        lines.append(f"  symbols={', '.join(summary.symbols[:30])}")
    if summary.imports:
        lines.append(f"  imports={', '.join(summary.imports[:20])}")
    return lines


def format_file_summary(record: FileRecord) -> str:
    """Format a FileRecord and any extracted LogFileSummary/CodeFileSummary."""
    header = (
        f"- file_id={record.id} filename={record.filename} "
        f"kind={record.kind.value} tokens={record.token_count}"
    )
    parts = [header]
    if record.log_summary is not None:
        parts.extend(_format_log_summary(record.log_summary))
    if record.code_summary is not None:
        parts.extend(_format_code_summary(record.code_summary))
    return "\n".join(parts)


def format_retrieval_context(
    files: Sequence[FileRecord],
    hits: Sequence[ScoredChunk],
    filenames_by_id: Mapping[str, str],
) -> str:
    """Render file catalog summaries and ranked hybrid retrieval chunks."""
    sections: list[str] = []
    if files:
        catalog = "\n".join(format_file_summary(record) for record in files)
        sections.append(f"Project file summaries:\n{catalog}")

    if hits:
        chunk_blocks: list[str] = []
        for hit in hits:
            chunk = hit.chunk
            filename = filenames_by_id.get(chunk.file_id, chunk.file_id)
            header = (
                f"[{filename}:L{chunk.start_line}-L{chunk.end_line}] "
                f"(file_id={chunk.file_id}, source={hit.source}, score={hit.score:.4f})"
            )
            chunk_blocks.append(f"{header}\n{chunk.text}")
        sections.append("Retrieved relevant excerpts:\n\n" + "\n\n".join(chunk_blocks))
    else:
        sections.append(
            "No initial keyword/vector chunks matched directly; use available tools "
            "(get_file_overview, search_files, read_file_lines) if needed."
        )

    return "\n\n".join(sections)


def _strip_window_header(window: TextWindow) -> str:
    """Remove the leading [filename:Lstart-Lend] header line from a TextWindow."""
    expected_prefix = f"[{window.filename}:L{window.start_line}-L{window.end_line}]"
    first_newline = window.text.find("\n")
    if first_newline != -1 and window.text[:first_newline] == expected_prefix:
        return window.text[first_newline + 1 :]
    return window.text


def build_map_window_messages(
    user_message: str,
    window: TextWindow,
    *,
    file_summary_text: str | None = None,
) -> tuple[InferenceMessage, ...]:
    """Build system and user messages for a single Map-Reduce window extraction pass."""
    body = _strip_window_header(window)
    numbered_window = format_line_numbered_block(
        window.filename,
        body,
        start_line=window.start_line,
    )
    parts = [
        f"User query: {user_message}",
        f"Window {window.window_index + 1} of {window.total_windows} for {window.filename}.",
    ]
    if file_summary_text:
        parts.append(f"File overview:\n{file_summary_text}")
    parts.append(f"Window content:\n{numbered_window}")
    return (
        InferenceMessage(role=MessageRole.SYSTEM, content=_MAP_SYSTEM_PROMPT),
        InferenceMessage(role=MessageRole.USER, content="\n\n".join(parts)),
    )


def build_compression_messages(
    user_message: str,
    findings: Sequence[str],
    *,
    batch_index: int,
    total_batches: int,
) -> tuple[InferenceMessage, ...]:
    """Build prompt messages for hierarchical compression of oversized window findings."""
    joined = "\n\n---\n\n".join(findings)
    user_content = (
        f"User query: {user_message}\n\n"
        f"Findings batch {batch_index + 1} of {total_batches}:\n\n{joined}"
    )
    return (
        InferenceMessage(role=MessageRole.SYSTEM, content=_COMPRESSION_SYSTEM_PROMPT),
        InferenceMessage(role=MessageRole.USER, content=user_content),
    )


def build_reduce_messages(
    user_message: str,
    file_summaries: Sequence[str],
    findings: Sequence[str],
    *,
    session_system_prompt: str | None = None,
    history_messages: Sequence[InferenceMessage] = (),
) -> tuple[InferenceMessage, ...]:
    """Build the final Reduce/Synthesis InferenceMessage tuple."""
    base_sys = build_system_prompt(session_system_prompt, has_files=True, has_tools=False)
    system_content = f"{base_sys}\n\n{_REDUCE_GUIDANCE}"

    sections: list[str] = []
    if file_summaries:
        sections.append("Deterministic file summaries:\n" + "\n".join(file_summaries))
    if findings:
        sections.append("Extracted window findings:\n\n" + "\n\n".join(findings))
    sections.append(f"User request:\n{user_message}")

    return (
        InferenceMessage(role=MessageRole.SYSTEM, content=system_content),
        *history_messages,
        InferenceMessage(role=MessageRole.USER, content="\n\n".join(sections)),
    )
