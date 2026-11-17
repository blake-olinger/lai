"""Unit tests for context-mode classifier, line-preserving windower, and helpers."""

from __future__ import annotations

import pytest

from lai.domain.chat import ContextMode, Message, MessageRole
from lai.domain.errors import ValidationFailedError
from lai.domain.files import FileKind, FileRecord, FileStatus
from lai.orchestrator._helpers import extract_citations, trim_history_to_budget
from lai.orchestrator.classifier import classify_context_mode
from lai.orchestrator.prompts import format_line_numbered_block
from lai.orchestrator.windower import TextWindow, slice_text_into_windows

BUDGET_TOKENS = 12000
SMALL_WINDOW_TOKENS = 90
OVERLAP_TOKENS = 24
MIN_MULTI_WINDOWS = 3
SMALL_HISTORY_BUDGET = 40
LONG_LINE_REPEATS = 80
EXPECTED_SINGLE_LINE = 1


def _make_file(
    file_id: str,
    filename: str,
    token_count: int,
    *,
    kind: FileKind = FileKind.LOG,
) -> FileRecord:
    return FileRecord(
        id=file_id,
        project_id="prj_1",
        filename=filename,
        size_bytes=max(1, token_count * 4),
        sha256="a" * 64,
        blob_path=f"prj_1/aa/{file_id}",
        kind=kind,
        status=FileStatus.READY,
        token_count=token_count,
    )


def test_classifier_direct_when_empty_or_within_budget() -> None:
    """AUTO chooses DIRECT with zero files or total token_count <= direct_budget_tokens."""
    assert (
        classify_context_mode(
            ContextMode.AUTO,
            "Summarize all errors",
            [],
            direct_budget_tokens=BUDGET_TOKENS,
        )
        is ContextMode.DIRECT
    )

    within = [
        _make_file("fil_1", "a.log", 7000),
        _make_file("fil_2", "b.log", 5000),
    ]
    assert (
        classify_context_mode(
            ContextMode.AUTO,
            "Summarize all errors across the log",
            within,
            direct_budget_tokens=BUDGET_TOKENS,
        )
        is ContextMode.DIRECT
    )


@pytest.mark.parametrize(
    "query",
    [
        "Please summarize what happened",
        "Check the ENTIRE log for crashes",
        "Inspect the whole file carefully",
        "List all   errors in the output",
        "Examine every file in this batch",
        "Build a failure timeline",
        "Give me an overview of the service",
        "Audit all authentication warnings",
        "Trace patterns across the log",
        "Count all HTTP 500 responses",
        "Produce a full summary of this incident",
    ],
)
def test_classifier_map_reduce_for_oversized_holistic_cues(query: str) -> None:
    """Oversized files with any holistic cue route to MAP_REDUCE."""
    oversized = [_make_file("fil_1", "big.log", BUDGET_TOKENS + 1)]
    assert (
        classify_context_mode(
            ContextMode.AUTO,
            query,
            oversized,
            direct_budget_tokens=BUDGET_TOKENS,
        )
        is ContextMode.MAP_REDUCE
    )


def test_classifier_retrieval_and_explicit_overrides() -> None:
    """Oversized targeted queries route to RETRIEVAL, and explicit modes always win."""
    oversized = [_make_file("fil_1", "app.py", BUDGET_TOKENS + 500, kind=FileKind.CODE)]
    assert (
        classify_context_mode(
            ContextMode.AUTO,
            "Why did verify_token fail at line 42 with ValueError?",
            oversized,
            direct_budget_tokens=BUDGET_TOKENS,
        )
        is ContextMode.RETRIEVAL
    )

    for explicit in (ContextMode.DIRECT, ContextMode.RETRIEVAL, ContextMode.MAP_REDUCE):
        assert (
            classify_context_mode(
                explicit,
                "Summarize the whole log",
                oversized,
                direct_budget_tokens=BUDGET_TOKENS,
            )
            is explicit
        )
        assert (
            classify_context_mode(
                explicit,
                "Targeted lookup",
                [],
                direct_budget_tokens=BUDGET_TOKENS,
            )
            is explicit
        )


def test_windower_preserves_lines_with_overlap_and_headers() -> None:
    """slice_text_into_windows emits line-bounded overlapping windows with headers."""
    lines = [
        f"2026-11-17T10:00:{idx:02d}Z ERROR worker_{idx} failed connecting to upstream"
        for idx in range(1, 35)
    ]
    text = "\n".join(lines)

    windows = slice_text_into_windows(
        "fil_log",
        "service.log",
        text,
        window_tokens=SMALL_WINDOW_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
    )
    assert len(windows) >= MIN_MULTI_WINDOWS
    assert all(isinstance(win, TextWindow) for win in windows)

    covered_lines: set[int] = set()
    for idx, win in enumerate(windows):
        assert win.file_id == "fil_log"
        assert win.filename == "service.log"
        assert win.window_index == idx
        assert win.total_windows == len(windows)
        assert win.text.startswith(f"[service.log:L{win.start_line}-L{win.end_line}]\n")
        body_lines = win.text.split("\n", 1)[1].splitlines()
        assert body_lines == lines[win.start_line - 1 : win.end_line]
        covered_lines.update(range(win.start_line, win.end_line + 1))
        if idx > 0:
            prev = windows[idx - 1]
            assert win.start_line > prev.start_line
            assert win.start_line <= prev.end_line

    assert covered_lines == set(range(1, len(lines) + 1))

    no_overlap = slice_text_into_windows(
        "fil_log",
        "service.log",
        text,
        window_tokens=SMALL_WINDOW_TOKENS,
        overlap_tokens=0,
    )
    for idx in range(1, len(no_overlap)):
        assert no_overlap[idx].start_line == no_overlap[idx - 1].end_line + 1


def test_windower_edge_cases_and_validation() -> None:
    """Windower never splits an oversized single line and validates bad arguments."""
    assert slice_text_into_windows("fil_1", "empty.log", "") == []

    long_line = " ".join(f"token_{i}" for i in range(LONG_LINE_REPEATS))
    single = slice_text_into_windows(
        "fil_1",
        "single.log",
        long_line,
        window_tokens=SMALL_WINDOW_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
    )
    assert len(single) == EXPECTED_SINGLE_LINE
    assert single[0].start_line == EXPECTED_SINGLE_LINE
    assert single[0].end_line == EXPECTED_SINGLE_LINE
    assert long_line in single[0].text

    with pytest.raises(ValidationFailedError):
        slice_text_into_windows(" ", "a.log", "hello")
    with pytest.raises(ValidationFailedError):
        slice_text_into_windows("fil_1", " ", "hello")
    with pytest.raises(ValidationFailedError):
        slice_text_into_windows("fil_1", "a.log", "hello", window_tokens=0)
    with pytest.raises(ValidationFailedError):
        slice_text_into_windows("fil_1", "a.log", "hello", overlap_tokens=-1)


def test_prompt_formatting_citations_and_history_trimming() -> None:
    """Line numbering, citation extraction, and history trimming work deterministically."""
    numbered = format_line_numbered_block("app.py", "alpha\nbeta", start_line=10)
    assert numbered == "[app.py:L10-L11]\n10: alpha\n11: beta"

    files = [_make_file("fil_1", "logs/app.log", 100)]
    cites = extract_citations(
        "Crash at [app.log:L12-L15], repeat [logs/app.log:L20], and [app.log:L12-L15].",
        files,
    )
    assert [(c.file_id, c.start_line, c.end_line) for c in cites] == [
        ("fil_1", 12, 15),
        ("fil_1", 20, 20),
    ]

    history = [
        Message(
            session_id="ses_1", project_id="prj_1", role=MessageRole.USER, content="oldest " * 45
        ),
        Message(
            session_id="ses_1", project_id="prj_1", role=MessageRole.ASSISTANT, content="reply 1"
        ),
        Message(
            session_id="ses_1", project_id="prj_1", role=MessageRole.USER, content="latest question"
        ),
    ]
    trimmed = trim_history_to_budget(history, max_history_tokens=SMALL_HISTORY_BUDGET)
    assert [m.content for m in trimmed] == ["reply 1", "latest question"]
