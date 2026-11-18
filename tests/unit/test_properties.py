"""Hypothesis property-based tests for core algorithmic invariants."""

from __future__ import annotations

import itertools
import sqlite3
from contextlib import closing
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from lai.db.file_repo import sanitize_fts5_query
from lai.domain.files import FileChunk, ScoredChunk
from lai.domain.rrf import reciprocal_rank_fusion
from lai.inference.thinking import ThinkingStreamParser, split_thinking_content
from lai.ingestion.detector import estimate_tokens
from lai.observability.scrubber import scrub_sensitive_text
from lai.orchestrator.windower import slice_text_into_windows

_OVERLAP_FRACTION_DIVISOR = 3
_MAX_FTS_TERMS = 32
_RRF_POOL_SIZE = 12
_WINDOW_LINE_TOKEN_BASE = 1

_TEXT_STRATEGY = st.text(min_size=0, max_size=400)
_THINK_TOKENS = [
    "<thought>",
    "</thought>",
    "<think>",
    "</think>",
    "<|thought|>",
    "<|/thought|>",
    "<tho",
    "</th",
    "<|th",
]
_FTS_TOKENS = [
    '"',
    "'",
    "(",
    ")",
    "\x00",
    "AND",
    "OR",
    "NOT",
    "NEAR",
    "*",
    "^",
    ":",
    "col:term",
    "NEAR/3",
]
_SECRET_TOKENS = [
    "Bearer ",
    "lai_sk_test_12345678",
    "sk-proj-1234567890",
    "ghp_1234567890",
    "github_pat_1234567890",
    "AKIA1234567890ABCDEF",
    "password=hunter2 ",
    "[REDACTED_",
    "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxIn0.sig123",
    'token: "raw_secret_val" ',
    "alice.ops@example.com ",
    "123-45-6789 ",
    "4532-0151-1283-0366 ",
]


def _mixed_text(
    tokens: list[str], *, max_text: int, max_parts: int, sep: str = ""
) -> st.SearchStrategy[str]:
    return st.lists(
        st.one_of(st.sampled_from(tokens), st.text(min_size=0, max_size=max_text)),
        min_size=0,
        max_size=max_parts,
    ).map(sep.join)


def _rrf_pairs(max_score: float) -> st.SearchStrategy[list[tuple[int, float]]]:
    return st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=_RRF_POOL_SIZE - 1),
            st.floats(min_value=0.0, max_value=max_score, allow_nan=False),
        ),
        max_size=20,
    )


_THINKING_TEXT_STRATEGY = _mixed_text(_THINK_TOKENS, max_text=30, max_parts=18)
_FTS_QUERY_STRATEGY = _mixed_text(_FTS_TOKENS, max_text=24, max_parts=40, sep=" ")
_SCRUB_TEXT_STRATEGY = _mixed_text(_SECRET_TOKENS, max_text=24, max_parts=14)


def _draw_chunks(data: st.DataObject, text: str) -> list[str]:
    """Partition text into arbitrary random chunks using Hypothesis data."""
    if not text:
        return []
    split_points = data.draw(
        st.lists(st.integers(min_value=0, max_value=len(text)), max_size=24),
        label="split_points",
    )
    cuts = sorted(idx for idx in split_points if 0 <= idx <= len(text))
    points = [0, *cuts, len(text)]
    return [text[start:end] for start, end in itertools.pairwise(points)]


def _stream_thinking_chunks(
    chunks: list[str],
    reasoning: str | None = None,
    *,
    tag_pair: tuple[str, str] | None = None,
    start_in_thought: bool = False,
) -> tuple[str, str]:
    """Feed content chunks and optional reasoning through ThinkingStreamParser."""
    parser = (
        ThinkingStreamParser(
            open_tag=tag_pair[0],
            close_tag=tag_pair[1],
            start_in_thought=start_in_thought,
        )
        if tag_pair is not None
        else ThinkingStreamParser(start_in_thought=start_in_thought)
    )
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    for idx, chunk in enumerate(chunks or [""]):
        delta = parser.feed(content_delta=chunk, reasoning_delta=reasoning if idx == 0 else None)
        content_parts.append(delta.content_delta)
        thinking_parts.append(delta.thinking_delta)
    tail = parser.finalize()
    content_parts.append(tail.content_delta)
    thinking_parts.append(tail.thinking_delta)
    return "".join(content_parts), "".join(thinking_parts)


def _build_hit_list(
    pairs: list[tuple[int, float]],
    source: Literal["fts", "vector"],
) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=FileChunk(
                id=f"chk_{idx}",
                file_id=f"fil_{idx % 3}",
                project_id="prj_prop",
                chunk_index=idx,
                start_line=idx * 5 + 1,
                end_line=idx * 5 + 5,
                token_count=10,
                text=f"chunk body chk_{idx}",
            ),
            score=score,
            source=source,
        )
        for idx, score in pairs
    ]


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    text=_TEXT_STRATEGY,
    window_tokens=st.integers(min_value=1, max_value=160),
    overlap_tokens=st.integers(min_value=0, max_value=80),
)
def test_slice_text_into_windows_line_coverage_and_overlap_invariants(
    text: str,
    window_tokens: int,
    overlap_tokens: int,
) -> None:
    """Windows cover 1..N lines without gaps, respect overlap cap, and stay monotonic."""
    lines = text.splitlines()
    windows = slice_text_into_windows(
        "fil_prop",
        "sample.log",
        text,
        window_tokens=window_tokens,
        overlap_tokens=overlap_tokens,
    )
    if not lines:
        assert windows == []
        return

    total_lines = len(lines)
    total_windows = len(windows)
    assert total_windows >= 1

    effective_overlap = (
        overlap_tokens
        if overlap_tokens < window_tokens
        else max(0, window_tokens // _OVERLAP_FRACTION_DIVISOR)
    )
    line_costs = [max(_WINDOW_LINE_TOKEN_BASE, estimate_tokens(line) + 1) for line in lines]

    covered: set[int] = set()
    for idx, win in enumerate(windows):
        assert win.window_index == idx
        assert win.total_windows == total_windows
        assert 1 <= win.start_line <= win.end_line <= total_lines
        assert win.text.startswith(f"[sample.log:L{win.start_line}-L{win.end_line}]\n")
        covered.update(range(win.start_line, win.end_line + 1))

    assert windows[0].start_line == 1
    assert windows[-1].end_line == total_lines
    assert covered == set(range(1, total_lines + 1))

    for prev, curr in itertools.pairwise(windows):
        assert prev.start_line < curr.start_line
        assert curr.start_line <= prev.end_line + 1
        if overlap_tokens == 0 or effective_overlap == 0:
            assert curr.start_line == prev.end_line + 1
        elif curr.start_line <= prev.end_line:
            overlap_cost = sum(line_costs[curr.start_line - 1 : prev.end_line])
            assert overlap_cost <= effective_overlap


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    content=_THINKING_TEXT_STRATEGY,
    reasoning=st.one_of(st.none(), _THINKING_TEXT_STRATEGY),
    start_in_thought=st.booleans(),
    data=st.data(),
)
def test_thinking_stream_parser_chunk_invariance_matches_split(
    content: str,
    reasoning: str | None,
    start_in_thought: bool,
    data: st.DataObject,
) -> None:
    """Streaming arbitrary splits through ThinkingStreamParser matches split_thinking_content."""
    got_content, got_thinking = _stream_thinking_chunks(
        _draw_chunks(data, content),
        reasoning,
        start_in_thought=start_in_thought,
    )
    expected = split_thinking_content(content, reasoning, start_in_thought=start_in_thought)
    assert got_content == expected.content_delta
    assert got_thinking == expected.thinking_delta
    assert "<thought>" not in got_content
    assert "</thought>" not in got_content
    assert "</thought>" not in got_thinking


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    content=_THINKING_TEXT_STRATEGY,
    tag_pair=st.sampled_from(
        [("<thought>", "</thought>"), ("<think>", "</think>"), ("<|thought|>", "<|/thought|>")]
    ),
    start_in_thought=st.booleans(),
    data=st.data(),
)
def test_thinking_stream_parser_custom_tags_chunk_invariance(
    content: str,
    tag_pair: tuple[str, str],
    start_in_thought: bool,
    data: st.DataObject,
) -> None:
    """Custom <think> and <|thought|> tag pairs remain invariant under arbitrary chunk splits."""
    ref_content, ref_thinking = _stream_thinking_chunks(
        [content],
        tag_pair=tag_pair,
        start_in_thought=start_in_thought,
    )
    got_content, got_thinking = _stream_thinking_chunks(
        _draw_chunks(data, content),
        tag_pair=tag_pair,
        start_in_thought=start_in_thought,
    )
    assert got_content == ref_content
    assert got_thinking == ref_thinking
    assert tag_pair[0] not in got_content
    assert tag_pair[1] not in got_content
    assert tag_pair[1] not in got_thinking


@settings(max_examples=50, deadline=None, derandomize=True)
@given(raw_query=_FTS_QUERY_STRATEGY)
def test_sanitize_fts5_query_never_raises_sqlite_operational_error(raw_query: str) -> None:
    """Every non-empty sanitize_fts5_query output executes cleanly against SQLite FTS5."""
    sanitized = sanitize_fts5_query(raw_query)
    assert "\x00" not in sanitized
    if not sanitized:
        return

    assert len(sanitized.split(" OR ")) <= _MAX_FTS_TERMS
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        conn.execute("INSERT INTO t(content) VALUES (?)", ("error timeout alpha beta",))
        rows = conn.execute("SELECT * FROM t WHERE t MATCH ?", (sanitized,)).fetchall()
        assert isinstance(rows, list)


@settings(max_examples=50, deadline=None, derandomize=True)
@given(text=_SCRUB_TEXT_STRATEGY, max_chars=st.integers(min_value=0, max_value=300))
def test_scrub_sensitive_text_idempotence_and_length_bound(text: str, max_chars: int) -> None:
    """scrub_sensitive_text is strictly bounded by max_chars and idempotent."""
    first = scrub_sensitive_text(text, max_chars=max_chars)
    second = scrub_sensitive_text(first, max_chars=max_chars)
    assert len(first) <= max_chars
    assert second == first


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    fts_pairs=_rrf_pairs(100.0),
    vector_pairs=_rrf_pairs(1.0),
    k=st.integers(min_value=1, max_value=100),
    limit=st.integers(min_value=0, max_value=25),
)
def test_reciprocal_rank_fusion_uniqueness_determinism_and_provenance(
    fts_pairs: list[tuple[int, float]],
    vector_pairs: list[tuple[int, float]],
    k: int,
    limit: int,
) -> None:
    """RRF enforces limit, unique chunk IDs, deterministic order, and exact provenance tags."""
    fts_hits = _build_hit_list(fts_pairs, "fts")
    vector_hits = _build_hit_list(vector_pairs, "vector")

    first_run = reciprocal_rank_fusion(fts_hits, vector_hits, k=k, limit=limit)
    second_run = reciprocal_rank_fusion(fts_hits, vector_hits, k=k, limit=limit)
    assert first_run == second_run
    assert len(first_run) <= limit

    out_ids = [hit.chunk.id for hit in first_run]
    assert len(out_ids) == len(set(out_ids))

    fts_ids = {hit.chunk.id for hit in fts_hits}
    vector_ids = {hit.chunk.id for hit in vector_hits}
    for hit in first_run:
        in_fts = hit.chunk.id in fts_ids
        in_vector = hit.chunk.id in vector_ids
        expected_source = "hybrid" if in_fts and in_vector else ("fts" if in_fts else "vector")
        assert hit.source == expected_source

    for prev, curr in itertools.pairwise(first_run):
        assert prev.score >= curr.score
