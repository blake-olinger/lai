"""Structure-aware source code chunker and deterministic CodeFileSummary builder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lai.domain.files import CodeFileSummary, FileChunk
from lai.ingestion.detector import detect_language, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CODE_CHUNK_TOKENS = 600
_HEADER_TOKEN_BUDGET = 16

_SYMBOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\b"),
    re.compile(
        r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)"
    ),
    re.compile(
        r"^\s*(?:export\s+(?:default\s+)?(?:async\s+)?)?function\*?\s+(?P<name>[A-Za-z_$][\w$]*)"
    ),
    re.compile(
        r"^\s*export\s+(?:default\s+)?(?:const|let|var|type|enum)\s+(?P<name>[A-Za-z_$][\w$]*)\b"
    ),
    re.compile(r"^\s*(?:export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)"),
    re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+"
        r"(?P<name>[A-Za-z_]\w*)"
    ),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)\b"),
    re.compile(r"^\s*type\s+(?P<name>[A-Za-z_]\w*)\s+(?:struct|interface)\b"),
    re.compile(
        r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed)\s+)+"
        r"(?:class|interface|struct|enum)\s+(?P<name>[A-Za-z_]\w*)"
    ),
)

_DECORATOR_RE = re.compile(r"^\s*(?:@\S|#\[.+\]|\[[A-Za-z_].*\]\s*$)")

_IMPORT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(import\s+.+)$"),
    re.compile(r"^\s*(from\s+\S+\s+import\s+.+)$"),
    re.compile(r"^\s*((?:pub\s+)?use\s+.+;)\s*$"),
    re.compile(r"^\s*(using\s+(?:module\s+)?.+)$"),
    re.compile(r"^\s*(Import-Module\s+.+)$", re.IGNORECASE),
    re.compile(r"^\s*(#\s*include\s+[<\"][^>\"]+[>\"])"),
    re.compile(r"^\s*((?:const|let|var)\s+.+=\s*require\(.+\).*)$"),
)


@dataclass(slots=True)
class _CodeSection:
    """Contiguous range of source lines anchored to an optional structural symbol."""

    start_line: int
    end_line: int
    symbol_name: str | None


@dataclass(slots=True)
class _CodeChunkContext:
    """Target file identity and language metadata for materialized code chunks."""

    file_id: str
    project_id: str
    display_name: str
    language: str


def _match_symbol(line: str) -> str | None:
    """Return structural symbol name if line starts a function/class/struct/interface."""
    for pattern in _SYMBOL_PATTERNS:
        match = pattern.match(line)
        if match is not None:
            return match.group("name")
    return None


def _extract_imports(raw_lines: Sequence[str]) -> tuple[str, ...]:
    """Extract deduplicated import statements preserving source order."""
    seen: set[str] = set()
    ordered: list[str] = []
    in_go_import_block = False

    for line in raw_lines:
        stripped = line.strip()
        if in_go_import_block:
            if stripped == ")":
                in_go_import_block = False
                continue
            if stripped and stripped not in seen:
                seen.add(stripped)
                ordered.append(stripped)
            continue

        if re.match(r"^\s*import\s*\(\s*$", line):
            in_go_import_block = True
            continue

        for pattern in _IMPORT_PATTERNS:
            match = pattern.match(line)
            if match is not None:
                stmt = match.group(1).strip().rstrip(";")
                if stmt not in seen:
                    seen.add(stmt)
                    ordered.append(stmt)
                break
    return tuple(ordered)


def _find_sections(raw_lines: Sequence[str]) -> tuple[list[_CodeSection], tuple[str, ...]]:
    """Partition source lines into structural sections and collect symbol names."""
    starts: list[tuple[int, str]] = []
    symbols: list[str] = []
    seen_symbols: set[str] = set()

    for idx, line in enumerate(raw_lines):
        symbol = _match_symbol(line)
        if symbol is None:
            continue
        if symbol not in seen_symbols:
            seen_symbols.add(symbol)
            symbols.append(symbol)
        start_idx = idx
        while start_idx > 0 and _DECORATOR_RE.match(raw_lines[start_idx - 1]):
            start_idx -= 1
        line_no = start_idx + 1
        if not starts or line_no > starts[-1][0]:
            starts.append((line_no, symbol))

    total = len(raw_lines)
    if not starts:
        return [_CodeSection(start_line=1, end_line=total, symbol_name=None)], ()

    sections: list[_CodeSection] = []
    first_start = starts[0][0]
    if first_start > 1:
        sections.append(_CodeSection(start_line=1, end_line=first_start - 1, symbol_name=None))

    for i, (sec_start, sym) in enumerate(starts):
        sec_end = starts[i + 1][0] - 1 if i + 1 < len(starts) else total
        if sec_start <= sec_end:
            sections.append(_CodeSection(start_line=sec_start, end_line=sec_end, symbol_name=sym))
    return sections, tuple(symbols)


def _format_chunk_text(
    display_name: str,
    raw_lines: Sequence[str],
    start_line: int,
    end_line: int,
) -> str:
    """Format a source code chunk with its [filename:Lstart-Lend] header."""
    header = f"[{display_name}:L{start_line}-L{end_line}]"
    body = "\n".join(raw_lines[start_line - 1 : end_line])
    return f"{header}\n{body}"


def _build_chunks_from_ranges(
    ctx: _CodeChunkContext,
    raw_lines: Sequence[str],
    ranges: list[tuple[int, int, str | None]],
) -> list[FileChunk]:
    """Materialize FileChunk models from packed (start_line, end_line, symbol_name) tuples."""
    chunks: list[FileChunk] = []
    for idx, (start_line, end_line, symbol_name) in enumerate(ranges):
        text = _format_chunk_text(ctx.display_name, raw_lines, start_line, end_line)
        chunks.append(
            FileChunk(
                file_id=ctx.file_id,
                project_id=ctx.project_id,
                chunk_index=idx,
                start_line=start_line,
                end_line=end_line,
                token_count=estimate_tokens(text),
                text=text,
                symbol_name=symbol_name,
                metadata={"language": ctx.language},
            )
        )
    return chunks


def _split_oversized_section(
    *,
    raw_lines: Sequence[str],
    section: _CodeSection,
    target_tokens: int,
) -> list[tuple[int, int, str | None]]:
    """Split a large structural section along line boundaries within target_tokens."""
    ranges: list[tuple[int, int, str | None]] = []
    cur_start = section.start_line
    budget = max(32, target_tokens - _HEADER_TOKEN_BUDGET)

    for line_no in range(section.start_line, section.end_line + 1):
        candidate = "\n".join(raw_lines[cur_start - 1 : line_no])
        if line_no > cur_start and estimate_tokens(candidate) > budget:
            ranges.append((cur_start, line_no - 1, section.symbol_name))
            cur_start = line_no
    if cur_start <= section.end_line:
        ranges.append((cur_start, section.end_line, section.symbol_name))
    return ranges


def _pack_sections(
    *,
    raw_lines: Sequence[str],
    sections: list[_CodeSection],
    display_name: str,
    target_tokens: int,
) -> list[tuple[int, int, str | None]]:
    """Pack small adjacent sections or split oversized ones into target_tokens slices."""
    packed: list[tuple[int, int, str | None]] = []
    pending: tuple[int, int, str | None] | None = None

    for section in sections:
        sec_text = _format_chunk_text(display_name, raw_lines, section.start_line, section.end_line)
        if estimate_tokens(sec_text) > target_tokens:
            if pending is not None:
                packed.append(pending)
                pending = None
            packed.extend(
                _split_oversized_section(
                    raw_lines=raw_lines,
                    section=section,
                    target_tokens=target_tokens,
                )
            )
            continue

        if pending is None:
            pending = (section.start_line, section.end_line, section.symbol_name)
            continue

        cur_start, _cur_end, cur_symbol = pending
        merged_text = _format_chunk_text(display_name, raw_lines, cur_start, section.end_line)
        if estimate_tokens(merged_text) <= target_tokens:
            pending = (cur_start, section.end_line, cur_symbol or section.symbol_name)
        else:
            packed.append(pending)
            pending = (section.start_line, section.end_line, section.symbol_name)

    if pending is not None:
        packed.append(pending)
    return packed


def chunk_code_file(
    *,
    file_id: str,
    project_id: str,
    filename: str,
    text: str,
    target_tokens: int = DEFAULT_CODE_CHUNK_TOKENS,
) -> tuple[list[FileChunk], CodeFileSummary]:
    """Split source code along structural symbols and build a deterministic CodeFileSummary."""
    language = detect_language(filename)
    if not text:
        return [], CodeFileSummary(language=language, total_lines=0)

    display_name = Path(filename).name or filename
    raw_lines = text.splitlines()
    imports = _extract_imports(raw_lines)
    sections, symbols = _find_sections(raw_lines)
    ranges = _pack_sections(
        raw_lines=raw_lines,
        sections=sections,
        display_name=display_name,
        target_tokens=max(64, target_tokens),
    )

    ctx = _CodeChunkContext(
        file_id=file_id,
        project_id=project_id,
        display_name=display_name,
        language=language,
    )
    chunks = _build_chunks_from_ranges(ctx, raw_lines, ranges)
    summary = CodeFileSummary(
        language=language,
        total_lines=len(raw_lines),
        symbols=symbols,
        imports=imports,
    )
    return chunks, summary
