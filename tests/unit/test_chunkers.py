"""Unit tests for log, code, and prose structural chunkers."""

from __future__ import annotations

import json
import textwrap

from lai.domain.files import FileKind, LogSeverity
from lai.ingestion.chunkers.code import chunk_code_file
from lai.ingestion.chunkers.logs import chunk_log_file
from lai.ingestion.chunkers.prose import chunk_prose_file

SMALL_TOKEN_BUDGET = 40
CODE_TOKEN_BUDGET = 64
MAX_TOP_EXCEPTIONS = 20
TOTAL_SYNTHETIC_ERRORS = 25
EXPECTED_TWO = 2
EXPECTED_THREE = 3
EXPECTED_LAST_LINE_TWELVE = 12


def test_log_chunker_coalesces_python_traceback_across_tiny_budget() -> None:
    """Python tracebacks and chained exceptions are never split across chunk boundaries."""
    log_text = textwrap.dedent(
        """\
        2026-11-17T10:00:00Z INFO Booting service worker alpha
        2026-11-17T10:00:01Z INFO Listening on 127.0.0.1:8400
        2026-11-17T10:00:02Z ERROR Request handler failed
        Traceback (most recent call last):
          File "src/app/router.py", line 42, in dispatch
            return parse_payload(body)
          File "src/app/parser.py", line 19, in parse_payload
            raise ValueError("invalid payload 42")
        ValueError: invalid payload 42

        During handling of the above exception, another exception occurred:

        Traceback (most recent call last):
          File "src/app/server.py", line 88, in handle_request
            resp = dispatch(req)
        RuntimeError: request aborted
        2026-11-17T10:00:03Z INFO Recovery probe passed"""
    )
    chunks, summary = chunk_log_file(
        file_id="fil_1",
        project_id="prj_1",
        text=log_text,
        target_tokens=SMALL_TOKEN_BUDGET,
    )

    tb_chunks = [c for c in chunks if "Traceback (most recent call last):" in c.text]
    assert len(tb_chunks) == 1
    assert "RuntimeError: request aborted" in tb_chunks[0].text
    assert "ValueError: invalid payload 42" in tb_chunks[0].text
    assert tb_chunks[0].severity is LogSeverity.ERROR
    assert summary.error_count == 1
    assert summary.first_timestamp == "2026-11-17T10:00:00Z"
    assert summary.last_timestamp == "2026-11-17T10:00:03Z"
    assert summary.top_exceptions[0].exception_type == "RuntimeError"


def test_log_chunker_coalesces_java_dotnet_and_powershell_traces() -> None:
    """Java/Caused-by/.NET indented frames and PowerShell CategoryInfo stay atomic."""
    cat_line = (
        "    + CategoryInfo          : InvalidOperation: "
        "(HttpWebRequest) [Invoke-RestMethod], WebException"
    )
    log_text = textwrap.dedent(
        f"""\
        2026-11-17 10:00:00.000 INFO Starting batch job
        2026-11-17 10:00:01.100 ERROR Batch step failed
        java.lang.IllegalStateException: Pool exhausted
        \tat com.example.Pool.borrow(Pool.java:42)
        \tat com.example.Service.run(Service.java:18)
        Caused by: java.sql.SQLException: Timeout waiting for DB
        \tat org.db.Driver.connect(Driver.java:101)
        \t... 14 more
           --- End of inner exception stack trace ---

        Invoke-RestMethod : The remote server returned an error: (502) Bad Gateway.
        At C:\\scripts\\sync.ps1:19 char:5
        +     Invoke-RestMethod -Uri $endpoint
        +     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {cat_line}
            + FullyQualifiedErrorId : WebCmdletWebResponseException

        2026-11-17 10:00:05.000 WARNING Retrying in 5s"""
    )
    chunks, summary = chunk_log_file(
        file_id="fil_2",
        project_id="prj_1",
        text=log_text,
        target_tokens=SMALL_TOKEN_BUDGET,
    )

    java_chunks = [c for c in chunks if "IllegalStateException" in c.text]
    assert len(java_chunks) == 1
    assert "Caused by: java.sql.SQLException: Timeout waiting for DB" in java_chunks[0].text
    assert "--- End of inner exception stack trace ---" in java_chunks[0].text

    ps_chunks = [c for c in chunks if "CategoryInfo" in c.text]
    assert len(ps_chunks) == 1
    assert "Invoke-RestMethod : The remote server returned an error" in ps_chunks[0].text
    assert "FullyQualifiedErrorId" in ps_chunks[0].text

    assert summary.error_count == EXPECTED_TWO
    assert summary.warning_count == 1
    assert summary.total_lines == len(log_text.splitlines())


def test_log_summary_deduplicates_and_caps_top_20_exceptions() -> None:
    """ExceptionFingerprints deduplicate numeric variations and cap deterministically at 20."""
    lines: list[str] = []
    for idx in range(EXPECTED_THREE):
        lines.extend(
            [
                f"2026-11-17T10:00:0{idx}Z CRITICAL Request crashed",
                "Traceback (most recent call last):",
                '  File "app.py", line 10, in run',
                f"ValueError: invalid user id {100 + idx}",
            ]
        )
    for idx in range(TOTAL_SYNTHETIC_ERRORS):
        lines.append(f"2026-11-17T10:01:{idx:02d}Z ERROR CustomService{idx}Error: failure alpha")
        if idx == 0:
            lines.append("2026-11-17T10:02:00Z ERROR CustomService0Error: failure alpha")

    chunks, summary = chunk_log_file(file_id="fil_3", project_id="prj_1", text="\n".join(lines))
    assert chunks
    assert len(summary.top_exceptions) == MAX_TOP_EXCEPTIONS
    top = summary.top_exceptions[0]
    assert top.exception_type == "ValueError"
    assert top.count == EXPECTED_THREE
    assert top.first_line == 1
    assert top.last_line == EXPECTED_LAST_LINE_TWELVE
    assert summary.top_exceptions[1].exception_type == "CustomService0Error"
    assert summary.top_exceptions[1].count == EXPECTED_TWO
    assert summary.severity_counts[LogSeverity.CRITICAL] == EXPECTED_THREE

    empty_chunks, empty_summary = chunk_log_file(file_id="fil_e", project_id="prj_1", text="")
    assert empty_chunks == []
    assert empty_summary.total_lines == 0


def test_code_chunker_extracts_symbols_imports_and_headers() -> None:
    """Code chunker adds [filename:Lstart-Lend] headers and builds CodeFileSummary."""
    py_code = textwrap.dedent(
        """\
        from __future__ import annotations
        import json
        from pathlib import Path

        @dataclass
        class Worker:
            name: str
            retries: int = 3

            def start(self, config_path: Path) -> dict[str, object]:
                raw = config_path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                return {"worker": self.name, "retries": self.retries, "cfg": parsed}

        async def shutdown(code: int, reason: str) -> tuple[int, str]:
            normalized_reason = reason.strip() or "normal_exit"
            return code, normalized_reason"""
    )
    chunks, summary = chunk_code_file(
        file_id="fil_py",
        project_id="prj_1",
        filename="src/worker.py",
        text=py_code,
        target_tokens=CODE_TOKEN_BUDGET,
    )
    assert summary.language == "python"
    assert summary.total_lines == len(py_code.splitlines())
    assert summary.symbols == ("Worker", "start", "shutdown")
    assert summary.imports == (
        "from __future__ import annotations",
        "import json",
        "from pathlib import Path",
    )
    assert len(chunks) >= EXPECTED_TWO
    for chunk in chunks:
        expected_prefix = f"[worker.py:L{chunk.start_line}-L{chunk.end_line}]"
        assert chunk.text.startswith(expected_prefix)
        assert chunk.metadata["language"] == "python"

    go_code = textwrap.dedent(
        """\
        package main
        import (
            "fmt"
            "net/http"
        )
        type Server struct {
            Port int
        }
        type Handler interface {
            Serve()
        }
        func (s *Server) Listen() {
            fmt.Println(s.Port)
        }"""
    )
    go_chunks, go_summary = chunk_code_file(
        file_id="fil_go",
        project_id="prj_1",
        filename="main.go",
        text=go_code,
    )
    assert go_chunks
    assert go_summary.language == "go"
    assert go_summary.symbols == ("Server", "Handler", "Listen")
    assert go_summary.imports == ('"fmt"', '"net/http"')

    empty_code_chunks, empty_code_summary = chunk_code_file(
        file_id="fil_empty",
        project_id="prj_1",
        filename="empty.rs",
        text="",
    )
    assert empty_code_chunks == []
    assert empty_code_summary.language == "rust"
    assert empty_code_summary.total_lines == 0


def test_prose_chunker_handles_markdown_text_and_json() -> None:
    """Prose chunker respects markdown headings, fenced blocks, text paragraphs, and JSON."""
    md = textwrap.dedent(
        """\
        # Overview

        First paragraph explaining architecture.

        ```bash
        # Not a markdown heading inside code fence
        echo hello
        ```

        ## Deployment

        Second section paragraph with details."""
    )
    md_chunks = chunk_prose_file(
        file_id="fil_md",
        project_id="prj_1",
        kind=FileKind.MARKDOWN,
        text=md,
        target_tokens=80,
    )
    assert len(md_chunks) >= EXPECTED_TWO
    assert md_chunks[0].symbol_name == "Overview"
    assert any("# Not a markdown heading inside code fence" in c.text for c in md_chunks)
    assert md_chunks[-1].symbol_name == "Deployment"
    assert md_chunks[-1].metadata["heading"] == "Deployment"

    long_text = " ".join(f"word{i}" for i in range(120))
    txt_chunks = chunk_prose_file(
        file_id="fil_txt",
        project_id="prj_1",
        kind=FileKind.TEXT,
        text=long_text,
        target_tokens=40,
    )
    assert len(txt_chunks) >= EXPECTED_TWO
    assert all(c.start_line == 1 and c.end_line == 1 for c in txt_chunks)

    jsonl = "\n".join(json.dumps({"id": i, "msg": f"entry {i}"}) for i in range(10))
    json_chunks = chunk_prose_file(
        file_id="fil_json",
        project_id="prj_1",
        kind=FileKind.JSON,
        text=jsonl,
        target_tokens=50,
    )
    assert len(json_chunks) >= EXPECTED_TWO
    assert all(c.metadata["kind"] == "json" for c in json_chunks)
    assert (
        chunk_prose_file(file_id="fil_e", project_id="prj_1", kind=FileKind.TEXT, text="   ") == []
    )

    structured_log = textwrap.dedent(
        """\
        {"timestamp": "2026-11-17T11:00:00Z", "level": "warning", "msg": "slow query"}
        Nov 17 11:00:05 ERROR Disk latency spike"""
    )
    slog_chunks, slog_summary = chunk_log_file(
        file_id="fil_slog", project_id="prj_1", text=structured_log
    )
    assert len(slog_chunks) == 1
    assert slog_summary.warning_count == 1
    assert slog_summary.error_count == 1
    assert slog_summary.first_timestamp == "2026-11-17T11:00:00Z"
    assert slog_summary.last_timestamp == "Nov 17 11:00:05"
