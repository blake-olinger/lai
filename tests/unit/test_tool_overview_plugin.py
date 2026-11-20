"""Unit tests for GetFileOverviewPlugin and default READ-tier tool registry."""

from __future__ import annotations

import json

import pytest

from lai.domain.errors import AuthorizationError, NotFoundError
from lai.domain.files import (
    CodeFileSummary,
    ExceptionFingerprint,
    FileKind,
    LogFileSummary,
    LogSeverity,
)
from lai.domain.tools import PermissionTier
from lai.tools import GetFileOverviewPlugin, build_default_tool_registry
from tests.unit.test_tool_plugins import (
    EXPECTED_THREE,
    RecordingBlobStore,
    RecordingFileRepo,
    make_file_record,
    make_tool_ctx,
)


async def test_get_file_overview_and_default_registry() -> None:
    """GetFileOverviewPlugin returns summaries and default registry exposes READ tools."""
    repo, blob = RecordingFileRepo(), RecordingBlobStore()
    fp = ExceptionFingerprint(
        fingerprint="fp1",
        exception_type="TimeoutError",
        message_sample="t",
        count=3,
        first_line=12,
        last_line=99,
    )
    log_sum = LogFileSummary(
        total_lines=100, severity_counts={LogSeverity.ERROR: 3}, error_count=3, top_exceptions=(fp,)
    )
    code_sum = CodeFileSummary(
        language="python", total_lines=40, symbols=("main",), imports=("json",)
    )
    log_rec = make_file_record("fil_log", "prj_1", "srv.log", FileKind.LOG).model_copy(
        update={"log_summary": log_sum}
    )
    code_rec = make_file_record("fil_code", "prj_1", "main.py", FileKind.CODE).model_copy(
        update={"code_summary": code_sum}
    )
    for rec in (
        log_rec,
        code_rec,
        make_file_record("fil_other", "prj_2", "other.py", FileKind.CODE),
    ):
        await repo.upsert_file(rec)

    overview = GetFileOverviewPlugin(repo)
    log_out = json.loads(
        (await overview.execute({"file_id": "fil_log"}, make_tool_ctx("prj_1"))).output
    )
    assert log_out["log_summary"]["error_count"] == EXPECTED_THREE
    assert log_out["code_summary"] is None
    code_out = json.loads(
        (await overview.execute({"file_id": "fil_code"}, make_tool_ctx("prj_1"))).output
    )
    assert code_out["code_summary"]["language"] == "python"
    assert code_out["code_summary"]["symbols"] == ["main"]

    for bad_args, ctx in (
        ({"file_id": "fil_other"}, make_tool_ctx("prj_1")),
        ({"file_id": "fil_log"}, make_tool_ctx("prj_1", ("fil_code",))),
    ):
        with pytest.raises(AuthorizationError):
            await overview.execute(bad_args, ctx)
    with pytest.raises(NotFoundError):
        await overview.execute({"file_id": "fil_missing"}, make_tool_ctx("prj_1"))

    specs = build_default_tool_registry(repo, blob).list_specs()
    assert {s.name for s in specs} == {"search_files", "read_file_lines", "get_file_overview"}
    assert all(s.permission_tier is PermissionTier.READ for s in specs)
    assert all(not s.requires_approval_by_default for s in specs)
