"""Integration tests validating OpenAPI 3.1 schema parity and CI/CD workflow YAML files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from openapi_spec_validator import validate as validate_openapi_spec

from lai.api.app import create_app
from lai.config.settings import LaiSettings

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import export_openapi  # noqa: E402
from scripts._openapi_markdown import render_api_reference_markdown  # noqa: E402

_SHA_USES_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _load_yaml_mapping(path: Path) -> dict[object, object]:
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path} must parse as a YAML mapping"
    return cast("dict[object, object]", loaded)


def _get_triggers(workflow: dict[object, object]) -> dict[str, object]:
    # PyYAML (YAML 1.1) parses bare `on:` as boolean True.
    raw_on = workflow.get("on") if "on" in workflow else workflow.get(True)
    assert isinstance(raw_on, dict), "Workflow must define an 'on' trigger mapping"
    return {str(k): v for k, v in cast("dict[object, object]", raw_on).items()}


def _assert_all_uses_sha_pinned(workflow: dict[object, object], name: str) -> None:
    jobs = cast("dict[str, dict[str, object]]", workflow["jobs"])
    for job_name, job_cfg in jobs.items():
        steps = cast("list[dict[str, object]]", job_cfg.get("steps", []))
        for step in steps:
            if "uses" in step:
                uses_val = str(step["uses"]).strip()
                assert _SHA_USES_RE.fullmatch(uses_val) is not None, (
                    f"{name}:{job_name} uses unpinned action {uses_val!r}"
                )


def test_openapi_json_and_markdown_match_app_and_export_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checked-in openapi.json and docs/api-reference.md match create_app() and --check."""
    openapi_path = REPO_ROOT / "openapi.json"
    markdown_path = REPO_ROOT / "docs" / "api-reference.md"
    assert openapi_path.exists()
    assert markdown_path.exists()
    checked_in_json = openapi_path.read_text(encoding="utf-8")
    checked_in_md = markdown_path.read_text(encoding="utf-8")

    app = create_app(LaiSettings())
    live_schema = app.openapi()
    validate_openapi_spec(cast("Any", live_schema))

    rendered = json.dumps(live_schema, indent=2, sort_keys=True) + "\n"
    assert checked_in_json == rendered
    assert checked_in_md == render_api_reference_markdown(live_schema)

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["export_openapi.py", "--check"])
    assert export_openapi.main() == 0

    paths = cast("dict[str, object]", live_schema["paths"])
    for expected_path in (
        "/v1/health",
        "/v1/system/status",
        "/v1/system/alerts",
        "/v1/projects",
        "/v1/files",
        "/v1/files/stream",
        "/v1/sessions",
        "/v1/sessions/{session_id}/messages",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/tools",
        "/v1/tool-approvals/{approval_id}",
        "/v1/tool-approvals/{approval_id}/resolve",
        "/v1/tool-approvals/{approval_id}/result",
        "/v1/client-apps",
        "/v1/client-apps/{app_id}",
    ):
        assert expected_path in paths
    app_ops = cast("dict[str, object]", paths["/v1/client-apps/{app_id}"])
    assert {"patch", "delete"}.issubset(app_ops)
    schemas = cast("dict[str, dict[str, object]]", live_schema["components"]["schemas"])
    for schema_name in ("Project", "ProjectCreateRequest"):
        props = cast("dict[str, object]", schemas[schema_name]["properties"])
        assert "workspace_root" not in props


def test_github_workflows_and_dependabot_parse_with_expected_triggers_and_jobs() -> None:
    """Validate CI, docs, security, release workflows, SHA pinning, and dependabot config."""
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    expected_workflows = {"ci.yml", "docs.yml", "release.yml", "security.yml"}
    assert {p.name for p in workflows_dir.glob("*.yml")} == expected_workflows

    ci = _load_yaml_mapping(workflows_dir / "ci.yml")
    assert ci.get("permissions") == {"contents": "read"}
    assert {"push", "pull_request"}.issubset(_get_triggers(ci))
    assert "verify" in cast("dict[str, object]", ci["jobs"])
    _assert_all_uses_sha_pinned(ci, "ci.yml")

    security = _load_yaml_mapping(workflows_dir / "security.yml")
    assert security.get("permissions") == {"contents": "read"}
    assert {"push", "pull_request", "schedule"}.issubset(_get_triggers(security))
    assert "security" in cast("dict[str, object]", security["jobs"])
    _assert_all_uses_sha_pinned(security, "security.yml")

    release = _load_yaml_mapping(workflows_dir / "release.yml")
    rel_triggers = _get_triggers(release)
    assert "push" in rel_triggers
    assert cast("dict[str, object]", rel_triggers["push"]).get("tags") == ["v*.*.*"]
    rel_jobs = cast("dict[str, dict[str, object]]", release["jobs"])
    assert "release" in rel_jobs
    assert rel_jobs["release"].get("environment") == "release"
    rel_steps = cast("list[dict[str, object]]", rel_jobs["release"].get("steps", []))
    assert any("bump_version.py --check-tag" in str(step.get("run", "")) for step in rel_steps)
    _assert_all_uses_sha_pinned(release, "release.yml")

    docs = _load_yaml_mapping(workflows_dir / "docs.yml")
    assert docs.get("permissions") == {"contents": "read"}
    docs_triggers = _get_triggers(docs)
    assert {"push", "pull_request", "workflow_dispatch"}.issubset(docs_triggers)
    docs_jobs = cast("dict[str, dict[str, object]]", docs["jobs"])
    assert {"build", "deploy"}.issubset(docs_jobs)
    deploy_job = docs_jobs["deploy"]
    assert deploy_job.get("if") == "github.event_name != 'pull_request'"
    deploy_env = cast("dict[str, object]", deploy_job.get("environment"))
    assert deploy_env.get("name") == "github-pages"
    _assert_all_uses_sha_pinned(docs, "docs.yml")

    dependabot = _load_yaml_mapping(REPO_ROOT / ".github" / "dependabot.yml")
    updates = cast("list[dict[str, object]]", dependabot["updates"])
    ecosystems = {entry["package-ecosystem"] for entry in updates}
    assert ecosystems == {"uv", "github-actions"}
