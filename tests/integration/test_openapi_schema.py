"""Integration tests validating OpenAPI 3.1 schema parity and CI/CD workflow YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml
from openapi_spec_validator import validate as validate_openapi_spec

from lai.api.app import create_app
from lai.config.settings import LaiSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_mapping(path: Path) -> dict[object, object]:
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path} must parse as a YAML mapping"
    return cast("dict[object, object]", loaded)


def _get_triggers(workflow: dict[object, object]) -> dict[str, object]:
    # PyYAML (YAML 1.1) parses bare `on:` as boolean True.
    raw_on = workflow.get("on") if "on" in workflow else workflow.get(True)
    assert isinstance(raw_on, dict), "Workflow must define an 'on' trigger mapping"
    return {str(k): v for k, v in cast("dict[object, object]", raw_on).items()}


def test_openapi_json_matches_app_and_passes_spec_validator() -> None:
    """Checked-in openapi.json matches create_app() and validates as OpenAPI 3.1."""
    openapi_path = REPO_ROOT / "openapi.json"
    assert openapi_path.exists()
    checked_in = openapi_path.read_text(encoding="utf-8")

    app = create_app(LaiSettings())
    live_schema = app.openapi()
    validate_openapi_spec(cast("Any", live_schema))

    rendered = json.dumps(live_schema, indent=2, sort_keys=True) + "\n"
    assert checked_in == rendered

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


def test_github_workflows_and_dependabot_parse_with_expected_triggers_and_jobs() -> None:
    """Validate all CI, security, release workflows and dependabot config."""
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    assert {p.name for p in workflows_dir.glob("*.yml")} == {
        "ci.yml",
        "release.yml",
        "security.yml",
    }

    ci = _load_yaml_mapping(workflows_dir / "ci.yml")
    ci_triggers = _get_triggers(ci)
    assert {"push", "pull_request"}.issubset(ci_triggers)
    ci_jobs = cast("dict[str, object]", ci["jobs"])
    assert "verify" in ci_jobs

    security = _load_yaml_mapping(workflows_dir / "security.yml")
    sec_triggers = _get_triggers(security)
    assert {"push", "pull_request", "schedule"}.issubset(sec_triggers)
    sec_jobs = cast("dict[str, object]", security["jobs"])
    assert "security" in sec_jobs

    release = _load_yaml_mapping(workflows_dir / "release.yml")
    rel_triggers = _get_triggers(release)
    assert "push" in rel_triggers
    push_cfg = cast("dict[str, object]", rel_triggers["push"])
    assert push_cfg.get("tags") == ["v*.*.*"]
    rel_jobs = cast("dict[str, object]", release["jobs"])
    assert "release" in rel_jobs

    dependabot = _load_yaml_mapping(REPO_ROOT / ".github" / "dependabot.yml")
    updates = cast("list[dict[str, object]]", dependabot["updates"])
    ecosystems = {entry["package-ecosystem"] for entry in updates}
    assert ecosystems == {"uv", "github-actions"}
