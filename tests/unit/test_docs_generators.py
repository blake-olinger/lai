"""Unit tests for OpenAPI Markdown renderer, export_openapi.py, and build_pages_docs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openapi_spec_validator import validate as validate_openapi_spec

from lai.api.app import create_app
from lai.config.settings import LaiSettings

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_pages_docs, export_openapi  # noqa: E402
from scripts._openapi_markdown import render_api_reference_markdown  # noqa: E402


def test_render_api_reference_markdown_live_and_synthetic_schemas() -> None:
    """render_api_reference_markdown deterministically formats live and edge-case schemas."""
    live_schema = create_app(LaiSettings()).openapi()
    live_md = render_api_reference_markdown(live_schema)
    assert "# LAI Local AI Server API Reference" in live_md
    assert "`Authorization: Bearer <google_id_token>`" in live_md
    assert "oauth_client_id" in live_md
    assert "`404 Not Found`" in live_md
    assert "ErrorResponse" in live_md
    assert "SSE Event Framing" in live_md
    assert "## Endpoints" in live_md
    assert "## Component Schemas" in live_md
    assert render_api_reference_markdown(live_schema) == live_md

    synthetic: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {"title": "Custom API", "version": "2.0.0", "description": "Top level overview."},
        "paths": {
            "/v1/items": {
                "parameters": [{"name": "ignored", "in": "header"}],
                "get": {
                    "summary": "List | Items",
                    "description": "Fetch items.",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "schema": {
                                "type": "string",
                                "minLength": 1,
                                "description": "A|B\nline",
                            },
                        },
                        {"name": "raw", "in": "query", "required": False},
                    ],
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ComplexInput"}
                            },
                            "text/plain": {"schema": {"type": "string"}},
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "OK | Done",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ScalarAlias"}
                                }
                            },
                        },
                        "204": {"description": ""},
                    },
                },
            }
        },
        "components": {
            "schemas": {
                "ColorEnum": {
                    "description": "Palette.",
                    "type": "string",
                    "enum": [f"c{i}" for i in range(10)],
                },
                "ScalarAlias": {"type": "string", "format": "uuid"},
                "ComplexInput": {
                    "type": "object",
                    "required": ["kind"],
                    "properties": {
                        "kind": {"const": "demo", "type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "bare_arr": {"type": "array"},
                        "meta": {"type": "object", "additionalProperties": {"type": "integer"}},
                        "bare_obj": {"type": "object"},
                        "opt": {
                            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
                            "default": None,
                        },
                        "choice": {"oneOf": [{"type": "string"}, {"type": "boolean"}]},
                        "merged": {"allOf": [{"$ref": "#/components/schemas/ScalarAlias"}]},
                        "enum_field": {
                            "type": "string",
                            "enum": [f"e{i}" for i in range(10)],
                        },
                    },
                },
            }
        },
    }
    md = render_api_reference_markdown(synthetic)
    assert "# Custom API API Reference" in md
    assert "Top level overview." in md
    assert "### Tag: `default`" in md
    assert "### `GET /v1/items`" in md
    assert "**List \\| Items**" in md
    assert "A\\|B line" in md
    assert "**Request Body** (optional)" in md
    assert "array[string]" in md
    assert "object[str, integer]" in md
    assert "integer \\| null" in md
    assert "string \\| boolean" in md
    assert "[`ScalarAlias`](#schema-scalaralias)" in md
    assert ", ..." in md
    assert "**Enum values**:" in md
    assert "**Type**: string (uuid)" in md


def test_export_openapi_write_and_check_drift_and_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """export_openapi.py writes deterministic artifacts and catches missing or drifted files."""
    json_out = tmp_path / "out" / "openapi.json"
    md_out = tmp_path / "docs" / "api-reference.md"

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_openapi.py", "--output", str(json_out), "--markdown-output", str(md_out)],
    )
    assert export_openapi.main() == 0
    assert json_out.is_file()
    assert md_out.is_file()
    validate_openapi_spec(json.loads(json_out.read_text(encoding="utf-8")))
    assert "OK: wrote validated OpenAPI 3.1 schema" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_openapi.py",
            "--check",
            "--output",
            str(json_out),
            "--markdown-output",
            str(md_out),
        ],
    )
    assert export_openapi.main() == 0
    assert "match validated OpenAPI 3.1 schema" in capsys.readouterr().out

    orig_md = md_out.read_text(encoding="utf-8")
    md_out.write_text("# drifted\n", encoding="utf-8")
    assert export_openapi.main() == 1
    assert "is out of date" in capsys.readouterr().err

    md_out.write_text(orig_md, encoding="utf-8")
    orig_json = json_out.read_text(encoding="utf-8")
    json_out.write_text("{}\n", encoding="utf-8")
    assert export_openapi.main() == 1
    assert "is out of date" in capsys.readouterr().err

    json_out.unlink()
    assert export_openapi.main() == 1
    assert "does not exist" in capsys.readouterr().err

    json_out.write_text(orig_json, encoding="utf-8")
    md_out.unlink()
    assert export_openapi.main() == 1
    assert "does not exist" in capsys.readouterr().err


def test_build_pages_docs_outputs_scalar_site_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """build_pages_docs.py emits index.html, openapi.json, and empty .nojekyll."""
    site_dir = tmp_path / "site"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_pages_docs.py", "--output-dir", str(site_dir)],
    )
    assert build_pages_docs.main() == 0
    assert "OK: built GitHub Pages API reference" in capsys.readouterr().out

    index_path = site_dir / "index.html"
    openapi_path = site_dir / "openapi.json"
    nojekyll_path = site_dir / ".nojekyll"

    assert index_path.is_file()
    assert openapi_path.is_file()
    assert nojekyll_path.is_file()
    assert nojekyll_path.read_text(encoding="utf-8") == ""

    html = index_path.read_text(encoding="utf-8")
    assert "<title>LAI API Reference</title>" in html
    assert 'data-url="./openapi.json"' in html
    assert "@scalar/api-reference@1.81.0" in html
    assert (
        'integrity="sha384-DXupxC0wR1MllQMEexwTowkgKaBvZGJuwz+TU+7cHC09aC+q5vPMiJS2TW3GWjPN"'
        in html
    )
    assert 'crossorigin="anonymous"' in html
    assert '"hideModels":false' in html

    schema = json.loads(openapi_path.read_text(encoding="utf-8"))
    validate_openapi_spec(schema)
    assert schema == create_app(LaiSettings()).openapi()
