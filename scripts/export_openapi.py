"""Export or validate the FastAPI OpenAPI 3.1 schema and Markdown API reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from openapi_spec_validator import validate as validate_openapi_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lai.api.app import create_app  # noqa: E402
from lai.config.settings import LaiSettings  # noqa: E402
from scripts._openapi_markdown import render_api_reference_markdown  # noqa: E402


def _check_artifacts(expected_files: tuple[tuple[Path, str], ...]) -> int:
    """Verify every generated OpenAPI/docs artifact exists and matches expected content."""
    failed = False
    for path, expected in expected_files:
        if not path.exists():
            sys.stderr.write(f"ERROR: {path} does not exist. Run export_openapi.py first.\n")
            failed = True
            continue
        current = path.read_text(encoding="utf-8")
        if current != expected:
            sys.stderr.write(f"ERROR: {path} is out of date.\n")
            failed = True
    if failed:
        return 1
    names = ", ".join(str(path) for path, _ in expected_files)
    sys.stdout.write(f"OK: {names} match validated OpenAPI 3.1 schema.\n")
    return 0


def main() -> int:
    """Export openapi.json and docs/api-reference.md or validate checked-in artifacts."""
    parser = argparse.ArgumentParser(
        description="Export or validate OpenAPI 3.1 JSON and Markdown API reference."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate OpenAPI 3.1 spec and verify checked-in JSON and Markdown match.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openapi.json"),
        help="Target OpenAPI JSON file path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/api-reference.md"),
        help="Target Markdown API reference file path.",
    )
    args = parser.parse_args()

    app = create_app(LaiSettings())
    schema = app.openapi()
    validate_openapi_spec(schema)
    rendered_json = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_api_reference_markdown(schema)

    output_path = cast("Path", args.output)
    markdown_path = cast("Path", args.markdown_output)
    artifacts = (
        (output_path, rendered_json),
        (markdown_path, rendered_markdown),
    )

    if bool(args.check):
        return _check_artifacts(artifacts)

    for target_path, content in artifacts:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8", newline="\n")
    sys.stdout.write(
        f"OK: wrote validated OpenAPI 3.1 schema to {output_path} "
        f"and Markdown reference to {markdown_path}.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
