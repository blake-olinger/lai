"""Build the static Scalar OpenAPI documentation site for GitHub Pages."""

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

_SCALAR_VERSION = "1.81.0"
_SCALAR_SCRIPT_URL = f"https://cdn.jsdelivr.net/npm/@scalar/api-reference@{_SCALAR_VERSION}"
_SCALAR_SCRIPT_INTEGRITY = "sha384-DXupxC0wR1MllQMEexwTowkgKaBvZGJuwz+TU+7cHC09aC+q5vPMiJS2TW3GWjPN"

_SCALAR_CONFIG_JSON = json.dumps(
    {
        "theme": "default",
        "layout": "modern",
        "hideModels": False,
        "showSidebar": True,
        "defaultHttpClient": {"targetKey": "shell", "clientKey": "curl"},
    },
    separators=(",", ":"),
)

_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>LAI API Reference</title>
    <style>
      :root {{
        color-scheme: light dark;
      }}
      html,
      body {{
        margin: 0;
        padding: 0;
        min-height: 100%;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }}
      @media (prefers-color-scheme: dark) {{
        body {{
          background-color: #0f1117;
          color: #e6edf3;
        }}
      }}
      @media (prefers-color-scheme: light) {{
        body {{
          background-color: #ffffff;
          color: #1f2328;
        }}
      }}
    </style>
  </head>
  <body>
    <script
      id="api-reference"
      data-url="./openapi.json"
      data-configuration='{scalar_config}'
    ></script>
    <script
      src="{scalar_script_url}"
      integrity="{scalar_script_integrity}"
      crossorigin="anonymous"
    ></script>
  </body>
</html>
"""


def build_pages_site(output_dir: Path) -> Path:
    """Validate OpenAPI 3.1 schema and emit openapi.json, .nojekyll, and index.html."""
    app = create_app(LaiSettings())
    schema = app.openapi()
    validate_openapi_spec(schema)

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_json = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    rendered_html = _INDEX_HTML_TEMPLATE.format(
        scalar_config=_SCALAR_CONFIG_JSON,
        scalar_script_url=_SCALAR_SCRIPT_URL,
        scalar_script_integrity=_SCALAR_SCRIPT_INTEGRITY,
    )

    (output_dir / "openapi.json").write_text(rendered_json, encoding="utf-8", newline="\n")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8", newline="\n")
    index_path = output_dir / "index.html"
    index_path.write_text(rendered_html, encoding="utf-8", newline="\n")
    return index_path


def main() -> int:
    """CLI entry point for building static GitHub Pages API documentation."""
    parser = argparse.ArgumentParser(
        description="Build static Scalar OpenAPI documentation for GitHub Pages."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("site"),
        help="Target output directory for generated static site files.",
    )
    args = parser.parse_args()
    output_dir = cast("Path", args.output_dir)
    index_path = build_pages_site(output_dir)
    sys.stdout.write(f"OK: built GitHub Pages API reference at {index_path}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
