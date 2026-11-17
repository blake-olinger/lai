"""Export or verify the FastAPI OpenAPI schema (placeholder until lai.api lands)."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol


class _OpenApiApp(Protocol):
    def openapi(self) -> dict[str, object]: ...


def _load_app() -> _OpenApiApp | None:
    """Try importing the FastAPI app factory from lai.api.app."""
    try:
        module = importlib.import_module("lai.api.app")
    except ModuleNotFoundError:
        return None
    create_app = getattr(module, "create_app", None)
    if callable(create_app):
        app_obj: object = create_app()
        return app_obj  # type: ignore[return-value]
    app_attr: object = getattr(module, "app", None)
    return app_attr  # type: ignore[return-value]


def main() -> int:
    """Export openapi.json or verify it matches the checked-in schema."""
    parser = argparse.ArgumentParser(description="Export or validate OpenAPI schema.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify checked-in schema matches generated schema without overwriting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openapi.json"),
        help="Target OpenAPI JSON file path.",
    )
    args = parser.parse_args()

    app = _load_app()
    if app is None:
        sys.stdout.write("INFO: lai.api not implemented yet; OpenAPI check skipped cleanly.\n")
        return 0

    schema = app.openapi()
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    output_path: Path = args.output
    if args.check:
        if not output_path.exists():
            sys.stderr.write(f"ERROR: {output_path} does not exist. Run export_openapi.py first.\n")
            return 1
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            sys.stderr.write(f"ERROR: {output_path} is out of date.\n")
            return 1
        sys.stdout.write(f"OK: {output_path} matches generated OpenAPI schema.\n")
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(f"OK: wrote OpenAPI schema to {output_path}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
