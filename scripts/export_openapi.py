"""Export or validate the FastAPI OpenAPI 3.1 schema without GPU or model dependencies."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol, cast

from openapi_spec_validator import validate as validate_openapi_spec

from lai.config.settings import LaiSettings


class _OpenApiApp(Protocol):
    def openapi(self) -> dict[str, object]: ...


def _load_app() -> _OpenApiApp:
    """Construct the FastAPI application with default settings without entering lifespan."""
    module = importlib.import_module("lai.api.app")
    app_obj: object = module.create_app(LaiSettings())
    return cast("_OpenApiApp", app_obj)


def main() -> int:
    """Export openapi.json or validate the checked-in OpenAPI 3.1 schema."""
    parser = argparse.ArgumentParser(description="Export or validate OpenAPI schema.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate OpenAPI 3.1 spec and verify checked-in schema matches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openapi.json"),
        help="Target OpenAPI JSON file path.",
    )
    args = parser.parse_args()

    app = _load_app()
    schema = app.openapi()
    validate_openapi_spec(schema)  # type: ignore[arg-type]
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
        sys.stdout.write(f"OK: {output_path} is valid OpenAPI 3.1 and matches current routes.\n")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(f"OK: wrote validated OpenAPI 3.1 schema to {output_path}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
