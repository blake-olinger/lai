"""Export or validate the FastAPI OpenAPI 3.1 schema without GPU or model dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openapi_spec_validator import validate as validate_openapi_spec

from lai.api.app import create_app
from lai.config.settings import LaiSettings


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

    app = create_app(LaiSettings())
    schema = app.openapi()
    validate_openapi_spec(schema)
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
