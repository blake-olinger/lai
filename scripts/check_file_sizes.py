"""Check that all Python files under src/ and tests/ stay within the 350-line budget."""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 350
TARGET_DIRS = ("src", "tests")


def find_oversized_files(root: Path, max_lines: int = MAX_LINES) -> list[tuple[Path, int]]:
    """Return paths and line counts for Python files exceeding max_lines."""
    violations: list[tuple[Path, int]] = []
    for dirname in TARGET_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > max_lines:
                violations.append((path.relative_to(root), line_count))
    return violations


def main() -> int:
    """Run the file-size check from the repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    violations = find_oversized_files(repo_root)
    if violations:
        for rel_path, count in violations:
            sys.stderr.write(f"ERROR: {rel_path} has {count} lines (limit: {MAX_LINES})\n")
        return 1
    sys.stdout.write(f"OK: all Python files under {TARGET_DIRS} are <= {MAX_LINES} lines.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
