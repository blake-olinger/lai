"""Cross-platform verification runner avoiding Windows App-Control-blocked .exe shims."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._verify_targets import (  # noqa: E402
    CRITICAL_MUTATION_TARGETS,
    MutationTier,
    resolve_mutation_targets,
)

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_COV_FAIL_UNDER = 90
DEFAULT_MUTATION_TARGETS = ",".join(CRITICAL_MUTATION_TARGETS)
DEFAULT_MIN_MUTATION_SCORE = 85.0
DEFAULT_MUTANT_TIMEOUT_SECONDS = 20
DEFAULT_MUTATION_WORKERS = min(os.cpu_count() or 4, 8)
GREMLINS_JSON_REL = Path("coverage/gremlins/gremlins.json")
_GREMLINS_JSON_REL = GREMLINS_JSON_REL


def _run_python_module(module: str, *args: str) -> int:
    """Run a Python module as a subprocess using the active interpreter."""
    cmd = [sys.executable, "-m", module, *args]
    sys.stdout.write(f">>> {' '.join(cmd)}\n")
    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)  # noqa: S603
    return completed.returncode


def _run_script(script_rel_path: str, *args: str) -> int:
    """Run a repository Python script using the active interpreter."""
    script_path = REPO_ROOT / script_rel_path
    cmd = [sys.executable, str(script_path), *args]
    sys.stdout.write(f">>> {' '.join(cmd)}\n")
    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)  # noqa: S603
    return completed.returncode


def cmd_lint() -> int:
    """Run ruff lint, ruff format check, and 350-line file size guard."""
    steps: tuple[Callable[[], int], ...] = (
        lambda: _run_python_module("ruff", "check", "src", "tests", "scripts"),
        lambda: _run_python_module("ruff", "format", "--check", "src", "tests", "scripts"),
        lambda: _run_script("scripts/check_file_sizes.py"),
    )
    for step in steps:
        rc = step()
        if rc != 0:
            return rc
    return 0


def cmd_typecheck() -> int:
    """Run basedpyright (strict) and mypy (strict)."""
    steps: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("basedpyright", ("--pythonpath", sys.executable, "src", "tests", "scripts")),
        ("mypy", ("src", "tests", "scripts")),
    )
    for module, args in steps:
        rc = _run_python_module(module, *args)
        if rc != 0:
            return rc
    return 0


def cmd_boundaries() -> int:
    """Run import-linter architectural contracts without spawning lint-imports.exe."""
    import importlinter.cli as importlinter_cli  # noqa: PLC0415

    sys.stdout.write(">>> importlinter.cli.lint_imports_command()\n")
    sys.stdout.flush()
    prev_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        result = importlinter_cli.lint_imports_command(args=[], standalone_mode=False)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    finally:
        os.chdir(prev_cwd)
    return result if isinstance(result, int) else 0


def cmd_test(cov_fail_under: int = DEFAULT_COV_FAIL_UNDER) -> int:
    """Run unit and integration tests with branch coverage."""
    args = [
        "tests/unit",
        "tests/integration",
        "-v",
        "--cov=lai",
        "--cov-branch",
        f"--cov-fail-under={cov_fail_under}",
    ]
    return _run_python_module("pytest", *args)


def cmd_schema() -> int:
    """Validate exported OpenAPI schema."""
    return _run_script("scripts/export_openapi.py", "--check")


def _cleanup_generated_coverage_dir(*, existed_before: bool) -> None:
    """Remove generated coverage/gremlins directory when mutation run succeeds."""
    gremlins_dir = REPO_ROOT / "coverage" / "gremlins"
    if gremlins_dir.exists():
        shutil.rmtree(gremlins_dir, ignore_errors=True)
    coverage_dir = REPO_ROOT / "coverage"
    if coverage_dir.exists() and (not existed_before or not any(coverage_dir.iterdir())):
        shutil.rmtree(coverage_dir, ignore_errors=True)


def _extract_summary_metric(
    summary: dict[str, object],
    key: str,
    *,
    allow_float: bool = False,
) -> float:
    """Extract a numeric non-boolean metric from a gremlins summary mapping."""
    value = summary.get(key, 0.0 if allow_float else 0)
    valid_type = (int, float) if allow_float else int
    if isinstance(value, bool) or not isinstance(value, valid_type):
        msg = f"Invalid numeric field {key!r} in mutation report summary."
        raise TypeError(msg)
    return float(value)


def _load_gremlins_summary(report_path: Path) -> dict[str, object] | None:
    """Load and validate the top-level summary mapping from a gremlins JSON report."""
    if not report_path.exists():
        sys.stderr.write(f"ERROR: Mutation JSON report not found at {report_path}\n")
        return None
    try:
        raw_obj: object = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"ERROR: Failed to parse mutation JSON report at {report_path}: {exc}\n")
        return None
    if not isinstance(raw_obj, dict):
        sys.stderr.write("ERROR: Invalid mutation JSON report structure.\n")
        return None
    data = cast("dict[str, object]", raw_obj)
    summary_obj = data.get("summary")
    if not isinstance(summary_obj, dict):
        sys.stderr.write("ERROR: Mutation JSON report missing 'summary'.\n")
        return None
    return cast("dict[str, object]", summary_obj)


def evaluate_gremlins_report(report_path: Path, min_score: float) -> int:
    """Parse pytest-gremlins JSON report and enforce minimum mutation kill percentage."""
    summary = _load_gremlins_summary(report_path)
    if summary is None:
        return 1
    try:
        total = int(_extract_summary_metric(summary, "total"))
        zapped = int(_extract_summary_metric(summary, "zapped"))
        pardoned = int(_extract_summary_metric(summary, "pardoned"))
        percentage = _extract_summary_metric(summary, "percentage", allow_float=True)
    except TypeError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    if total <= 0:
        sys.stderr.write("ERROR: Zero gremlins evaluated; mutation targets may be invalid.\n")
        return 1
    sys.stdout.write(
        f"Mutation score: {zapped}/{total} zapped ({percentage:.2f}%, pardoned={pardoned}) "
        f"[threshold >= {min_score:.1f}%]\n"
    )
    sys.stdout.flush()
    if percentage < min_score:
        sys.stderr.write(
            f"ERROR: Mutation score {percentage:.2f}% is below threshold {min_score:.1f}%.\n"
        )
        return 1
    return 0


_evaluate_gremlins_report = evaluate_gremlins_report


def cmd_mutate(
    targets: str | None = None,
    *,
    tier: MutationTier = "critical",
    min_score: float = DEFAULT_MIN_MUTATION_SCORE,
    workers: int = DEFAULT_MUTATION_WORKERS,
    mutant_timeout: int = DEFAULT_MUTANT_TIMEOUT_SECONDS,
) -> int:
    """Run mutation tests with pytest-gremlins and enforce the minimum mutation score."""
    resolved_targets = resolve_mutation_targets(tier=tier, targets=targets)
    existed_before = (REPO_ROOT / "coverage").exists()
    report_path = REPO_ROOT / GREMLINS_JSON_REL
    rc = _run_python_module(
        "pytest",
        "--gremlins",
        "--gremlin-parallel",
        f"--gremlin-workers={workers}",
        f"--gremlin-mutant-timeout={mutant_timeout}",
        f"--gremlin-targets={resolved_targets}",
        "--gremlin-report=console,json",
        "tests/unit",
        "tests/integration",
    )
    if rc == 0:
        rc = evaluate_gremlins_report(report_path, min_score)
    if rc != 0:
        if report_path.exists():
            sys.stderr.write(f"Preserving mutation report at {report_path} for inspection.\n")
        return rc
    _cleanup_generated_coverage_dir(existed_before=existed_before)
    return 0


def cmd_all(
    *,
    skip_mutate: bool = False,
    cov_fail_under: int = DEFAULT_COV_FAIL_UNDER,
    tier: MutationTier = "critical",
    targets: str | None = None,
    min_score: float = DEFAULT_MIN_MUTATION_SCORE,
) -> int:
    """Run the full verification suite in strict fail-fast order."""
    steps: list[Callable[[], int]] = [
        cmd_lint,
        cmd_typecheck,
        cmd_boundaries,
        lambda: cmd_test(cov_fail_under=cov_fail_under),
        cmd_schema,
    ]
    if not skip_mutate:
        steps.append(lambda: cmd_mutate(targets=targets, tier=tier, min_score=min_score))

    for step in steps:
        rc = step()
        if rc != 0:
            return rc
    return 0


def _add_mutation_args(subparser: argparse.ArgumentParser) -> None:
    """Register shared mutation testing CLI options on a subcommand parser."""
    subparser.add_argument(
        "--tier",
        choices=("critical", "full"),
        default="critical",
        help="Mutation target tier to run when --targets is omitted (default: critical).",
    )
    subparser.add_argument(
        "--targets",
        default=None,
        help="Optional comma-separated source targets override for pytest-gremlins.",
    )
    subparser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_MUTATION_SCORE,
        help="Minimum required mutation kill percentage.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for repository verification subcommands."""
    parser = argparse.ArgumentParser(description="Run repository verification checks.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparsers.add_parser("lint", help="Run ruff lint, format check, and file size check.")
    subparsers.add_parser("typecheck", help="Run basedpyright and mypy.")
    subparsers.add_parser("boundaries", help="Run import-linter architectural contracts.")

    test_parser = subparsers.add_parser("test", help="Run pytest unit and integration suites.")
    test_parser.add_argument(
        "--cov-fail-under",
        type=int,
        default=DEFAULT_COV_FAIL_UNDER,
        help=f"Minimum branch coverage percentage threshold (default: {DEFAULT_COV_FAIL_UNDER}).",
    )

    subparsers.add_parser("schema", help="Validate checked-in OpenAPI schema.")

    mutate_parser = subparsers.add_parser("mutate", help="Run pytest-gremlins mutation tests.")
    _add_mutation_args(mutate_parser)

    all_parser = subparsers.add_parser("all", help="Run all verification checks.")
    all_parser.add_argument(
        "--skip-mutate",
        action="store_true",
        help="Skip pytest-gremlins mutation testing.",
    )
    all_parser.add_argument(
        "--cov-fail-under",
        type=int,
        default=DEFAULT_COV_FAIL_UNDER,
        help=f"Minimum branch coverage percentage threshold (default: {DEFAULT_COV_FAIL_UNDER}).",
    )
    _add_mutation_args(all_parser)
    return parser


_build_parser = build_parser


def main() -> int:
    """Dispatch the requested verification subcommand."""
    parser = _build_parser()
    args = parser.parse_args()
    subcommand = cast("str", args.subcommand)

    handlers: dict[str, Callable[[], int]] = {
        "lint": cmd_lint,
        "typecheck": cmd_typecheck,
        "boundaries": cmd_boundaries,
        "test": lambda: cmd_test(cov_fail_under=cast("int", args.cov_fail_under)),
        "schema": cmd_schema,
        "mutate": lambda: cmd_mutate(
            targets=cast("str | None", args.targets),
            tier=cast("MutationTier", args.tier),
            min_score=cast("float", args.min_score),
        ),
        "all": lambda: cmd_all(
            skip_mutate=cast("bool", args.skip_mutate),
            cov_fail_under=cast("int", args.cov_fail_under),
            tier=cast("MutationTier", args.tier),
            targets=cast("str | None", args.targets),
            min_score=cast("float", args.min_score),
        ),
    }
    return handlers[subcommand]()


if __name__ == "__main__":
    raise SystemExit(main())
