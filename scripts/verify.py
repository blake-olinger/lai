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

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MUTATION_TARGETS = (
    "src/lai/observability/alerts.py,"
    "src/lai/observability/scrubber.py,"
    "src/lai/security/keys.py,"
    "src/lai/security/scopes.py,"
    "src/lai/security/ratelimit.py,"
    "src/lai/guardrails/prompts.py,"
    "src/lai/tools/policy.py,"
    "src/lai/scheduler/fair_queue.py,"
    "src/lai/scheduler/session_gate.py,"
    "src/lai/ingestion/rrf.py,"
    "src/lai/orchestrator/classifier.py,"
    "src/lai/orchestrator/windower.py"
)
DEFAULT_MIN_MUTATION_SCORE = 85.0
DEFAULT_MUTANT_TIMEOUT_SECONDS = 20
_GREMLINS_JSON_REL = Path("coverage/gremlins/gremlins.json")


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
    for step in (
        lambda: _run_python_module("ruff", "check", "src", "tests"),
        lambda: _run_python_module("ruff", "format", "--check", "src", "tests"),
        lambda: _run_script("scripts/check_file_sizes.py"),
    ):
        rc = step()
        if rc != 0:
            return rc
    return 0


def cmd_typecheck() -> int:
    """Run basedpyright (strict) and mypy (strict)."""
    for module in ("basedpyright", "mypy"):
        rc = _run_python_module(module, "src", "tests")
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


def cmd_test(cov_fail_under: int | None = None) -> int:
    """Run unit and integration tests with coverage."""
    args = ["tests/unit", "tests/integration", "-v", "--cov=lai"]
    if cov_fail_under is not None:
        args.append(f"--cov-fail-under={cov_fail_under}")
    return _run_python_module("pytest", *args)


def cmd_schema() -> int:
    """Validate exported OpenAPI schema."""
    return _run_script("scripts/export_openapi.py", "--check")


def _cleanup_generated_coverage_dir(*, existed_before: bool) -> None:
    """Remove generated coverage/gremlins directory when created during verify."""
    gremlins_dir = REPO_ROOT / "coverage" / "gremlins"
    if gremlins_dir.exists():
        shutil.rmtree(gremlins_dir, ignore_errors=True)
    coverage_dir = REPO_ROOT / "coverage"
    if not existed_before and coverage_dir.exists():
        shutil.rmtree(coverage_dir, ignore_errors=True)


def _evaluate_gremlins_report(report_path: Path, min_score: float) -> int:
    """Parse pytest-gremlins JSON report and enforce minimum mutation kill percentage."""
    if not report_path.exists():
        sys.stderr.write(f"ERROR: Mutation JSON report not found at {report_path}\n")
        return 1
    raw_obj: object = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        sys.stderr.write("ERROR: Invalid mutation JSON report structure.\n")
        return 1
    data = cast("dict[str, object]", raw_obj)
    summary_obj = data.get("summary")
    if not isinstance(summary_obj, dict):
        sys.stderr.write("ERROR: Mutation JSON report missing 'summary'.\n")
        return 1
    summary = cast("dict[str, object]", summary_obj)
    total = int(cast("int", summary.get("total", 0)))
    zapped = int(cast("int", summary.get("zapped", 0)))
    pardoned = int(cast("int", summary.get("pardoned", 0)))
    percentage = float(cast("float | int", summary.get("percentage", 0.0)))
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


def cmd_mutate(
    targets: str = DEFAULT_MUTATION_TARGETS,
    *,
    min_score: float = DEFAULT_MIN_MUTATION_SCORE,
    workers: int = 4,
    mutant_timeout: int = DEFAULT_MUTANT_TIMEOUT_SECONDS,
) -> int:
    """Run mutation tests with pytest-gremlins and enforce the minimum mutation score."""
    coverage_dir = REPO_ROOT / "coverage"
    existed_before = coverage_dir.exists()
    report_path = REPO_ROOT / _GREMLINS_JSON_REL
    try:
        rc = _run_python_module(
            "pytest",
            "--gremlins",
            "--gremlin-parallel",
            f"--gremlin-workers={workers}",
            f"--gremlin-mutant-timeout={mutant_timeout}",
            f"--gremlin-targets={targets}",
            "--gremlin-report=console,json",
            "tests/unit",
        )
        if rc != 0:
            return rc
        return _evaluate_gremlins_report(report_path, min_score)
    finally:
        _cleanup_generated_coverage_dir(existed_before=existed_before)


def cmd_all(
    *,
    skip_mutate: bool = False,
    cov_fail_under: int | None = None,
    targets: str = DEFAULT_MUTATION_TARGETS,
    min_score: float = DEFAULT_MIN_MUTATION_SCORE,
) -> int:
    """Run the full verification suite in strict fail-fast order."""
    steps = [
        cmd_lint,
        cmd_typecheck,
        cmd_boundaries,
        lambda: cmd_test(cov_fail_under=cov_fail_under),
        cmd_schema,
    ]
    if not skip_mutate:
        steps.append(lambda: cmd_mutate(targets=targets, min_score=min_score))

    for step in steps:
        rc = step()
        if rc != 0:
            return rc
    return 0


def _build_parser() -> argparse.ArgumentParser:
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
        default=None,
        help="Optional minimum coverage percentage threshold.",
    )

    subparsers.add_parser("schema", help="Validate checked-in OpenAPI schema.")

    mutate_parser = subparsers.add_parser("mutate", help="Run pytest-gremlins mutation tests.")
    mutate_parser.add_argument(
        "--targets",
        default=DEFAULT_MUTATION_TARGETS,
        help="Comma-separated source targets for pytest-gremlins.",
    )
    mutate_parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_MUTATION_SCORE,
        help="Minimum required mutation kill percentage.",
    )

    all_parser = subparsers.add_parser("all", help="Run all verification checks.")
    all_parser.add_argument(
        "--skip-mutate",
        action="store_true",
        help="Skip pytest-gremlins mutation testing.",
    )
    all_parser.add_argument(
        "--cov-fail-under",
        type=int,
        default=None,
        help="Optional minimum coverage percentage threshold.",
    )
    all_parser.add_argument(
        "--targets",
        default=DEFAULT_MUTATION_TARGETS,
        help="Comma-separated source targets for pytest-gremlins.",
    )
    all_parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_MUTATION_SCORE,
        help="Minimum required mutation kill percentage.",
    )
    return parser


def main() -> int:
    """Dispatch the requested verification subcommand."""
    parser = _build_parser()
    args = parser.parse_args()
    subcommand = cast("str", args.subcommand)

    handlers: dict[str, Callable[[], int]] = {
        "lint": cmd_lint,
        "typecheck": cmd_typecheck,
        "boundaries": cmd_boundaries,
        "test": lambda: cmd_test(cov_fail_under=cast("int | None", args.cov_fail_under)),
        "schema": cmd_schema,
        "mutate": lambda: cmd_mutate(
            targets=cast("str", args.targets),
            min_score=cast("float", args.min_score),
        ),
        "all": lambda: cmd_all(
            skip_mutate=cast("bool", args.skip_mutate),
            cov_fail_under=cast("int | None", args.cov_fail_under),
            targets=cast("str", args.targets),
            min_score=cast("float", args.min_score),
        ),
    }
    return handlers[subcommand]()


if __name__ == "__main__":
    raise SystemExit(main())
