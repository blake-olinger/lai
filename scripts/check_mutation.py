"""Run pytest-gremlins mutation testing and enforce the minimum mutation score gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._mutation_targets import (  # noqa: E402
    MutationTier,
    isolated_temp_env,
    resolve_mutation_targets,
)

DEFAULT_MIN_MUTATION_SCORE = 85.0
DEFAULT_MUTANT_TIMEOUT_SECONDS = 20
DEFAULT_MUTATION_WORKERS = min(os.cpu_count() or 4, 8)
GREMLINS_JSON_REL = Path("coverage/gremlins/gremlins.json")


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable or return default when unset."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable or return default when unset."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _run_pytest(*args: str) -> int:
    """Execute pytest in a subprocess using the active Python interpreter."""
    cmd = [sys.executable, "-m", "pytest", *args]
    sys.stdout.write(f">>> {' '.join(cmd)}\n")
    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)  # noqa: S603
    return completed.returncode


def _cleanup_generated_coverage_dir(*, existed_before: bool) -> None:
    """Remove generated coverage/gremlins directory when the mutation run succeeds."""
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


def run_mutation_gate(
    targets: str | None = None,
    *,
    tier: MutationTier = "critical",
    min_score: float = DEFAULT_MIN_MUTATION_SCORE,
    workers: int = DEFAULT_MUTATION_WORKERS,
    mutant_timeout: int = DEFAULT_MUTANT_TIMEOUT_SECONDS,
) -> int:
    """Run pytest-gremlins against resolved targets and enforce the minimum mutation score."""
    resolved_targets = resolve_mutation_targets(tier=tier, targets=targets)
    existed_before = (REPO_ROOT / "coverage").exists()
    report_path = REPO_ROOT / GREMLINS_JSON_REL
    with isolated_temp_env():
        rc = _run_pytest(
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the mutation testing gate."""
    parser = argparse.ArgumentParser(
        description="Run pytest-gremlins mutation testing and enforce a score threshold.",
    )
    parser.add_argument(
        "--tier",
        choices=("critical", "full"),
        default="critical",
        help="Mutation target tier to run when --targets is omitted (default: critical).",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Optional comma-separated source targets override for pytest-gremlins.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=_env_float("LAI_MIN_MUTATION_SCORE", DEFAULT_MIN_MUTATION_SCORE),
        help="Minimum required mutation kill percentage.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MUTATION_WORKERS,
        help=f"Parallel gremlin worker count (default: {DEFAULT_MUTATION_WORKERS}).",
    )
    parser.add_argument(
        "--mutant-timeout",
        type=int,
        default=_env_int("LAI_MUTANT_TIMEOUT", DEFAULT_MUTANT_TIMEOUT_SECONDS),
        help="Per-mutant execution timeout in seconds.",
    )
    return parser


def main() -> int:
    """Parse CLI arguments and run the mutation gate."""
    args = build_parser().parse_args()
    return run_mutation_gate(
        targets=cast("str | None", args.targets),
        tier=cast("MutationTier", args.tier),
        min_score=cast("float", args.min_score),
        workers=cast("int", args.workers),
        mutant_timeout=cast("int", args.mutant_timeout),
    )


if __name__ == "__main__":
    raise SystemExit(main())
