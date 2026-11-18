"""Unit tests for scripts/verify.py and scripts/_verify_targets.py gates."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _verify_targets, verify  # noqa: E402

if TYPE_CHECKING:
    from scripts._verify_targets import MutationTier

_EXPECTED_COV_FLOOR = 90
_EXPECTED_MIN_SCORE = 85.0
_CUSTOM_MIN_SCORE = 90.5


def _write_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_mutation_target_lists_exist_and_contain_executable_statements() -> None:
    """Every target in CRITICAL and FULL lists exists, is unique, and has AST statements."""
    critical = _verify_targets.CRITICAL_MUTATION_TARGETS
    full = _verify_targets.FULL_MUTATION_TARGETS
    assert len(critical) == len(set(critical))
    assert len(full) == len(set(full))
    assert set(critical).issubset(set(full))
    assert len(full) > len(critical)

    for rel_path in full:
        target = REPO_ROOT / rel_path
        assert target.is_file(), f"Missing mutation target: {rel_path}"
        tree = ast.parse(target.read_text(encoding="utf-8"))
        assert tree.body, f"Mutation target has no statements: {rel_path}"


def test_resolve_mutation_targets_tiers_and_override() -> None:
    """Tier resolution maps critical/full and honors explicit --targets override."""
    assert _verify_targets.resolve_mutation_targets("critical") == ",".join(
        _verify_targets.CRITICAL_MUTATION_TARGETS
    )
    assert _verify_targets.resolve_mutation_targets("full") == ",".join(
        _verify_targets.FULL_MUTATION_TARGETS
    )
    assert (
        _verify_targets.resolve_mutation_targets("full", "src/lai/security/keys.py")
        == "src/lai/security/keys.py"
    )
    with pytest.raises(ValueError, match="empty"):
        _verify_targets.resolve_mutation_targets("critical", "   ")


def test_build_parser_defaults_and_tier_flags() -> None:
    """CLI parser defaults to 90% coverage, critical tier, and parses full/override options."""
    parser = verify.build_parser()

    test_args = parser.parse_args(["test"])
    assert test_args.cov_fail_under == _EXPECTED_COV_FLOOR

    mutate_default = parser.parse_args(["mutate"])
    assert mutate_default.tier == "critical"
    assert mutate_default.targets is None
    assert mutate_default.min_score == _EXPECTED_MIN_SCORE

    mutate_full = parser.parse_args(
        ["mutate", "--tier", "full", "--min-score", "90.5", "--targets", "src/a.py"]
    )
    assert mutate_full.tier == "full"
    assert mutate_full.min_score == _CUSTOM_MIN_SCORE
    assert mutate_full.targets == "src/a.py"

    all_args = parser.parse_args(["all", "--tier", "full"])
    assert all_args.cov_fail_under == _EXPECTED_COV_FLOOR
    assert all_args.tier == "full"
    assert all_args.targets is None
    assert all_args.skip_mutate is False


@pytest.mark.parametrize(
    ("payload", "min_score", "expected_rc"),
    [
        ({"summary": {"total": 20, "zapped": 17, "pardoned": 0, "percentage": 85.0}}, 85.0, 0),
        ({"summary": {"total": 20, "zapped": 16, "pardoned": 1, "percentage": 80.0}}, 85.0, 1),
        ({"summary": {"total": 0, "zapped": 0, "pardoned": 0, "percentage": 0.0}}, 85.0, 1),
        ({"other": {}}, 85.0, 1),
        ([1, 2, 3], 85.0, 1),
        ("{not valid json", 85.0, 1),
        ({"summary": {"total": "bad", "zapped": 1, "pardoned": 0, "percentage": 85.0}}, 85.0, 1),
        ({"summary": {"total": 1, "zapped": 1, "pardoned": 0, "percentage": False}}, 85.0, 1),
        ({"summary": {"total": True, "zapped": 1, "pardoned": 0, "percentage": 85.0}}, 85.0, 1),
    ],
)
def test_evaluate_gremlins_report_cases(
    tmp_path: Path,
    payload: object,
    min_score: float,
    expected_rc: int,
) -> None:
    """_evaluate_gremlins_report accepts passing scores and rejects invalid or low reports."""
    report_path = tmp_path / "gremlins.json"
    _write_report(report_path, payload)
    assert verify.evaluate_gremlins_report(report_path, min_score) == expected_rc


def test_evaluate_gremlins_report_missing_file(tmp_path: Path) -> None:
    """Missing gremlins JSON report returns failure code 1."""
    missing = tmp_path / "missing.json"
    assert verify.evaluate_gremlins_report(missing, _EXPECTED_MIN_SCORE) == 1


def test_cmd_mutate_cleans_up_on_success_and_runs_both_suites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful mutation test targets unit + integration suites and removes coverage/gremlins."""
    monkeypatch.setattr(verify, "REPO_ROOT", tmp_path)
    recorded: list[tuple[str, tuple[str, ...]]] = []

    def _fake_run(module: str, *args: str) -> int:
        recorded.append((module, args))
        report = tmp_path / verify.GREMLINS_JSON_REL
        _write_report(
            report,
            {"summary": {"total": 10, "zapped": 9, "pardoned": 0, "percentage": 90.0}},
        )
        return 0

    monkeypatch.setattr(verify, "_run_python_module", _fake_run)
    rc = verify.cmd_mutate(tier="critical")
    assert rc == 0
    assert not (tmp_path / "coverage").exists()
    assert len(recorded) == 1
    module, args = recorded[0]
    assert module == "pytest"
    assert "tests/unit" in args
    assert "tests/integration" in args
    assert f"--gremlin-targets={','.join(_verify_targets.CRITICAL_MUTATION_TARGETS)}" in args


@pytest.mark.parametrize(
    ("pytest_rc", "percentage", "tier"),
    [
        (1, 95.0, "critical"),
        (0, 70.0, "full"),
    ],
)
def test_cmd_mutate_preserves_gremlins_report_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytest_rc: int,
    percentage: float,
    tier: MutationTier,
) -> None:
    """Failed pytest or below-threshold score preserves coverage/gremlins/gremlins.json."""
    monkeypatch.setattr(verify, "REPO_ROOT", tmp_path)
    report = tmp_path / verify.GREMLINS_JSON_REL
    recorded_args: list[tuple[str, ...]] = []

    def _fake_run(_module: str, *args: str) -> int:
        recorded_args.append(args)
        _write_report(
            report,
            {"summary": {"total": 10, "zapped": 7, "pardoned": 0, "percentage": percentage}},
        )
        return pytest_rc

    monkeypatch.setattr(verify, "_run_python_module", _fake_run)
    rc = verify.cmd_mutate(tier=tier, min_score=_EXPECTED_MIN_SCORE)
    assert rc == 1
    assert report.exists()
    expected_targets = _verify_targets.resolve_mutation_targets(tier)
    assert f"--gremlin-targets={expected_targets}" in recorded_args[0]


def test_cmd_all_propagates_tier_and_cov_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_all forwards tier, targets, min_score, and cov_fail_under to subcommands."""
    calls: list[tuple[str, object]] = []

    def _record_step(name: str) -> int:
        calls.append((name, None))
        return 0

    def _fake_test(*, cov_fail_under: int = 90) -> int:
        calls.append(("test", cov_fail_under))
        return 0

    def _fake_mutate(
        *,
        targets: str | None = None,
        tier: MutationTier = "critical",
        min_score: float = 85.0,
    ) -> int:
        calls.append(("mutate", (targets, tier, min_score)))
        return 0

    monkeypatch.setattr(verify, "cmd_lint", lambda: _record_step("lint"))
    monkeypatch.setattr(verify, "cmd_typecheck", lambda: _record_step("typecheck"))
    monkeypatch.setattr(verify, "cmd_boundaries", lambda: _record_step("boundaries"))
    monkeypatch.setattr(verify, "cmd_test", _fake_test)
    monkeypatch.setattr(verify, "cmd_schema", lambda: _record_step("schema"))
    monkeypatch.setattr(verify, "cmd_mutate", _fake_mutate)

    assert verify.cmd_all(tier="full", cov_fail_under=92, min_score=88.0) == 0
    assert calls == [
        ("lint", None),
        ("typecheck", None),
        ("boundaries", None),
        ("test", 92),
        ("schema", None),
        ("mutate", (None, "full", 88.0)),
    ]
