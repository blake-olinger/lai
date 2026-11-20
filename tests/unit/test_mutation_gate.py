"""Unit tests for scripts/check_mutation.py and scripts/_mutation_targets.py."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _mutation_targets, check_mutation  # noqa: E402

if TYPE_CHECKING:
    from scripts._mutation_targets import MutationTier

_EXPECTED_MIN_SCORE = 85.0
_CUSTOM_MIN_SCORE = 90.5
_DEFAULT_TIMEOUT = 20
_CUSTOM_TIMEOUT = 35
_CUSTOM_WORKERS = 3


def _write_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_mutation_target_lists_exist_and_exclude_removed_ollama_provider() -> None:
    """Every target in CRITICAL and FULL exists, is unique, and omits ollama_provider."""
    critical = _mutation_targets.CRITICAL_MUTATION_TARGETS
    full = _mutation_targets.FULL_MUTATION_TARGETS
    assert len(critical) == len(set(critical))
    assert len(full) == len(set(full))
    assert set(critical).issubset(set(full))
    assert len(full) > len(critical)
    assert "src/lai/guardrails/ollama_provider.py" not in full

    for rel_path in full:
        target = REPO_ROOT / rel_path
        assert target.is_file(), f"Missing mutation target: {rel_path}"
        tree = ast.parse(target.read_text(encoding="utf-8"))
        assert tree.body, f"Mutation target has no statements: {rel_path}"


def test_resolve_mutation_targets_tiers_and_override() -> None:
    """Tier resolution maps critical/full and honors explicit --targets override."""
    assert _mutation_targets.resolve_mutation_targets("critical") == ",".join(
        _mutation_targets.CRITICAL_MUTATION_TARGETS
    )
    assert _mutation_targets.resolve_mutation_targets("full") == ",".join(
        _mutation_targets.FULL_MUTATION_TARGETS
    )
    assert (
        _mutation_targets.resolve_mutation_targets("full", "src/lai/security/keys.py")
        == "src/lai/security/keys.py"
    )
    with pytest.raises(ValueError, match="empty"):
        _mutation_targets.resolve_mutation_targets("critical", "   ")


def test_build_parser_defaults_env_vars_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI parser honors env defaults and explicit tier/threshold/timeout flags."""
    monkeypatch.delenv("LAI_MIN_MUTATION_SCORE", raising=False)
    monkeypatch.delenv("LAI_MUTANT_TIMEOUT", raising=False)
    default_args = check_mutation.build_parser().parse_args([])
    assert default_args.tier == "critical"
    assert default_args.targets is None
    assert default_args.min_score == _EXPECTED_MIN_SCORE
    assert default_args.mutant_timeout == _DEFAULT_TIMEOUT

    monkeypatch.setenv("LAI_MIN_MUTATION_SCORE", "88.5")
    monkeypatch.setenv("LAI_MUTANT_TIMEOUT", "35")
    env_args = check_mutation.build_parser().parse_args([])
    assert env_args.min_score == 88.5
    assert env_args.mutant_timeout == _CUSTOM_TIMEOUT

    explicit_args = check_mutation.build_parser().parse_args(
        [
            "--tier",
            "full",
            "--min-score",
            "90.5",
            "--targets",
            "src/a.py",
            "--workers",
            "3",
            "--mutant-timeout",
            "45",
        ]
    )
    assert explicit_args.tier == "full"
    assert explicit_args.min_score == _CUSTOM_MIN_SCORE
    assert explicit_args.targets == "src/a.py"
    assert explicit_args.workers == _CUSTOM_WORKERS
    assert explicit_args.mutant_timeout == 45


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
    """evaluate_gremlins_report accepts passing scores and rejects invalid or low reports."""
    report_path = tmp_path / "gremlins.json"
    _write_report(report_path, payload)
    assert check_mutation.evaluate_gremlins_report(report_path, min_score) == expected_rc


def test_evaluate_gremlins_report_missing_file(tmp_path: Path) -> None:
    """Missing gremlins JSON report returns failure code 1."""
    missing = tmp_path / "missing.json"
    assert check_mutation.evaluate_gremlins_report(missing, _EXPECTED_MIN_SCORE) == 1


def test_run_mutation_gate_cleans_up_on_success_and_runs_both_suites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful mutation gate targets unit + integration suites and cleans coverage/gremlins."""
    monkeypatch.setattr(check_mutation, "REPO_ROOT", tmp_path)
    recorded: list[tuple[str, ...]] = []

    def _fake_run(*args: str) -> int:
        recorded.append(args)
        report = tmp_path / check_mutation.GREMLINS_JSON_REL
        _write_report(
            report,
            {"summary": {"total": 10, "zapped": 9, "pardoned": 0, "percentage": 90.0}},
        )
        return 0

    monkeypatch.setattr(check_mutation, "_run_pytest", _fake_run)
    rc = check_mutation.run_mutation_gate(tier="critical")
    assert rc == 0
    assert not (tmp_path / "coverage").exists()
    assert len(recorded) == 1
    args = recorded[0]
    assert "tests/unit" in args
    assert "tests/integration" in args
    assert f"--gremlin-targets={','.join(_mutation_targets.CRITICAL_MUTATION_TARGETS)}" in args


@pytest.mark.parametrize(
    ("pytest_rc", "percentage", "tier"),
    [
        (1, 95.0, "critical"),
        (0, 70.0, "full"),
    ],
)
def test_run_mutation_gate_preserves_report_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytest_rc: int,
    percentage: float,
    tier: MutationTier,
) -> None:
    """Failed pytest or below-threshold score preserves coverage/gremlins/gremlins.json."""
    monkeypatch.setattr(check_mutation, "REPO_ROOT", tmp_path)
    report = tmp_path / check_mutation.GREMLINS_JSON_REL
    recorded_args: list[tuple[str, ...]] = []

    def _fake_run(*args: str) -> int:
        recorded_args.append(args)
        _write_report(
            report,
            {"summary": {"total": 10, "zapped": 7, "pardoned": 0, "percentage": percentage}},
        )
        return pytest_rc

    monkeypatch.setattr(check_mutation, "_run_pytest", _fake_run)
    rc = check_mutation.run_mutation_gate(tier=tier, min_score=_EXPECTED_MIN_SCORE)
    assert rc == 1
    assert report.exists()
    expected_targets = _mutation_targets.resolve_mutation_targets(tier)
    assert f"--gremlin-targets={expected_targets}" in recorded_args[0]


def test_isolated_temp_env_sets_and_restores_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """isolated_temp_env isolates TEMP/TMP/TMPDIR and restores prior env and cleans up."""
    monkeypatch.setenv("TEMP", "orig-temp")
    monkeypatch.setenv("TMP", "orig-tmp")
    monkeypatch.delenv("TMPDIR", raising=False)

    created_dir: Path | None = None
    with _mutation_targets.isolated_temp_env() as tmp_dir:
        created_dir = tmp_dir
        assert tmp_dir.is_dir()
        for key in ("TEMP", "TMP", "TMPDIR"):
            assert os.environ.get(key) == str(tmp_dir)

    assert created_dir is not None
    assert not created_dir.exists()
    assert os.environ.get("TEMP") == "orig-temp"
    assert os.environ.get("TMP") == "orig-tmp"
    assert "TMPDIR" not in os.environ
