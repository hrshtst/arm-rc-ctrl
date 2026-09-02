# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The reproduction script rebuilds data, model, evaluation, and report or names the mismatched input (M3-012)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from arm_rc_ctrl.data.records import ProcessedDatasetRecord, RawDemonstrationRecord, load_record
from arm_rc_ctrl.dependencies import submodule_revisions
from arm_rc_ctrl.experiments.reproduce_1a import (
    CONFIRMATORY_REPORT,
    DOCS,
    STEPS,
    Reproducer,
    ReproductionError,
    compare_suites,
    main,
    prepare_evidence_checkout,
    prepare_scratch,
    reproduce,
)
from arm_rc_ctrl.experiments.robustness import RobustnessSuite, load_suite
from arm_rc_ctrl.metrics.report import RunReport
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()


def configured_store() -> StorageRoot:
    """The configured external store, or a skip when it is not available."""
    try:
        return open_storage()
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store not available: {exc}")


def test_compare_suites_covers_every_report_field_except_the_rerun_identity() -> None:
    """Identical suites deviate by zero; any float field counts towards the deviation; counts and labels are exact."""
    suite = load_suite(CONFIRMATORY_REPORT)
    assert compare_suites(suite, suite) == (0.0, [])
    first = suite.runs[0]
    assert first.report.joint_rmse is not None
    assert first.report.demand is not None
    assert first.report.dwell is not None

    def rebuilt(report: RunReport, run_id: str = first.run_id) -> RobustnessSuite:
        run = replace(first, run_id=run_id, report=replace(report, run_id=run_id))
        return replace(suite, runs=(run, *suite.runs[1:]), aggregates=(), effects=())

    renamed = rebuilt(first.report, run_id="run-20260831-000000000000")
    assert compare_suites(suite, renamed) == (0.0, [])  # the rerun's own ID is the only exempt field
    bumped_demand = replace(first.report.demand, torque_rms=first.report.demand.torque_rms + 1.0)
    demand = replace(first.report, demand=bumped_demand)
    assert compare_suites(suite, rebuilt(demand)) == (pytest.approx(1.0), [])
    peaks = first.report.demand.per_joint_peak
    per_joint = replace(first.report, demand=replace(first.report.demand, per_joint_peak=(peaks[0] + 0.5, *peaks[1:])))
    assert compare_suites(suite, rebuilt(per_joint)) == (pytest.approx(0.5), [])
    rmse = replace(
        first.report, joint_rmse=replace(first.report.joint_rmse, aggregate=first.report.joint_rmse.aggregate + 1e-6)
    )
    worst, differences = compare_suites(suite, rebuilt(rmse))
    assert worst == pytest.approx(1e-6)
    assert differences == []
    counted = replace(first.report, dwell=replace(first.report.dwell, samples=first.report.dwell.samples + 1))
    _, differences = compare_suites(suite, rebuilt(counted))
    samples = first.report.dwell.samples
    assert differences == [f"{first.arm}/{first.scenario_id}.dwell.samples: {samples + 1} vs {samples}"]
    source = replace(first.report, effort_source="tau_requested")
    _, differences = compare_suites(suite, rebuilt(source))
    assert differences == [f"{first.arm}/{first.scenario_id}.effort_source: 'tau_requested' vs 'tau_applied'"]
    windows = replace(first.report, windows=replace(first.report.windows, dwell=(0.0, 1.0)))
    worst, differences = compare_suites(suite, rebuilt(windows))
    assert worst >= 3.0  # the report windows are compared like every other float field
    assert differences == []
    shorter = replace(
        first.report, joint_rmse=replace(first.report.joint_rmse, per_joint=first.report.joint_rmse.per_joint[:1])
    )
    _, differences = compare_suites(suite, rebuilt(shorter))
    assert differences == [f"{first.arm}/{first.scenario_id}.joint_rmse.per_joint: length 1 vs 2"]
    failed = replace(first.report, success=False, failed_criteria=("completed",))
    _, differences = compare_suites(suite, rebuilt(failed))
    assert any(".success: False vs True" in d for d in differences)
    assert any(".failed_criteria: length 1 vs 0" in d for d in differences)


def test_scratch_must_be_a_fresh_directory_outside_the_repository(tmp_path: Path) -> None:
    """Repository-local, symbolic-link, and non-empty scratch targets are refused and nothing is deleted."""
    with pytest.raises(ValueError, match="outside the repository"):
        reproduce(scratch=REPO_ROOT / "build" / "scratch", classes=("nominal",))
    assert not (REPO_ROOT / "build" / "scratch").exists()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        reproduce(scratch=occupied, classes=("nominal",))
    assert sentinel.read_text(encoding="utf-8") == "do not delete"
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symbolic link"):
        reproduce(scratch=link, classes=("nominal",))
    assert real.is_dir()
    with pytest.raises(ValueError, match="not a directory"):
        reproduce(scratch=sentinel, classes=("nominal",))
    fresh = prepare_scratch(tmp_path / "new" / "deep")
    assert fresh.is_dir()
    assert not any(fresh.iterdir())


def test_tolerance_must_be_finite_and_non_negative(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """NaN, infinite, and negative tolerances are rejected before any work; the command exits with usage."""
    for bad in (float("nan"), float("inf"), -1e-9):
        with pytest.raises(ValueError, match="tolerance must be a finite non-negative number"):
            reproduce(scratch=tmp_path / f"scratch-{bad}", classes=("nominal",), tolerance=bad)
    with pytest.raises(SystemExit) as info:
        main(["--tolerance", "nan", "--scratch", str(tmp_path / "s"), "--exploratory"])
    assert info.value.code == 2
    assert "--tolerance must be a finite non-negative number" in capsys.readouterr().err


def worktree_is_clean() -> bool:
    """Whether the repository has no uncommitted change (the confirmatory rerun needs that)."""
    status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return status.stdout.strip() == ""


def at_evidence_pins() -> bool:
    """Whether the checked-out submodules match the confirmatory evidence (required by the environment step)."""
    suite = load_suite(CONFIRMATORY_REPORT)
    recorded = {s.name: (s.checked_out or s.recorded) for s in suite.provenance.submodules}
    current = {s.name: (s.checked_out or s.recorded) for s in submodule_revisions(REPO_ROOT)}
    return all(current.get(name) == revision for name, revision in recorded.items())


def test_reproduction_rebuilds_data_model_and_report_and_names_the_rerun_requirement(tmp_path: Path) -> None:
    """In a dirty worktree every step but the confirmatory rerun passes; the rerun names its clean-checkout need."""
    store = configured_store()
    result = reproduce(
        scratch=tmp_path / "scratch", classes=("nominal",), exploratory=True, keep_going=True, store=store
    )
    assert [c.name for c in result.checks] == list(STEPS)  # keep_going runs every step
    if not at_evidence_pins():
        # A checkout whose pins differ from the evidence must fail the environment step and rebuild nothing.
        environment = result.checks[0]
        assert not environment.ok
        assert "submodule pins differ" in environment.detail
        assert result.ok is False
        return
    outcomes = {c.name: c.ok for c in result.checks}
    assert outcomes == {**dict.fromkeys(STEPS, True), "evaluation": False}
    evaluation = next(c for c in result.checks if c.name == "evaluation")
    assert "clean checkout" in evaluation.detail
    assert result.ok is False
    assert result.inputs["raw"].startswith("raw-")
    assert result.inputs["reproduction_project_dirty"] == "True"
    assert len(result.inputs["reproduction_project_commit"]) == 40
    assert result.inputs["evidence_project_commit"] != result.inputs["reproduction_project_commit"]
    assert (tmp_path / "scratch" / "store" / "processed").is_dir()
    assert next(c for c in result.checks if c.name == "data").detail.endswith("(identical)")


def test_reproduction_reruns_the_nominal_evidence_exactly_from_a_clean_checkout(tmp_path: Path) -> None:
    """From a clean checkout at the evidence pins the nominal class re-evaluates bitwise."""
    store = configured_store()
    if not worktree_is_clean():
        pytest.skip("the confirmatory rerun needs a clean worktree")
    if not at_evidence_pins():
        pytest.skip("the checked-out submodule pins differ from the evidence; covered by the worktree test")
    result = reproduce(scratch=tmp_path / "scratch", classes=("nominal",), store=store)
    assert result.ok, [c for c in result.checks if not c.ok]
    assert result.max_deviation == 0.0
    assert (tmp_path / "scratch" / "store" / "runs").is_dir()


def test_from_evidence_prepares_a_worktree_with_the_recorded_pins(tmp_path: Path) -> None:
    """The worktree sits at the recorded audit commit with every evidence submodule pin checked out."""
    suite = load_suite(CONFIRMATORY_REPORT)
    audit = json.loads((DOCS / "reproduction_audit.json").read_text(encoding="utf-8"))
    try:
        checkout, commit = prepare_evidence_checkout(tmp_path / "scratch", keep=True)
    except ReproductionError as exc:
        pytest.skip(f"evidence worktree unavailable here: {exc}")
    try:
        assert commit == audit["inputs"]["reproduction_project_commit"]
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        assert head == commit
        assert (checkout / "scripts" / "reproduce_1a.py").is_file()
        status = subprocess.run(
            ["git", "-C", str(checkout), "submodule", "status"], check=True, capture_output=True, text=True
        ).stdout
        recorded = {s.name: (s.checked_out or s.recorded) for s in suite.provenance.submodules}
        for line in status.splitlines():
            revision, name = line.split()[0].lstrip("+-U"), line.split()[1].split("/")[-1]
            assert recorded[name] == revision, name
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(checkout)], check=False, capture_output=True)


def test_a_mismatched_payload_fails_the_payload_step_clearly(tmp_path: Path) -> None:
    """A store whose processed payload differs from the record is refused before anything is rebuilt.

    The payload step is exercised directly (past the environment step), so this
    holds on any checkout regardless of the current submodule pins.
    """
    store = configured_store()
    suite = load_suite(CONFIRMATORY_REPORT)
    recipe_datasets = json.loads((REPO_ROOT / "docs" / "experiments" / "task_1a" / "training_v4.json").read_text())[
        "datasets"
    ]
    processed = load_record(
        REPO_ROOT / "data" / "records" / "processed" / f"{recipe_datasets[0]}.toml", ProcessedDatasetRecord
    )
    (raw_id,) = processed.artifact.origin.sources
    raw = load_record(REPO_ROOT / "data" / "records" / "raw" / f"{raw_id}.toml", RawDemonstrationRecord)
    root = tmp_path / "corrupt"
    root.mkdir()
    corrupt = StorageRoot(root, repositories=(REPO_ROOT,))
    corrupt.path(raw.artifact.payload.uri, mode="write").write_bytes(
        store.path(raw.artifact.payload.uri, mode="read").read_bytes()
    )
    corrupt.path(processed.artifact.payload.uri, mode="write").write_bytes(b"not the payload")
    reproducer = Reproducer(
        scratch=tmp_path / "scratch",
        classes=("nominal",),
        tolerance=0.0,
        exploratory=True,
        configured_store=corrupt,
        docs=DOCS,
        now=None,
    )
    reproducer.suite = load_suite(CONFIRMATORY_REPORT)
    assert reproducer.step("storage").ok
    assert reproducer.step("records").ok
    check = reproducer.step("payloads")
    assert not check.ok
    assert "digest" in check.detail.lower() or "size" in check.detail.lower()
    assert suite.reference_artifact in check.detail or "run.json" not in check.detail


def test_command_writes_a_summary_and_reports_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints and writes the summary; a missing storage configuration is a clear non-zero exit."""
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(tmp_path / "missing"))
    summary = tmp_path / "summary.json"
    audit = tmp_path / "audit.md"
    argv = ["--classes", "nominal", "--scratch", str(tmp_path / "scratch"), "--summary", str(summary), "--exploratory"]
    status = main([*argv, "--audit", str(audit)])
    assert status == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    if at_evidence_pins():
        assert [c["name"] for c in printed["checks"]] == ["environment", "storage"]
    else:
        assert [c["name"] for c in printed["checks"]] == ["environment"]
        assert "submodule pins differ" in printed["checks"][0]["detail"]
    assert printed["checks"][-1]["ok"] is False
    assert json.loads(summary.read_text(encoding="utf-8")) == printed
    note = audit.read_text(encoding="utf-8")
    assert note.startswith("# Task 1-a reproduction audit")
    assert "- Outcome: FAIL" in note
    if at_evidence_pins():
        assert "| storage | FAILED |" in note
        assert "Steps not run after the first failure: records, payloads, data, model, evaluation, report." in note
    else:
        assert "| environment | FAILED |" in note
        assert (
            "Steps not run after the first failure: storage, records, payloads, data, model, evaluation, report."
            in note
        )
    assert str(tmp_path) not in note  # records never carry machine-specific paths
    assert "- Command: `python -m arm_rc_ctrl.experiments.reproduce_1a --classes nominal --scratch scratch" in note
