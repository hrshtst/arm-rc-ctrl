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
from arm_rc_ctrl.experiments.reproduce_1a import CONFIRMATORY_REPORT, STEPS, compare_suites, main, reproduce
from arm_rc_ctrl.experiments.robustness import load_suite
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


def test_compare_suites_reports_deviations_and_categorical_differences() -> None:
    """Identical suites deviate by zero; a changed metric or outcome is reported."""
    suite = load_suite(CONFIRMATORY_REPORT)
    assert compare_suites(suite, suite) == (0.0, [])
    first = suite.runs[0]
    assert first.report.joint_rmse is not None
    bumped = replace(
        first.report, joint_rmse=replace(first.report.joint_rmse, aggregate=first.report.joint_rmse.aggregate + 1e-6)
    )
    worst, differences = compare_suites(
        suite, replace(suite, runs=(replace(first, report=bumped), *suite.runs[1:]), aggregates=(), effects=())
    )
    assert worst == pytest.approx(1e-6)
    assert differences == []
    failed = replace(first.report, success=False, failed_criteria=("completed",))
    _, differences = compare_suites(
        suite, replace(suite, runs=(replace(first, report=failed), *suite.runs[1:]), aggregates=(), effects=())
    )
    assert differences
    assert "outcome" in differences[0]


def worktree_is_clean() -> bool:
    """Whether the repository has no uncommitted change (the confirmatory rerun needs that)."""
    status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return status.stdout.strip() == ""


def test_reproduction_rebuilds_data_model_and_report_and_names_the_rerun_requirement(tmp_path: Path) -> None:
    """In a dirty worktree every step but the confirmatory rerun passes; the rerun names its clean-checkout need."""
    store = configured_store()
    result = reproduce(
        scratch=tmp_path / "scratch", classes=("nominal",), exploratory=True, keep_going=True, store=store
    )
    assert [c.name for c in result.checks] == list(STEPS)
    outcomes = {c.name: c.ok for c in result.checks}
    assert outcomes == {**dict.fromkeys(STEPS, True), "evaluation": False}
    evaluation = next(c for c in result.checks if c.name == "evaluation")
    assert "clean checkout" in evaluation.detail
    assert result.ok is False
    assert result.inputs["raw"].startswith("raw-")
    assert (tmp_path / "scratch" / "store" / "processed").is_dir()
    assert next(c for c in result.checks if c.name == "data").detail.endswith("(identical)")


def test_reproduction_reruns_the_nominal_evidence_exactly_from_a_clean_checkout(tmp_path: Path) -> None:
    """From a clean checkout the nominal class re-evaluates bitwise (skipped while the worktree is dirty)."""
    store = configured_store()
    if not worktree_is_clean():
        pytest.skip("the confirmatory rerun needs a clean worktree")
    result = reproduce(scratch=tmp_path / "scratch", classes=("nominal",), store=store)
    assert result.ok, [c for c in result.checks if not c.ok]
    assert result.max_deviation == 0.0
    assert (tmp_path / "scratch" / "store" / "runs").is_dir()


def test_a_mismatched_payload_fails_the_payload_step_clearly(tmp_path: Path) -> None:
    """A store whose processed payload differs from the record is refused before anything is rebuilt."""
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
    result = reproduce(scratch=tmp_path / "scratch", classes=("nominal",), exploratory=True, store=corrupt)
    names = [c.name for c in result.checks]
    assert names == ["environment", "storage", "records", "payloads"]
    assert not result.ok
    assert not result.checks[-1].ok
    assert result.checks[-1].name == "payloads"
    assert (
        suite.reference_artifact in result.checks[-1].detail
        or "digest" in result.checks[-1].detail.lower()
        or "size" in result.checks[-1].detail.lower()
    )


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
    assert [c["name"] for c in printed["checks"]] == ["environment", "storage"]
    assert printed["checks"][-1]["ok"] is False
    assert json.loads(summary.read_text(encoding="utf-8")) == printed
    note = audit.read_text(encoding="utf-8")
    assert note.startswith("# Task 1-a reproduction audit")
    assert "- Outcome: FAIL" in note
    assert "| storage | FAILED |" in note
    assert "Steps not run after the first failure: records, payloads, data, model, evaluation, report." in note
