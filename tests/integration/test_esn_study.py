# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""ESN study runs: Optuna study plus parent/child MLflow mirror, resume, selection report, CLI (M3-005)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.esn_objective import TrialEvaluation
from arm_rc_ctrl.experiments.esn_search import ComparisonPoint, EsnSearchProtocol, TrialPoint, load_esn_search
from arm_rc_ctrl.experiments.esn_study import (
    STUDY_TAG,
    TRIAL_TAG,
    EsnStudyResult,
    load_report,
    main,
    report_to_json,
    run_esn_study,
)
from arm_rc_ctrl.experiments.studies import PrunerSpec, SamplerSpec
from arm_rc_ctrl.experiments.tracking import MlflowTracker
from arm_rc_ctrl.experiments.tuning import DevelopmentScenarios
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 3, 11, 0, 0, tzinfo=UTC)
ANCHOR = TrialPoint(100, 0.9, 0.9, 0.3, 0.5, 31, 1e-2, 20.0, 20.0)


@pytest.fixture(scope="module")
def prepared(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path]:
    """A store with the fixture dataset and a four-trial protocol on the fixture scenario."""
    base = tmp_path_factory.mktemp("esn-study")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    protocol = replace(
        load_esn_search(PROTOCOL),
        name="esn-search-fixture",
        scenario=SCENARIO,
        model=MODEL,
        budget=4,
        sampler=SamplerSpec(seed=5, n_startup_trials=2),
        pruner=PrunerSpec(kind="none"),
        comparison=(ComparisonPoint("anchor", ANCHOR),),
        development=DevelopmentScenarios(((0.0, 0.0),)),
    )
    protocol_file = base / "esn_search_fixture.toml"
    protocol_file.write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")
    return store, records, processed, protocol, protocol_file


def test_study_is_mirrored_as_parent_and_child_runs_and_resumes(
    prepared: tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path], tmp_path: Path
) -> None:
    """Two invocations reach the budget; every trial is a traceable child run with its components stored."""
    store, records, processed, protocol, protocol_file = prepared
    tracker = MlflowTracker(store)

    def run(max_trials: int | None = None) -> EsnStudyResult:
        return run_esn_study(
            protocol,
            protocol_file,
            store=store,
            dataset_file=processed.record_file,
            records_root=records,
            exploratory=True,
            max_trials=max_trials,
            tracker=tracker,
            now=FIXED_TIME,
        )

    first = run(max_trials=2)
    assert first.report.trials_run == 2
    assert len(first.report.summary.trials) == 2
    assert first.report.protocol_sha256 == first.report.summary.identity["armrc.protocol_sha256"]
    assert sorted(first.child_runs) == [0, 1]
    parent = first.report.mlflow_parent_run
    assert parent is not None
    experiment = tracker.experiment_id(protocol.name)
    assert tracker.find_by_tags(experiment, {STUDY_TAG: protocol.name, "armrc.kind": "study"}) == parent

    params = tracker.params(parent)
    assert params["protocol.budget"] == "4"
    assert params["protocol.sha256"] == first.report.protocol_sha256
    assert params["dataset.artifact_id"] == processed.record.artifact.artifact_id
    assert params["tracker.sha256"] == first.report.tracker_sha256
    assert params["protocol.search.alpha.low"] == "0.001"
    assert any(key.startswith("revisions.") for key in params)
    assert params["provenance.project_commit"] == first.report.provenance.project_commit
    assert set(tracker.logged_artifacts(parent)) >= {"protocol.toml", "provenance.json"}

    child = first.child_runs[0]
    tags = tracker.tags(child)
    assert tags["mlflow.parentRunId"] == parent
    assert tags[TRIAL_TAG] == "0"
    assert tags["armrc.comparison"] == "anchor"
    assert tags["armrc.state"] == "COMPLETE"
    child_params = tracker.params(child)
    assert child_params["point.alpha"] == "0.01"
    assert child_params["point.n_neurons"] == "100"
    metrics = tracker.metrics(child)
    evaluation = first.evaluations[0]
    assert metrics["objective"] == evaluation.objective
    assert metrics["feasible"] == float(evaluation.feasible)
    assert metrics["scenarios_evaluated"] == 1.0
    component = evaluation.components[0]
    assert metrics["component.0.feasible"] == float(component.feasible)
    if component.move_joint_rmse is not None:
        assert metrics["component.0.move_joint_rmse"] == component.move_joint_rmse
    assert metrics["component.0.criteria.completed"] == float(component.criteria["completed"])
    artifacts = tracker.logged_artifacts(child)
    assert artifacts == ["evaluation.json"]
    downloaded = Path(tracker.client.download_artifacts(child, "evaluation.json", str(tmp_path / "dl")))
    stored = from_mapping(json.loads(downloaded.read_text()), TrialEvaluation)
    assert stored == evaluation
    history = tracker.client.get_metric_history(child, "objective_running")
    assert [float(cast("float", m.value)) for m in history] == list(evaluation.running)

    second = run()
    assert second.report.trials_run == 2
    assert len(second.report.summary.trials) == 4
    assert second.report.mlflow_parent_run == parent
    assert sorted(second.child_runs) == [2, 3]
    for number in range(4):
        hits = tracker.client.search_runs([experiment], filter_string=f"tags.`{TRIAL_TAG}` = '{number}'")
        assert len(hits) == 1, number
    assert "study_summary.json" in tracker.logged_artifacts(parent)
    assert "report.json" in tracker.logged_artifacts(parent)
    assert tracker.metrics(parent)["n_complete"] == 4.0
    assert tracker.metrics(parent)["trials_run"] == 2.0
    assert second.report.best_point is not None
    best = second.report.summary.best_number
    assert best is not None
    assert second.report.best_point.params() == second.study.trials[best].params
    assert second.report.n_feasible == sum(1 for t in second.report.summary.trials if t.flags["feasible"])

    third = run()
    assert third.report.trials_run == 0
    assert third.child_runs == {}
    assert third.evaluations == ()
    report_file = tmp_path / "report.json"
    report_file.write_text(report_to_json(third.report) + "\n", encoding="utf-8")
    assert load_report(report_file) == third.report


def test_command_runs_a_bounded_number_of_trials_and_writes_the_report(
    prepared: tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The command runs the study from a protocol file, honours --max-trials, and refuses to overwrite reports."""
    store, records, processed, _, _ = prepared
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    text = PROTOCOL.read_text(encoding="utf-8")
    text = re.sub(r"^name = .*$", 'name = "esn-search-cli"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^scenario = .*$", f'scenario = "{SCENARIO}"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^model = .*$", f'model = "{MODEL}"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^budget = .*$", "budget = 6", text, count=1, flags=re.MULTILINE)
    protocol_file = tmp_path / "esn_search_cli.toml"
    protocol_file.write_text(text, encoding="utf-8")
    report = tmp_path / "esn_search.json"
    argv = [
        "--protocol",
        str(protocol_file),
        "--dataset",
        str(processed.record_file),
        "--report",
        str(report),
        "--records-root",
        str(records),
        "--exploratory",
        "--no-mlflow",
        "--max-trials",
        "1",
    ]
    assert main([*argv, "--markdown", str(tmp_path / "esn_search.md")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["trials_run"] == 1
    assert printed["trials_stored"] == 1
    assert printed["budget"] == 6
    assert printed["mlflow_parent_run"] is None
    loaded = load_report(report)
    assert (tmp_path / "esn_search.md").read_text(encoding="utf-8").startswith("# ESN search `esn-search-cli`")
    assert loaded.protocol == "esn-search-cli"
    assert loaded.summary.trials[0].labels["armrc.comparison"] == "anchor-alpha-1e-2"
    assert loaded.provenance.exploratory
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)
    with pytest.raises(ValueError, match="max-trials"):
        main([*argv[:-2], "--max-trials", "0", "--report", str(tmp_path / "other.json")])
