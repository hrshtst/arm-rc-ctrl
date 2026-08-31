# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Freezing a study's selected point into versioned configurations and rendering the selection report (M3-006)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec, load_nominal_config
from arm_rc_ctrl.experiments.esn_freeze import (
    frozen_evaluation,
    frozen_model,
    main,
    render_evaluation_toml,
    render_model_toml,
)
from arm_rc_ctrl.experiments.esn_search import EsnSearchProtocol, TrialPoint, load_esn_search, protocol_digest
from arm_rc_ctrl.experiments.esn_study import EsnStudyReport, render_markdown, report_to_json
from arm_rc_ctrl.experiments.studies import StudySummary, TrialRecord
from arm_rc_ctrl.provenance import collect_provenance
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
FIXED_TIME = datetime(2026, 9, 3, 12, 0, 0, tzinfo=UTC)
BEST = TrialPoint(350, 1.05, 0.8, 0.2, 0.7, 77, 0.08, 15.0, 25.0)


def trial(
    number: int, value: float, point: TrialPoint, *, feasible: bool, reason: str = "", label: str | None = None
) -> TrialRecord:
    """A stored trial with one development component."""
    labels = {"reason": reason, "components.0.kind": "posture"}
    if label is not None:
        labels["armrc.comparison"] = label
    return TrialRecord(
        number=number,
        state="COMPLETE",
        value=value,
        params={k: float(v) for k, v in point.params().items()},
        metrics={"fit_rmse": 1e-4, "components.0.move_joint_rmse": value, "components.0.saturation_fraction": 0.0},
        flags={"feasible": feasible, "components.0.feasible": feasible, "components.0.criteria.completed": True},
        labels=labels,
    )


@pytest.fixture(scope="module")
def protocol() -> EsnSearchProtocol:
    """The committed protocol with a budget of three."""
    return replace(load_esn_search(PROTOCOL), budget=3, comparison=())


@pytest.fixture(scope="module")
def report(protocol: EsnSearchProtocol) -> EsnStudyReport:
    """A finished three-trial study selecting trial 1."""
    anchor = TrialPoint(200, 0.9, 0.9, 0.3, 0.5, 31, 1e-2, 20.0, 20.0)
    trials = (
        trial(0, 0.2, anchor, feasible=True, label="anchor"),
        trial(1, 0.05, BEST, feasible=True),
        trial(2, 10.0, replace(BEST, seed=78), feasible=False, reason="scenario 0: limit_violation:torque"),
    )
    summary = StudySummary(
        name=protocol.name,
        storage=f"armrc://optuna/{protocol.name}.db",
        direction="minimize",
        identity={"armrc.protocol_sha256": protocol_digest(protocol)},
        trials=trials,
        n_complete=3,
        n_pruned=0,
        best_number=1,
        best_value=0.05,
    )
    provenance = collect_provenance(
        {"protocol": protocol.name}, seeds={"sampler": 5}, artifacts=[], exploratory=True, now=FIXED_TIME
    )
    return EsnStudyReport(
        protocol=protocol.name,
        protocol_file="configs/studies/esn_search_1a.toml",
        protocol_sha256=protocol_digest(protocol),
        dataset="processed-20260830-feaf73e6663c",
        tracker=protocol.tracker,
        tracker_sha256="0" * 64,
        budget=3,
        trials_run=3,
        summary=summary,
        best_point=BEST,
        n_feasible=2,
        provenance=provenance,
    )


def test_frozen_model_applies_the_selected_point(report: EsnStudyReport, protocol: EsnSearchProtocol) -> None:
    """Only the tuned reservoir/readout fields change; the transform and solver come from the base model."""
    base = protocol.base_model()
    model = frozen_model(report, protocol, name="esn-task-1a-v3")
    assert model == BEST.model_config(base, name="esn-task-1a-v3")
    assert model.input_transform == base.input_transform
    assert model.esn.readout.alpha == 0.08
    assert model.esn.reservoir.n_neurons == 350
    evaluation = frozen_evaluation(
        report, protocol, name="task-1a-nominal-v3", tracker_file=Path("../controllers/x.toml")
    )
    assert evaluation.estimator == EstimatorSpec(15.0, 25.0, protocol.max_dt_ratio)
    assert evaluation.tracker == Path("../controllers/x.toml")


def test_freeze_refuses_foreign_unfinished_or_empty_reports(
    report: EsnStudyReport, protocol: EsnSearchProtocol
) -> None:
    """The report must come from the protocol, hold the whole budget, and select a trial."""
    with pytest.raises(ValueError, match="was not produced by protocol"):
        frozen_model(report, replace(protocol, budget=4), name="x")
    with pytest.raises(ValueError, match="selects no trial"):
        frozen_model(replace(report, best_point=None), protocol, name="x")
    partial = replace(report, summary=replace(report.summary, trials=report.summary.trials[:2], n_complete=2))
    with pytest.raises(ValueError, match="finish it before freezing"):
        frozen_model(partial, protocol, name="x")


def test_rendered_configurations_load_back(report: EsnStudyReport, protocol: EsnSearchProtocol, tmp_path: Path) -> None:
    """The TOML files cite the study and parse into the frozen configurations."""
    model_text = render_model_toml(
        report, protocol, name="esn-task-1a-v3", report_file="docs/experiments/task_1a/esn_search.json"
    )
    assert model_text.startswith("# Task ESN model configuration (docs/TASKS.md M3-006)")
    assert "docs/experiments/task_1a/esn_search.json" in model_text
    assert report.protocol_sha256[:12] in model_text
    model_file = tmp_path / "configs" / "models" / "esn_task_1a_v3.toml"
    model_file.parent.mkdir(parents=True)
    model_file.write_text(model_text, encoding="utf-8")
    assert load_model_config(model_file) == frozen_model(report, protocol, name="esn-task-1a-v3")
    tracker = Path("../controllers/task_1a_pd_v2.toml")
    evaluation_text = render_evaluation_toml(
        report,
        protocol,
        name="task-1a-nominal-v3",
        tracker_file=tracker,
        report_file="docs/experiments/task_1a/esn_search.json",
    )
    evaluation_file = tmp_path / "configs" / "evaluations" / "task_1a_nominal_v3.toml"
    evaluation_file.parent.mkdir(parents=True)
    evaluation_file.write_text(evaluation_text, encoding="utf-8")
    loaded = load_nominal_config(evaluation_file)
    assert loaded.name == "task-1a-nominal-v3"
    assert loaded.estimator == EstimatorSpec(15.0, 25.0, protocol.max_dt_ratio)
    assert loaded.tracker == (tmp_path / "configs" / "controllers" / "task_1a_pd_v2.toml").resolve()


def test_markdown_report_lists_selection_comparisons_and_reasons(report: EsnStudyReport) -> None:
    """The Markdown report records the budget, failures, the chosen trial, its point, and development metrics."""
    text = render_markdown(report)
    assert text.startswith("# ESN search `esn-search-1a`")
    assert "Budget 3; stored 3 trials (3 complete, 0 pruned)" in text
    assert "2 feasible" in text
    assert "Trial 1 with objective 0.05 rad" in text
    assert "| alpha | 0.08 |" in text
    assert "| 0 | posture | 0.05 | 0 | completed=ok |" in text
    assert "| anchor | 0 | 0.2 | True |  |" in text
    assert "| limit_violation:torque | 1 |" in text
    assert "| 1 | 0.05 | 350 |" in text


def test_command_writes_both_files_once(
    report: EsnStudyReport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command freezes from the report/protocol pair and refuses to overwrite."""
    report_file = tmp_path / "esn_search.json"
    report_file.write_text(report_to_json(report) + "\n", encoding="utf-8")
    protocol_file = tmp_path / "esn_search_1a.toml"
    protocol_file.write_text(
        PROTOCOL.read_text(encoding="utf-8")
        .replace("budget = 120", "budget = 3")
        .replace(
            'scenario = "../tasks/task_1a.toml"', f'scenario = "{REPO_ROOT / "configs" / "tasks" / "task_1a.toml"}"'
        )
        .replace(
            'model = "../models/esn_task_1a_v2.toml"',
            f'model = "{REPO_ROOT / "configs" / "models" / "esn_task_1a_v2.toml"}"',
        ),
        encoding="utf-8",
    )
    stripped = protocol_file.read_text(encoding="utf-8")
    stripped = stripped[: stripped.index("[[comparison]]")] + stripped[stripped.index("[development]") :]
    protocol_file.write_text(stripped, encoding="utf-8")
    model = tmp_path / "esn_task_1a_v3.toml"
    evaluation = tmp_path / "task_1a_nominal_v3.toml"
    argv = [
        "--report",
        str(report_file),
        "--protocol",
        str(protocol_file),
        "--name",
        "esn-task-1a-v3",
        "--model",
        str(model),
        "--evaluation",
        str(evaluation),
    ]
    assert main(argv) == 0
    assert "trial 1" in capsys.readouterr().out
    assert load_model_config(model).esn.reservoir.seed == 77
    assert load_nominal_config(evaluation).name == "task-1a-nominal-v3"
    assert 'tracker = "../controllers/task_1a_pd_v2.toml"' in evaluation.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)
