# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3-006/M3-016: the frozen v3/v4 configurations adopt the recorded ESN searches' feasible selections."""

from __future__ import annotations

import json

import pytest

from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_catalog
from arm_rc_ctrl.data.recovery import load_processed_record
from arm_rc_ctrl.experiments.closed_loop import load_nominal_config
from arm_rc_ctrl.experiments.esn_freeze import frozen_evaluation, frozen_model
from arm_rc_ctrl.experiments.esn_search import load_esn_search, point_from_params, protocol_digest
from arm_rc_ctrl.experiments.esn_stability import leading_trials, load_stability
from arm_rc_ctrl.experiments.esn_stability import render_markdown as stability_markdown
from arm_rc_ctrl.experiments.esn_study import load_report, render_markdown
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
REPORT = REPO_ROOT / "docs" / "experiments" / "task_1a" / "esn_search.json"
MARKDOWN = REPO_ROOT / "docs" / "experiments" / "task_1a" / "esn_search.md"
MODEL_V3 = REPO_ROOT / "configs" / "models" / "esn_task_1a_v3.toml"
EVALUATION_V3 = REPO_ROOT / "configs" / "evaluations" / "task_1a_nominal_v3.toml"
TRAINING_V3 = REPO_ROOT / "docs" / "experiments" / "task_1a" / "training_v3.json"
CONFIRMATORY_SEEDS = {20260901, 20260902, 20260903, 20260904, 20260905}


def test_study_ran_the_committed_protocol_to_its_budget_from_a_clean_checkout() -> None:
    """The report belongs to the committed protocol, holds every budgeted trial, and is development-grade."""
    protocol = load_esn_search(PROTOCOL)
    report = load_report(REPORT)
    assert report.protocol == protocol.name
    assert report.protocol_sha256 == protocol_digest(protocol)
    assert report.budget == protocol.budget
    assert len(report.summary.trials) == protocol.budget
    assert report.summary.n_complete + report.summary.n_pruned == protocol.budget
    assert report.provenance.project_dirty is False
    assert not CONFIRMATORY_SEEDS & set(report.provenance.seeds.values())
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.toml")
    processed = [load_processed_record(REPO_ROOT / e.record) for e in catalog.artifacts if e.kind == "processed"]
    (dataset,) = [
        r
        for r in processed
        if isinstance(r, ProcessedDatasetRecord) and r.artifact.origin.sources == ("raw-20260830-b5adde395f1c",)
    ]
    assert report.dataset == dataset.artifact.artifact_id
    labels = {t.labels.get("armrc.comparison") for t in report.summary.trials}
    assert {c.label for c in protocol.comparison} <= labels
    assert render_markdown(report) == MARKDOWN.read_text(encoding="utf-8")


def test_frozen_configurations_adopt_the_selected_trial() -> None:
    """The v3 model and evaluation configurations equal the freeze of the report's best trial."""
    protocol = load_esn_search(PROTOCOL)
    report = load_report(REPORT)
    assert report.best_point is not None
    best = next(t for t in report.summary.trials if t.number == report.summary.best_number)
    assert best.flags["feasible"] is True
    assert report.best_point == point_from_params(protocol.search, best.params)
    model = load_model_config(MODEL_V3)
    assert model == frozen_model(report, protocol, name="esn-task-1a-v3")
    assert model.input_transform == protocol.base_model().input_transform
    evaluation = load_nominal_config(EVALUATION_V3)
    expected = frozen_evaluation(report, protocol, name="task-1a-nominal-v3", tracker_file=evaluation.tracker)
    assert evaluation == expected
    assert evaluation.tracker.name == f"task_1a_{protocol.tracker}.toml"


def test_recipe_v3_was_trained_from_the_frozen_configuration() -> None:
    """The recorded training report names the v3 configuration and the curated dataset, from a clean checkout."""
    report = load_report(REPORT)
    training = json.loads(TRAINING_V3.read_text(encoding="utf-8"))
    assert training["model_config"] == "configs/models/esn_task_1a_v3.toml"
    assert training["datasets"] == [report.dataset]
    assert training["refit_verified"] is True
    assert training["provenance"]["project_dirty"] is False
    recipe = load_recipe(REPO_ROOT / training["recipe_file"])
    assert recipe.name == "esn-task-1a-v3"
    assert recipe.esn == load_model_config(MODEL_V3).esn
    assert [source.artifact_id for source in recipe.datasets] == [report.dataset]
    assert training["fit"]["rmse"] == recipe.fit.rmse


V2_PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a_v2.toml"
V2_REPORT = REPO_ROOT / "docs" / "experiments" / "task_1a" / "esn_search_v2.json"
V2_MARKDOWN = REPO_ROOT / "docs" / "experiments" / "task_1a" / "esn_search_v2.md"
STABILITY_V2 = REPO_ROOT / "docs" / "experiments" / "task_1a" / "esn_stability_v2.json"
STABILITY_V2_MARKDOWN = REPO_ROOT / "docs" / "experiments" / "task_1a" / "esn_stability_v2.md"
MODEL_V4 = REPO_ROOT / "configs" / "models" / "esn_task_1a_v4.toml"
EVALUATION_V4 = REPO_ROOT / "configs" / "evaluations" / "task_1a_nominal_v4.toml"
TRAINING_V4 = REPO_ROOT / "docs" / "experiments" / "task_1a" / "training_v4.json"


def test_search_v2_ran_to_its_budget_from_a_clean_checkout_and_selected_a_feasible_trial() -> None:
    """The v2 report belongs to the committed v2 protocol, holds every trial, and selects a feasible one."""
    protocol = load_esn_search(V2_PROTOCOL)
    report = load_report(V2_REPORT)
    assert report.protocol == protocol.name
    assert report.protocol_sha256 == protocol_digest(protocol)
    assert len(report.summary.trials) == protocol.budget == report.budget
    assert report.summary.n_pruned == 0
    assert report.summary.selection_rule == "feasible"
    assert report.provenance.project_dirty is False
    assert not CONFIRMATORY_SEEDS & set(report.provenance.seeds.values())
    assert report.dataset == load_report(REPORT).dataset
    labels = {t.labels.get("armrc.comparison") for t in report.summary.trials}
    assert {c.label for c in protocol.comparison} <= labels
    best = next(t for t in report.summary.trials if t.number == report.summary.best_number)
    assert best.flags["feasible"] is True
    assert report.best_point == point_from_params(protocol.search, best.params)
    assert render_markdown(report) == V2_MARKDOWN.read_text(encoding="utf-8")


def test_stability_panel_binds_to_search_v2_and_its_leading_trials() -> None:
    """The committed panel re-evaluated the leading feasible v2 trials with its recorded seeds from a clean checkout."""
    protocol = load_esn_search(V2_PROTOCOL)
    report = load_report(V2_REPORT)
    panel = load_stability(STABILITY_V2)
    assert panel.protocol == protocol.name
    assert panel.protocol_sha256 == report.protocol_sha256
    assert panel.study_report == "docs/experiments/task_1a/esn_search_v2.json"
    assert panel.dataset == report.dataset
    assert panel.provenance.project_dirty is False
    assert [c.trial for c in panel.configurations] == leading_trials(report, len(panel.configurations))
    assert panel.configurations[0].trial == report.summary.best_number
    for configuration in panel.configurations:
        assert [o.seed for o in configuration.outcomes] == list(panel.panel_seeds)
        assert configuration.point == point_from_params(
            protocol.search, next(t for t in report.summary.trials if t.number == configuration.trial).params
        )
    assert stability_markdown(panel) == STABILITY_V2_MARKDOWN.read_text(encoding="utf-8")


def test_frozen_v4_configurations_adopt_search_v2_selection() -> None:
    """The v4 model and evaluation configurations equal the freeze of the v2 report's best trial."""
    protocol = load_esn_search(V2_PROTOCOL)
    report = load_report(V2_REPORT)
    model = load_model_config(MODEL_V4)
    assert model == frozen_model(report, protocol, name="esn-task-1a-v4")
    evaluation = load_nominal_config(EVALUATION_V4)
    assert evaluation == frozen_evaluation(report, protocol, name="task-1a-nominal-v4", tracker_file=evaluation.tracker)
    assert evaluation.tracker.name == f"task_1a_{protocol.tracker}.toml"


def test_recipe_v4_was_trained_from_the_frozen_configuration() -> None:
    """The recorded v4 training report names the v4 configuration and the curated dataset, from a clean checkout."""
    report = load_report(V2_REPORT)
    training = json.loads(TRAINING_V4.read_text(encoding="utf-8"))
    assert training["model_config"] == "configs/models/esn_task_1a_v4.toml"
    assert training["datasets"] == [report.dataset]
    assert training["refit_verified"] is True
    assert training["provenance"]["project_dirty"] is False
    recipe = load_recipe(REPO_ROOT / training["recipe_file"])
    assert recipe.name == "esn-task-1a-v4"
    assert recipe.esn == load_model_config(MODEL_V4).esn
    assert [source.artifact_id for source in recipe.datasets] == [report.dataset]
