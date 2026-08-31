# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3-010: the confirmatory suite of task 1-a ran exactly once, on the frozen recipe, under the locked protocol."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.closed_loop import load_nominal_config
from arm_rc_ctrl.experiments.confirmatory import load_confirmatory
from arm_rc_ctrl.experiments.perturbations import robustness_scenarios
from arm_rc_ctrl.experiments.robustness import load_suite
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
CONFIRMATORY = REPO_ROOT / "configs" / "evaluations" / "task_1a_confirmatory_v2.toml"
REPORT = DOCS / "robustness_confirmatory_v2_recipe_v4.json"
MODEL_V4 = REPO_ROOT / "configs" / "models" / "esn_task_1a_v4.toml"
EVALUATION_V4 = REPO_ROOT / "configs" / "evaluations" / "task_1a_nominal_v4.toml"
RECIPE_V4 = REPO_ROOT / "data" / "records" / "models" / "model-20260831-1b9477aaa246.toml"


def test_exactly_one_confirmatory_suite_exists_for_the_study_version() -> None:
    """One report carries the confirmatory label; any rerun must be labelled separately."""
    suites = [load_suite(f) for f in sorted(DOCS.glob("robustness_*.json"))]
    confirmatory = [s for s in suites if s.label == "confirmatory"]
    assert len(confirmatory) == 1
    assert confirmatory[0] == load_suite(REPORT)
    assert all(s.label in ("development", "confirmatory-rerun") for s in suites if s is not confirmatory[0])


def test_confirmatory_suite_used_the_locked_protocol_and_the_frozen_recipe() -> None:
    """Protocol, seeds, scenarios, recipe, and estimator are the locked/frozen ones; the run was clean."""
    protocol = load_confirmatory(CONFIRMATORY)
    suite = load_suite(REPORT)
    assert suite.label == "confirmatory"
    assert suite.protocol == protocol.name
    assert suite.protocol_file == "configs/evaluations/task_1a_confirmatory_v2.toml"
    assert suite.provenance.project_dirty is False
    assert suite.provenance.exploratory is False
    assert set(suite.provenance.seeds.values()) == set(protocol.seeds)
    scenario = load_scenario(protocol.scenario)
    lower = [link.q_min for link in scenario.robot.links]
    upper = [link.q_max for link in scenario.robot.links]
    expected = robustness_scenarios(protocol, nominal=scenario.task.initial_q, lower=lower, upper=upper)
    assert suite.scenarios == expected
    assert suite.scenario == scenario.name
    recipe = load_recipe(RECIPE_V4)
    assert suite.recipe == recipe.name == "esn-task-1a-v4"
    assert suite.recipe_file == "data/records/models/model-20260831-1b9477aaa246.toml"
    assert recipe.esn == load_model_config(MODEL_V4).esn
    assert suite.estimator == load_nominal_config(EVALUATION_V4).estimator
    assert {a.tracker for a in suite.arms} == {"pd_v2", "computed_torque"}
    assert len(suite.runs) == len(suite.arms) * len(suite.scenarios)
    assert all(run.pointer and run.mlflow_run_id for run in suite.runs)
