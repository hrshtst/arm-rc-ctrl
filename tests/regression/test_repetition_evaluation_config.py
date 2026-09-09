# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-004: the versioned pilot evaluation config is the locked development envelope plus the approved abort."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.perturbations import load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.repetition_evaluation import load_evaluation_config
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
EVALUATION = REPO_ROOT / "configs" / "evaluations" / "task_1a_repetition_dev_v1.toml"
MANIFEST = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration" / "panel_manifest_v1.json"


def test_pilot_evaluation_config_binds_the_locked_levels_and_doubles_the_abort() -> None:
    """The config names the development file the panel manifest bound and relaxes only the simulation abort."""
    config = load_evaluation_config(EVALUATION)
    manifest = load_panel(MANIFEST)
    assert config.name == "task-1a-repetition-dev-v1"
    assert config.development == (REPO_ROOT / manifest.configs.development_file).resolve()
    assert sha256_file(config.development) == manifest.configs.development_sha256
    levels = load_development_robustness(config.development)
    scenario = load_scenario(REPO_ROOT / manifest.configs.scenario_file)
    assert config.simulation.velocity_abort == tuple(2.0 * v for v in scenario.limits.velocity) == (12.0, 12.0)
    lower = tuple(link.q_min for link in scenario.robot.links)
    upper = tuple(link.q_max for link in scenario.robot.links)
    scenarios = robustness_scenarios(levels, nominal=scenario.task.initial_q, lower=lower, upper=upper)
    assert len(scenarios) == 65
    assert [s.kind for s in scenarios].count("nominal") == 1
    assert [s.kind for s in scenarios].count("force") == 4
    # The scenario file (training validation, augmentation, dataset binding) keeps its own 6 rad/s limit.
    assert scenario.limits.velocity == (6.0, 6.0)
