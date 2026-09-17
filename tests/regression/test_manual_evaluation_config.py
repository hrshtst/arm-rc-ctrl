# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the committed manual evaluation configuration binds the locked draws and the study's task.

The revised horizon, dwell rule, and force timing need their own configuration
identity: historical scenario identities and replay caches cannot be reused
(plan section 6). What is reused is the locked development envelope itself --
the same posture draws and force directions, by reference, so this pilot adds
no seed of its own and stays a development diagnostic rather than a new
held-out test.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest, frozen_baseline_file, load_frozen_baseline
from arm_rc_ctrl.experiments.manual_evaluation import load_manual_evaluation_config
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.experiments.perturbations import load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
EVALUATION = REPO_ROOT / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"
DEVELOPMENT = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
STUDY = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration" / "study_manifest_v1.json"

HORIZON_S = 30.0
"""The D2-approved common evaluation horizon from activation, for every arm and scenario."""

TRIGGER_HOLD_S = 0.5
"""Continuous target dwell that arms the pulse (plan section 6)."""


def test_committed_configuration_binds_the_locked_draws_and_the_manual_task() -> None:
    """It names the locked development levels and the very task configuration the frozen study bound."""
    config = load_manual_evaluation_config(EVALUATION)
    assert config.name == "task-1a-manual-dev-v1"
    assert config.development == DEVELOPMENT.resolve()
    assert sha256_file(config.development) == sha256_file(DEVELOPMENT)
    study = load_study(STUDY)
    assert config.scenario == (REPO_ROOT / study.scenario.path).resolve()
    assert sha256_file(config.scenario) == study.scenario.sha256


def test_committed_configuration_keeps_the_canonical_abort_not_the_repetition_relaxation() -> None:
    """D3: 6 rad/s per joint for all evaluation and training validation; 12 rad/s does not carry over."""
    config = load_manual_evaluation_config(EVALUATION)
    scenario = load_manual_scenario(config.scenario)
    assert config.simulation.velocity_abort == tuple(scenario.limits.velocity) == (6.0, 6.0)
    assert config.simulation.velocity_abort != (12.0, 12.0)
    assert tuple(scenario.limits.torque) == (10.0, 5.0)
    assert all(link.q_min == -3.0 and link.q_max == 3.0 for link in scenario.robot.links)


def test_committed_configuration_states_the_revised_horizon_and_force_timing() -> None:
    """The horizon, the trigger hold, and the pulse it fires, with room for the final dwell after it."""
    config = load_manual_evaluation_config(EVALUATION)
    scenario = load_manual_scenario(config.scenario)
    levels = load_development_robustness(config.development)
    assert config.horizon_s == HORIZON_S
    assert config.trigger.hold_s == TRIGGER_HOLD_S
    # The pulse itself is inherited unchanged; only its timing rule is new.
    assert config.trigger.magnitude_n == levels.force.magnitude_n == 12.0
    assert config.trigger.duration_s == levels.force.duration_s == 0.2
    # The inherited fixed start is superseded: a state-triggered pulse has no task-clock start.
    assert levels.force.start_s == 1.0
    room = config.horizon_s - (config.trigger.hold_s + config.trigger.duration_s)
    assert room > scenario.task.dwell_min_duration_s


def test_committed_configuration_yields_the_approved_sixty_five_cases() -> None:
    """The locked draws transfer to the manual reset posture unchanged, in the approved allocation."""
    config = load_manual_evaluation_config(EVALUATION)
    scenario = load_manual_scenario(config.scenario)
    levels = load_development_robustness(config.development)
    lower = tuple(link.q_min for link in scenario.robot.links)
    upper = tuple(link.q_max for link in scenario.robot.links)
    scenarios = robustness_scenarios(levels, nominal=scenario.task.initial_q, lower=lower, upper=upper)
    kinds = [s.kind for s in scenarios]
    assert len(scenarios) == 65
    assert kinds.count("nominal") == 1
    assert kinds.count("posture_small") == 20
    assert kinds.count("posture_large") == 20
    assert kinds.count("force") == 4
    assert kinds.count("combined") == 20
    assert len({s.scenario_id for s in scenarios}) == 65
    directions = sorted({s.direction_deg for s in scenarios if s.direction_deg is not None})
    assert directions == [0.0, 90.0, 180.0, 270.0]


def test_both_frozen_trackers_are_evaluated_and_bound_by_their_values() -> None:
    """Both frozen trackers run every scenario, and each is bound by its gains rather than by its file.

    The digest is the canonical JSON of the loaded gains, not the bytes of the
    TOML: a comment or a reformatting of the gain file leaves it unchanged,
    while any change to a gain moves it. That is what makes it usable as part
    of a run identity, so the binding is asserted rather than assumed.
    """
    assert RECOVERY_TRACKERS == ("pd_v2", "computed_torque")
    digests: dict[str, str] = {}
    for name in RECOVERY_TRACKERS:
        file = frozen_baseline_file(name)
        assert file.is_file()
        gains = load_frozen_baseline(name)
        assert len(gains.kp) == len(gains.kd) == 2
        canonical = json.dumps(to_mapping(gains), sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert frozen_baseline_digest(name) == expected
        assert frozen_baseline_digest(name) != sha256_file(file)
        digests[name] = expected
    assert len(set(digests.values())) == len(RECOVERY_TRACKERS)
