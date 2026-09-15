# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-002 (I6): the manual-demonstration task configuration copies task 1-a without its fixed intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError, from_mapping, load_config, to_mapping
from arm_rc_ctrl.data.manual_scenario import (
    ManualScenarioConfig,
    load_manual_scenario,
    manual_build_skeleton,
    manual_endpoint_positions,
    manual_joint_limits,
)
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

REPO_ROOT = repository_root()
TASK_1A = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
MANUAL_V1 = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v1.toml"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_manual_fixture.toml"


def test_manual_v1_copies_the_robot_limits_and_target_of_task_1a() -> None:
    """Geometry, limits, start posture, target, and tolerance are the historical values; only the rules differ."""
    manual = load_manual_scenario(MANUAL_V1)
    legacy = load_scenario(TASK_1A)
    assert manual.robot == legacy.robot
    assert manual.limits == legacy.limits
    assert manual.task.initial_q == legacy.task.initial_q
    assert manual.task.target == legacy.task.target
    assert manual.task.tolerance == legacy.task.tolerance
    assert manual.task.dwell_max_velocity == legacy.task.dwell_max_velocity
    assert manual.timing.dt == legacy.timing.dt
    assert manual.protocol == "task_1a_manual_v1"
    assert manual.dof == 2


def test_manual_v1_declares_the_continuous_dwell_rule_and_no_fixed_intervals() -> None:
    """One second inside 1 cm at 0.05 rad/s is 101 consecutive samples at 100 Hz; intervals do not exist."""
    manual = load_manual_scenario(MANUAL_V1)
    assert manual.task.dwell_min_duration_s == 1.0
    assert manual.dwell_min_samples == 101
    assert not hasattr(manual.timing, "intervals")
    assert not hasattr(manual.task, "dwell_min_fraction")
    rules = manual.acquisition
    assert rules.velocity_bound_rad_s == 6.0  # D3: the canonical bound until the pilot tightens it
    assert rules.start_tolerance_rad > 0
    assert rules.max_sample_gap_s > manual.timing.dt
    assert rules.min_frames >= 2


def test_task_1a_toml_is_untouched_by_the_new_configuration() -> None:
    """The historical task file still loads with its old semantics and keeps the digest bound into the evidence."""
    legacy = load_scenario(TASK_1A)
    assert legacy.task.dwell_min_fraction == 0.9
    assert legacy.timing.intervals.prime == (0.0, 1.0)
    record_dir = REPO_ROOT / "data" / "records" / "processed"
    digests: set[str] = set()
    for path in sorted(record_dir.glob("processed-*.toml")):
        try:
            record = load_record(path, ProcessedDatasetRecord)
        except ConfigError:
            record = load_record(path, RecoveryDatasetRecord)
        if record.scenario.config_path == "configs/tasks/task_1a.toml":
            digests.add(record.scenario.config_sha256)
    assert digests == {sha256_file(TASK_1A)}


def test_legacy_keys_and_inconsistent_rules_are_rejected() -> None:
    """The strict schema refuses the legacy dwell fraction and intervals and rules that contradict the limits."""
    data = cast("dict[str, Any]", to_mapping(load_manual_scenario(FIXTURE)))

    def variant(section: str, **changes: object) -> dict[str, Any]:
        table = cast("dict[str, Any]", data[section])
        return {**data, section: {**table, **changes}}

    with pytest.raises(ConfigError, match="dwell_min_fraction"):
        from_mapping(variant("task", dwell_min_fraction=0.9), ManualScenarioConfig)
    with pytest.raises(ConfigError, match="intervals"):
        from_mapping(variant("timing", intervals={"prime": [0.0, 1.0]}), ManualScenarioConfig)
    with pytest.raises(ConfigError, match="velocity_bound_rad_s"):
        from_mapping(variant("acquisition", velocity_bound_rad_s=100.0), ManualScenarioConfig)
    with pytest.raises(ConfigError, match="whole number of samples"):
        from_mapping(variant("task", dwell_min_duration_s=0.105), ManualScenarioConfig)


def test_helpers_pose_the_skeleton_and_compute_the_endpoint(tmp_path: Path) -> None:  # noqa: ARG001
    """The limits, the posed skeleton, and forward kinematics come from the shared robot description."""
    manual = load_manual_scenario(MANUAL_V1)
    limits = manual_joint_limits(manual)
    assert limits.lower == (-3.0, -3.0)
    assert limits.speed == (6.0, 6.0)
    skeleton = manual_build_skeleton(manual)
    assert np.allclose(skeleton.q, manual.task.initial_q)
    tips = manual_endpoint_positions(manual, np.asarray([manual.task.initial_q], dtype=np.float64))
    assert tips.shape == (1, 2)
    assert np.allclose(tips[0], (skeleton.links[-1].xe, skeleton.links[-1].ye))
    assert np.allclose(tips[0], (0.34, 0.31), atol=0.01)


def test_fixture_configuration_loads_with_the_manual_schema() -> None:
    """The planar fixture of the manual protocol is a valid, self-consistent configuration."""
    fixture = load_config(FIXTURE, ManualScenarioConfig)
    assert fixture.dof == 2
    assert fixture.dwell_min_samples == round(fixture.task.dwell_min_duration_s / fixture.timing.dt) + 1
