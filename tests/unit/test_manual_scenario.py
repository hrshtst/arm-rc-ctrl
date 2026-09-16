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
from arm_rc_ctrl.data.recovery import load_processed_record
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

REPO_ROOT = repository_root()
TASK_1A = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
MANUAL_V1 = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v1.toml"
MANUAL_V2 = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v2.toml"
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
    assert rules.min_frames is not None
    assert rules.min_frames >= 2


def test_task_1a_toml_is_untouched_by_the_new_configuration() -> None:
    """The historical task file still loads with its old semantics and keeps the digest bound into the evidence.

    Every committed processed record is read through the dispatcher, so a manual-take dataset (which binds a
    versioned manual task file, never ``task_1a.toml``) neither breaks the sweep nor loosens the digest lock.
    """
    legacy = load_scenario(TASK_1A)
    assert legacy.task.dwell_min_fraction == 0.9
    assert legacy.timing.intervals.prime == (0.0, 1.0)
    record_dir = REPO_ROOT / "data" / "records" / "processed"
    digests: set[str] = set()
    other_tasks: set[str] = set()
    for path in sorted(record_dir.glob("processed-*.toml")):
        record = load_processed_record(path)
        if record.scenario.config_path == "configs/tasks/task_1a.toml":
            digests.add(record.scenario.config_sha256)
        else:
            other_tasks.add(record.scenario.config_path)
    assert digests == {sha256_file(TASK_1A)}
    assert other_tasks <= {"configs/tasks/task_1a_manual_v1.toml", "configs/tasks/task_1a_manual_v2.toml"}


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


def test_manual_v2_separates_the_acquisition_rate_from_the_training_grid() -> None:
    """M3MAN-013 (I13): 50 Hz acquisition with rules in actual time; the task and its 0.01 s grid are v1's."""
    v1 = load_manual_scenario(MANUAL_V1)
    v2 = load_manual_scenario(MANUAL_V2)
    assert (v2.name, v2.protocol, v2.robot, v2.limits, v2.task, v2.timing) == (
        v1.name,
        v1.protocol,
        v1.robot,
        v1.limits,
        v1.task,
        v1.timing,
    )
    assert v2.protocol == "task_1a_manual_v1"  # the experiment label
    assert v2.timing.dt == 0.01
    assert v2.dwell_min_samples == 101  # one actual second on the training grid
    rules = v2.acquisition
    assert rules.sample_rate_hz == 50.0
    assert v2.acquisition_period_s == 0.02
    assert rules.min_duration_s == 1.0
    assert rules.min_frames is None
    assert rules.max_sample_gap_s == 0.06  # provisional: three acquisition periods
    assert rules.start_tolerance_rad == v1.acquisition.start_tolerance_rad
    assert rules.velocity_bound_rad_s == 6.0
    assert v1.acquisition.sample_rate_hz is None
    assert v1.acquisition.min_duration_s is None
    assert v1.acquisition_period_s == v1.timing.dt  # v1: acquisition implied by the grid


def _variant(data: dict[str, Any], section: str, changes: dict[str, object]) -> dict[str, Any]:
    table = dict(cast("dict[str, object]", data[section]))
    table.update(changes)
    result = dict(data)
    result[section] = table
    return result


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        ("acquisition", {"min_frames": 101}, "mixes"),
        ("acquisition", {"min_duration_s": None}, "min_duration_s"),
        ("acquisition", {"sample_rate_hz": None, "min_duration_s": None}, "either"),
        ("acquisition", {"sample_rate_hz": 0.0}, "sample_rate_hz"),
        ("acquisition", {"min_duration_s": -1.0}, "min_duration_s"),
        ("acquisition", {"max_sample_gap_s": 0.01}, "acquisition period"),
        ("timing", {"dt": 0.05}, "at least as fine"),
    ],
)
def test_acquisition_rules_use_one_style_and_a_grid_at_least_as_fine_as_the_acquisition(
    section: str, changes: dict[str, object], message: str
) -> None:
    """v1 and v2 acquisition keys never mix; the gap limit spans an acquisition period; the grid is not coarser."""
    data = cast("dict[str, Any]", to_mapping(load_manual_scenario(MANUAL_V2)))
    assert from_mapping(data, ManualScenarioConfig) == load_manual_scenario(MANUAL_V2)
    with pytest.raises(ConfigError, match=message):
        from_mapping(_variant(data, section, changes), ManualScenarioConfig)
