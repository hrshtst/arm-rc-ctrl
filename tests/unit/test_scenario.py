# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-012: the canonical task 1-a scenario is versioned, self-consistent, and drives skelarm."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.data.records import Intervals
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import (
    LimitsConfig,
    LinkConfig,
    RobotConfig,
    ScenarioConfig,
    TaskConfig,
    TimingConfig,
    build_skeleton,
    endpoint_positions,
    joint_limits,
    load_scenario,
)

REPO_ROOT = repository_root()
TASK_1A = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"


@pytest.fixture(scope="module")
def scenario() -> ScenarioConfig:
    """The committed canonical scenario."""
    return load_scenario(TASK_1A)


def test_canonical_scenario_fixes_every_required_parameter(scenario: ScenarioConfig) -> None:
    """Robot, target, dt, limits, duration, prime, and dwell are all pinned by the TOML."""
    assert scenario.name == "task-1a-reach"
    assert scenario.dof == 2
    assert [link.length for link in scenario.robot.links] == [0.30, 0.25]
    assert scenario.robot.gravity == (0.0, 0.0)
    assert scenario.limits.velocity == (6.0, 6.0)
    assert scenario.limits.torque == (10.0, 5.0)
    assert scenario.limits.endpoint_radius == pytest.approx(scenario.robot.reach)
    assert scenario.task.initial_q == (0.2, 1.2)
    assert scenario.task.target == (0.10, 0.45)
    assert scenario.task.tolerance == 0.01
    assert scenario.task.dwell_min_fraction == 0.9
    assert scenario.task.dwell_max_velocity == 0.05
    assert scenario.task.dwell_criteria.names == ("dwell_in_tolerance", "dwell_stationary")
    assert scenario.timing.dt == 0.01
    assert scenario.timing.intervals == Intervals((0.0, 1.0), (1.0, 4.0), (4.0, 5.0))
    assert scenario.timing.intervals.duration_s == 5.0


def test_target_is_reachable_with_a_bent_elbow(scenario: ScenarioConfig) -> None:
    """Closed-form IK reaches the target inside the joint limits, near the documented (0.83, 1.16) rad."""
    l1, l2 = (link.length for link in scenario.robot.links)
    x, y = scenario.task.target
    assert math.hypot(x, y) < 0.9 * scenario.robot.reach
    cos_q2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    q2 = math.acos(cos_q2)  # elbow-up branch
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    for angle, link in zip((q1, q2), scenario.robot.links, strict=True):
        assert link.q_min < angle < link.q_max
    assert abs(q1 - 0.83) < 0.02
    assert abs(q2 - 1.16) < 0.02
    tip = endpoint_positions(scenario, np.array([[q1, q2]]))[0]
    assert np.linalg.norm(tip - np.asarray(scenario.task.target)) < 1e-9


def test_forward_kinematics_matches_planar_trigonometry(scenario: ScenarioConfig) -> None:
    """Endpoint positions from skelarm equal the closed-form two-link kinematics."""
    q = np.array([[0.2, 1.2], [0.0, 0.0], [1.0, -0.5], [-2.0, 2.5]])
    tips = endpoint_positions(scenario, q)
    l1, l2 = (link.length for link in scenario.robot.links)
    expected = np.column_stack(
        [l1 * np.cos(q[:, 0]) + l2 * np.cos(q.sum(axis=1)), l1 * np.sin(q[:, 0]) + l2 * np.sin(q.sum(axis=1))]
    )
    assert tips.shape == (4, 2)
    assert np.allclose(tips, expected, atol=1e-12)


def test_skeleton_is_posed_at_the_initial_or_given_posture(scenario: ScenarioConfig) -> None:
    """build_skeleton reproduces the configuration and posture."""
    skeleton = build_skeleton(scenario)
    assert skeleton.num_joints == 2
    assert np.allclose(skeleton.q, scenario.task.initial_q)
    assert np.allclose(skeleton.dq, 0.0)
    posed = build_skeleton(scenario, np.array([0.5, -0.5]))
    assert np.allclose(posed.q, [0.5, -0.5])
    with pytest.raises(ValueError, match=r"q must have shape \(2,\)"):
        build_skeleton(scenario, np.zeros(3))


def test_joint_limits_for_validation(scenario: ScenarioConfig) -> None:
    """Position bounds come from the links and speed bounds from the limits table."""
    limits = joint_limits(scenario)
    assert limits.lower == (-3.0, -3.0)
    assert limits.upper == (3.0, 3.0)
    assert limits.speed == (6.0, 6.0)


def test_endpoint_positions_validate_input(scenario: ScenarioConfig) -> None:
    """Wrong shapes and non-finite postures are rejected."""
    with pytest.raises(ValueError, match=r"q must have shape \(N, 2\)"):
        endpoint_positions(scenario, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        endpoint_positions(scenario, np.array([[0.0, np.nan]]))


def _variant(base: ScenarioConfig, key: str) -> dict[str, object]:
    if key == "velocity_length":
        return {"limits": dataclasses.replace(base.limits, velocity=(6.0,))}
    if key == "initial_outside":
        return {"task": dataclasses.replace(base.task, initial_q=(3.5, 0.0))}
    if key == "target_far":
        return {"task": dataclasses.replace(base.task, target=(0.5, 0.3))}
    if key == "radius_small":
        return {"limits": dataclasses.replace(base.limits, endpoint_radius=0.4)}
    if key == "radius_large":
        return {"limits": dataclasses.replace(base.limits, endpoint_radius=0.9)}
    if key == "blank_name":
        return {"name": " "}
    msg = f"unknown variant {key!r}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("velocity_length", "limits.velocity must have 2 entries"),
        ("initial_outside", r"initial_q\[0\]=3.5 lies outside"),
        ("target_far", "exceeds the arm's reach"),
        ("radius_small", "outside limits.endpoint_radius"),
        ("radius_large", "exceeds the arm's reach"),
        ("blank_name", "name must not be empty"),
    ],
)
def test_cross_field_consistency_is_enforced(key: str, message: str) -> None:
    """Limits, posture, and target are checked against the robot."""
    base = load_scenario(TASK_1A)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **_variant(base, key))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: LinkConfig(0.3, 1.0, 0.01, (0.15,), -1.0, 1.0), r"com must be an \[x, y\] pair"),
        (lambda: LinkConfig(0.0, 1.0, 0.01, (0.15, 0.0), -1.0, 1.0), "must be positive"),
        (lambda: LinkConfig(0.3, 1.0, 0.01, (0.15, 0.0), 1.0, 1.0), "q_min 1.0 must be below q_max 1.0"),
        (lambda: RobotConfig(links=()), "at least one link"),
        (lambda: RobotConfig(links=(LinkConfig(0.3, 1.0, 0.01, (0.15, 0.0), -1.0, 1.0),), gravity=(0.0,)), "gravity"),
        (lambda: LimitsConfig((0.0,), (1.0,), 1.0), "must be positive"),
        (lambda: LimitsConfig((1.0,), (1.0,), 0.0), "endpoint_radius must be positive"),
        (lambda: TaskConfig((0.0,), (0.1,), 0.01, 0.9, 0.05), r"target must be an \[x, y\]"),
        (lambda: TaskConfig((0.0,), (0.1, 0.1), 0.0, 0.9, 0.05), "tolerance must be positive"),
        (lambda: TaskConfig((0.0,), (0.1, 0.1), 0.01, 1.2, 0.05), r"dwell_min_fraction must be in \[0, 1\]"),
        (lambda: TaskConfig((0.0,), (0.1, 0.1), 0.01, 0.9, 0.0), "dwell_max_velocity must be positive"),
        (lambda: TimingConfig(0.0, Intervals((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))), "dt must be positive"),
    ],
)
def test_section_invariants(build: object, message: str) -> None:
    """Each section validates its own fields."""
    with pytest.raises(ValueError, match=message):
        build()  # type: ignore[operator]


def test_unknown_keys_in_the_scenario_file_are_rejected(tmp_path: Path) -> None:
    """Typos in the versioned TOML fail loudly."""
    text = TASK_1A.read_text().replace("tolerance = 0.01", "tolerance = 0.01\ntolerence = 0.02")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=r"task: unknown key\(s\) 'tolerence'"):
        load_scenario(path)
