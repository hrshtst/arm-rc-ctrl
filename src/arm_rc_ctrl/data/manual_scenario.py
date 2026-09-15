# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Task configuration of the manual-demonstration protocol (``task_1a_manual_v1``, plan clarification I6).

The schema copies the robot description, limits, start posture, target, and
tolerance of the historical task 1-a configuration but replaces the fixed
prime/move/dwell intervals and the dwell-occupancy rule with the protocol's
continuous final-dwell rule and the offline acquisition rules that the
acquisition pilot freezes. The historical ``configs/tasks/task_1a.toml`` keeps
its own schema and semantics; every committed dataset is digest-bound to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.data.validate import JointLimits
from arm_rc_ctrl.scenario import LimitsConfig, RobotConfig, build_robot_skeleton, robot_endpoint_positions
from arm_rc_ctrl.validation import require_finite

__all__ = [
    "AcquisitionRules",
    "ManualScenarioConfig",
    "ManualTaskConfig",
    "ManualTimingConfig",
    "load_manual_scenario",
    "manual_build_skeleton",
    "manual_endpoint_positions",
    "manual_joint_limits",
]

_PLANE = 2
_GRID_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ManualTaskConfig:
    """Start posture, endpoint target, and the continuous final-dwell rule."""

    initial_q: tuple[float, ...]
    target: tuple[float, ...]
    tolerance: float
    """Radius (m) of the target region every dwell sample must lie inside."""
    dwell_min_duration_s: float
    """Minimum uninterrupted final dwell (s); at 100 Hz one second is 101 consecutive samples."""
    dwell_max_velocity: float
    """Maximum absolute joint speed (rad/s) at every dwell sample."""

    def __post_init__(self) -> None:
        """Validate dimensions, finiteness, and the dwell rule."""
        require_finite(self.initial_q, "task.initial_q")
        if len(self.target) != _PLANE:
            msg = f"task.target must be an [x, y] endpoint, got {list(self.target)}"
            raise ValueError(msg)
        require_finite(self.target, "task.target")
        for name, value in (
            ("tolerance", self.tolerance),
            ("dwell_min_duration_s", self.dwell_min_duration_s),
            ("dwell_max_velocity", self.dwell_max_velocity),
        ):
            if not (value > 0 and math.isfinite(value)):
                msg = f"task.{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class ManualTimingConfig:
    """The control/training period; there are no prescribed recording intervals."""

    dt: float

    def __post_init__(self) -> None:
        """Validate the period."""
        if not (self.dt > 0 and math.isfinite(self.dt)):
            msg = f"timing.dt must be positive and finite, got {self.dt!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class AcquisitionRules:
    """Offline acceptance rules applied to every saved take (frozen before the study bank)."""

    start_tolerance_rad: float
    """Serialization tolerance for the first logged posture against the reset posture (no physical allowance)."""
    max_sample_gap_s: float
    """Largest allowed interval between consecutive raw frames (a missing interval rejects the take)."""
    velocity_bound_rad_s: float
    """Per-joint bound on raw increments and processed joint speeds (D3: 6 rad/s until the pilot tightens it)."""
    min_frames: int
    """Fewest raw frames a take may have."""

    def __post_init__(self) -> None:
        """Validate positivity."""
        for name, value in (
            ("start_tolerance_rad", self.start_tolerance_rad),
            ("max_sample_gap_s", self.max_sample_gap_s),
            ("velocity_bound_rad_s", self.velocity_bound_rad_s),
        ):
            if not (value > 0 and math.isfinite(value)):
                msg = f"acquisition.{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)
        if self.min_frames < 2:  # noqa: PLR2004
            msg = f"acquisition.min_frames must be at least 2, got {self.min_frames}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ManualScenarioConfig:
    """A complete, self-consistent manual-demonstration task configuration."""

    name: str
    protocol: str
    robot: RobotConfig
    limits: LimitsConfig
    task: ManualTaskConfig
    timing: ManualTimingConfig
    acquisition: AcquisitionRules

    def __post_init__(self) -> None:
        """Cross-check limits, posture, target, the dwell grid, and the rules against the robot."""
        if not self.name.strip() or not self.protocol.strip():
            msg = "name and protocol must not be empty"
            raise ValueError(msg)
        self._check_geometry()
        self._check_rules()

    def _check_geometry(self) -> None:
        dof = self.robot.dof
        for label, values in (
            ("limits.velocity", self.limits.velocity),
            ("limits.torque", self.limits.torque),
            ("task.initial_q", self.task.initial_q),
        ):
            if len(values) != dof:
                msg = f"{label} must have {dof} entries (one per joint), got {len(values)}"
                raise ValueError(msg)
        for i, (angle, link) in enumerate(zip(self.task.initial_q, self.robot.links, strict=True)):
            if not link.q_min <= angle <= link.q_max:
                msg = f"task.initial_q[{i}]={angle} lies outside joint limits [{link.q_min}, {link.q_max}]"
                raise ValueError(msg)
        distance = math.hypot(*self.task.target)
        if distance + self.task.tolerance > self.robot.reach:
            msg = f"task.target at {distance:.4f} m (+ tolerance) exceeds the arm's reach {self.robot.reach:.4f} m"
            raise ValueError(msg)
        if distance > self.limits.endpoint_radius:
            msg = f"task.target at {distance:.4f} m lies outside limits.endpoint_radius {self.limits.endpoint_radius}"
            raise ValueError(msg)
        if self.limits.endpoint_radius > self.robot.reach + 1e-12:
            msg = (
                f"limits.endpoint_radius {self.limits.endpoint_radius} exceeds the arm's reach {self.robot.reach:.4f} m"
            )
            raise ValueError(msg)

    def _check_rules(self) -> None:
        samples = self.task.dwell_min_duration_s / self.timing.dt
        if abs(samples - round(samples)) > _GRID_TOLERANCE:
            msg = (
                f"task.dwell_min_duration_s {self.task.dwell_min_duration_s} must be a whole number of samples "
                f"at timing.dt {self.timing.dt}"
            )
            raise ValueError(msg)
        if self.acquisition.velocity_bound_rad_s > min(self.limits.velocity):
            msg = (
                f"acquisition.velocity_bound_rad_s {self.acquisition.velocity_bound_rad_s} exceeds "
                f"limits.velocity {list(self.limits.velocity)}"
            )
            raise ValueError(msg)
        if self.acquisition.max_sample_gap_s < self.timing.dt:
            msg = (
                f"acquisition.max_sample_gap_s {self.acquisition.max_sample_gap_s} is below timing.dt {self.timing.dt}"
            )
            raise ValueError(msg)

    @property
    def dof(self) -> int:
        """Number of actuated joints."""
        return self.robot.dof

    @property
    def dwell_min_samples(self) -> int:
        """Consecutive grid samples the final dwell must span (duration on the grid plus one)."""
        return round(self.task.dwell_min_duration_s / self.timing.dt) + 1


def load_manual_scenario(path: Path) -> ManualScenarioConfig:
    """Load and validate a manual-protocol task TOML."""
    return load_config(path, ManualScenarioConfig)


def manual_joint_limits(config: ManualScenarioConfig) -> JointLimits:
    """Position and speed limits for dataset validation."""
    return JointLimits(
        lower=tuple(link.q_min for link in config.robot.links),
        upper=tuple(link.q_max for link in config.robot.links),
        speed=config.limits.velocity,
    )


def manual_build_skeleton(config: ManualScenarioConfig, q: NDArray[np.float64] | None = None) -> Skeleton:
    """Construct the ``skelarm`` skeleton posed at ``q`` (default: the reset posture)."""
    return build_robot_skeleton(config.robot, np.asarray(config.task.initial_q if q is None else q, dtype=np.float64))


def manual_endpoint_positions(config: ManualScenarioConfig, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Endpoint ``(x, y)`` for each row of ``q`` via ``skelarm`` forward kinematics."""
    return robot_endpoint_positions(config.robot, q)
