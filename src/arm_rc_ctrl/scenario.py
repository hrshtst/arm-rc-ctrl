# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Typed scenario configuration (robot, limits, task, timing) and ``skelarm`` adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from skelarm import LinkProp, Skeleton, compute_forward_kinematics

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.data.records import Intervals
from arm_rc_ctrl.data.validate import JointLimits
from arm_rc_ctrl.metrics.dwell import DwellCriteria
from arm_rc_ctrl.validation import require_finite

__all__ = [
    "LimitsConfig",
    "LinkConfig",
    "RobotConfig",
    "ScenarioConfig",
    "TaskConfig",
    "TimingConfig",
    "build_robot_skeleton",
    "build_skeleton",
    "endpoint_positions",
    "joint_limits",
    "joint_target",
    "load_scenario",
    "robot_endpoint_positions",
    "robot_joint_limits",
]

_PLANE = 2


@dataclass(frozen=True)
class LinkConfig:
    """One movable planar link."""

    length: float
    mass: float
    inertia: float
    com: tuple[float, ...]
    q_min: float
    q_max: float

    def __post_init__(self) -> None:
        """Validate geometry, inertial parameters, and joint bounds."""
        if len(self.com) != _PLANE:
            msg = f"link com must be an [x, y] pair, got {list(self.com)}"
            raise ValueError(msg)
        require_finite((self.length, self.mass, self.inertia, *self.com, self.q_min, self.q_max), "link")
        if self.length <= 0 or self.mass <= 0 or self.inertia <= 0:
            msg = "link length, mass, and inertia must be positive"
            raise ValueError(msg)
        if self.q_min >= self.q_max:
            msg = f"link q_min {self.q_min} must be below q_max {self.q_max}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RobotConfig:
    """Planar arm description."""

    links: tuple[LinkConfig, ...]
    name: str = "planar"
    base_length: float = 0.0
    gravity: tuple[float, ...] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate link count, base, and gravity vector."""
        if not self.links:
            msg = "robot.links must contain at least one link"
            raise ValueError(msg)
        if not self.name.strip():
            msg = "robot.name must not be empty"
            raise ValueError(msg)
        if not math.isfinite(self.base_length) or self.base_length < 0:
            msg = f"robot.base_length must be finite and non-negative, got {self.base_length!r}"
            raise ValueError(msg)
        if len(self.gravity) != _PLANE:
            msg = f"robot.gravity must be an [x, y] vector, got {list(self.gravity)}"
            raise ValueError(msg)
        require_finite(self.gravity, "robot.gravity")

    @property
    def dof(self) -> int:
        """Number of actuated joints."""
        return len(self.links)

    @property
    def reach(self) -> float:
        """Maximum distance of the endpoint from the origin."""
        return self.base_length + sum(link.length for link in self.links)


@dataclass(frozen=True)
class LimitsConfig:
    """Joint velocity/torque bounds and the endpoint workspace radius."""

    velocity: tuple[float, ...]
    torque: tuple[float, ...]
    endpoint_radius: float

    def __post_init__(self) -> None:
        """Validate positivity and finiteness."""
        require_finite(self.velocity, "limits.velocity")
        require_finite(self.torque, "limits.torque")
        if any(v <= 0 for v in self.velocity) or any(t <= 0 for t in self.torque):
            msg = "limits.velocity and limits.torque entries must be positive"
            raise ValueError(msg)
        if not (self.endpoint_radius > 0 and math.isfinite(self.endpoint_radius)):
            msg = f"limits.endpoint_radius must be positive and finite, got {self.endpoint_radius!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TaskConfig:
    """Initial posture, endpoint target, and the dwell success criteria."""

    initial_q: tuple[float, ...]
    target: tuple[float, ...]
    tolerance: float
    dwell_min_fraction: float
    """Minimum fraction of dwell samples whose endpoint lies within ``tolerance``."""
    dwell_max_velocity: float
    """Maximum absolute joint velocity (rad/s) allowed at any dwell sample."""

    def __post_init__(self) -> None:
        """Validate dimensions, finiteness, tolerance, and dwell criteria."""
        require_finite(self.initial_q, "task.initial_q")
        if len(self.target) != _PLANE:
            msg = f"task.target must be an [x, y] endpoint, got {list(self.target)}"
            raise ValueError(msg)
        require_finite(self.target, "task.target")
        if not (self.tolerance > 0 and math.isfinite(self.tolerance)):
            msg = f"task.tolerance must be positive and finite, got {self.tolerance!r}"
            raise ValueError(msg)
        if not 0.0 <= self.dwell_min_fraction <= 1.0:
            msg = f"task.dwell_min_fraction must be in [0, 1], got {self.dwell_min_fraction!r}"
            raise ValueError(msg)
        if not (self.dwell_max_velocity > 0 and math.isfinite(self.dwell_max_velocity)):
            msg = f"task.dwell_max_velocity must be positive and finite, got {self.dwell_max_velocity!r}"
            raise ValueError(msg)

    @property
    def dwell_criteria(self) -> DwellCriteria:
        """The dwell criteria as evaluated by the dwell metrics."""
        return DwellCriteria(self.tolerance, self.dwell_min_fraction, self.dwell_max_velocity)


@dataclass(frozen=True)
class TimingConfig:
    """Control period and nominal recording intervals."""

    dt: float
    intervals: Intervals

    def __post_init__(self) -> None:
        """Validate the period."""
        if not (self.dt > 0 and math.isfinite(self.dt)):
            msg = f"timing.dt must be positive and finite, got {self.dt!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete, self-consistent task scenario."""

    name: str
    robot: RobotConfig
    limits: LimitsConfig
    task: TaskConfig
    timing: TimingConfig

    def __post_init__(self) -> None:
        """Cross-check limits, posture, and target against the robot."""
        dof = self.robot.dof
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
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

    @property
    def dof(self) -> int:
        """Number of actuated joints."""
        return self.robot.dof


def load_scenario(path: Path) -> ScenarioConfig:
    """Load and validate a scenario TOML."""
    return load_config(path, ScenarioConfig)


def robot_joint_limits(robot: RobotConfig, limits: LimitsConfig) -> JointLimits:
    """Position and speed limits of an arm, independent of which task schema declared them."""
    return JointLimits(
        lower=tuple(link.q_min for link in robot.links),
        upper=tuple(link.q_max for link in robot.links),
        speed=limits.velocity,
    )


def joint_limits(config: ScenarioConfig) -> JointLimits:
    """Position and speed limits for dataset validation."""
    return robot_joint_limits(config.robot, config.limits)


def build_robot_skeleton(robot: RobotConfig, q: NDArray[np.float64]) -> Skeleton:
    """Construct the ``skelarm`` skeleton of ``robot`` posed at ``q`` with zero velocity."""
    props = [
        LinkProp(
            length=link.length,
            m=link.mass,
            i=link.inertia,
            rgx=link.com[0],
            rgy=link.com[1],
            qmin=link.q_min,
            qmax=link.q_max,
        )
        for link in robot.links
    ]
    skeleton = Skeleton(props, base_length=robot.base_length)
    posture = np.asarray(q, dtype=np.float64)
    if posture.shape != (robot.dof,):
        msg = f"q must have shape ({robot.dof},), got {posture.shape}"
        raise ValueError(msg)
    skeleton.q = posture
    skeleton.dq = np.zeros(robot.dof, dtype=np.float64)
    return skeleton


def build_skeleton(config: ScenarioConfig, q: NDArray[np.float64] | None = None) -> Skeleton:
    """Construct the ``skelarm`` skeleton posed at ``q`` (default: the task's initial posture)."""
    return build_robot_skeleton(config.robot, np.asarray(config.task.initial_q if q is None else q, dtype=np.float64))


def robot_endpoint_positions(robot: RobotConfig, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Endpoint ``(x, y)`` for each row of ``q`` (shape ``(N, dof)``) via ``skelarm`` forward kinematics."""
    joints = np.asarray(q, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != robot.dof:  # noqa: PLR2004
        msg = f"q must have shape (N, {robot.dof}), got {joints.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(joints)):
        msg = "q contains non-finite values"
        raise ValueError(msg)
    out = np.empty((joints.shape[0], _PLANE), dtype=np.float64)
    if joints.shape[0] == 0:
        return out
    skeleton = build_robot_skeleton(robot, joints[0])
    for i, row in enumerate(joints):
        skeleton.q = row
        compute_forward_kinematics(skeleton)
        tip = skeleton.links[-1]
        out[i] = (tip.xe, tip.ye)
    return out


def endpoint_positions(config: ScenarioConfig, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Endpoint ``(x, y)`` for each row of ``q`` (shape ``(N, dof)``) via ``skelarm`` forward kinematics."""
    return robot_endpoint_positions(config.robot, q)


class TaskTarget(Protocol):
    """The task members every scenario schema shares: where the arm starts and what it must reach."""

    @property
    def initial_q(self) -> tuple[float, ...]:
        """Start posture (rad)."""
        ...

    @property
    def target(self) -> tuple[float, ...]:
        """Endpoint target (m)."""
        ...

    @property
    def tolerance(self) -> float:
        """Radius (m) of the target region."""
        ...


class TaskTiming(Protocol):
    """The timing members every scenario schema shares."""

    @property
    def dt(self) -> float:
        """Control period (s)."""
        ...


class TaskScenario(Protocol):
    """A scenario the simulator and the closed-form kinematics can work from, whichever schema declared it.

    The historical :class:`ScenarioConfig` and the manual protocol's own
    configuration carry the same robot, limits, start posture, target, and
    control period but differ in what else they require: the historical schema
    prescribes recording intervals and an occupancy fraction, which the manual
    protocol replaces with a continuous dwell rule (plan section 6). Reading
    the shared members structurally lets one simulator serve both without
    either schema growing a field its protocol does not define.
    """

    @property
    def robot(self) -> RobotConfig:
        """Link geometry and inertias."""
        ...

    @property
    def limits(self) -> LimitsConfig:
        """Joint speed, torque, and endpoint bounds."""
        ...

    @property
    def task(self) -> TaskTarget:
        """Start posture and endpoint target."""
        ...

    @property
    def timing(self) -> TaskTiming:
        """Control period."""
        ...

    @property
    def dof(self) -> int:
        """Number of actuated joints."""
        ...


def joint_target(config: TaskScenario, *, elbow_up: bool = True) -> tuple[float, ...]:
    """Closed-form inverse kinematics of the task target for a planar 2-DOF arm (rad).

    Raises
    ------
    ValueError
        If the robot is not a two-link arm, the target is unreachable, or the
        solution violates the joint limits.
    """
    if config.dof != _PLANE or config.robot.base_length != 0.0:
        msg = "joint_target supports only two-link arms with a zero base length"
        raise ValueError(msg)
    l1, l2 = (link.length for link in config.robot.links)
    x, y = config.task.target
    cos_q2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= cos_q2 <= 1.0:
        msg = f"target {config.task.target} is unreachable"
        raise ValueError(msg)
    q2 = math.acos(cos_q2) if elbow_up else -math.acos(cos_q2)
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    solution = (q1, q2)
    for i, (angle, link) in enumerate(zip(solution, config.robot.links, strict=True)):
        if not link.q_min <= angle <= link.q_max:
            msg = f"joint target[{i}]={angle:.4f} lies outside joint limits [{link.q_min}, {link.q_max}]"
            raise ValueError(msg)
    return solution
