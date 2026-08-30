# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""``skelarm.Controller`` adapter combining a target generator with a low-level tracker (docs/PLAN.md section 8).

The adapter is an explicit two-phase state machine (section 5.3): during the
priming interval the tracker holds the initial posture while the generator is
primed with the measured state; from the configured boundary on, every control
sample asks the generator for the next desired joint state, which the limited
PD or computed-torque tracker converts to torque. All internal signals of both
parts are exposed through :meth:`log_channels`. The adapter can also be built
by ``skelarm``'s config-driven registry (:func:`register_with_skelarm`) without
patching ``skelarm``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skelarm import Controller, Skeleton, Task, register_controller

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.contracts import DesiredJointState, RobotState, TargetGeneratorBase
from arm_rc_ctrl.controllers.estimator import EstimatorConfig
from arm_rc_ctrl.controllers.tracking import LimitedTracker, TrackerConfig
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.rc.runtime import generator_from_recipe, load_training_samples
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CONTROLLER_TYPE",
    "GeneratorTrackingController",
    "LatestTargetReference",
    "Phase",
    "build_rc_controller",
    "register_with_skelarm",
]

CONTROLLER_TYPE: Final = "rc_target"
"""``[controller].type`` under which the adapter is registered with ``skelarm``."""


class Phase:
    """Controller phases (recorded in telemetry as ``phase``: 0 hold, 1 generate)."""

    HOLD: Final = 0.0
    GENERATE: Final = 1.0


class LatestTargetReference:
    """``skelarm`` joint reference that returns whatever desired state was set last."""

    def __init__(self, initial: DesiredJointState) -> None:
        self._desired = initial

    @property
    def desired(self) -> DesiredJointState:
        """The current desired state."""
        return self._desired

    def set(self, desired: DesiredJointState) -> None:
        """Replace the desired state for the next tracker evaluation."""
        self._desired = desired

    def sample(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The desired position, velocity, and acceleration (independent of ``t``)."""
        del t
        return self._desired.q, self._desired.dq, self._desired.ddq


class GeneratorTrackingController(Controller):
    """Hold the initial posture while priming, then track the generator's targets within torque limits."""

    def __init__(
        self,
        generator: TargetGeneratorBase,
        tracker: TrackerConfig,
        torque_limits: tuple[float, ...],
        *,
        hold_until_s: float,
        task_code: NDArray[np.float64] | None = None,
    ) -> None:
        if not (math.isfinite(hold_until_s) and hold_until_s >= 0):
            msg = f"hold_until_s must be finite and non-negative, got {hold_until_s!r}"
            raise ValueError(msg)
        self.generator = generator
        self.tracker_config = tracker
        self.torque_limits = torque_limits
        self.hold_until_s = hold_until_s
        self.task_code = None if task_code is None else np.asarray(task_code, dtype=np.float64)
        self._reference: LatestTargetReference | None = None
        self._tracker: LimitedTracker | None = None
        self._hold: NDArray[np.float64] | None = None
        self._first_generated: NDArray[np.float64] | None = None
        self._channels: dict[str, NDArray[np.float64]] = {}

    @property
    def phase(self) -> float:
        """The phase of the last control evaluation (``Phase.HOLD`` before any)."""
        return float(self._channels.get("phase", np.array([Phase.HOLD]))[0])

    @property
    def boundary_jump(self) -> float | None:
        """Distance (rad, max over joints) between the first generated target and the held posture."""
        if self._first_generated is None or self._hold is None:
            return None
        return float(np.abs(self._first_generated - self._hold).max())

    def reset(self, skeleton: Skeleton) -> None:
        """Start an episode at the skeleton's posture: reset the generator and the tracker, hold there."""
        state = RobotState(0.0, cast("Any", skeleton.q), cast("Any", skeleton.dq))
        self._hold = state.q
        self._first_generated = None
        self.generator.reset(state)
        self._reference = LatestTargetReference(DesiredJointState.hold(state.q))
        self._tracker = LimitedTracker(self._reference, self.tracker_config, self.torque_limits)
        self._tracker.reset(skeleton)
        self._channels = {}

    def control(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        """Prime (holding) before the boundary, generate from it on; return the limited torque."""
        if self._reference is None or self._tracker is None or self._hold is None:
            msg = "reset() must be called before control()"
            raise RuntimeError(msg)
        state = RobotState(t, cast("Any", skeleton.q), cast("Any", skeleton.dq))
        if t < self.hold_until_s:
            self.generator.prime(state, self.task_code)
            desired = DesiredJointState.hold(self._hold)
            phase = Phase.HOLD
        else:
            desired = self.generator.step(state, self.task_code)
            if self._first_generated is None:
                self._first_generated = desired.q
            phase = Phase.GENERATE
        self._reference.set(desired)
        tau = self._tracker.control(t, skeleton)
        generator_channels = getattr(self.generator, "last", {})
        self._channels = {
            "phase": np.array([phase]),
            "q_desired": desired.q,
            "dq_desired": desired.dq,
            "ddq_desired": desired.ddq,
            **{k: np.asarray(v, dtype=np.float64) for k, v in cast("dict[str, Any]", generator_channels).items()},
            **self._tracker.last,
        }
        return tau

    def log_channels(self) -> dict[str, ArrayLike]:
        """Every internal signal of the generator and the tracker for the last evaluation."""
        return dict(self._channels)

    @property
    def last(self) -> dict[str, NDArray[np.float64]]:
        """Typed view of the last telemetry."""
        return dict(self._channels)


def build_rc_controller(params: Mapping[str, Any], skeleton: Skeleton, task: Task) -> Controller:
    """``skelarm`` controller builder (``recipe``, ``tracker``, ``torque_limits``, ``hold_until_s``, ``estimator``).

    Datasets referenced by the recipe are resolved through the configured
    external store (``records_root`` defaults to the repository root). ``dt``
    is injected by ``skelarm`` and becomes the estimator's nominal period.
    """
    del task
    required = ("recipe", "tracker", "torque_limits", "hold_until_s", "dt")
    missing = [key for key in required if key not in params]
    if missing:
        msg = f"controller params are missing {missing}"
        raise ValueError(msg)
    recipe = load_recipe(Path(str(params["recipe"])))
    tracker = load_config(Path(str(params["tracker"])), TrackerConfig)
    estimator_params = dict(cast("Mapping[str, Any]", params.get("estimator", {})))
    estimator = EstimatorConfig(nominal_dt_s=float(params["dt"]), **estimator_params)
    records_root = Path(str(params["records_root"])) if "records_root" in params else None
    samples = load_training_samples(recipe, open_storage(), records_root=records_root)
    movable = skeleton.links[1:]  # links[0] is the fixed base link
    lower = np.array([link.prop.qmin for link in movable], dtype=np.float64)
    upper = np.array([link.prop.qmax for link in movable], dtype=np.float64)
    generator = generator_from_recipe(recipe, samples, estimator=estimator, position_bounds=(lower, upper))
    limits = tuple(float(v) for v in cast("list[float]", params["torque_limits"]))
    return GeneratorTrackingController(generator, tracker, limits, hold_until_s=float(params["hold_until_s"]))


def register_with_skelarm() -> str:
    """Register the adapter as ``[controller].type = "rc_target"``; returns the type name."""
    register_controller(CONTROLLER_TYPE, build_rc_controller)
    return CONTROLLER_TYPE
