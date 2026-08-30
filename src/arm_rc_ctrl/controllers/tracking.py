# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Low-level trackers with torque limits and full telemetry (``docs/PLAN.md`` sections 5.4 and 6).

:class:`LimitedTracker` wraps ``skelarm``'s ``JointPD`` or ``ComputedTorque``
around any joint reference, clamps the requested torque to the scenario's
per-joint limits, and exposes the reference, tracking error, requested and
applied torque, and saturation flags as log channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skelarm import ComputedTorque, Controller, JointPD, JointReference, Skeleton

from arm_rc_ctrl.validation import require_finite

__all__ = ["LimitedTracker", "TrackerConfig", "TrackerType"]

type TrackerType = Literal["pd", "computed_torque"]


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker type and per-joint gains (``configs/controllers/*.toml``)."""

    type: TrackerType
    kp: tuple[float, ...]
    kd: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require matching, finite, non-negative gains."""
        if not self.kp or len(self.kp) != len(self.kd):
            msg = f"kp and kd must have the same non-zero length, got {len(self.kp)} and {len(self.kd)}"
            raise ValueError(msg)
        require_finite(self.kp, "kp")
        require_finite(self.kd, "kd")
        if any(g < 0 for g in (*self.kp, *self.kd)):
            msg = "gains must be non-negative"
            raise ValueError(msg)

    @property
    def dof(self) -> int:
        """Number of joints the gains cover."""
        return len(self.kp)

    @property
    def method(self) -> str:
        """Report label of the tracker."""
        return "pd" if self.type == "pd" else "computed_torque"


class LimitedTracker(Controller):
    """A ``skelarm`` controller that tracks a reference within torque limits and logs everything."""

    def __init__(self, reference: JointReference, config: TrackerConfig, torque_limits: tuple[float, ...]) -> None:
        """Wrap the configured tracker around ``reference`` with per-joint torque bounds."""
        if len(torque_limits) != config.dof or any(not (lim > 0 and np.isfinite(lim)) for lim in torque_limits):
            msg = f"torque_limits must give {config.dof} positive finite bounds, got {torque_limits}"
            raise ValueError(msg)
        self.config = config
        self.reference = reference
        self.limits = np.asarray(torque_limits, dtype=np.float64)
        kp, kd = np.asarray(config.kp), np.asarray(config.kd)
        self._inner = JointPD(reference, kp, kd) if config.type == "pd" else ComputedTorque(reference, kp, kd)
        self._channels: dict[str, NDArray[np.float64]] = {}

    def reset(self, skeleton: Skeleton) -> None:
        """Clear the telemetry."""
        self._inner.reset(skeleton)
        self._channels = {}

    def control(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        """Return the applied (clamped) torque, recording reference, error, and saturation."""
        q_r, dq_r, ddq_r = self.reference.sample(t)
        requested = np.asarray(self._inner.control(t, skeleton), dtype=np.float64)
        applied = np.clip(requested, -self.limits, self.limits)
        saturated = np.abs(requested) >= self.limits
        self._channels = {
            "q_ref": np.asarray(q_r, dtype=np.float64),
            "dq_ref": np.asarray(dq_r, dtype=np.float64),
            "ddq_ref": np.asarray(ddq_r, dtype=np.float64),
            "error": np.asarray(q_r, dtype=np.float64) - skeleton.q,
            "tau_requested": requested,
            "tau_applied": applied,
            "saturation": saturated.astype(np.float64),
        }
        return applied

    def log_channels(self) -> dict[str, ArrayLike]:
        """Telemetry of the last control evaluation."""
        return dict(self._channels)

    @property
    def last(self) -> dict[str, NDArray[np.float64]]:
        """Typed view of the last telemetry."""
        return dict(self._channels)
