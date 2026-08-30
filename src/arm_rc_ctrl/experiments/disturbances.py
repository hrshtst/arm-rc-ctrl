# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Repeatable disturbances applied during simulated runs (docs/PLAN.md section 9.2).

M1-028 introduces the finite-duration endpoint force pulse used to calibrate
the confirmatory force level; M3-008 builds the full disturbance suite on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.experiments.run_record import Disturbance
from arm_rc_ctrl.validation import require_finite

__all__ = ["FORCE_PULSE_KIND", "ForcePulse"]

FORCE_PULSE_KIND = "endpoint_force_pulse"
_PLANE = 2


@dataclass(frozen=True)
class ForcePulse:
    """A constant planar force applied at the endpoint over ``[start_s, start_s + duration_s)``."""

    start_s: float
    duration_s: float
    force: tuple[float, float]
    """Force components (N) in the base frame."""

    def __post_init__(self) -> None:
        """Timing must be finite and non-negative with a positive duration; the force finite."""
        require_finite((self.start_s, self.duration_s, *self.force), "force pulse")
        if self.start_s < 0:
            msg = f"start_s must be >= 0, got {self.start_s}"
            raise ValueError(msg)
        if self.duration_s <= 0:
            msg = f"duration_s must be > 0, got {self.duration_s}"
            raise ValueError(msg)
        if len(self.force) != _PLANE:
            msg = f"force must have {_PLANE} components, got {len(self.force)}"
            raise ValueError(msg)

    @classmethod
    def from_polar(cls, start_s: float, duration_s: float, magnitude_n: float, direction_deg: float) -> ForcePulse:
        """Build a pulse from a magnitude (N) and a direction measured from the base x axis (degrees)."""
        require_finite((magnitude_n, direction_deg), "force pulse")
        if magnitude_n < 0:
            msg = f"magnitude_n must be >= 0, got {magnitude_n}"
            raise ValueError(msg)
        angle = math.radians(direction_deg)
        return cls(start_s, duration_s, (magnitude_n * math.cos(angle), magnitude_n * math.sin(angle)))

    @property
    def end_s(self) -> float:
        """First instant at which the pulse is no longer applied."""
        return self.start_s + self.duration_s

    @property
    def magnitude_n(self) -> float:
        """Euclidean magnitude of the force (N)."""
        return math.hypot(*self.force)

    def active(self, t: float) -> bool:
        """Whether the pulse acts at time ``t`` (half-open window)."""
        return self.start_s <= t < self.end_s

    def at(self, t: float) -> NDArray[np.float64]:
        """Force vector applied at time ``t`` (zero outside the window)."""
        if self.active(t):
            return np.asarray(self.force, dtype=np.float64)
        return np.zeros(_PLANE, dtype=np.float64)

    def to_disturbance(self) -> Disturbance:
        """The run-record description of this pulse."""
        return Disturbance(
            FORCE_PULSE_KIND,
            self.start_s,
            self.end_s,
            {"fx": float(self.force[0]), "fy": float(self.force[1]), "magnitude_n": self.magnitude_n},
        )
