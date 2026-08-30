# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Joint reference sampled from a canonical demonstration (``docs/PLAN.md`` section 6).

Direct-replay baselines track the demonstrated trajectory. The reference is
exact at the dataset's grid points, interpolates between them with the
dataset's own policy (``linear`` or ``cubic``), and beyond the recorded range
holds the boundary posture with zero velocity and acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

from arm_rc_ctrl.data.samples import SampleSet

__all__ = ["DemonstrationReference"]

type Interpolation = Literal["linear", "cubic"]
_CUBIC_MIN: Final = 4


@dataclass(frozen=True)
class DemonstrationReference:
    """``q``, ``dq``, ``ddq`` of a demonstration as a function of time."""

    t: NDArray[np.float64]
    q: NDArray[np.float64]
    dq: NDArray[np.float64]
    ddq: NDArray[np.float64]
    interpolation: Interpolation = "linear"

    def __post_init__(self) -> None:
        """Validate the grid and shapes; store read-only copies."""
        t = np.ascontiguousarray(self.t, dtype=np.float64)
        if t.ndim != 1 or t.shape[0] < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):  # noqa: PLR2004
            msg = "t must be a strictly increasing 1-D array with at least 2 samples"
            raise ValueError(msg)
        for name in ("q", "dq", "ddq"):
            array = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != t.shape[0] or array.shape[1] < 1:  # noqa: PLR2004
                msg = f"{name} must have shape ({t.shape[0]}, dof), got {array.shape}"
                raise ValueError(msg)
            if not np.all(np.isfinite(array)):
                msg = f"{name} must be finite"
                raise ValueError(msg)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.q.shape != self.dq.shape or self.q.shape != self.ddq.shape:
            msg = "q, dq, and ddq must share one shape"
            raise ValueError(msg)
        if self.interpolation == "cubic" and t.shape[0] < _CUBIC_MIN:
            msg = f"cubic interpolation needs at least {_CUBIC_MIN} samples"
            raise ValueError(msg)
        t.setflags(write=False)
        object.__setattr__(self, "t", t)

    @classmethod
    def from_samples(cls, samples: SampleSet, interpolation: Interpolation = "linear") -> DemonstrationReference:
        """Build from a canonical dataset."""
        return cls(samples.t, samples.q, samples.dq, samples.ddq, interpolation)

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.q.shape[1])

    @property
    def duration(self) -> float:
        """End time of the recorded reference."""
        return float(self.t[-1])

    def sample_many(
        self, times: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the reference at many times (``(M,)`` → three ``(M, dof)`` arrays)."""
        query = np.asarray(times, dtype=np.float64)
        if query.ndim != 1:
            msg = f"times must be 1-D, got shape {query.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(query)):
            msg = "times must be finite"
            raise ValueError(msg)
        clipped = np.clip(query, self.t[0], self.t[-1])
        q, dq, ddq = self._interpolate(clipped)
        outside = (query < self.t[0]) | (query > self.t[-1])
        dq[outside] = 0.0
        ddq[outside] = 0.0
        return q, dq, ddq

    def sample(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the reference at one time (``skelarm.JointReference`` protocol)."""
        q, dq, ddq = self.sample_many(np.array([t], dtype=np.float64))
        return q[0], dq[0], ddq[0]

    def _interpolate(
        self, times: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if self.interpolation == "linear":
            return tuple(_linear(times, self.t, arr) for arr in (self.q, self.dq, self.ddq))  # type: ignore[return-value]
        outs: list[NDArray[np.float64]] = []
        for arr in (self.q, self.dq, self.ddq):
            spline = make_interp_spline(self.t, arr, k=3, axis=0)
            outs.append(np.ascontiguousarray(cast("NDArray[Any]", spline(times)), dtype=np.float64))
        return outs[0], outs[1], outs[2]


def _linear(times: NDArray[np.float64], grid: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.column_stack([np.interp(times, grid, values[:, j]) for j in range(values.shape[1])])
    return np.ascontiguousarray(out, dtype=np.float64)
