# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Torque effort, peak, and saturation metrics (``docs/PLAN.md`` section 9.1).

``effort`` is ``integral(sum(tau**2), t)`` by the trapezoidal rule, which is
exact for piecewise-linear integrands and valid on irregular time grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["EffortMetrics", "effort_metrics"]


@dataclass(frozen=True)
class EffortMetrics:
    """Torque statistics over the selected samples."""

    torque_rms: float
    torque_peak: float
    per_joint_peak: tuple[float, ...]
    saturation_fraction: float
    """Fraction of samples in which at least one joint reached its torque limit."""
    effort: float
    """``integral(sum_j tau_j^2, t)`` in N^2 m^2 s."""
    samples: int


def effort_metrics(
    t: NDArray[np.float64],
    tau: NDArray[np.float64],
    limits: tuple[float, ...],
    *,
    window: tuple[float, float] | None = None,
) -> EffortMetrics:
    """Compute effort metrics over the samples with ``window[0] <= t <= window[1]`` (all if ``None``)."""
    times = np.asarray(t, dtype=np.float64)
    torque = np.asarray(tau, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] < 2:  # noqa: PLR2004
        msg = f"t must be a 1-D array with at least 2 samples, got shape {times.shape}"
        raise ValueError(msg)
    if torque.ndim != 2 or torque.shape[0] != times.shape[0]:  # noqa: PLR2004
        msg = f"tau must have shape ({times.shape[0]}, dof), got {torque.shape}"
        raise ValueError(msg)
    if len(limits) != torque.shape[1] or any(not (lim > 0 and np.isfinite(lim)) for lim in limits):
        msg = f"limits must give {torque.shape[1]} positive finite bounds, got {limits}"
        raise ValueError(msg)
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(torque))):
        msg = "t and tau must be finite"
        raise ValueError(msg)
    if not np.all(np.diff(times) > 0):
        msg = "t must be strictly increasing"
        raise ValueError(msg)
    if window is None:
        mask = np.ones(times.shape[0], dtype=np.bool_)
    else:
        lo, hi = window
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            msg = f"window must be a finite [start, end] with start < end, got {window}"
            raise ValueError(msg)
        mask = (times >= lo) & (times <= hi)
        if np.count_nonzero(mask) < 2:  # noqa: PLR2004
            msg = f"fewer than 2 samples fall inside the window {window}"
            raise ValueError(msg)
    sel_t, sel_tau = times[mask], torque[mask]
    bounds = np.asarray(limits, dtype=np.float64)
    magnitude = np.abs(sel_tau)
    saturated = np.any(magnitude >= bounds, axis=1)
    sum_sq = np.sum(sel_tau * sel_tau, axis=1)
    return EffortMetrics(
        torque_rms=float(np.sqrt(np.mean(sel_tau * sel_tau))),
        torque_peak=float(np.max(magnitude)),
        per_joint_peak=tuple(float(v) for v in np.max(magnitude, axis=0)),
        saturation_fraction=float(np.mean(saturated)),
        effort=float(np.trapezoid(sum_sq, sel_t)),
        samples=int(sel_t.shape[0]),
    )
