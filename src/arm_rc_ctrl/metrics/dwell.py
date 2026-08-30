# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Final-dwell endpoint and stationarity metrics (``docs/PLAN.md`` section 9.1).

Over the samples inside the dwell window: endpoint error mean, RMS, maximum,
and 95th percentile (NumPy's default linear interpolation), the fraction of
samples inside the target tolerance, the longest continuous in-tolerance
duration (time between the first and last sample of the longest run), and the
joint-velocity RMS and maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DwellCriteria",
    "DwellMetrics",
    "EndpointErrorStats",
    "dwell_metrics",
    "endpoint_error_stats",
    "longest_run_duration",
]

_PERCENTILE: Final = 95.0
_PLANE: Final = 2


@dataclass(frozen=True)
class EndpointErrorStats:
    """Distribution of the endpoint distance to the target (m)."""

    mean: float
    rms: float
    max: float
    p95: float
    samples: int


@dataclass(frozen=True)
class DwellMetrics:
    """Endpoint accuracy and stationarity inside the dwell window."""

    endpoint: EndpointErrorStats
    in_tolerance_fraction: float
    longest_in_tolerance_s: float
    velocity_rms: float
    velocity_max: float
    window_s: float
    samples: int


@dataclass(frozen=True)
class DwellCriteria:
    """Versioned dwell success criteria (``docs/PLAN.md`` section 9.1)."""

    tolerance: float
    min_fraction: float
    max_velocity: float

    def __post_init__(self) -> None:
        """Validate ranges."""
        if not (self.tolerance > 0 and np.isfinite(self.tolerance)):
            msg = f"tolerance must be positive and finite, got {self.tolerance!r}"
            raise ValueError(msg)
        if not 0.0 <= self.min_fraction <= 1.0:
            msg = f"min_fraction must be in [0, 1], got {self.min_fraction!r}"
            raise ValueError(msg)
        if not (self.max_velocity > 0 and np.isfinite(self.max_velocity)):
            msg = f"max_velocity must be positive and finite, got {self.max_velocity!r}"
            raise ValueError(msg)

    def evaluate(self, metrics: DwellMetrics) -> dict[str, bool]:
        """Named criteria: in-tolerance fraction and stationarity."""
        return {
            "dwell_in_tolerance": metrics.in_tolerance_fraction >= self.min_fraction,
            "dwell_stationary": metrics.velocity_max <= self.max_velocity,
        }

    @property
    def names(self) -> tuple[str, ...]:
        """Criterion names in evaluation order."""
        return ("dwell_in_tolerance", "dwell_stationary")


def _finite(array: NDArray[np.float64], name: str) -> None:
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)


def _distances(positions: NDArray[np.float64], goal: NDArray[np.float64]) -> NDArray[np.float64]:
    offsets: NDArray[np.float64] = positions - goal[None, :]
    return np.ascontiguousarray(np.sqrt(np.sum(offsets * offsets, axis=1)), dtype=np.float64)


def endpoint_error_stats(tip: NDArray[np.float64], target: NDArray[np.float64]) -> EndpointErrorStats:
    """Mean/RMS/max/p95 of ``||tip - target||`` over the given samples."""
    positions = np.asarray(tip, dtype=np.float64)
    goal = np.asarray(target, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != _PLANE or positions.shape[0] == 0:  # noqa: PLR2004
        msg = f"tip must have shape (N, 2) with N > 0, got {positions.shape}"
        raise ValueError(msg)
    if goal.shape != (_PLANE,):
        msg = f"target must have shape (2,), got {goal.shape}"
        raise ValueError(msg)
    _finite(positions, "tip")
    _finite(goal, "target")
    distance = _distances(positions, goal)
    return EndpointErrorStats(
        mean=float(np.mean(distance)),
        rms=float(np.sqrt(np.mean(distance * distance))),
        max=float(np.max(distance)),
        p95=float(np.percentile(distance, _PERCENTILE)),
        samples=int(distance.shape[0]),
    )


def longest_run_duration(t: NDArray[np.float64], inside: NDArray[np.bool_]) -> float:
    """Time between the first and last sample of the longest run of ``True`` values (0 for runs of one)."""
    best = 0.0
    start: int | None = None
    for i, flag in enumerate(inside.tolist()):
        if flag and start is None:
            start = i
        if (not flag or i == len(inside) - 1) and start is not None:
            end = i if flag else i - 1
            best = max(best, float(t[end] - t[start]))
            start = None
    return best


def dwell_metrics(
    t: NDArray[np.float64],
    tip: NDArray[np.float64],
    dq: NDArray[np.float64],
    target: NDArray[np.float64],
    tolerance: float,
    *,
    window: tuple[float, float] | None = None,
) -> DwellMetrics:
    """Compute dwell metrics over the samples with ``window[0] <= t <= window[1]`` (all samples if ``None``)."""
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0:
        msg = f"t must be a non-empty 1-D array, got shape {times.shape}"
        raise ValueError(msg)
    _finite(times, "t")
    if not np.all(np.diff(times) > 0):
        msg = "t must be strictly increasing"
        raise ValueError(msg)
    velocities = np.asarray(dq, dtype=np.float64)
    positions = np.asarray(tip, dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[0] != times.shape[0] or positions.shape[0] != times.shape[0]:  # noqa: PLR2004
        msg = f"tip and dq must have {times.shape[0]} rows, got {positions.shape} and {velocities.shape}"
        raise ValueError(msg)
    _finite(velocities, "dq")
    if not (tolerance > 0 and np.isfinite(tolerance)):
        msg = f"tolerance must be positive and finite, got {tolerance!r}"
        raise ValueError(msg)
    if window is None:
        mask = np.ones(times.shape[0], dtype=np.bool_)
    else:
        lo, hi = window
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            msg = f"window must be a finite [start, end] with start < end, got {window}"
            raise ValueError(msg)
        mask = (times >= lo) & (times <= hi)
        if not mask.any():
            msg = f"no samples fall inside the window {window}"
            raise ValueError(msg)
    sel_t, sel_tip, sel_dq = times[mask], positions[mask], velocities[mask]
    stats = endpoint_error_stats(sel_tip, target)
    goal = np.asarray(target, dtype=np.float64)
    distance = _distances(sel_tip, goal)
    inside = distance <= tolerance
    return DwellMetrics(
        endpoint=stats,
        in_tolerance_fraction=float(np.mean(inside)),
        longest_in_tolerance_s=longest_run_duration(sel_t, inside),
        velocity_rms=float(np.sqrt(np.mean(sel_dq * sel_dq))),
        velocity_max=float(np.max(np.abs(sel_dq))),
        window_s=float(sel_t[-1] - sel_t[0]),
        samples=int(sel_t.shape[0]),
    )
