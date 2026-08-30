# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Resampling of demonstrations onto the uniform control-period grid.

The output grid is ``t[0] + k * period`` for ``k = 0 .. K`` where the last
point is the largest grid time not beyond the source end (within a small
relative tolerance, so an end time carrying accumulated round-off such as
``0.30000000000000004`` still yields the ``0.30`` sample). Values are
interpolated per column; the grid never leaves the source range, so no
extrapolation occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

__all__ = ["ENDPOINT_TOLERANCE", "ResamplingConfig", "resample", "uniform_grid"]

ENDPOINT_TOLERANCE: Final = 1e-6
"""Fraction of the period by which the last grid point may exceed the source end time."""

_MIN_SOURCE_SAMPLES: Final = 2
_CUBIC_MIN_SAMPLES: Final = 4


@dataclass(frozen=True)
class ResamplingConfig:
    """Target period and interpolation method."""

    period_s: float
    interpolation: Literal["linear", "cubic"] = "linear"

    def __post_init__(self) -> None:
        """Validate the period."""
        if not (self.period_s > 0 and self.period_s < float("inf")):
            msg = f"period_s must be positive and finite, got {self.period_s!r}"
            raise ValueError(msg)


def uniform_grid(t_source: NDArray[np.float64], period_s: float) -> NDArray[np.float64]:
    """Return the grid ``t_source[0] + k * period_s`` covering the source range, endpoint included."""
    times = np.asarray(t_source, dtype=np.float64)
    _check_times(times)
    if not (period_s > 0 and period_s < float("inf")):
        msg = f"period_s must be positive and finite, got {period_s!r}"
        raise ValueError(msg)
    span = float(times[-1] - times[0])
    count = int(np.floor(span / period_s + ENDPOINT_TOLERANCE)) + 1
    if count < _MIN_SOURCE_SAMPLES:
        msg = f"period {period_s} s is too coarse for a recording spanning {span!r} s (fewer than 2 samples)"
        raise ValueError(msg)
    grid = float(times[0]) + np.arange(count, dtype=np.float64) * period_s
    # Clamp round-off overshoot of the final point so interpolation never extrapolates.
    grid[-1] = min(float(grid[-1]), float(times[-1]))
    return grid


def resample(
    t_source: NDArray[np.float64], values: NDArray[np.float64], config: ResamplingConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate ``values`` (samples along axis 0) onto the uniform grid.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        The grid times and the interpolated values with the same trailing shape as ``values``.

    Raises
    ------
    ValueError
        If times are not strictly increasing and finite, values are not finite
        or have a different sample count, or the period is too coarse.
    """
    times = np.asarray(t_source, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim not in (1, 2) or data.shape[0] != times.shape[0]:
        msg = f"values must have shape (N,) or (N, k) with N == len(t), got {data.shape} for N={times.shape[0]}"
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "values contain non-finite entries; resampling never repairs data"
        raise ValueError(msg)
    grid = uniform_grid(times, config.period_s)
    if config.interpolation == "linear":
        if data.ndim == 1:
            out = np.interp(grid, times, data)
        else:
            out = np.column_stack([np.interp(grid, times, data[:, j]) for j in range(data.shape[1])])
        return grid, np.ascontiguousarray(out, dtype=np.float64)
    if times.shape[0] < _CUBIC_MIN_SAMPLES:
        msg = f"cubic interpolation needs at least {_CUBIC_MIN_SAMPLES} source samples, got {times.shape[0]}"
        raise ValueError(msg)
    spline = make_interp_spline(times, data, k=3, axis=0)
    out = cast("NDArray[Any]", spline(grid))
    return grid, np.ascontiguousarray(out, dtype=np.float64)


def _check_times(times: NDArray[np.float64]) -> None:
    if times.ndim != 1 or times.shape[0] < _MIN_SOURCE_SAMPLES:
        msg = f"t must be a 1-D array with at least {_MIN_SOURCE_SAMPLES} samples, got shape {times.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(times)):
        msg = "t contains non-finite values"
        raise ValueError(msg)
    if not bool(np.all(np.diff(times) > 0)):
        msg = "t must be strictly increasing"
        raise ValueError(msg)
