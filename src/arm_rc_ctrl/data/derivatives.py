# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Offline first and second derivatives on the uniform control-period grid.

``central``
    Second-order central differences in the interior and second-order
    one-sided formulas at both ends, so every sample is O(h^2) accurate:
    first derivative ``(x[i+1] - x[i-1]) / 2h``, second derivative
    ``(x[i+1] - 2 x[i] + x[i-1]) / h^2``, boundary second derivative
    ``(2 x0 - 5 x1 + 4 x2 - x3) / h^2`` (and mirrored at the end).
``spline``
    Analytic derivatives of a not-a-knot cubic spline through the samples.

These are offline, non-causal estimators for demonstrations only; the online
desired-derivative estimator (``docs/PLAN.md`` section 5.4) is causal and
lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

__all__ = ["DerivativeConfig", "differentiate", "first_derivative", "second_derivative"]

_MIN_SAMPLES: Final = 4


@dataclass(frozen=True)
class DerivativeConfig:
    """Derivative scheme."""

    method: Literal["central", "spline"] = "central"

    @property
    def label(self) -> str:
        """Record label of the scheme."""
        return "central-difference" if self.method == "central" else "cubic-spline"


def _check(values: NDArray[np.float64], period_s: float) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim not in (1, 2) or data.shape[0] < _MIN_SAMPLES:
        msg = f"values must have shape (N,) or (N, k) with N >= {_MIN_SAMPLES}, got {data.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "values contain non-finite entries; differentiation never repairs data"
        raise ValueError(msg)
    if not (period_s > 0 and period_s < float("inf")):
        msg = f"period_s must be positive and finite, got {period_s!r}"
        raise ValueError(msg)
    return data


def first_derivative(values: NDArray[np.float64], period_s: float, config: DerivativeConfig) -> NDArray[np.float64]:
    """First derivative along axis 0 (uniform spacing ``period_s``)."""
    data = _check(values, period_s)
    if config.method == "central":
        out = np.gradient(data, period_s, axis=0, edge_order=2)
        return np.ascontiguousarray(out, dtype=np.float64)
    return _spline_derivative(data, period_s, 1)


def second_derivative(values: NDArray[np.float64], period_s: float, config: DerivativeConfig) -> NDArray[np.float64]:
    """Second derivative along axis 0 (uniform spacing ``period_s``)."""
    data = _check(values, period_s)
    if config.method == "spline":
        return _spline_derivative(data, period_s, 2)
    h2 = period_s * period_s
    out = np.empty_like(data)
    out[1:-1] = (data[2:] - 2.0 * data[1:-1] + data[:-2]) / h2
    out[0] = (2.0 * data[0] - 5.0 * data[1] + 4.0 * data[2] - data[3]) / h2
    out[-1] = (2.0 * data[-1] - 5.0 * data[-2] + 4.0 * data[-3] - data[-4]) / h2
    return np.ascontiguousarray(out, dtype=np.float64)


def differentiate(
    values: NDArray[np.float64], period_s: float, config: DerivativeConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(first, second)`` derivatives of ``values``."""
    return first_derivative(values, period_s, config), second_derivative(values, period_s, config)


def _spline_derivative(data: NDArray[np.float64], period_s: float, order: int) -> NDArray[np.float64]:
    t = np.arange(data.shape[0], dtype=np.float64) * period_s
    spline = make_interp_spline(t, data, k=3, axis=0)
    out = cast("NDArray[Any]", spline.derivative(order)(t))
    return np.ascontiguousarray(out, dtype=np.float64)
