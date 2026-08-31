# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Causal desired-derivative estimation (docs/PLAN.md section 5.4).

The target generator emits a desired position at every control sample. This
estimator turns that stream into desired velocity and acceleration with
backward differences over the *measured* sample interval, followed by
optional first-order low-pass filtering. The contract, shared with the C++
parity tests of a later phase, is:

- ``reset()`` clears all state; the first sample after a reset yields zero
  raw and filtered velocity and acceleration;
- for sample ``k`` with interval ``dt = t_k - t_(k-1)``:
  ``dq_raw = (q_k - q_(k-1)) / dt`` and ``ddq_raw = (dq_raw_k - dq_raw_(k-1)) / dt``;
- each filtered value ``y`` follows ``y_k = y_(k-1) + a (x_k - y_(k-1))`` with
  ``a = dt / (tau + dt)`` and ``tau = 1 / (2 pi f_c)`` (a backward-Euler first-order
  low-pass evaluated with the measured ``dt``); a channel without a cutoff passes
  its raw value through;
- a non-positive interval, or one above ``max_dt_ratio`` times the nominal
  period, is rejected before any state changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.controllers.contracts import as_joint_vector

__all__ = ["CausalDerivativeEstimator", "DerivativeEstimate", "EstimatorConfig", "EstimatorError"]


class EstimatorError(ValueError):
    """The sample interval violates the estimator contract (a ``stale_time`` failure when it ends a run)."""

    category = "stale_time"


@dataclass(frozen=True)
class EstimatorConfig:
    """Timing bounds and low-pass cutoffs of the estimator."""

    nominal_dt_s: float
    """Expected control period (s)."""
    max_dt_ratio: float = 3.0
    """Intervals longer than ``max_dt_ratio * nominal_dt_s`` are rejected as excessive."""
    velocity_cutoff_hz: float | None = None
    """First-order low-pass cutoff for the desired velocity; ``None`` passes the raw value."""
    acceleration_cutoff_hz: float | None = None
    """First-order low-pass cutoff for the desired acceleration; ``None`` passes the raw value."""

    def __post_init__(self) -> None:
        """Timing and cutoffs are positive and finite."""
        if not (math.isfinite(self.nominal_dt_s) and self.nominal_dt_s > 0):
            msg = f"nominal_dt_s must be positive and finite, got {self.nominal_dt_s!r}"
            raise ValueError(msg)
        if not (math.isfinite(self.max_dt_ratio) and self.max_dt_ratio >= 1):
            msg = f"max_dt_ratio must be finite and >= 1, got {self.max_dt_ratio!r}"
            raise ValueError(msg)
        for name in ("velocity_cutoff_hz", "acceleration_cutoff_hz"):
            cutoff = getattr(self, name)
            if cutoff is not None and not (math.isfinite(cutoff) and cutoff > 0):
                msg = f"{name} must be positive and finite or null, got {cutoff!r}"
                raise ValueError(msg)

    @property
    def max_dt_s(self) -> float:
        """Largest accepted sample interval (s)."""
        return self.max_dt_ratio * self.nominal_dt_s


@dataclass(frozen=True)
class DerivativeEstimate:
    """Raw and filtered desired derivatives of one sample (telemetry-ready)."""

    t: float
    dt: float
    """Measured interval to the previous sample (0 on the first sample)."""
    q: NDArray[np.float64]
    dq_raw: NDArray[np.float64]
    ddq_raw: NDArray[np.float64]
    dq: NDArray[np.float64]
    ddq: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate timing and vectors; store finite, read-only copies so telemetry consumers cannot alter state."""
        if not (math.isfinite(self.t) and self.t >= 0) or not (math.isfinite(self.dt) and self.dt >= 0):
            msg = f"t and dt must be finite and non-negative, got {self.t!r} and {self.dt!r}"
            raise ValueError(msg)
        q = as_joint_vector(self.q, "q")
        object.__setattr__(self, "q", q)
        for name in ("dq_raw", "ddq_raw", "dq", "ddq"):
            object.__setattr__(self, name, as_joint_vector(getattr(self, name), name, dof=q.shape[0]))

    def channels(self) -> dict[str, NDArray[np.float64]]:
        """Telemetry channels named as in the run record."""
        return {
            "dq_desired_raw": self.dq_raw,
            "ddq_desired_raw": self.ddq_raw,
            "dq_desired": self.dq,
            "ddq_desired": self.ddq,
        }


def _smooth(
    previous: NDArray[np.float64], raw: NDArray[np.float64], dt: float, cutoff_hz: float | None
) -> NDArray[np.float64]:
    if cutoff_hz is None:
        return raw
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (tau + dt)
    return previous + alpha * (raw - previous)


class CausalDerivativeEstimator:
    """Stateful backward-difference estimator with optional first-order filtering."""

    def __init__(self, config: EstimatorConfig, dof: int) -> None:
        if dof < 1:
            msg = f"dof must be >= 1, got {dof}"
            raise ValueError(msg)
        self._config = config
        self._dof = dof
        self._last: DerivativeEstimate | None = None

    @property
    def config(self) -> EstimatorConfig:
        """The estimator's configuration."""
        return self._config

    @property
    def dof(self) -> int:
        """Number of joints."""
        return self._dof

    @property
    def last(self) -> DerivativeEstimate | None:
        """The most recent estimate, or ``None`` after a reset."""
        return self._last

    def reset(self) -> None:
        """Forget every previous sample (episode start)."""
        self._last = None

    def update(self, t: float, q: NDArray[np.float64]) -> DerivativeEstimate:
        """Consume the desired position at time ``t`` and return the derivative estimate."""
        if not (math.isfinite(t) and t >= 0):
            msg = f"t must be finite and non-negative, got {t!r}"
            raise EstimatorError(msg)
        position = as_joint_vector(q, "q", dof=self._dof)
        previous = self._last
        if previous is None:
            zeros = [np.zeros(self._dof, dtype=np.float64) for _ in range(4)]  # distinct arrays, copied and frozen
            estimate = DerivativeEstimate(t, 0.0, position, *zeros)
        else:
            dt = t - previous.t
            if dt <= 0:
                msg = f"sample interval must be positive, got {dt!r} s (t = {t!r} after {previous.t!r})"
                raise EstimatorError(msg)
            if dt > self._config.max_dt_s:
                msg = f"sample interval {dt!r} s exceeds the accepted maximum {self._config.max_dt_s!r} s"
                raise EstimatorError(msg)
            dq_raw = (position - previous.q) / dt
            ddq_raw = (dq_raw - previous.dq_raw) / dt
            dq = _smooth(previous.dq, dq_raw, dt, self._config.velocity_cutoff_hz)
            ddq = _smooth(previous.ddq, ddq_raw, dt, self._config.acceleration_cutoff_hz)
            estimate = DerivativeEstimate(t, dt, position, dq_raw, ddq_raw, dq, ddq)
        self._last = estimate
        return estimate
