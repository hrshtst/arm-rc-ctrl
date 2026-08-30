# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-008: causal desired-derivative estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig, EstimatorError

DT = 0.01


def _estimator(
    *,
    max_dt_ratio: float = 3.0,
    velocity_cutoff_hz: float | None = None,
    acceleration_cutoff_hz: float | None = None,
) -> CausalDerivativeEstimator:
    config = EstimatorConfig(
        nominal_dt_s=DT,
        max_dt_ratio=max_dt_ratio,
        velocity_cutoff_hz=velocity_cutoff_hz,
        acceleration_cutoff_hz=acceleration_cutoff_hz,
    )
    return CausalDerivativeEstimator(config, dof=2)


def test_first_sample_after_reset_has_zero_derivatives() -> None:
    """The first sample emits zero raw and filtered velocity/acceleration, then reset repeats that."""
    estimator = _estimator()
    assert estimator.last is None
    first = estimator.update(0.0, np.array([0.5, -0.5]))
    assert first.dt == 0.0
    assert np.array_equal(first.q, [0.5, -0.5])
    for name in ("dq_raw", "ddq_raw", "dq", "ddq"):
        assert not getattr(first, name).any(), name
    estimator.update(DT, np.array([0.6, -0.5]))
    assert estimator.last is not None
    assert estimator.last.dq_raw[0] == pytest.approx(10.0)
    estimator.reset()
    assert estimator.last is None
    again = estimator.update(7.0, np.array([1.0, 1.0]))
    assert not again.dq_raw.any()


def test_backward_differences_are_exact_for_polynomials() -> None:
    """A linear ramp gives its slope; a quadratic gives the exact backward-difference acceleration."""
    estimator = _estimator()
    for k in range(5):  # q = 2 t on joint 0, q = -1 + 3 t on joint 1
        t = k * DT
        estimate = estimator.update(t, np.array([2.0 * t, -1.0 + 3.0 * t]))
        if k >= 1:
            assert estimate.dq_raw == pytest.approx([2.0, 3.0])
            assert estimate.dq == pytest.approx([2.0, 3.0])  # no cutoff: filtered equals raw
        if k >= 2:
            assert estimate.ddq_raw == pytest.approx([0.0, 0.0], abs=1e-9)
    estimator.reset()
    for k in range(5):  # q = 0.5 * a t^2 with a = 4: backward differences give exactly a
        t = k * DT
        estimate = estimator.update(t, np.array([2.0 * t * t, 0.0]))
        if k >= 2:
            assert estimate.ddq_raw[0] == pytest.approx(4.0)
            assert estimate.ddq[0] == pytest.approx(4.0)


def test_measured_intervals_are_used() -> None:
    """Irregular sampling divides by the actual interval, and the interval is reported."""
    estimator = _estimator(max_dt_ratio=5.0)
    estimator.update(0.0, np.zeros(2))
    estimate = estimator.update(0.03, np.array([0.3, 0.0]))
    assert estimate.dt == pytest.approx(0.03)
    assert estimate.dq_raw[0] == pytest.approx(10.0)
    estimate = estimator.update(0.035, np.array([0.31, 0.0]))
    assert estimate.dt == pytest.approx(0.005)
    assert estimate.dq_raw[0] == pytest.approx(2.0)
    assert estimate.ddq_raw[0] == pytest.approx((2.0 - 10.0) / 0.005)


def test_nonpositive_and_excessive_intervals_are_rejected_without_state_change() -> None:
    """Time standing still, going backwards, or jumping too far is an error and leaves the last estimate intact."""
    estimator = _estimator(max_dt_ratio=2.0)
    estimator.update(0.0, np.zeros(2))
    kept = estimator.update(DT, np.array([0.1, 0.0]))
    with pytest.raises(EstimatorError, match=r"sample interval must be positive, got 0\.0"):
        estimator.update(DT, np.array([0.2, 0.0]))
    with pytest.raises(EstimatorError, match="must be positive"):
        estimator.update(0.005, np.array([0.2, 0.0]))
    with pytest.raises(EstimatorError, match=r"exceeds the accepted maximum 0\.02"):
        estimator.update(DT + 0.021, np.array([0.2, 0.0]))
    assert estimator.last is kept
    with pytest.raises(EstimatorError, match="t must be finite and non-negative"):
        estimator.update(math.nan, np.zeros(2))
    with pytest.raises(ValueError, match="q must have 2 entries"):
        estimator.update(2 * DT, np.zeros(3))
    with pytest.raises(ValueError, match="q must be finite"):
        estimator.update(2 * DT, np.array([math.inf, 0.0]))
    assert estimator.last is kept


def test_first_order_filter_matches_its_analytic_step_response() -> None:
    """With a cutoff, the filtered velocity follows the backward-Euler step response 1 - (tau/(tau+dt))^n."""
    cutoff = 5.0
    estimator = _estimator(velocity_cutoff_hz=cutoff, acceleration_cutoff_hz=cutoff)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    estimator.update(0.0, np.zeros(2))
    filtered = 0.0
    for n in range(1, 40):  # constant velocity 1 rad/s from the second sample on: a unit step in dq_raw
        estimate = estimator.update(n * DT, np.array([n * DT, 0.0]))
        expected = 1.0 - (tau / (tau + DT)) ** n
        assert estimate.dq_raw[0] == pytest.approx(1.0)
        assert estimate.dq[0] == pytest.approx(expected)
        assert estimate.dq[1] == 0.0
        filtered = float(estimate.dq[0])
    assert filtered > 0.99  # converged well inside 40 samples at 5 Hz


def test_telemetry_channels_expose_raw_and_filtered_values() -> None:
    """The estimate publishes the four run-record channels."""
    estimator = _estimator(velocity_cutoff_hz=10.0)
    estimator.update(0.0, np.zeros(2))
    estimate = estimator.update(DT, np.array([0.1, 0.2]))
    channels = estimate.channels()
    assert set(channels) == {"dq_desired_raw", "ddq_desired_raw", "dq_desired", "ddq_desired"}
    assert np.array_equal(channels["dq_desired_raw"], [10.0, 20.0])
    assert channels["dq_desired"][0] < 10.0  # filtered lags the raw value
    assert np.array_equal(channels["ddq_desired"], channels["ddq_desired_raw"])  # no acceleration cutoff


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nominal_dt_s": 0.0}, "nominal_dt_s must be positive"),
        ({"nominal_dt_s": DT, "max_dt_ratio": 0.5}, "max_dt_ratio must be finite and >= 1"),
        ({"nominal_dt_s": DT, "velocity_cutoff_hz": 0.0}, "velocity_cutoff_hz must be positive"),
        ({"nominal_dt_s": DT, "acceleration_cutoff_hz": math.inf}, "acceleration_cutoff_hz must be positive"),
    ],
)
def test_config_validation(kwargs: dict[str, float], message: str) -> None:
    """Timing and cutoff parameters are validated."""
    with pytest.raises(ValueError, match=message):
        EstimatorConfig(**kwargs)
    with pytest.raises(ValueError, match="dof must be >= 1"):
        CausalDerivativeEstimator(EstimatorConfig(DT), 0)
