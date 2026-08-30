# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-006: resampling reproduces constant/linear signals and includes the endpoint within tolerance."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.resampling import ResamplingConfig, resample, uniform_grid

LINEAR = ResamplingConfig(period_s=0.01, interpolation="linear")
CUBIC = ResamplingConfig(period_s=0.01, interpolation="cubic")


def _irregular_times(n: int = 200, end: float = 1.0) -> NDArray[np.float64]:
    """Strictly increasing, jittered timestamps from 0 to ``end``."""
    rng = np.random.default_rng(0)
    steps = rng.uniform(0.5, 1.5, size=n - 1)
    t = np.concatenate([[0.0], np.cumsum(steps)])
    return t / t[-1] * end


@pytest.mark.parametrize("config", [LINEAR, CUBIC])
def test_constant_and_linear_signals_are_reproduced(config: ResamplingConfig) -> None:
    """Constants and ramps are exact (to round-off) on the new grid for both methods."""
    t = _irregular_times()
    constant = np.full_like(t, 2.5)
    ramp = 3.0 * t - 1.0
    grid, out_c = resample(t, constant, config)
    _, out_r = resample(t, ramp, config)
    assert np.allclose(out_c, 2.5, atol=1e-12)
    assert np.allclose(out_r, 3.0 * grid - 1.0, atol=1e-12)


def test_grid_starts_at_source_start_and_includes_the_endpoint() -> None:
    """A source ending at 0.30000000000000004 s still yields the 0.30 s sample."""
    t = np.cumsum(np.full(31, 0.01)) - 0.01  # accumulated round-off at the end
    assert t[-1] != 0.3
    grid = uniform_grid(t, 0.01)
    assert grid.shape == (31,)
    assert grid[0] == 0.0
    assert grid[-1] <= t[-1]
    assert abs(grid[-1] - 0.3) < 1e-9
    assert np.allclose(np.diff(grid)[:-1], 0.01, atol=1e-15)


def test_endpoint_is_excluded_when_it_is_not_reached_within_tolerance() -> None:
    """A source that stops just short of the next grid point does not extrapolate to it."""
    t = np.linspace(0.0, 0.295, 60)
    grid = uniform_grid(t, 0.01)
    assert grid[-1] == pytest.approx(0.29)
    assert grid.shape == (30,)


def test_identity_when_source_is_already_on_the_grid() -> None:
    """Resampling a uniformly sampled signal at its own period returns it unchanged."""
    t = np.arange(50, dtype=np.float64) * 0.01
    x = np.column_stack([np.sin(t), np.cos(2 * t)])
    for config in (LINEAR, CUBIC):
        grid, out = resample(t, x, config)
        assert np.allclose(grid, t, atol=1e-15)
        assert np.allclose(out, x, atol=1e-12)


def test_cubic_reproduces_a_cubic_polynomial_between_samples() -> None:
    """Cubic interpolation is exact for cubic polynomials; linear is not."""
    t = _irregular_times(n=60)
    x = t**3 - 0.5 * t**2 + 0.25 * t + 1.0
    grid, cubic = resample(t, x, CUBIC)
    _, linear = resample(t, x, LINEAR)
    exact = grid**3 - 0.5 * grid**2 + 0.25 * grid + 1.0
    assert np.allclose(cubic, exact, atol=1e-10)
    assert not np.allclose(linear, exact, atol=1e-10)


def test_columns_are_interpolated_independently() -> None:
    """Each column of a 2-D signal is resampled on its own."""
    t = _irregular_times(n=80)
    a, b = np.sin(3 * t), np.cos(5 * t)
    grid, out = resample(t, np.column_stack([a, b]), LINEAR)
    assert np.array_equal(out[:, 0], resample(t, a, LINEAR)[1])
    assert np.array_equal(out[:, 1], resample(t, b, LINEAR)[1])
    assert out.shape == (grid.shape[0], 2)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("t", "x", "config", "message"),
    [
        (np.array([0.0, 0.1, 0.1, 0.3]), np.zeros(4), LINEAR, "strictly increasing"),
        (np.array([0.0, np.nan, 0.2]), np.zeros(3), LINEAR, "t contains non-finite"),
        (np.array([0.0]), np.zeros(1), LINEAR, "at least 2 samples"),
        (np.linspace(0, 1, 10), np.zeros(9), LINEAR, r"N == len\(t\)"),
        (np.linspace(0, 1, 10), np.full(10, np.inf), LINEAR, "never repairs"),
        (np.linspace(0, 0.005, 10), np.zeros(10), LINEAR, "too coarse"),
        (np.array([0.0, 0.5, 1.0]), np.zeros(3), CUBIC, "cubic interpolation needs at least 4"),
        (np.linspace(0, 1, 10), np.zeros((10, 2, 2)), LINEAR, r"shape \(N,\) or \(N, k\)"),
    ],
)
def test_invalid_inputs_are_rejected(
    t: NDArray[np.float64], x: NDArray[np.float64], config: ResamplingConfig, message: str
) -> None:
    """Bad time bases, non-finite values, shape mismatches, and coarse periods fail."""
    with pytest.raises(ValueError, match=message):
        resample(t, x, config)


def test_configuration_is_validated() -> None:
    """The period must be positive and finite."""
    with pytest.raises(ValueError, match="period_s must be positive"):
        ResamplingConfig(period_s=0.0)
    with pytest.raises(ValueError, match="period_s must be positive"):
        uniform_grid(np.array([0.0, 1.0]), float("inf"))
