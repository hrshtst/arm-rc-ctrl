# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-018: effort integration on irregular grids and the saturation fraction match analytic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.metrics.effort import effort_metrics


def _irregular(n: int = 400, end: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    steps = rng.uniform(0.2, 1.8, size=n - 1)
    t = np.concatenate([[0.0], np.cumsum(steps)])
    return t / t[-1] * end


def test_effort_of_a_constant_sum_of_squares_is_exact_on_an_irregular_grid() -> None:
    """Tau = [sin t, cos t]: sum tau^2 = 1, so effort = t_end - t_0 exactly (trapezoid is exact for constants)."""
    t = _irregular()
    tau = np.column_stack([np.sin(t), np.cos(t)])
    metrics = effort_metrics(t, tau, limits=(2.0, 2.0))
    assert metrics.effort == pytest.approx(t[-1] - t[0], abs=1e-12)
    assert metrics.torque_rms == pytest.approx(np.sqrt(0.5), abs=1e-12)
    assert metrics.saturation_fraction == 0.0
    assert metrics.samples == t.shape[0]


def test_effort_of_a_linear_sum_of_squares_is_exact_on_an_irregular_grid() -> None:
    """Tau = [sqrt(t)]: sum tau^2 = t (piecewise linear), so effort = T^2 / 2 exactly."""
    t = _irregular(end=3.0)
    tau = np.sqrt(t)[:, None]
    metrics = effort_metrics(t, tau, limits=(10.0,))
    assert metrics.effort == pytest.approx(3.0**2 / 2, abs=1e-12)
    assert metrics.torque_peak == pytest.approx(np.sqrt(3.0))
    assert metrics.per_joint_peak == (pytest.approx(np.sqrt(3.0)),)


def test_peak_and_saturation_fraction_by_joint_limits() -> None:
    """Saturation counts samples where any joint reaches its own limit."""
    t = np.array([0.0, 0.1, 0.25, 0.3, 0.5])
    tau = np.array([[0.5, 0.0], [1.0, 0.0], [0.2, -2.5], [-1.2, 1.0], [0.0, 0.0]])
    metrics = effort_metrics(t, tau, limits=(1.0, 2.0))
    assert metrics.torque_peak == pytest.approx(2.5)
    assert metrics.per_joint_peak == (pytest.approx(1.2), pytest.approx(2.5))
    assert metrics.saturation_fraction == pytest.approx(3 / 5)  # samples 1, 2, 3
    expected = np.trapezoid(np.sum(tau * tau, axis=1), t)
    assert metrics.effort == pytest.approx(expected)


def test_window_selects_samples_inclusively() -> None:
    """Effort over a sub-window integrates only that span."""
    t = np.linspace(0.0, 1.0, 11)
    tau = np.ones((11, 1))
    metrics = effort_metrics(t, tau, limits=(5.0,), window=(0.2, 0.5))
    assert metrics.samples == 4
    assert metrics.effort == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"t": np.array([0.0])}, "at least 2 samples"),
        ({"t": np.array([0.0, 0.1, 0.1])}, "strictly increasing"),
        ({"tau": np.zeros((2, 2))}, r"tau must have shape \(3, dof\)"),
        ({"limits": (1.0,)}, "limits must give 2 positive finite bounds"),
        ({"limits": (1.0, 0.0)}, "positive finite bounds"),
        ({"tau": np.full((3, 2), np.inf)}, "must be finite"),
        ({"window": (0.05, 0.06)}, "fewer than 2 samples"),
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict[str, object], message: str) -> None:
    """Time base, shapes, limits, finiteness, and windows are validated."""
    args: dict[str, object] = {"t": np.array([0.0, 0.1, 0.2]), "tau": np.zeros((3, 2)), "limits": (1.0, 1.0)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        effort_metrics(**args)  # type: ignore[arg-type]
