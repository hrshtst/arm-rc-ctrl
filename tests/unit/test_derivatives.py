# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-007: offline derivatives meet declared interior and boundary tolerances on analytic fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate, first_derivative, second_derivative

H = 0.01
N = 201
T: NDArray[np.float64] = np.arange(N, dtype=np.float64) * H
CENTRAL = DerivativeConfig("central")
SPLINE = DerivativeConfig("spline")
OMEGA = 2 * np.pi * 1.0  # 1 Hz test tone

# Declared tolerances (absolute), derived from the truncation errors of the schemes at h = 10 ms
# on sin(omega t) with omega = 2 pi rad/s:
#   central first derivative:  interior  h^2 omega^3 / 6  ~ 4.2e-3  -> 5e-3;  boundary (one-sided, 2nd order) -> 2e-2
#   central second derivative: interior  h^2 omega^4 / 12 ~ 1.3e-2  -> 2e-2;  boundary (4-point)              -> 2e-1
#   spline first derivative:   interior  ~ h^3 omega^4    ~ 1.6e-3  -> 2e-3;  boundary (not-a-knot)           -> 2e-2
#   spline second derivative:  interior  ~ h^2 omega^4 /12 -> 2e-2;  boundary                                 -> 3e-1
TOLERANCES = {
    ("central", 1): (5e-3, 2e-2),
    ("central", 2): (2e-2, 2e-1),
    ("spline", 1): (2e-3, 2e-2),
    ("spline", 2): (2e-2, 3e-1),
}


def _split(error: NDArray[np.float64], edge: int = 2) -> tuple[float, float]:
    """Max absolute error in the interior and at the boundary samples."""
    return float(np.max(error[edge:-edge])), float(np.max(np.concatenate([error[:edge], error[-edge:]])))


@pytest.mark.parametrize("config", [CENTRAL, SPLINE])
def test_polynomials_up_to_quadratic_are_exact(config: DerivativeConfig) -> None:
    """Constant, linear, and quadratic signals have exact derivatives everywhere (to round-off)."""
    for x, dx, ddx in (
        (np.full_like(T, 1.5), np.zeros_like(T), np.zeros_like(T)),
        (2.0 * T + 1.0, np.full_like(T, 2.0), np.zeros_like(T)),
        (3.0 * T**2 - T + 0.5, 6.0 * T - 1.0, np.full_like(T, 6.0)),
    ):
        first, second = differentiate(x, H, config)
        assert np.allclose(first, dx, atol=1e-9)
        assert np.allclose(second, ddx, atol=1e-6)


def test_cubic_polynomial_meets_second_order_error_bound() -> None:
    """For t^3 the central first derivative has the known interior error h^2 * f'''/6 = h^2."""
    x = T**3
    first = first_derivative(x, H, CENTRAL)
    interior_error = np.abs(first - 3.0 * T**2)[2:-2]
    assert np.allclose(interior_error, H**2, atol=1e-9)
    assert np.allclose(first_derivative(x, H, SPLINE)[2:-2], (3.0 * T**2)[2:-2], atol=1e-9)


@pytest.mark.parametrize("method", ["central", "spline"])
@pytest.mark.parametrize("order", [1, 2])
def test_sinusoid_meets_declared_tolerances(method: str, order: int) -> None:
    """Interior and boundary errors on sin(omega t) stay within the declared bounds."""
    config = DerivativeConfig(method)  # type: ignore[arg-type]
    x = np.sin(OMEGA * T)
    exact = OMEGA * np.cos(OMEGA * T) if order == 1 else -(OMEGA**2) * np.sin(OMEGA * T)
    got = first_derivative(x, H, config) if order == 1 else second_derivative(x, H, config)
    interior, boundary = _split(np.abs(got - exact))
    tol_interior, tol_boundary = TOLERANCES[(method, order)]
    assert interior <= tol_interior, (method, order, interior)
    assert boundary <= tol_boundary, (method, order, boundary)


def test_columns_are_differentiated_independently() -> None:
    """2-D inputs are handled column by column with float64 contiguous output."""
    x = np.column_stack([np.sin(OMEGA * T), np.cos(OMEGA * T)])
    for config in (CENTRAL, SPLINE):
        first, second = differentiate(x, H, config)
        assert np.array_equal(first[:, 0], first_derivative(x[:, 0], H, config))
        assert np.array_equal(second[:, 1], second_derivative(x[:, 1], H, config))
        assert first.dtype == second.dtype == np.float64
        assert first.flags["C_CONTIGUOUS"]
        assert second.flags["C_CONTIGUOUS"]


def test_labels_for_records() -> None:
    """Scheme labels feed the processed record."""
    assert CENTRAL.label == "central-difference"
    assert SPLINE.label == "cubic-spline"


@pytest.mark.parametrize(
    ("x", "period", "message"),
    [
        (np.zeros(3), H, "N >= 4"),
        (np.zeros((5, 2, 2)), H, r"shape \(N,\) or \(N, k\)"),
        (np.array([0.0, np.nan, 1.0, 2.0]), H, "never repairs"),
        (np.zeros(5), 0.0, "period_s must be positive"),
    ],
)
def test_invalid_inputs_are_rejected(x: NDArray[np.float64], period: float, message: str) -> None:
    """Short, mis-shaped, or non-finite inputs and bad periods fail."""
    with pytest.raises(ValueError, match=message):
        differentiate(x, period, CENTRAL)
