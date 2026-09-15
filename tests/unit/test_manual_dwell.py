# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-002: the continuous final-dwell predicate (1 s inside 1 cm at joint speeds <= 0.05 rad/s)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.manual import DwellPredicate, continuous_dwell

PREDICATE = DwellPredicate(tolerance_m=0.01, max_velocity_rad_s=0.05, min_duration_s=1.0, min_samples=101)
TARGET: NDArray[np.float64] = np.array([0.10, 0.45])
DT = 0.01


Arrays = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _run(n: int, inside_from: int, *, excursions: dict[int, str] | None = None) -> Arrays:
    """A grid of ``n`` samples whose tip is on target and still from ``inside_from`` on, with optional excursions."""
    t: NDArray[np.float64] = np.arange(n, dtype=np.float64) * DT
    tip: NDArray[np.float64] = np.tile(TARGET + np.array([0.05, 0.0]), (n, 1))  # 5 cm away before arriving
    dq: NDArray[np.float64] = np.full((n, 2), 0.5)
    tip[inside_from:] = TARGET
    dq[inside_from:] = 0.0
    for index, kind in (excursions or {}).items():
        if kind == "position":
            tip[index] = TARGET + np.array([0.011, 0.0])
        else:
            dq[index, 1] = 0.051
    return t, tip, dq


def test_exactly_101_final_samples_qualify() -> None:
    """One second at 100 Hz is 101 consecutive samples ending at the last sample."""
    t, tip, dq = _run(300, 199)
    dwell = continuous_dwell(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert dwell.ok
    assert dwell.final_samples == 101
    assert dwell.final_duration_s == pytest.approx(1.0)
    assert dwell.start_s == pytest.approx(t[199])
    assert dwell.end_s == pytest.approx(t[-1])
    assert dwell.max_endpoint_error_m == 0.0
    assert dwell.max_joint_speed_rad_s == 0.0
    assert dwell.earliest_start_s == pytest.approx(t[199])


def test_100_final_samples_do_not_qualify() -> None:
    """Off by one: 0.99 s of dwell is not a second."""
    t, tip, dq = _run(300, 200)
    dwell = continuous_dwell(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert not dwell.ok
    assert dwell.final_samples == 100
    assert dwell.final_duration_s == pytest.approx(0.99)
    assert dwell.earliest_start_s is None


@pytest.mark.parametrize("kind", ["position", "velocity"])
def test_any_excursion_restarts_the_timer(kind: str) -> None:
    """A single sample outside 1 cm or above 0.05 rad/s on any joint ends the run."""
    t, tip, dq = _run(400, 100, excursions={349: kind})
    dwell = continuous_dwell(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert not dwell.ok
    assert dwell.final_samples == 50  # samples 350..399
    assert dwell.longest_samples == 249  # samples 100..348 qualified earlier
    assert dwell.earliest_start_s == pytest.approx(t[100])  # a second was reached before the excursion


def test_the_tolerance_boundary_counts_as_inside() -> None:
    """Exactly 1 cm and exactly 0.05 rad/s satisfy the predicate (closed bounds)."""
    t, tip, dq = _run(200, 50)
    tip[50:] = TARGET + np.array([0.01, 0.0])
    dq[50:, 0] = 0.05
    dwell = continuous_dwell(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert dwell.ok
    assert dwell.max_endpoint_error_m == pytest.approx(0.01)
    assert dwell.max_joint_speed_rad_s == pytest.approx(0.05)


def test_predicate_requires_consistent_samples_and_duration() -> None:
    """The sample count must be the duration on the grid plus one; malformed inputs are refused."""
    with pytest.raises(ValueError, match="min_samples"):
        DwellPredicate(tolerance_m=0.01, max_velocity_rad_s=0.05, min_duration_s=1.0, min_samples=0)
    t, tip, dq = _run(10, 5)
    with pytest.raises(ValueError, match="shape"):
        continuous_dwell(t, tip[:5], dq, target=TARGET, predicate=PREDICATE)
