# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-017: dwell metrics match hand-calculated fixtures."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.metrics.dwell import (
    DwellCriteria,
    DwellMetrics,
    EndpointErrorStats,
    dwell_metrics,
    endpoint_error_stats,
    longest_run_duration,
)

TARGET = np.array([0.1, 0.45])
DIST = np.array([0.0, 0.01, 0.02, 0.005, 0.03])  # planned endpoint distances per sample
T: NDArray[np.float64] = np.arange(5, dtype=np.float64) * 0.01
TIP = np.column_stack([TARGET[0] + DIST, np.full(5, TARGET[1])])  # error along x only
DQ = np.array([[0.1, 0.0], [0.2, -0.1], [0.0, 0.3], [0.05, 0.05], [-0.4, 0.0]])


def test_endpoint_error_statistics_match_hand_calculation() -> None:
    """Mean, RMS, max, and p95 of the planned distances."""
    stats = endpoint_error_stats(TIP, TARGET)
    assert stats.samples == 5
    assert stats.mean == pytest.approx(DIST.mean())
    assert stats.rms == pytest.approx(math.sqrt(np.mean(DIST**2)))
    assert stats.max == pytest.approx(0.03)
    assert stats.p95 == pytest.approx(np.percentile(DIST, 95))  # 0.028 with linear interpolation
    assert stats.p95 == pytest.approx(0.028)


def test_in_tolerance_fraction_longest_run_and_velocity() -> None:
    """Tolerance 0.01 m: samples 0, 1, 3 inside; longest run = samples 0-1 = 0.01 s."""
    metrics = dwell_metrics(T, TIP, DQ, TARGET, tolerance=0.01)
    assert metrics.samples == 5
    assert metrics.window_s == pytest.approx(0.04)
    assert metrics.in_tolerance_fraction == pytest.approx(3 / 5)
    assert metrics.longest_in_tolerance_s == pytest.approx(0.01)
    assert metrics.velocity_max == pytest.approx(0.4)
    assert metrics.velocity_rms == pytest.approx(math.sqrt(np.mean(DQ**2)))
    assert metrics.endpoint == endpoint_error_stats(TIP, TARGET)


def test_window_selection_is_inclusive() -> None:
    """Only samples with window[0] <= t <= window[1] contribute."""
    metrics = dwell_metrics(T, TIP, DQ, TARGET, tolerance=0.01, window=(0.02, 0.03))
    assert metrics.samples == 2
    assert metrics.endpoint.max == pytest.approx(0.02)
    assert metrics.in_tolerance_fraction == pytest.approx(0.5)
    assert metrics.longest_in_tolerance_s == 0.0  # a single in-tolerance sample spans no time
    assert metrics.velocity_max == pytest.approx(0.3)


def test_longest_run_duration_cases() -> None:
    """Runs at the start, middle, end, and a single sample."""
    t: NDArray[np.float64] = np.arange(6, dtype=np.float64) * 0.1
    assert longest_run_duration(t, np.array([True, True, True, False, True, True])) == pytest.approx(0.2)
    assert longest_run_duration(t, np.array([False, True, True, True, True, False])) == pytest.approx(0.3)
    assert longest_run_duration(t, np.array([False, False, False, True, True, True])) == pytest.approx(0.2)
    assert longest_run_duration(t, np.array([False, True, False, False, False, False])) == 0.0
    assert longest_run_duration(t, np.zeros(6, dtype=bool)) == 0.0
    assert longest_run_duration(t, np.ones(6, dtype=bool)) == pytest.approx(0.5)


def test_perfect_dwell_gives_zero_error_and_full_fraction() -> None:
    """Metrics are zero/one for a stationary endpoint on the target."""
    tip = np.tile(TARGET, (5, 1))
    metrics = dwell_metrics(T, tip, np.zeros((5, 2)), TARGET, tolerance=0.01)
    assert metrics.endpoint.max == 0.0
    assert metrics.in_tolerance_fraction == 1.0
    assert metrics.longest_in_tolerance_s == pytest.approx(0.04)
    assert metrics.velocity_rms == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"t": np.array([0.0, 0.01, 0.01, 0.03, 0.04])}, "strictly increasing"),
        ({"tip": np.zeros((4, 2))}, "must have 5 rows"),
        ({"dq": np.zeros(5)}, "must have 5 rows"),
        ({"tolerance": 0.0}, "tolerance must be positive"),
        ({"window": (0.5, 0.6)}, "no samples fall inside"),
        ({"window": (0.03, 0.01)}, "start < end"),
        ({"target": np.array([0.1, 0.2, 0.3])}, r"target must have shape \(2,\)"),
        ({"tip": np.full((5, 2), np.nan)}, "tip must be finite"),
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict[str, object], message: str) -> None:
    """Time base, shapes, tolerance, window, and finiteness are validated."""
    args: dict[str, object] = {"t": T, "tip": TIP, "dq": DQ, "target": TARGET, "tolerance": 0.01}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        dwell_metrics(**args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: EndpointErrorStats(-0.1, 0.1, 0.2, 0.1, 3), "non-negative"),
        (lambda: EndpointErrorStats(0.3, 0.2, 0.2, 0.1, 3), "mean <= rms"),
        (lambda: EndpointErrorStats(0.1, 0.15, 0.2, 0.5, 3), "p95 <= max"),
        (lambda: EndpointErrorStats(0.1, 0.15, 0.2, 0.1, 0), ">= 1 sample"),
        (lambda: _metrics(in_tolerance_fraction=1.5), r"\[0, 1\]"),
        (lambda: _metrics(longest_in_tolerance_s=0.4), "cannot exceed the window"),
        (lambda: _metrics(velocity_rms=9.0), "cannot exceed velocity_max"),
        (lambda: _metrics(samples=7), "equal the endpoint sample count"),
    ],
)
def test_semantic_validation_rejects_tampered_dwell_values(build: object, message: str) -> None:
    """M3R review: negative errors, invalid fractions, and inconsistent counts cannot be constructed."""
    with pytest.raises(ValueError, match=message):
        build()  # type: ignore[operator]


def _metrics(**changes: object) -> DwellMetrics:
    base = DwellMetrics(
        endpoint=EndpointErrorStats(0.01, 0.012, 0.02, 0.018, 5),
        in_tolerance_fraction=0.8,
        longest_in_tolerance_s=0.03,
        velocity_rms=0.1,
        velocity_max=0.2,
        window_s=0.05,
        samples=5,
    )
    return dataclasses.replace(base, **changes)


def test_dwell_criteria_evaluate_fraction_and_stationarity() -> None:
    """Criteria are named booleans derived from the dwell metrics; ranges are validated."""
    metrics = dwell_metrics(T, TIP, DQ, TARGET, tolerance=0.01)  # fraction 0.6, velocity_max 0.4
    strict = DwellCriteria(tolerance=0.01, min_fraction=0.9, max_velocity=0.05)
    assert strict.evaluate(metrics) == {"dwell_in_tolerance": False, "dwell_stationary": False}
    loose = DwellCriteria(tolerance=0.01, min_fraction=0.5, max_velocity=0.5)
    assert loose.evaluate(metrics) == {"dwell_in_tolerance": True, "dwell_stationary": True}
    assert strict.names == ("dwell_in_tolerance", "dwell_stationary")
    with pytest.raises(ValueError, match=r"min_fraction must be in \[0, 1\]"):
        DwellCriteria(0.01, 1.5, 0.1)
    with pytest.raises(ValueError, match="max_velocity must be positive"):
        DwellCriteria(0.01, 0.5, 0.0)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        DwellCriteria(0.0, 0.5, 0.1)
