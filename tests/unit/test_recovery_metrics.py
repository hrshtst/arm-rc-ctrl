# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-008: pure recovery-mechanism metrics with tamper tests; movement RMSE stays diagnostic only."""

from __future__ import annotations

import dataclasses
import json
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.config import ConfigError, from_mapping
from arm_rc_ctrl.metrics.dwell import dwell_metrics
from arm_rc_ctrl.metrics.effort import effort_metrics
from arm_rc_ctrl.metrics.recovery import (
    EARLY_WINDOW_S,
    SATURATION_BOUND,
    AlignmentMetrics,
    DecayFit,
    GeneratedReferenceCriteria,
    RecoveryMetricsReport,
    SettlingMetrics,
    SmoothnessMetrics,
    WindowSummary,
    activation_jump,
    compute_recovery_metrics,
    gap_series,
    gap_summary,
    paired_summary,
    recovery_report_from_json,
    recovery_report_to_json,
    restoring_alignment,
    saturation_within_bound,
    settling_metrics,
    smoothness_metrics,
)

N = 101
DT = 0.01
T: NDArray[np.float64] = np.arange(N, dtype=np.float64) * DT


def _reference() -> NDArray[np.float64]:
    start = np.array([0.3, 0.6])
    goal = np.array([0.8, 0.4])
    s = np.clip(T / 0.8, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    return start[None, :] + blend[:, None] * (goal - start)[None, :]


def _recovering_run() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Actual motion starting offset from the reference and decaying toward it; desired points at the reference."""
    q_ref = _reference()
    offset = np.array([0.1, -0.08])
    q = q_ref + offset[None, :] * np.exp(-T / 0.2)[:, None]
    q_desired = q + 0.5 * (q_ref - q)
    return q, q_desired


def test_activation_jump_is_the_first_sample_gap() -> None:
    """The jump is the Euclidean norm between the first desired and first actual posture."""
    q, q_desired = _recovering_run()
    difference: NDArray[np.float64] = q_desired[0] - q[0]
    expected = float(np.sqrt(np.sum(difference * difference)))
    assert activation_jump(q_desired[0], q[0]) == expected
    assert activation_jump(q[0], q[0]) == 0.0
    with pytest.raises(ValueError, match="shape"):
        activation_jump(q_desired[0], q[0, :1])
    with pytest.raises(ValueError, match="finite"):
        activation_jump(np.array([np.nan, 0.0]), q[0])


def test_gap_series_and_early_window_integral() -> None:
    """The command gap is the per-sample norm; the early integral covers exactly [0, 0.5] s."""
    q, q_desired = _recovering_run()
    gap = gap_series(q_desired, q)
    expected = np.sqrt(np.sum((q_desired - q) ** 2, axis=1))
    assert np.array_equal(gap, expected)
    constant = np.full(N, 0.2)
    summary = gap_summary(T, constant, window=(0.0, EARLY_WINDOW_S))
    assert summary.integral == pytest.approx(0.2 * EARLY_WINDOW_S)
    assert summary.mean == pytest.approx(0.2)
    assert summary.max == pytest.approx(0.2)
    assert summary.samples == 51
    full = gap_summary(T, constant, window=None)
    assert full.integral == pytest.approx(0.2 * float(T[-1]))


def test_tampered_desired_positions_increase_the_gap() -> None:
    """Corrupting the generated reference is visible in the early integral."""
    q, q_desired = _recovering_run()
    honest = gap_summary(T, gap_series(q_desired, q), window=(0.0, EARLY_WINDOW_S))
    tampered = gap_summary(T, gap_series(q_desired + 0.05, q), window=(0.0, EARLY_WINDOW_S))
    assert tampered.integral > honest.integral


def test_gap_summary_rejects_bad_inputs() -> None:
    """Negative gaps, non-finite values, and windows without two samples are errors."""
    with pytest.raises(ValueError, match="negative"):
        gap_summary(T, np.full(N, -1.0), window=None)
    with pytest.raises(ValueError, match="finite"):
        gap_summary(T, np.full(N, np.nan), window=None)
    with pytest.raises(ValueError, match="window"):
        gap_summary(T, np.full(N, 0.1), window=(0.0, 0.001))


def test_restoring_alignment_signs() -> None:
    """A desired command toward the reference scores +1; away scores -1; zero directions are skipped."""
    q_ref = _reference()
    offset = np.array([0.1, -0.08])
    q = q_ref + offset[None, :]  # constant offset, never zero
    toward = q + 0.4 * (q_ref - q)
    away = q - 0.4 * (q_ref - q)
    aligned = restoring_alignment(toward, q, q_ref)
    assert aligned.mean_cosine == pytest.approx(1.0)
    assert aligned.positive_fraction == pytest.approx(1.0)
    assert aligned.samples == N
    assert aligned.skipped == 0
    opposed = restoring_alignment(away, q, q_ref)
    assert opposed.mean_cosine == pytest.approx(-1.0)
    assert opposed.positive_fraction == 0.0
    perfect = restoring_alignment(q_ref, q_ref, q_ref)
    assert perfect.samples == 0
    assert perfect.skipped == N
    assert perfect.mean_cosine is None


def test_settling_time_enters_and_remains() -> None:
    """Settling requires staying inside the band; a late excursion voids it; the decay fit recovers the rate."""
    tau_c = 0.15
    series = 0.2 * np.exp(-T / tau_c)
    band = 0.05
    result = settling_metrics(T, series, band_rad=band)
    inside = series <= band
    first = int(np.argmax(inside))
    assert result.settling_time_s == pytest.approx(float(T[first]))
    assert result.band_rad == band
    assert result.decay is not None
    assert result.decay.rate_per_s == pytest.approx(1.0 / tau_c, rel=1e-6)
    bumped = series.copy()
    bumped[-1] = 2 * band
    assert settling_metrics(T, bumped, band_rad=band).settling_time_s is None
    with pytest.raises(ValueError, match="band"):
        settling_metrics(T, series, band_rad=0.0)


def test_smoothness_of_a_quadratic_and_a_tampered_series() -> None:
    """A quadratic position has constant acceleration and (near) zero jerk; noise raises the jerk."""
    accel = np.array([0.6, -0.4])
    q = 0.5 * accel[None, :] * (T**2)[:, None]
    smooth = smoothness_metrics(T, q)
    assert smooth.accel_max == pytest.approx(float(np.max(np.abs(accel))), rel=1e-6)
    assert smooth.jerk_max == pytest.approx(0.0, abs=1e-6)
    noisy = q.copy()
    noisy[50, 0] += 0.01
    assert smoothness_metrics(T, noisy).jerk_max > smooth.jerk_max + 1.0


def test_paired_summary_medians_and_improvements() -> None:
    """Ratios are rc/replay per scenario; improving counts strict reductions."""
    summary = paired_summary((1.0, 2.0, 3.0, 4.0), (2.0, 2.0, 2.0, 2.0))
    assert summary.median_ratio == pytest.approx(1.25)
    assert summary.improving == 1
    assert summary.n == 4
    assert summary.ratios == (0.5, 1.0, 1.5, 2.0)
    with pytest.raises(ValueError, match="positive"):
        paired_summary((1.0,), (0.0,))
    with pytest.raises(ValueError, match="length"):
        paired_summary((1.0, 2.0), (1.0,))


def test_generated_reference_dwell_criteria() -> None:
    """The generated reference must dwell within 1 cm for 90% and stay below 0.05 rad/s."""
    criteria = GeneratedReferenceCriteria()
    assert criteria.tolerance_m == 0.01
    assert criteria.min_fraction == 0.9
    assert criteria.max_desired_velocity == 0.05
    target = np.array([0.2, 0.3])
    tip = np.tile(target, (N, 1)) + 0.002
    dq = np.full((N, 2), 0.01)
    metrics = dwell_metrics(T, tip, dq, target, criteria.tolerance_m, window=(0.8, 1.0))
    outcome = criteria.evaluate(metrics)
    assert outcome == {"generated_dwell_in_tolerance": True, "generated_dwell_stationary": True}
    fast = dwell_metrics(T, tip, np.full((N, 2), 0.2), target, criteria.tolerance_m, window=(0.8, 1.0))
    assert criteria.evaluate(fast)["generated_dwell_stationary"] is False
    with pytest.raises(ValueError, match="min_fraction"):
        GeneratedReferenceCriteria(min_fraction=1.5)


def test_saturation_bound_gate() -> None:
    """The 0.5% torque-saturation bound is a named constant applied to effort metrics."""
    assert SATURATION_BOUND == 0.005
    tau = np.zeros((N, 2))
    ok = effort_metrics(T, tau, (10.0, 5.0))
    assert saturation_within_bound(ok)
    tau_bad = tau.copy()
    tau_bad[:3, 0] = 10.0  # 3 of 101 samples saturated (~3%)
    bad = effort_metrics(T, tau_bad, (10.0, 5.0))
    assert not saturation_within_bound(bad)


def test_recovery_report_composes_the_pure_metrics() -> None:
    """The strict report aggregates the mechanism metrics and round-trips through canonical JSON."""
    q_ref = _reference()
    q, q_desired = _recovering_run()
    dq = np.gradient(q, DT, axis=0)
    dq_desired = np.gradient(q_desired, DT, axis=0)
    target = np.array([0.2996, 0.4482])
    tip_desired = np.tile(target, (N, 1)) + 0.001
    report = compute_recovery_metrics(
        T,
        q,
        dq,
        q_desired,
        dq_desired,
        tip_desired,
        q_ref,
        target=target,
        dwell_window=(0.8, 1.0),
        settling_band_rad=0.05,
    )
    assert report.activation_jump_rad == activation_jump(q_desired[0], q[0])
    assert report.command_gap_early.integral > 0
    assert report.command_gap_full.integral >= report.command_gap_early.integral
    assert report.reference_deviation_early.integral > 0
    assert report.alignment.mean_cosine == pytest.approx(1.0)
    assert report.reference_settling.settling_time_s is not None
    assert report.generated_dwell_criteria["generated_dwell_in_tolerance"] is True
    assert report.smoothness_desired.samples == N
    assert report.smoothness_actual.samples == N
    text = recovery_report_to_json(report)
    assert recovery_report_from_json(text) == report
    payload = cast("dict[str, object]", json.loads(text))
    payload["movement_rmse"] = 0.1  # diagnostic only: the strict schema has no such field
    with pytest.raises(ConfigError):
        from_mapping(payload, RecoveryMetricsReport)


def test_report_rejects_a_shifted_task_clock() -> None:
    """The mechanism metrics are defined from task time zero."""
    q_ref = _reference()
    q, q_desired = _recovering_run()
    dq = np.gradient(q, DT, axis=0)
    with pytest.raises(ValueError, match=r"start at 0.0"):
        compute_recovery_metrics(
            T + DT,
            q,
            dq,
            q_desired,
            dq,
            np.tile(np.array([0.2, 0.3]), (N, 1)),
            q_ref,
            target=np.array([0.2, 0.3]),
            dwell_window=(0.8, 1.0),
            settling_band_rad=0.05,
        )


def _report() -> RecoveryMetricsReport:
    q_ref = _reference()
    q, q_desired = _recovering_run()
    dq = np.gradient(q, DT, axis=0)
    dq_desired = np.gradient(q_desired, DT, axis=0)
    target = np.array([0.2996, 0.4482])
    return compute_recovery_metrics(
        T,
        q,
        dq,
        q_desired,
        dq_desired,
        np.tile(target, (N, 1)) + 0.001,
        q_ref,
        target=target,
        dwell_window=(0.8, 1.0),
        settling_band_rad=0.05,
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: WindowSummary(-0.1, 0.1, 0.2, (0.0, 0.5), 10), "non-negative"),
        (lambda: WindowSummary(0.1, 0.3, 0.2, (0.0, 0.5), 10), "max >= mean"),
        (lambda: WindowSummary(0.1, 0.1, 0.2, (0.5, 0.0), 10), "increasing"),
        (lambda: AlignmentMetrics(1.5, 0.5, 3, 0), r"\[-1, 1\]"),
        (lambda: AlignmentMetrics(None, 0.5, 3, 0), "None exactly"),
        (lambda: AlignmentMetrics(0.5, 0.5, 0, 3), "None exactly"),
        (lambda: DecayFit(float("nan"), 0.1, 5), "finite"),
        (lambda: DecayFit(1.0, -0.1, 5), "non-negative residual"),
        (lambda: SettlingMetrics(0.05, -0.1, None), "non-negative"),
        (lambda: SettlingMetrics(0.0, 0.1, None), "band_rad"),
        (lambda: SmoothnessMetrics(0.5, 0.1, 0.0, 0.0, 10), "max >= rms"),
        (lambda: SmoothnessMetrics(0.1, 0.5, -0.1, 0.0, 10), "non-negative"),
    ],
)
def test_semantic_validation_rejects_tampered_nested_metrics(build: object, message: str) -> None:
    """Every nested metric dataclass rejects negative, non-finite, or internally contradictory values."""
    with pytest.raises(ValueError, match=message):
        build()  # type: ignore[operator]


def test_derived_fields_reject_tampering() -> None:
    """Swapped windows, flipped criteria, inconsistent paired summaries, and bad jumps are rejected."""
    report = _report()
    with pytest.raises(ValueError, match="start at task time 0 inside"):
        dataclasses.replace(
            report, command_gap_early=report.command_gap_full, command_gap_full=report.command_gap_early
        )
    flipped = dict(report.generated_dwell_criteria)
    flipped["generated_dwell_in_tolerance"] = not flipped["generated_dwell_in_tolerance"]
    with pytest.raises(ValueError, match="contradict"):
        dataclasses.replace(report, generated_dwell_criteria=flipped)
    with pytest.raises(ValueError, match="activation_jump_rad"):
        dataclasses.replace(report, activation_jump_rad=-1.0)
    summary = paired_summary((1.0, 2.0), (2.0, 2.0))
    with pytest.raises(ValueError, match="contradict"):
        dataclasses.replace(summary, median_ratio=0.1)
    with pytest.raises(ValueError, match="contradict"):
        dataclasses.replace(summary, improving=2)


def test_tampered_settling_band_is_rejected_by_the_schema() -> None:
    """A report whose recorded band disagrees with its settling metrics cannot be constructed."""
    q_ref = _reference()
    q, q_desired = _recovering_run()
    dq = np.gradient(q, DT, axis=0)
    dq_desired = np.gradient(q_desired, DT, axis=0)
    target = np.array([0.2996, 0.4482])
    report = compute_recovery_metrics(
        T,
        q,
        dq,
        q_desired,
        dq_desired,
        np.tile(target, (N, 1)),
        q_ref,
        target=target,
        dwell_window=(0.8, 1.0),
        settling_band_rad=0.05,
    )
    with pytest.raises(ValueError, match="schema_version"):
        dataclasses.replace(report, schema_version=2)
