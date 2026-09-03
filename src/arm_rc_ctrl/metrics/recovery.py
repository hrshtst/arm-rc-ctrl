# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Recovery-mechanism metrics on the task clock (M3R-008; recovery plan sections 7.2 and 7.3).

Pure functions evaluate the state-conditioned recovery mechanism from task time
zero: the activation command jump, command-gap and reference-deviation window
summaries (the first-0.5-s integral is the primary early quantity), restoring
alignment, settling into a declared band with a fitted decay diagnostic,
generated-reference dwell criteria (1 cm for at least 90% of the dwell with
every desired joint below 0.05 rad/s), desired/actual smoothness, the 0.5%
torque-saturation bound, and paired rc/replay ratios. Values are exactly what
the pure functions return; the strict :class:`RecoveryMetricsReport` composes
them and rejects unknown fields on load.

Time-aligned movement RMSE against the original demonstration remains a
diagnostic (computed with :mod:`arm_rc_ctrl.metrics.joint` as in M3) and is
deliberately **not** part of this report or any freeze criterion.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.metrics.dwell import DwellMetrics, dwell_metrics
from arm_rc_ctrl.metrics.effort import EffortMetrics
from arm_rc_ctrl.provenance import canonical_json

__all__ = [
    "EARLY_WINDOW_S",
    "RECOVERY_REPORT_SCHEMA_VERSION",
    "SATURATION_BOUND",
    "AlignmentMetrics",
    "DecayFit",
    "GeneratedReferenceCriteria",
    "PairedSummary",
    "RecoveryMetricsReport",
    "SettlingMetrics",
    "SmoothnessMetrics",
    "WindowSummary",
    "activation_jump",
    "compute_recovery_metrics",
    "gap_series",
    "gap_summary",
    "paired_summary",
    "recovery_report_from_json",
    "recovery_report_to_json",
    "restoring_alignment",
    "saturation_within_bound",
    "settling_metrics",
    "smoothness_metrics",
]

RECOVERY_REPORT_SCHEMA_VERSION: Final = 1
EARLY_WINDOW_S: Final = 0.5
"""The early command-gap window ``[0, 0.5]`` s after activation (plan section 7.2)."""
SATURATION_BOUND: Final = 0.005
"""A development run is ineligible above this torque-saturation fraction (plan section 6)."""

_DIRECTION_EPS: Final = 1e-12
_DECAY_EPS: Final = 1e-15
_MIN_WINDOW_SAMPLES: Final = 2
_GRID_TOLERANCE_S: Final = 1e-9


def _vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] == 0:
        msg = f"{name} must be a non-empty 1-D vector, got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return array


def _matrix(values: NDArray[np.float64], name: str, rows: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or (rows is not None and array.shape[0] != rows):  # noqa: PLR2004
        expected = f"({rows}, dof)" if rows is not None else "(N, dof)"
        msg = f"{name} must have shape {expected}, got {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return array


def _norms(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.ascontiguousarray(np.sqrt(np.sum(values * values, axis=1)), dtype=np.float64)


def activation_jump(q_desired0: NDArray[np.float64], q0: NDArray[np.float64]) -> float:
    """``norm(q_desired[0] - q[0])`` — the command jump at task activation (rad)."""
    desired = _vector(q_desired0, "q_desired0")
    actual = _vector(q0, "q0")
    if desired.shape != actual.shape:
        msg = f"q_desired0 and q0 must share one shape, got {desired.shape} and {actual.shape}"
        raise ValueError(msg)
    difference = desired - actual
    return float(np.sqrt(np.sum(difference * difference)))


def gap_series(q_desired: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-sample Euclidean norm ``norm(q_desired_k - q_k)`` (rad)."""
    desired = _matrix(q_desired, "q_desired")
    actual = _matrix(q, "q", desired.shape[0])
    if desired.shape != actual.shape:
        msg = f"q_desired and q must share one shape, got {desired.shape} and {actual.shape}"
        raise ValueError(msg)
    return _norms(desired - actual)


@dataclass(frozen=True)
class WindowSummary:
    """Integral, mean, and maximum of a non-negative series over one time window."""

    integral: float
    mean: float
    max: float
    window: tuple[float, ...]
    samples: int


def gap_summary(
    t: NDArray[np.float64], gap: NDArray[np.float64], *, window: tuple[float, float] | None
) -> WindowSummary:
    """Trapezoidal integral plus mean/max of ``gap`` over ``window`` (the whole run when ``None``)."""
    times = _vector(t, "t")
    series = np.asarray(gap, dtype=np.float64)
    if series.shape != times.shape:
        msg = f"gap must have shape {times.shape}, got {series.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(series)):
        msg = "gap must be finite"
        raise ValueError(msg)
    if bool(np.any(series < 0)):
        msg = "gap must be non-negative"
        raise ValueError(msg)
    if not np.all(np.diff(times) > 0):
        msg = "t must be strictly increasing"
        raise ValueError(msg)
    if window is None:
        mask = np.ones(times.shape[0], dtype=np.bool_)
        bounds = (float(times[0]), float(times[-1]))
    else:
        lo, hi = window
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            msg = f"window must be a finite [start, end] with start < end, got {window}"
            raise ValueError(msg)
        mask = (times >= lo) & (times <= hi)
        bounds = (float(lo), float(hi))
    if int(np.count_nonzero(mask)) < _MIN_WINDOW_SAMPLES:
        msg = f"fewer than {_MIN_WINDOW_SAMPLES} samples fall inside the window {window}"
        raise ValueError(msg)
    sel_t, sel = times[mask], series[mask]
    return WindowSummary(
        integral=float(np.trapezoid(sel, sel_t)),
        mean=float(np.mean(sel)),
        max=float(np.max(sel)),
        window=bounds,
        samples=int(sel.shape[0]),
    )


@dataclass(frozen=True)
class AlignmentMetrics:
    """Cosine alignment of the desired command direction with the direction toward the reference."""

    mean_cosine: float | None
    positive_fraction: float | None
    samples: int
    """Samples where both directions were non-degenerate."""
    skipped: int
    """Samples excluded because either direction was (numerically) zero."""


def restoring_alignment(
    q_desired: NDArray[np.float64], q: NDArray[np.float64], q_ref: NDArray[np.float64]
) -> AlignmentMetrics:
    """Cosine between ``q_desired - q`` and ``q_ref - q`` per sample (degenerate directions skipped)."""
    desired = _matrix(q_desired, "q_desired")
    actual = _matrix(q, "q", desired.shape[0])
    reference = _matrix(q_ref, "q_ref", desired.shape[0])
    command = desired - actual
    restoring = reference - actual
    n1 = _norms(command)
    n2 = _norms(restoring)
    valid = (n1 > _DIRECTION_EPS) & (n2 > _DIRECTION_EPS)
    skipped = int(np.count_nonzero(~valid))
    if not bool(valid.any()):
        return AlignmentMetrics(mean_cosine=None, positive_fraction=None, samples=0, skipped=skipped)
    cosine = np.sum(command[valid] * restoring[valid], axis=1) / (n1[valid] * n2[valid])
    return AlignmentMetrics(
        mean_cosine=float(np.mean(cosine)),
        positive_fraction=float(np.mean(cosine > 0)),
        samples=int(np.count_nonzero(valid)),
        skipped=skipped,
    )


@dataclass(frozen=True)
class DecayFit:
    """Least-squares line fit of ``ln(series)`` against time (the contraction-rate diagnostic)."""

    rate_per_s: float
    """``-slope``: positive for a decaying series."""
    log_residual_rms: float
    samples: int


@dataclass(frozen=True)
class SettlingMetrics:
    """When the series enters and remains within the declared band, with the decay diagnostic."""

    band_rad: float
    settling_time_s: float | None
    """First time from which every later sample stays inside the band (``None`` when it never settles)."""
    decay: DecayFit | None


def _line_fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Closed-form least-squares slope and intercept (no BLAS)."""
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    dx = x - mean_x
    variance = float(np.sum(dx * dx))
    slope = float(np.sum(dx * (y - mean_y)) / variance) if variance > 0 else 0.0
    return slope, mean_y - slope * mean_x


def settling_metrics(t: NDArray[np.float64], series: NDArray[np.float64], *, band_rad: float) -> SettlingMetrics:
    """Enter-and-remain settling time of a non-negative deviation series, plus its decay fit."""
    times = _vector(t, "t")
    values = np.asarray(series, dtype=np.float64)
    if values.shape != times.shape:
        msg = f"series must have shape {times.shape}, got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or bool(np.any(values < 0)):
        msg = "series must be finite and non-negative"
        raise ValueError(msg)
    if not (band_rad > 0 and np.isfinite(band_rad)):
        msg = f"band_rad must be positive and finite, got {band_rad!r}"
        raise ValueError(msg)
    inside = values <= band_rad
    settled: float | None = None
    settle_index = values.shape[0]
    if bool(inside[-1]):
        outside = np.argwhere(~inside).ravel()
        settle_index = int(outside[-1]) + 1 if outside.size else 0
        settled = float(times[settle_index])
    fit: DecayFit | None = None
    fit_mask = values > _DECAY_EPS
    fit_mask[settle_index:] = False
    if int(np.count_nonzero(fit_mask)) >= _MIN_WINDOW_SAMPLES:
        log_values = np.log(values[fit_mask])
        slope, intercept = _line_fit(times[fit_mask], log_values)
        residual = log_values - (slope * times[fit_mask] + intercept)
        fit = DecayFit(
            rate_per_s=-slope,
            log_residual_rms=float(np.sqrt(np.mean(residual * residual))),
            samples=int(np.count_nonzero(fit_mask)),
        )
    return SettlingMetrics(band_rad=band_rad, settling_time_s=settled, decay=fit)


@dataclass(frozen=True)
class SmoothnessMetrics:
    """Finite-difference acceleration and jerk statistics of a position series."""

    accel_rms: float
    accel_max: float
    jerk_rms: float
    jerk_max: float
    samples: int


def smoothness_metrics(t: NDArray[np.float64], q: NDArray[np.float64]) -> SmoothnessMetrics:
    """Second/third finite differences of ``q`` on the uniform grid ``t``."""
    times = _vector(t, "t")
    positions = _matrix(q, "q", times.shape[0])
    steps = np.diff(times)
    if times.shape[0] < 4 or not np.all(steps > 0):  # noqa: PLR2004
        msg = "t must be strictly increasing with at least 4 samples"
        raise ValueError(msg)
    dt = float(steps[0])
    if float(np.max(np.abs(steps - dt))) > _GRID_TOLERANCE_S:
        msg = "t must be uniformly sampled for finite-difference smoothness"
        raise ValueError(msg)
    accel = np.diff(positions, n=2, axis=0) / dt**2
    jerk = np.diff(positions, n=3, axis=0) / dt**3
    return SmoothnessMetrics(
        accel_rms=float(np.sqrt(np.mean(accel * accel))),
        accel_max=float(np.max(np.abs(accel))),
        jerk_rms=float(np.sqrt(np.mean(jerk * jerk))),
        jerk_max=float(np.max(np.abs(jerk))),
        samples=int(times.shape[0]),
    )


@dataclass(frozen=True)
class PairedSummary:
    """Per-scenario rc/replay ratios of one metric with the class-level summaries used by the freeze rule."""

    ratios: tuple[float, ...]
    median_ratio: float
    improving: int
    """Scenarios where the rc value is strictly below replay's."""
    n: int


def paired_summary(rc: tuple[float, ...], replay: tuple[float, ...]) -> PairedSummary:
    """Ratios ``rc/replay`` per paired scenario; replay values must be positive."""
    if len(rc) != len(replay) or not rc:
        msg = f"rc and replay must have the same non-zero length, got {len(rc)} and {len(replay)}"
        raise ValueError(msg)
    values = (*rc, *replay)
    if any(not np.isfinite(v) or v < 0 for v in values):
        msg = "paired values must be finite and non-negative"
        raise ValueError(msg)
    if any(v <= 0 for v in replay):
        msg = "replay values must be positive to define ratios"
        raise ValueError(msg)
    ratios = tuple(a / b for a, b in zip(rc, replay, strict=True))
    return PairedSummary(
        ratios=ratios,
        median_ratio=float(statistics.median(ratios)),
        improving=sum(1 for a, b in zip(rc, replay, strict=True) if a < b),
        n=len(ratios),
    )


@dataclass(frozen=True)
class GeneratedReferenceCriteria:
    """Dwell gates of the generated reference (plan section 7.3): 1 cm, 90%, and 0.05 rad/s."""

    tolerance_m: float = 0.01
    min_fraction: float = 0.9
    max_desired_velocity: float = 0.05

    def __post_init__(self) -> None:
        """Validate ranges."""
        if not (self.tolerance_m > 0 and np.isfinite(self.tolerance_m)):
            msg = f"tolerance_m must be positive and finite, got {self.tolerance_m!r}"
            raise ValueError(msg)
        if not 0.0 <= self.min_fraction <= 1.0:
            msg = f"min_fraction must be in [0, 1], got {self.min_fraction!r}"
            raise ValueError(msg)
        if not (self.max_desired_velocity > 0 and np.isfinite(self.max_desired_velocity)):
            msg = f"max_desired_velocity must be positive and finite, got {self.max_desired_velocity!r}"
            raise ValueError(msg)

    def evaluate(self, metrics: DwellMetrics) -> dict[str, bool]:
        """Named generated-reference criteria."""
        return {
            "generated_dwell_in_tolerance": metrics.in_tolerance_fraction >= self.min_fraction,
            "generated_dwell_stationary": metrics.velocity_max <= self.max_desired_velocity,
        }


def saturation_within_bound(effort: EffortMetrics, *, bound: float = SATURATION_BOUND) -> bool:
    """Whether the run stays at or below the torque-saturation eligibility bound."""
    return effort.saturation_fraction <= bound


@dataclass(frozen=True)
class RecoveryMetricsReport:
    """Strict aggregate of the recovery-mechanism metrics of one task-time run segment."""

    activation_jump_rad: float
    command_gap_early: WindowSummary
    command_gap_full: WindowSummary
    reference_deviation_early: WindowSummary
    reference_deviation_full: WindowSummary
    alignment: AlignmentMetrics
    reference_settling: SettlingMetrics
    generated_dwell: DwellMetrics
    generated_dwell_criteria: dict[str, bool]
    smoothness_desired: SmoothnessMetrics
    smoothness_actual: SmoothnessMetrics
    schema_version: int = RECOVERY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require the supported schema version."""
        if self.schema_version != RECOVERY_REPORT_SCHEMA_VERSION:
            msg = (
                f"unsupported recovery report schema_version {self.schema_version}; "
                f"expected {RECOVERY_REPORT_SCHEMA_VERSION}"
            )
            raise ValueError(msg)


def compute_recovery_metrics(
    t: NDArray[np.float64],
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    q_desired: NDArray[np.float64],
    dq_desired: NDArray[np.float64],
    tip_desired: NDArray[np.float64],
    q_ref: NDArray[np.float64],
    *,
    target: NDArray[np.float64],
    dwell_window: tuple[float, float],
    settling_band_rad: float,
    criteria: GeneratedReferenceCriteria | None = None,
    early_window_s: float = EARLY_WINDOW_S,
) -> RecoveryMetricsReport:
    """Compose the pure mechanism metrics of one active task segment (task clock starting at zero)."""
    del dq  # reserved for future actual-motion metrics; the actual dwell gate uses the M3 report
    times = _vector(t, "t")
    if float(times[0]) != 0.0:
        msg = f"the task clock must start at 0.0 s, got {float(times[0])!r}"
        raise ValueError(msg)
    gates = criteria if criteria is not None else GeneratedReferenceCriteria()
    command_gap = gap_series(q_desired, q)
    deviation = gap_series(q_desired, q_ref)
    generated_dwell = dwell_metrics(
        times, tip_desired, dq_desired, np.asarray(target, dtype=np.float64), gates.tolerance_m, window=dwell_window
    )
    return RecoveryMetricsReport(
        activation_jump_rad=activation_jump(np.asarray(q_desired)[0], np.asarray(q)[0]),
        command_gap_early=gap_summary(times, command_gap, window=(0.0, early_window_s)),
        command_gap_full=gap_summary(times, command_gap, window=None),
        reference_deviation_early=gap_summary(times, deviation, window=(0.0, early_window_s)),
        reference_deviation_full=gap_summary(times, deviation, window=None),
        alignment=restoring_alignment(q_desired, q, q_ref),
        reference_settling=settling_metrics(times, deviation, band_rad=settling_band_rad),
        generated_dwell=generated_dwell,
        generated_dwell_criteria=gates.evaluate(generated_dwell),
        smoothness_desired=smoothness_metrics(times, q_desired),
        smoothness_actual=smoothness_metrics(times, q),
        schema_version=RECOVERY_REPORT_SCHEMA_VERSION,
    )


def recovery_report_to_json(report: RecoveryMetricsReport) -> str:
    """Canonical JSON of the report (loads back with ``from_mapping``)."""
    return canonical_json(to_mapping(report))


def recovery_report_from_json(text: str) -> RecoveryMetricsReport:
    """Strictly rebuild a recovery report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(text)), RecoveryMetricsReport)
