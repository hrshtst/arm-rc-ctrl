# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-002: task-time annotation is move/dwell only, starts at zero, and rejects invalid annotations."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.phases import PhaseAnnotationError
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    TaskIntervals,
    annotate_task_phases,
    check_task_annotation,
    task_intervals_from_phases,
)
from arm_rc_ctrl.data.samples import PHASE_CODES, PHASE_DWELL, PHASE_MOVE, PHASE_PRIME

TASK = TaskIntervals(move=(0.0, 0.15), dwell=(0.15, 0.2))
T: NDArray[np.float64] = np.arange(21, dtype=np.float64) * 0.01  # 0.00 .. 0.20


def test_task_phase_codes_reuse_the_global_encoding() -> None:
    """Move and dwell keep their canonical integer codes; prime has no task-time code."""
    assert TASK_PHASE_CODES == {"move": PHASE_MOVE, "dwell": PHASE_DWELL}
    assert all(PHASE_CODES[name] == code for name, code in TASK_PHASE_CODES.items())
    assert PHASE_PRIME not in TASK_PHASE_CODES.values()


def test_every_sample_gets_move_or_dwell_with_half_open_boundaries() -> None:
    """Move covers [0, 0.15), dwell [0.15, 0.2]; no sample is ever prime."""
    phase = annotate_task_phases(T, TASK)
    assert phase.dtype == np.int64
    assert phase.shape == T.shape
    assert list(phase[:15]) == [PHASE_MOVE] * 15
    assert list(phase[15:]) == [PHASE_DWELL] * 6
    assert not bool(np.any(phase == PHASE_PRIME))


def test_task_duration_is_the_dwell_end() -> None:
    """The task clock runs from zero to the dwell end."""
    assert TASK.duration_s == 0.2


def test_accumulated_timestamp_error_at_the_dwell_end_is_tolerated() -> None:
    """A final timestamp a few ULPs beyond the dwell end still belongs to the dwell."""
    t: NDArray[np.float64] = T.copy()
    t[-1] = 0.2000000000000001
    assert annotate_task_phases(t, TASK)[-1] == PHASE_DWELL
    with pytest.raises(PhaseAnnotationError, match="outside"):
        annotate_task_phases(t, TASK, tolerance_s=0.0)


@pytest.mark.parametrize(
    ("t", "message"),
    [
        (T + 0.5, r"21 sample\(s\) fall outside \[0.0, 0.2\] s, first at index 0"),
        (np.array([-0.01, 0.0, 0.1, 0.2]), r"1 sample\(s\) fall outside .* first at index 0"),
        (np.array([0.15, 0.2]), r"interval\(s\) \['move'\] contain no samples"),
        (np.array([0.0, 0.1]), r"interval\(s\) \['dwell'\] contain no samples"),
        (np.array([], dtype=np.float64), "non-empty 1-D array"),
        (np.array([0.0, np.nan, 0.2]), "non-finite"),
    ],
)
def test_unmatched_samples_or_empty_task_intervals_fail(t: NDArray[np.float64], message: str) -> None:
    """Samples outside the task intervals and intervals without samples are errors."""
    with pytest.raises(PhaseAnnotationError, match=message):
        annotate_task_phases(t, TASK)


@pytest.mark.parametrize(
    ("move", "dwell", "message"),
    [
        ((0.1, 0.15), (0.15, 0.2), r"task.move must start at 0.0"),
        ((0.0, 0.15), (0.16, 0.2), "contiguous"),
        ((0.15, 0.0), (0.15, 0.2), "start < end"),
        ((0.0, 0.15, 0.2), (0.2, 0.3), "pair"),
        ((0.0, float("nan")), (float("nan"), 0.2), "finite"),
    ],
)
def test_task_interval_invariants(move: tuple[float, ...], dwell: tuple[float, ...], message: str) -> None:
    """Task intervals must be finite, increasing, contiguous pairs starting at task time zero."""
    with pytest.raises(ValueError, match=message):
        TaskIntervals(move=move, dwell=dwell)


def test_check_task_annotation_accepts_the_derived_annotation() -> None:
    """A phase array that matches the intervals passes."""
    check_task_annotation(T, annotate_task_phases(T, TASK), TASK)


def test_check_task_annotation_rejects_prime_and_disagreements() -> None:
    """A prime code or any disagreeing sample is an invalid annotation, never absorbed."""
    phase = annotate_task_phases(T, TASK)
    tampered = phase.copy()
    tampered[0] = PHASE_PRIME
    with pytest.raises(PhaseAnnotationError, match="disagrees"):
        check_task_annotation(T, tampered, TASK)
    with pytest.raises(PhaseAnnotationError, match="shape"):
        check_task_annotation(T, phase[:-1], TASK)


def test_task_intervals_round_trip_at_grid_resolution() -> None:
    """Annotate then recover: boundaries land on the sample grid."""
    phase = annotate_task_phases(T, TASK)
    recovered = task_intervals_from_phases(T, phase)
    boundary = float(T[15])
    assert recovered == TaskIntervals(move=(0.0, boundary), dwell=(boundary, float(T[-1])))


@pytest.mark.parametrize(
    ("t", "phase", "message"),
    [
        (np.array([0.0, 0.1, 0.2]), np.array([0, 1, 2]), "only move/dwell codes"),
        (np.array([0.0, 0.1, 0.2]), np.array([2, 1, 2]), "ordered move -> dwell"),
        (np.array([0.0, 0.1, 0.2]), np.array([2, 2, 2]), r"no 'move' samples"),
        (np.array([0.0, 0.1, 0.2]), np.array([1, 1, 1]), r"no 'dwell' samples"),
        (np.array([0.0, 0.1]), np.array([1, 2, 2]), "equal length"),
    ],
)
def test_invalid_task_phase_arrays_fail(t: NDArray[np.float64], phase: NDArray[np.int64], message: str) -> None:
    """Prime codes, disorder, missing phases, and shape mismatches are rejected."""
    with pytest.raises(PhaseAnnotationError, match=message):
        task_intervals_from_phases(t, phase)


def test_task_time_must_start_at_zero() -> None:
    """A cropped episode whose clock does not start at zero cannot yield task intervals."""
    t: NDArray[np.float64] = np.array([0.01, 0.02, 0.03, 0.04])
    phase: NDArray[np.int64] = np.array([1, 1, 2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match=r"task.move must start at 0.0"):
        task_intervals_from_phases(t, phase)
