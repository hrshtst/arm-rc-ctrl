# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-008: every sample receives exactly one phase; bad or unmatched intervals fail."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.phases import PhaseAnnotationError, annotate_phases, check_annotation, intervals_from_phases
from arm_rc_ctrl.data.records import Intervals
from arm_rc_ctrl.data.samples import PHASE_DWELL, PHASE_MOVE, PHASE_PRIME

INTERVALS = Intervals(prime=(0.0, 0.1), move=(0.1, 0.25), dwell=(0.25, 0.3))
T: NDArray[np.float64] = np.arange(31, dtype=np.float64) * 0.01  # 0.00 .. 0.30


def test_every_sample_gets_exactly_one_phase_with_half_open_boundaries() -> None:
    """Prime covers [0, 0.1), move [0.1, 0.25), dwell [0.25, 0.3]; counts sum to N."""
    phase = annotate_phases(T, INTERVALS)
    assert phase.dtype == np.int64
    assert phase.shape == T.shape
    assert list(phase[:10]) == [PHASE_PRIME] * 10
    assert phase[10] == PHASE_MOVE
    assert list(phase[10:25]) == [PHASE_MOVE] * 15
    assert phase[25] == PHASE_DWELL
    assert list(phase[25:]) == [PHASE_DWELL] * 6
    counts = [int(np.count_nonzero(phase == code)) for code in (PHASE_PRIME, PHASE_MOVE, PHASE_DWELL)]
    assert sum(counts) == T.size


def test_accumulated_timestamp_error_at_the_end_is_tolerated() -> None:
    """A final timestamp a few ULPs beyond the dwell end still belongs to the dwell."""
    t: NDArray[np.float64] = T.copy()
    t[-1] = 0.3000000000000001
    assert annotate_phases(t, INTERVALS)[-1] == PHASE_DWELL
    with pytest.raises(PhaseAnnotationError, match="outside"):
        annotate_phases(t, INTERVALS, tolerance_s=0.0)


@pytest.mark.parametrize(
    ("t", "message"),
    [
        (T + 0.5, r"31 sample\(s\) fall outside \[0.0, 0.3\] s, first at index 0"),
        (np.append(T, 0.31), r"1 sample\(s\) fall outside .* first at index 31 \(t=0.31\)"),
        (np.array([-0.01, 0.0, 0.15, 0.3]), r"1 sample\(s\) fall outside .* first at index 0"),
        (np.array([], dtype=np.float64), "non-empty 1-D array"),
        (np.array([0.0, np.nan, 0.3]), "non-finite"),
        (np.array([0.15, 0.2, 0.3]), r"interval\(s\) \['prime'\] contain no samples"),
        (np.array([0.0, 0.05, 0.3]), r"interval\(s\) \['move'\] contain no samples"),
        (np.array([0.0, 0.15]), r"interval\(s\) \['dwell'\] contain no samples"),
    ],
)
def test_unmatched_samples_or_empty_intervals_fail(t: NDArray[np.float64], message: str) -> None:
    """Samples outside the intervals and intervals without samples are errors."""
    with pytest.raises(PhaseAnnotationError, match=message):
        annotate_phases(t, INTERVALS)


def test_negative_tolerance_is_rejected() -> None:
    """Tolerance must be non-negative."""
    with pytest.raises(PhaseAnnotationError, match="tolerance_s must be non-negative"):
        annotate_phases(T, INTERVALS, tolerance_s=-1e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prime": (0.0, 0.1), "move": (0.05, 0.25), "dwell": (0.25, 0.3)}, "contiguous"),  # overlapping
        ({"prime": (0.0, 0.1), "move": (0.15, 0.25), "dwell": (0.25, 0.3)}, "contiguous"),  # gap
        ({"prime": (0.0, 0.1), "move": (0.25, 0.1), "dwell": (0.25, 0.3)}, "start < end"),  # reversed
        ({"prime": (0.0, 0.1), "move": (0.1, 0.25), "dwell": (0.3, 0.25)}, "start < end"),  # reversed dwell
        ({"prime": (0.02, 0.1), "move": (0.1, 0.25), "dwell": (0.25, 0.3)}, "must start at 0.0"),
    ],
)
def test_overlapping_gapped_or_reversed_intervals_are_rejected(
    kwargs: dict[str, tuple[float, float]], message: str
) -> None:
    """Interval geometry is validated before any annotation happens."""
    with pytest.raises(ValueError, match=message):
        Intervals(**kwargs)


def test_missing_interval_is_rejected() -> None:
    """All three intervals are required."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dwell'"):
        Intervals(prime=(0.0, 0.1), move=(0.1, 0.25))  # type: ignore[call-arg]


def test_check_annotation_accepts_matching_and_rejects_drift() -> None:
    """An existing phase array must equal the derived annotation sample by sample."""
    phase = annotate_phases(T, INTERVALS)
    check_annotation(T, phase, INTERVALS)
    drifted = phase.copy()
    drifted[12] = PHASE_DWELL
    with pytest.raises(PhaseAnnotationError, match=r"1 sample\(s\) carry a phase that disagrees .* first at index 12"):
        check_annotation(T, drifted, INTERVALS)
    with pytest.raises(PhaseAnnotationError, match="phase has shape"):
        check_annotation(T, phase[:-1], INTERVALS)


def test_intervals_round_trip_through_phases() -> None:
    """Boundaries recovered from an annotation reproduce the intervals on the sample grid."""
    phase = annotate_phases(T, INTERVALS)
    recovered = intervals_from_phases(T, phase)
    assert recovered.prime == (0.0, pytest.approx(0.1))
    assert recovered.move == (pytest.approx(0.1), pytest.approx(0.25))
    assert recovered.dwell == (pytest.approx(0.25), pytest.approx(0.3))
    assert np.array_equal(annotate_phases(T, recovered), phase)


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        (np.array([0, 1, 0, 2]), "ordered prime -> move -> dwell"),
        (np.array([1, 1, 2, 2]), "no 'prime' samples"),
        (np.array([0, 0, 2, 2]), "no 'move' samples"),
        (np.array([0, 1]), "equal length"),
    ],
)
def test_intervals_from_phases_rejects_bad_arrays(phase: NDArray[np.int64], message: str) -> None:
    """Unordered, incomplete, or mismatched phase arrays cannot yield intervals."""
    t: NDArray[np.float64] = np.arange(4, dtype=np.float64) * 0.1
    with pytest.raises(PhaseAnnotationError, match=message):
        intervals_from_phases(t, phase.astype(np.int64))
