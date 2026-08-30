# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Prime/move/dwell annotation of sampled time (``docs/PLAN.md`` section 5.3).

A demonstration carries three contiguous intervals in seconds (see
:class:`~arm_rc_ctrl.data.records.Intervals`, which already rejects missing,
overlapping, reversed, or non-contiguous intervals). :func:`annotate_phases`
assigns every sample exactly one phase code from its timestamp; a sample that
falls outside the intervals or an interval that contains no sample is an
error, never silently absorbed.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.records import Intervals
from arm_rc_ctrl.data.samples import PHASE_CODES, PHASE_DWELL, PHASE_MOVE, PHASE_PRIME

__all__ = [
    "DEFAULT_TOLERANCE_S",
    "PhaseAnnotationError",
    "annotate_phases",
    "check_annotation",
    "intervals_from_phases",
]

DEFAULT_TOLERANCE_S: Final = 1e-9
"""Slack applied at the outer boundaries (start of prime, end of dwell) for accumulated timestamp error."""

_ORDER: Final = (PHASE_PRIME, PHASE_MOVE, PHASE_DWELL)


class PhaseAnnotationError(ValueError):
    """Samples and intervals cannot be reconciled one-to-one."""


def annotate_phases(
    t: NDArray[np.float64], intervals: Intervals, *, tolerance_s: float = DEFAULT_TOLERANCE_S
) -> NDArray[np.int64]:
    """Assign one phase code per sample by timestamp.

    Boundaries are half-open, ``[start, end)``, except that the dwell end is
    inclusive (within ``tolerance_s``) so the final sample belongs to the dwell.

    Raises
    ------
    PhaseAnnotationError
        If any sample lies outside the intervals or any interval has no sample.
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        msg = f"t must be a non-empty 1-D array, got shape {times.shape}"
        raise PhaseAnnotationError(msg)
    if not np.all(np.isfinite(times)):
        msg = "t contains non-finite values"
        raise PhaseAnnotationError(msg)
    if tolerance_s < 0:
        msg = f"tolerance_s must be non-negative, got {tolerance_s}"
        raise PhaseAnnotationError(msg)
    start, end = intervals.prime[0], intervals.dwell[1]
    outside = np.argwhere((times < start - tolerance_s) | (times > end + tolerance_s)).ravel()
    if outside.size:
        first = int(outside[0])
        msg = (
            f"{outside.size} sample(s) fall outside [{start}, {end}] s, first at index {first} "
            f"(t={float(times[first])!r})"
        )
        raise PhaseAnnotationError(msg)
    phase = np.full(times.shape, PHASE_DWELL, dtype=np.int64)
    phase[times < intervals.move[0]] = PHASE_PRIME
    phase[(times >= intervals.move[0]) & (times < intervals.dwell[0])] = PHASE_MOVE
    empty = [name for name, code in PHASE_CODES.items() if not np.any(phase == code)]
    if empty:
        msg = f"interval(s) {empty} contain no samples"
        raise PhaseAnnotationError(msg)
    return phase


def check_annotation(
    t: NDArray[np.float64], phase: NDArray[np.int64], intervals: Intervals, *, tolerance_s: float = DEFAULT_TOLERANCE_S
) -> None:
    """Verify an existing ``phase`` array equals the annotation derived from ``intervals``."""
    expected = annotate_phases(t, intervals, tolerance_s=tolerance_s)
    given = np.asarray(phase)
    if given.shape != expected.shape:
        msg = f"phase has shape {given.shape}, expected {expected.shape}"
        raise PhaseAnnotationError(msg)
    mismatch = np.argwhere(given != expected).ravel()
    if mismatch.size:
        first = int(mismatch[0])
        msg = f"{mismatch.size} sample(s) carry a phase that disagrees with the intervals, first at index {first}"
        raise PhaseAnnotationError(msg)


def intervals_from_phases(t: NDArray[np.float64], phase: NDArray[np.int64]) -> Intervals:
    """Recover interval boundaries from an ordered phase array (sample-grid resolution).

    The prime interval starts at ``t[0]``; each later interval starts at the
    first sample of its phase; the dwell ends at ``t[-1]``.
    """
    times = np.asarray(t, dtype=np.float64)
    codes = np.asarray(phase, dtype=np.int64)
    if times.shape != codes.shape or times.ndim != 1 or times.size == 0:
        msg = f"t and phase must be non-empty 1-D arrays of equal length, got {times.shape} and {codes.shape}"
        raise PhaseAnnotationError(msg)
    if not np.all(np.diff(codes) >= 0):
        msg = "phase must be ordered prime -> move -> dwell"
        raise PhaseAnnotationError(msg)
    starts: list[float] = []
    for code in _ORDER:
        where = np.argwhere(codes == code).ravel()
        if where.size == 0:
            name = next(k for k, v in PHASE_CODES.items() if v == code)
            msg = f"phase array has no {name!r} samples"
            raise PhaseAnnotationError(msg)
        starts.append(float(times[int(where[0])]))
    end = float(times[-1])
    return Intervals(prime=(starts[0], starts[1]), move=(starts[1], starts[2]), dwell=(starts[2], end))
